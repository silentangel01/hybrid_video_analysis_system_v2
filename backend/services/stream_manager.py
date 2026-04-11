# backend/services/stream_manager.py
"""
StreamManager - RTSP stream lifecycle management with MongoDB persistence.

Responsibilities:
  - add / remove / update RTSP streams at runtime
  - persist stream configs to MongoDB `streams` collection
  - restore streams on startup
  - route captured frames to per-stream StreamRuntime
"""

import time
import os
import sys
import logging
import threading
import subprocess
from typing import Dict, List, Optional

from backend.utils.frame_capture import VideoFrameCapture
from backend.services.parking_zone_checker import load_zones_from_file
from backend.services.stream_runtime import StreamRuntimeFactory, StreamRuntime

logger = logging.getLogger(__name__)


# LEGACY: the previous shared-dispatcher mode used a lightweight StreamConfig and
# a global TaskDispatcher. It is kept here as comments for later comparison.
# from dataclasses import dataclass, field
# from backend.services.task_dispatcher import TaskDispatcher
#
# @dataclass
# class StreamConfig:
#     stream_id: str
#     url: str
#     tasks: List[str]
#     camera_id: str
#     status: str = "connecting"  # running / stopped / error / connecting
#     capture: Optional[VideoFrameCapture] = field(default=None, repr=False)
#     created_at: float = field(default_factory=time.time)


class StreamManager:
    VALID_TASKS = {"parking_violation", "smoke_flame", "common_space"}

    def __init__(self, runtime_factory: StreamRuntimeFactory, mongo_client):
        self.streams: Dict[str, StreamRuntime] = {}
        self.runtime_factory = runtime_factory
        self.mongo = mongo_client
        self.zone_checker = runtime_factory.resources.zone_checker
        self.zone_config_path = getattr(self.zone_checker, "config_path", None)
        self._streams_col = mongo_client.db["streams"] if mongo_client is not None and mongo_client.db is not None else None
        self.lock = threading.Lock()
        self.zone_lock = threading.Lock()
        self._counter = 0

        # LEGACY:
        # self.services = services
        # self.dispatcher = TaskDispatcher(services)

        self._restore_streams()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_stream(self, url: str, tasks: List[str], camera_id: Optional[str] = None) -> str:
        """Add a new RTSP stream. Returns stream_id."""
        tasks = [t for t in tasks if t in self.VALID_TASKS]
        if not tasks:
            raise ValueError(f"No valid tasks provided. Choose from: {self.VALID_TASKS}")

        if "parking_violation" in tasks and not camera_id:
            raise ValueError("camera_id is required when tasks include parking_violation")

        with self.lock:
            self._counter += 1
            stream_id = f"stream_{self._counter}"
            while stream_id in self.streams:
                self._counter += 1
                stream_id = f"stream_{self._counter}"

        # LEGACY:
        # config = StreamConfig(
        #     stream_id=stream_id,
        #     url=url,
        #     tasks=tasks,
        #     status="connecting",
        #     camera_id=camera_id or stream_id,
        #     created_at=time.time(),
        # )

        if "parking_violation" in tasks:
            self._ensure_zone_ready(camera_id or stream_id, url)

        runtime = self.runtime_factory.create_runtime(
            stream_id=stream_id,
            url=url,
            tasks=tasks,
            camera_id=camera_id or stream_id,
            created_at=time.time(),
        )

        with self.lock:
            self.streams[stream_id] = runtime

        try:
            self._start_capture(runtime)
        except Exception:
            with self.lock:
                self.streams.pop(stream_id, None)
            raise

        self._persist_stream(runtime)
        logger.info(f"[StreamManager] Added {stream_id}: {url} | camera_id={runtime.camera_id} | tasks={tasks}")
        return stream_id

    def _reload_zone_config(self):
        """Reload zones from the same config file used by the shared zone_checker instance."""
        if self.zone_config_path:
            new_zones = load_zones_from_file(self.zone_config_path)
        else:
            new_zones = load_zones_from_file()

        self.zone_checker.zones = new_zones

        # LEGACY:
        # parking_service = self.services.get("parking_service")
        # if parking_service and getattr(parking_service, "zone_checker", None):
        #     parking_service.zone_checker.zones = new_zones

        logger.info(
            f"[StreamManager] Reloaded {len(new_zones)} zone key(s) from "
            f"{self.zone_config_path or 'default config path'}"
        )
        return new_zones

    def _ensure_zone_ready(self, zone_key: str, rtsp_url: str):
        with self.zone_lock:
            self._reload_zone_config()

            zones = self.zone_checker.get_zones_for_source(zone_key)
            if zones:
                logger.info(f"[StreamManager] Zone already exists for camera_id={zone_key}")
                return

            logger.warning(f"[StreamManager] No zone for camera_id={zone_key}, launching GUI before stream starts")
            self._launch_zone_gui(rtsp_url, zone_key)
            self._reload_zone_config()

            if not self.zone_checker.get_zones_for_source(zone_key):
                raise ValueError(f"No zone saved for camera_id='{zone_key}', stream aborted")

            logger.info(f"[StreamManager] Zone confirmed for camera_id={zone_key}, stream can start")

    def _launch_zone_gui(self, rtsp_url: str, zone_key: str):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        gui_script = os.path.join(project_root, "scripts", "draw_fence_gui.py")

        cmd = [
            sys.executable,
            gui_script,
            "--source", rtsp_url,
            "--zone-key", zone_key,
            "--test-mode"
        ]

        logger.info(f"[StreamManager] Launching GUI: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    def remove_stream(self, stream_id: str) -> bool:
        """Stop and remove a stream."""
        with self.lock:
            runtime = self.streams.pop(stream_id, None)
        if runtime is None:
            return False

        runtime.stop()
        self._remove_persisted(stream_id)
        logger.info(f"[StreamManager] Removed {stream_id}")
        return True

    def update_tasks(self, stream_id: str, tasks: List[str]) -> bool:
        """
        Rebuild the per-stream runtime when task composition changes.

        LEGACY:
        # "Update the task list for a running stream (no reconnect needed)."
        """
        tasks = [t for t in tasks if t in self.VALID_TASKS]
        if not tasks:
            return False

        with self.lock:
            current = self.streams.get(stream_id)
            if current is None:
                return False
            current.status = "switching"

        try:
            if "parking_violation" in tasks:
                self._ensure_zone_ready(current.camera_id, current.url)
        except Exception:
            current.status = "running"
            raise

        new_runtime = self.runtime_factory.create_runtime(
            stream_id=current.stream_id,
            url=current.url,
            tasks=tasks,
            camera_id=current.camera_id,
            created_at=current.created_at,
        )
        new_runtime.status = "switching"

        try:
            self._start_capture(new_runtime)
            new_ready = self._wait_capture_ready(new_runtime, timeout_sec=3.0)
        except Exception as e:
            current.status = "running"
            logger.error(f"[StreamManager] Failed to rebuild runtime for {stream_id}: {e}")
            return False

        # LEGACY:
        # with self.lock:
        #     old_runtime = self.streams.get(stream_id)
        #     self.streams[stream_id] = new_runtime
        # if old_runtime is not None:
        #     old_runtime.stop()

        if not new_ready:
            current.status = "running"
            new_runtime.stop()
            logger.error(
                f"[StreamManager] Timed out waiting for rebuilt runtime of {stream_id} "
                f"to receive its first batch; keeping old runtime alive"
            )
            return False

        with self.lock:
            old_runtime = self.streams.get(stream_id)
            if old_runtime is not current:
                new_runtime.stop()
                logger.warning(
                    f"[StreamManager] Runtime for {stream_id} changed during task update; "
                    f"discarding rebuilt runtime"
                )
                return False
            self.streams[stream_id] = new_runtime

        if old_runtime is not None:
            old_runtime.stop()

        new_runtime.status = "running"
        self._persist_stream(new_runtime)
        logger.info(f"[StreamManager] Updated tasks for {stream_id}: {tasks}")
        return True

    def get_streams(self) -> List[Dict]:
        """Return a serialisable snapshot of all streams."""
        result = []
        with self.lock:
            for runtime in self.streams.values():
                if runtime.capture:
                    capture_details = runtime.capture.get_source_details()
                    stream_detail = capture_details.get(runtime.stream_id, {})
                    capture_status = stream_detail.get("status", "connecting")

                    # LEGACY:
                    # statuses = runtime.capture.get_source_status()
                    # alive = any(s == "alive" for s in statuses.values())
                    # if alive and runtime.status == "connecting":
                    #     runtime.status = "running"
                    # elif not alive and runtime.status == "running":
                    #     runtime.status = "error"
                    if capture_status == "running":
                        runtime.status = "running"
                    elif capture_status in {"error", "stopped"}:
                        runtime.status = "error"
                    elif capture_status in {"connecting", "reconnecting"} and runtime.status != "switching":
                        runtime.status = "connecting"
                else:
                    stream_detail = {}

                metrics = runtime.get_metrics(stream_detail)
                bottleneck_hints = runtime.get_bottleneck_hints(stream_detail)

                result.append({
                    "stream_id": runtime.stream_id,
                    "url": runtime.url,
                    "tasks": runtime.tasks,
                    "camera_id": runtime.camera_id,
                    "status": runtime.status,
                    "last_error": stream_detail.get("last_error"),
                    "retry_count": stream_detail.get("retry_count", 0),
                    "last_connected_at": stream_detail.get("last_connected_at"),
                    "last_frame_at": stream_detail.get("last_frame_at"),
                    "metrics": metrics,
                    "bottleneck_hints": bottleneck_hints,
                    "created_at": runtime.created_at,
                })
        return result

    def get_stream_metrics(self, stream_id: str) -> Optional[Dict]:
        with self.lock:
            runtime = self.streams.get(stream_id)
        if runtime is None:
            return None

        capture_detail = {}
        if runtime.capture:
            capture_detail = runtime.capture.get_source_details().get(runtime.stream_id, {})

        return {
            "stream_id": runtime.stream_id,
            "camera_id": runtime.camera_id,
            "url": runtime.url,
            "tasks": runtime.tasks,
            "status": runtime.status,
            "metrics": runtime.get_metrics(capture_detail),
            "bottleneck_hints": runtime.get_bottleneck_hints(capture_detail),
            "created_at": runtime.created_at,
        }

    def stop_all(self):
        """
        Stop all streams.

        LEGACY:
        # self.dispatcher.shutdown()
        """
        with self.lock:
            ids = list(self.streams.keys())
        for sid in ids:
            self.remove_stream(sid)
        logger.info("[StreamManager] All streams stopped.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _start_capture(self, runtime: StreamRuntime):
        """Create a VideoFrameCapture for this stream and wire the callback."""
        cap = VideoFrameCapture()
        cap.register_batch_callback(self._make_callback(runtime))
        cap.add_rtsp_source(
            source_id=runtime.stream_id,
            rtsp_url=runtime.url,
            batch_size=8,
            batch_sec=1.0,
            reconnect_delay=5,
        )
        runtime.capture = cap

    def _wait_capture_ready(self, runtime: StreamRuntime, timeout_sec: float = 3.0) -> bool:
        """
        Wait until the rebuilt runtime receives its first captured batch.
        This avoids switching the stream map to a runtime that has not warmed up yet.
        """
        return runtime.started_event.wait(timeout=timeout_sec)

    def _make_callback(self, runtime: StreamRuntime):
        """Return the batch callback bound to a specific StreamRuntime object."""
        stream_id = runtime.stream_id

        def _on_frames(source_id, frames):
            if frames:
                runtime.mark_capture_started()
            with self.lock:
                current_runtime = self.streams.get(stream_id)
            if current_runtime is not runtime:
                return
            for frame in frames:
                # LEGACY:
                # frame.source_id = config.camera_id
                # self.dispatcher.dispatch(stream_id, frame, config.tasks)
                frame.source_id = runtime.camera_id
                runtime.dispatch_frame(frame)
        return _on_frames

    # ------------------------------------------------------------------
    # MongoDB persistence
    # ------------------------------------------------------------------
    def _persist_stream(self, runtime: StreamRuntime):
        if self._streams_col is None:
            return
        try:
            self._streams_col.update_one(
                {"stream_id": runtime.stream_id},
                {"$set": {
                    "stream_id": runtime.stream_id,
                    "url": runtime.url,
                    "tasks": runtime.tasks,
                    "camera_id": runtime.camera_id,
                    "created_at": runtime.created_at,
                }},
                upsert=True,
            )
        except Exception as e:
            logger.error(f"[StreamManager] Persist failed for {runtime.stream_id}: {e}")

    def _remove_persisted(self, stream_id: str):
        if self._streams_col is None:
            return
        try:
            self._streams_col.delete_one({"stream_id": stream_id})
        except Exception as e:
            logger.error(f"[StreamManager] Remove-persist failed for {stream_id}: {e}")

    def _restore_streams(self):
        """On startup, reload streams from MongoDB and re-create captures."""
        if self._streams_col is None:
            return
        try:
            docs = list(self._streams_col.find())
            if not docs:
                return
            logger.info(f"[StreamManager] Restoring {len(docs)} stream(s) from MongoDB...")
            for doc in docs:
                sid = doc.get("stream_id", "")
                url = doc.get("url", "")
                tasks = doc.get("tasks", [])
                camera_id = doc.get("camera_id", sid)
                created = doc.get("created_at", time.time())
                if not url or not tasks:
                    continue

                # LEGACY:
                # config = StreamConfig(
                #     stream_id=sid,
                #     url=url,
                #     tasks=tasks,
                #     camera_id=camera_id,
                #     status="connecting",
                #     created_at=created,
                # )

                try:
                    if "parking_violation" in tasks:
                        self._ensure_zone_ready(camera_id, url)
                    runtime = self.runtime_factory.create_runtime(
                        stream_id=sid,
                        url=url,
                        tasks=tasks,
                        camera_id=camera_id,
                        created_at=created,
                    )
                except Exception as e:
                    logger.error(
                        f"[StreamManager] Skip restoring {sid}: zone not ready or GUI failed: {e}"
                    )
                    continue

                with self.lock:
                    self.streams[sid] = runtime
                    try:
                        idx = int(sid.split("_")[-1])
                        if idx >= self._counter:
                            self._counter = idx
                    except ValueError:
                        pass

                try:
                    self._start_capture(runtime)
                except Exception as e:
                    with self.lock:
                        self.streams.pop(sid, None)
                    logger.error(f"[StreamManager] Failed to start restored stream {sid}: {e}")
                    continue

                logger.info(
                    f"[StreamManager] Restored {sid}: {url} | camera_id={camera_id} | tasks={tasks}"
                )
        except Exception as e:
            logger.error(f"[StreamManager] Restore failed: {e}")
