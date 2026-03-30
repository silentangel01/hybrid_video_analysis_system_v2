# backend/services/stream_manager.py
"""
StreamManager - RTSP stream lifecycle management with MongoDB persistence.

Responsibilities:
  - add / remove / update RTSP streams at runtime
  - persist stream configs to MongoDB `streams` collection
  - restore streams on startup
  - route captured frames to TaskDispatcher
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from backend.utils.frame_capture import VideoFrameCapture
from backend.services.task_dispatcher import TaskDispatcher

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    stream_id: str
    url: str
    tasks: List[str]
    status: str = "connecting"  # running / stopped / error / connecting
    capture: Optional[VideoFrameCapture] = field(default=None, repr=False)
    created_at: float = field(default_factory=time.time)


class StreamManager:
    VALID_TASKS = {"parking_violation", "smoke_flame", "common_space"}

    def __init__(self, services: Dict[str, Any], mongo_client):
        self.streams: Dict[str, StreamConfig] = {}
        self.dispatcher = TaskDispatcher(services)
        self.mongo = mongo_client
        self._streams_col = mongo_client.db["streams"] if mongo_client is not None and mongo_client.db is not None else None
        self.lock = threading.Lock()
        self._counter = 0
        self._restore_streams()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_stream(self, url: str, tasks: List[str]) -> str:
        """Add a new RTSP stream. Returns stream_id."""
        tasks = [t for t in tasks if t in self.VALID_TASKS]
        if not tasks:
            raise ValueError(f"No valid tasks provided. Choose from: {self.VALID_TASKS}")

        with self.lock:
            self._counter += 1
            stream_id = f"stream_{self._counter}"
            while stream_id in self.streams:
                self._counter += 1
                stream_id = f"stream_{self._counter}"

        config = StreamConfig(
            stream_id=stream_id,
            url=url,
            tasks=tasks,
            status="connecting",
            created_at=time.time(),
        )

        self._start_capture(config)

        with self.lock:
            self.streams[stream_id] = config

        self._persist_stream(config)
        logger.info(f"[StreamManager] Added {stream_id}: {url} | tasks={tasks}")
        return stream_id

    def remove_stream(self, stream_id: str) -> bool:
        """Stop and remove a stream."""
        with self.lock:
            config = self.streams.pop(stream_id, None)
        if config is None:
            return False

        if config.capture:
            config.capture.stop_all()
        config.status = "stopped"

        self._remove_persisted(stream_id)
        logger.info(f"[StreamManager] Removed {stream_id}")
        return True

    def update_tasks(self, stream_id: str, tasks: List[str]) -> bool:
        """Update the task list for a running stream (no reconnect needed)."""
        tasks = [t for t in tasks if t in self.VALID_TASKS]
        if not tasks:
            return False

        with self.lock:
            config = self.streams.get(stream_id)
            if config is None:
                return False
            config.tasks = tasks

        self._persist_stream(config)
        logger.info(f"[StreamManager] Updated tasks for {stream_id}: {tasks}")
        return True

    def get_streams(self) -> List[Dict]:
        """Return a serialisable snapshot of all streams."""
        result = []
        with self.lock:
            for cfg in self.streams.values():
                # Update live status from capture thread
                if cfg.capture:
                    statuses = cfg.capture.get_source_status()
                    alive = any(s == "alive" for s in statuses.values())
                    if alive and cfg.status == "connecting":
                        cfg.status = "running"
                    elif not alive and cfg.status == "running":
                        cfg.status = "error"

                result.append({
                    "stream_id": cfg.stream_id,
                    "url": cfg.url,
                    "tasks": cfg.tasks,
                    "status": cfg.status,
                    "created_at": cfg.created_at,
                })
        return result

    def stop_all(self):
        """Stop all streams and dispatcher."""
        with self.lock:
            ids = list(self.streams.keys())
        for sid in ids:
            self.remove_stream(sid)
        self.dispatcher.shutdown()
        logger.info("[StreamManager] All streams stopped.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _start_capture(self, config: StreamConfig):
        """Create a VideoFrameCapture for this stream and wire the callback."""
        cap = VideoFrameCapture()
        cap.register_batch_callback(self._make_callback(config.stream_id))
        cap.add_rtsp_source(
            source_id=config.stream_id,
            rtsp_url=config.url,
            batch_size=8,
            batch_sec=1.0,
            reconnect_delay=5,
        )
        config.capture = cap

    def _make_callback(self, stream_id: str):
        """Return the batch callback bound to a specific stream_id."""
        def _on_frames(source_id, frames):
            with self.lock:
                config = self.streams.get(stream_id)
            if config is None:
                return
            for frame in frames:
                self.dispatcher.dispatch(stream_id, frame, config.tasks)
        return _on_frames

    # ------------------------------------------------------------------
    # MongoDB persistence
    # ------------------------------------------------------------------
    def _persist_stream(self, config: StreamConfig):
        if self._streams_col is None:
            return
        try:
            self._streams_col.update_one(
                {"stream_id": config.stream_id},
                {"$set": {
                    "stream_id": config.stream_id,
                    "url": config.url,
                    "tasks": config.tasks,
                    "created_at": config.created_at,
                }},
                upsert=True,
            )
        except Exception as e:
            logger.error(f"[StreamManager] Persist failed for {config.stream_id}: {e}")

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
                created = doc.get("created_at", time.time())
                if not url or not tasks:
                    continue

                config = StreamConfig(
                    stream_id=sid,
                    url=url,
                    tasks=tasks,
                    status="connecting",
                    created_at=created,
                )
                self._start_capture(config)
                with self.lock:
                    self.streams[sid] = config
                    # Keep counter in sync
                    try:
                        idx = int(sid.split("_")[-1])
                        if idx >= self._counter:
                            self._counter = idx
                    except ValueError:
                        pass
                logger.info(f"[StreamManager] Restored {sid}: {url} | tasks={tasks}")
        except Exception as e:
            logger.error(f"[StreamManager] Restore failed: {e}")
