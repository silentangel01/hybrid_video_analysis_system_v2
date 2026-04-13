# -*- coding: utf-8 -*-
# backend/utils/frame_capture.py
"""
Frame capture utilities.

Features:
  1. Multi-stream RTSP capture with one thread per source
  2. Local video file capture with optional looping
  3. Frame sampling and batch delivery
  4. Batched callback handoff to downstream detection services
  5. Performance counters for capture diagnostics
"""

import cv2
import time
import logging
import threading
import os
from typing import List, Optional, Callable, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse

from backend.utils.performance_metrics import SlidingCounter

logger = logging.getLogger(__name__)


# -------------------- Data Container --------------------
@dataclass
class FrameWithMetadata:
    """Frame payload plus source metadata for downstream processing."""

    image: Any
    source_id: str
    timestamp: float
    frame_index: int
    original_time_str: str
    is_rtsp: bool


# -------------------- Batch Buffer --------------------
class FrameBuffer:
    """Buffer frames and flush them as a batch by size or time window."""

    def __init__(self, source_id: str, batch_size: int = 1, batch_sec: float = 1.0):
        self.source_id = source_id
        self.batch_size = batch_size
        self.batch_sec = batch_sec
        self.buffer: List[FrameWithMetadata] = []
        self.last_push = time.time()

    def add(self, frame: FrameWithMetadata) -> Optional[List[FrameWithMetadata]]:
        """Add a frame and return a batch when the flush condition is met."""
        self.buffer.append(frame)
        meet_size = len(self.buffer) >= self.batch_size
        meet_time = (frame.timestamp - self.last_push) >= self.batch_sec
        if meet_size or meet_time:
            self.last_push = frame.timestamp
            batch, self.buffer = self.buffer[:], []
            return batch
        return None


# -------------------- Unified Capture Manager --------------------
class VideoFrameCapture:
    """One thread per source plus buffered batch delivery to a callback."""

    def __init__(self):
        self.sources: Dict[str, threading.Thread] = {}
        self.source_details: Dict[str, Dict[str, Any]] = {}
        self.source_perf: Dict[str, Dict[str, Any]] = {}
        self.state_lock = threading.Lock()
        self.running = True
        # Callback signature: batch_callback(source_id, frame_list)
        self._callback: Optional[Callable[[str, List[FrameWithMetadata]], None]] = None

    def register_batch_callback(self, callback: Callable[[str, List[FrameWithMetadata]], None]):
        """Register the batched callback used by downstream detectors."""
        self._callback = callback

    # -------------------- Source Registration --------------------
    def _set_source_detail(self, source_id: str, **updates):
        with self.state_lock:
            detail = self.source_details.setdefault(source_id, {})
            detail.update(updates)

    def _get_source_detail(self, source_id: str) -> Dict[str, Any]:
        with self.state_lock:
            return dict(self.source_details.get(source_id, {}))

    def _build_rtsp_open_error(self, url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path.lstrip("/") or "<empty>"
        base = f"Failed to open RTSP source: {url}."
        hints = [
            "Verify the RTSP URL and path are correct.",
            "Make sure the publisher is already pushing before the detector subscribes.",
        ]
        if "localhost:8554" in url or "127.0.0.1:8554" in url:
            hints.append(
                f"If you are using MediaMTX, a DESCRIBE 404 usually means there is no published stream on path '{path}' yet."
            )
            hints.append(
                f"Wait until the RTSP server reports that a publisher is actively publishing to '{path}', then retry."
            )
        return " ".join([base] + hints)

    def _init_source_perf(self, source_id: str):
        with self.state_lock:
            if source_id in self.source_perf:
                return
            self.source_perf[source_id] = {
                "read_counter": SlidingCounter(window_sec=10.0),
                "emit_counter": SlidingCounter(window_sec=10.0),
                "batch_counter": SlidingCounter(window_sec=10.0),
                "open_failures_total": 0,
                "disconnect_count": 0,
                "consecutive_open_failures": 0,
                "last_batch_size": 0,
                "frames_skipped_total": 0,
            }

    def _with_source_perf(self, source_id: str) -> Dict[str, Any]:
        with self.state_lock:
            return self.source_perf[source_id]

    def _increment_source_perf(self, source_id: str, key: str, delta: int = 1):
        with self.state_lock:
            perf = self.source_perf.get(source_id)
            if perf is None:
                return
            perf[key] = int(perf.get(key, 0)) + delta

    def _set_source_perf_value(self, source_id: str, key: str, value: Any):
        with self.state_lock:
            perf = self.source_perf.get(source_id)
            if perf is None:
                return
            perf[key] = value

    def _get_source_perf_snapshot(self, source_id: str) -> Dict[str, Any]:
        with self.state_lock:
            perf = self.source_perf.get(source_id)
        if perf is None:
            return {}

        read_snapshot = perf["read_counter"].snapshot()
        emit_snapshot = perf["emit_counter"].snapshot()
        batch_snapshot = perf["batch_counter"].snapshot()

        return {
            "frames_read_total": read_snapshot["total"],
            "frames_emitted_total": emit_snapshot["total"],
            "batches_emitted_total": batch_snapshot["total"],
            "capture_fps_10s": read_snapshot["rate_per_sec"],
            "emit_fps_10s": emit_snapshot["rate_per_sec"],
            "batch_rate_10s": batch_snapshot["rate_per_sec"],
            "open_failures_total": perf["open_failures_total"],
            "disconnect_count": perf["disconnect_count"],
            "consecutive_open_failures": perf["consecutive_open_failures"],
            "last_batch_size": perf["last_batch_size"],
            "frames_skipped_total": perf.get("frames_skipped_total", 0),
        }

    def add_rtsp_source(
        self,
        source_id: str,
        rtsp_url: str,
        batch_size: int = 1,
        batch_sec: float = 1.0,
        reconnect_delay: int = 5,
        sample_interval_sec: float = 1.0,
    ):
        """Add an RTSP source and start its capture thread.

        Args:
            sample_interval_sec: Minimum seconds between frames that enter the
                processing pipeline.  The capture thread still calls cap.read()
                at the native frame-rate to keep the RTSP buffer drained, but
                only frames spaced at least this far apart are forwarded.
        """
        if source_id in self.sources:
            logger.warning(f"[{source_id}] Already exists.")
            return
        self._init_source_perf(source_id)
        self._set_source_detail(
            source_id,
            kind="rtsp",
            url=rtsp_url,
            status="connecting",
            last_error=None,
            retry_count=0,
            last_connected_at=None,
            last_frame_at=None,
        )
        thread = threading.Thread(
            target=self._rtsp_loop,
            args=(source_id, rtsp_url, batch_size, batch_sec, reconnect_delay, sample_interval_sec),
            daemon=True,
        )
        self.sources[source_id] = thread
        thread.start()

    def add_local_video_source(
        self,
        source_id: str,
        video_path: str,
        batch_size: int = 1,
        batch_sec: float = 1.0,
        loop_play: bool = True,
    ):
        """Add a local video source and start its capture thread."""
        if not os.path.exists(video_path):
            logger.error(f"[{source_id}] File not found: {video_path}")
            return
        if source_id in self.sources:
            logger.warning(f"[{source_id}] Already exists.")
            return
        self._init_source_perf(source_id)
        self._set_source_detail(
            source_id,
            kind="local",
            url=video_path,
            status="connecting",
            last_error=None,
            retry_count=0,
            last_connected_at=None,
            last_frame_at=None,
        )
        thread = threading.Thread(
            target=self._local_loop,
            args=(source_id, video_path, batch_size, batch_sec, loop_play),
            daemon=True,
        )
        self.sources[source_id] = thread
        thread.start()

    # -------------------- RTSP Thread --------------------
    def _rtsp_loop(
        self,
        source_id: str,
        url: str,
        batch_size: int,
        batch_sec: float,
        reconnect_delay: int,
        sample_interval_sec: float = 1.0,
    ):
        """Capture RTSP frames and emit them in batches.

        Every frame is read (to keep the RTSP buffer drained), but only
        frames spaced >= *sample_interval_sec* apart are forwarded into
        the processing pipeline.
        """
        buffer = FrameBuffer(source_id, batch_size, batch_sec)
        perf = self._with_source_perf(source_id)
        frame_idx = 0
        last_sample_time = 0.0
        frames_skipped = 0

        logger.debug(f"[{source_id}] RTSP capture started: {url} (sample_interval={sample_interval_sec}s)")

        while self.running:
            self._set_source_detail(source_id, status="connecting")
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                detail = self._get_source_detail(source_id)
                retry_count = int(detail.get("retry_count", 0)) + 1
                error_message = self._build_rtsp_open_error(url)
                self._increment_source_perf(source_id, "open_failures_total")
                self._increment_source_perf(source_id, "consecutive_open_failures")
                self._set_source_detail(
                    source_id,
                    status="reconnecting",
                    last_error=error_message,
                    retry_count=retry_count,
                )
                logger.error(f"[{source_id}] {error_message} Retry in {reconnect_delay}s")
                time.sleep(reconnect_delay)
                continue

            self._set_source_detail(
                source_id,
                status="running",
                last_error=None,
                retry_count=0,
                last_connected_at=time.time(),
            )
            self._set_source_perf_value(source_id, "consecutive_open_failures", 0)
            logger.debug(f"[{source_id}] RTSP connected.")

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                perf["read_counter"].add()

                current_time = time.time()

                # Always update last_frame_at so liveness checks stay fresh.
                self._set_source_detail(
                    source_id,
                    status="running",
                    last_error=None,
                    last_frame_at=current_time,
                )

                # --- Time-based sampling ---
                if current_time - last_sample_time < sample_interval_sec:
                    frames_skipped += 1
                    continue
                last_sample_time = current_time

                current_dt = datetime.fromtimestamp(current_time)

                meta = FrameWithMetadata(
                    image=frame,
                    source_id=source_id,
                    timestamp=current_time,
                    frame_index=frame_idx,
                    original_time_str=current_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    is_rtsp=True,
                )

                # Emit a lightweight debug sample every 30 *sampled* frames.
                if frame_idx % 30 == 1:
                    logger.debug(
                        f"[{source_id}] Frame {frame_idx} (skipped {frames_skipped}) - "
                        f"System time: {current_dt} | "
                        f"Timestamp: {current_time}"
                    )

                batch = buffer.add(meta)
                if batch and self._callback:
                    perf["emit_counter"].add(len(batch))
                    perf["batch_counter"].add()
                    self._set_source_perf_value(source_id, "last_batch_size", len(batch))
                    self._callback(source_id, batch)

            cap.release()
            self._increment_source_perf(source_id, "disconnect_count")
            self._set_source_detail(
                source_id,
                status="reconnecting",
                last_error="RTSP stream disconnected; waiting to reconnect.",
            )
            logger.warning(f"[{source_id}] RTSP disconnected, reconnecting...")
            time.sleep(reconnect_delay)

    # -------------------- Local File Thread --------------------
    def _local_loop(
        self,
        source_id: str,
        path: str,
        batch_size: int,
        batch_sec: float,
        loop_play: bool,
    ):
        """Capture frames from a local video file and emit them in batches."""
        buffer = FrameBuffer(source_id, batch_size, batch_sec)
        perf = self._with_source_perf(source_id)

        logger.debug(f"[{source_id}] Local video capture started: {path}")

        while self.running:
            self._set_source_detail(source_id, status="connecting")
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                self._set_source_detail(
                    source_id,
                    status="error",
                    last_error=f"Cannot open local file: {path}",
                )
                logger.error(f"[{source_id}] Cannot open file: {path}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            logger.debug(
                f"[{source_id}] Video info: FPS={fps:.1f}, Frames={total_frames}, Duration={duration:.1f}s"
            )

            frame_idx = 0
            start_system_time = time.time()

            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                perf["read_counter"].add()

                # Local files use playback offset plus a wall-clock base.
                video_pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                # Option 1: use a fixed wall-clock base plus playback offset.
                current_system_time = start_system_time + video_pos_sec
                current_dt = datetime.fromtimestamp(current_system_time)

                # Option 2: use the current system time directly.
                # current_system_time = time.time()
                # current_dt = datetime.now()

                meta = FrameWithMetadata(
                    image=frame,
                    source_id=source_id,
                    timestamp=current_system_time,
                    frame_index=frame_idx,
                    original_time_str=current_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    is_rtsp=False,
                )

                # Emit a lightweight debug sample every 100 frames.
                if frame_idx % 100 == 1:
                    logger.debug(
                        f"[{source_id}] Frame {frame_idx}/{total_frames} - "
                        f"Video time: {video_pos_sec:.1f}s - "
                        f"System time: {current_dt} | "
                        f"Timestamp: {current_system_time}"
                    )

                batch = buffer.add(meta)
                self._set_source_detail(
                    source_id,
                    status="running",
                    last_error=None,
                    last_frame_at=current_system_time,
                )
                if batch and self._callback:
                    perf["emit_counter"].add(len(batch))
                    perf["batch_counter"].add()
                    self._set_source_perf_value(source_id, "last_batch_size", len(batch))
                    self._callback(source_id, batch)

                # Simulate near-real-time playback for very small batch windows.
                if batch_sec <= 0.1:
                    time.sleep(1.0 / fps)

            cap.release()

            # Debug marker for file loop restarts.
            logger.debug(f"[{source_id}] Video finished at frame {frame_idx}")

            if not loop_play:
                self._set_source_detail(source_id, status="stopped", last_error=None)
                logger.debug(f"[{source_id}] Single play completed.")
                break

            self._set_source_detail(source_id, status="connecting")
            logger.debug(f"[{source_id}] Local video looping...")
            time.sleep(1)

    # -------------------- Graceful Shutdown --------------------
    def stop_all(self):
        """Stop all capture threads."""
        self.running = False
        for tid, thread in self.sources.items():
            if thread.is_alive():
                thread.join(timeout=2.0)
                self._set_source_detail(tid, status="stopped")
                logger.debug(f"Stopped: {tid}")
        logger.debug("All capture threads stopped.")

    # -------------------- Status Inspection --------------------
    def get_source_status(self) -> Dict[str, str]:
        """Return the status of all registered sources."""
        status = {}
        for source_id, thread in self.sources.items():
            detail = self._get_source_detail(source_id)
            if not thread.is_alive():
                status[source_id] = "stopped"
            else:
                status[source_id] = detail.get("status", "connecting")
        return status

    def get_source_details(self) -> Dict[str, Dict[str, Any]]:
        """Return detailed capture and performance state for all sources."""
        details = {}
        for source_id, thread in self.sources.items():
            detail = self._get_source_detail(source_id)
            detail.update(self._get_source_perf_snapshot(source_id))
            detail["thread_alive"] = thread.is_alive()
            if not thread.is_alive():
                detail["status"] = "stopped"
            details[source_id] = detail
        return details
