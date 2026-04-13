# backend/services/violation_detection.py
"""
Violation Detection Service

Pipeline per frame:
  1. Whole-frame YOLO inference (vehicle model)
  2. Batch-filter violations in no-parking zones
  3. Single-pass rendering with all detections + violations highlighted
  4. MinIO upload and MongoDB event save
"""

import logging
import threading
import time
from typing import Any, Dict, List

import numpy as np

from backend.services.dwell_tracker import DwellTracker
from backend.services.event_generator import handle_parking_violation_events
from backend.services.parking_zone_checker import NoParkingZoneChecker
from backend.utils.frame_capture import FrameWithMetadata
from backend.utils.performance_metrics import LatencyRecorder, SlidingCounter
from backend.utils.visualization import render_official_frame
from ml_models.yolov8.inference import YOLOInference
from ml_models.yolov8.model_loader import YOLOModelLoader
from storage.minio_client import MinIOClient
from storage.mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)


class ViolationDetectionService:
    """Whole-frame batch detection service for parking violations."""

    def __init__(self, dwell_threshold: int = 5):
        self.minio_client = None
        self.mongo_client = None
        self.model_loader = None
        self.zone_checker = None
        self._dwell_threshold = dwell_threshold
        # Per-source DwellTracker instances (created lazily)
        self._dwell_trackers: Dict[str, DwellTracker] = {}
        self._dwell_lock = threading.Lock()
        # Ultralytics YOLO is not thread-safe for concurrent predict calls on
        # the same model instance; serialise inference when RTSP frames overlap.
        self._inference_lock = threading.Lock()
        # Location metadata (set by StreamRuntimeFactory)
        self.lat_lng: str = ""
        self.location: str = ""

        # Metrics
        self.metrics_lock = threading.Lock()
        self.frame_counter = SlidingCounter(window_sec=10.0)
        self.event_counter = SlidingCounter(window_sec=10.0)
        self.yolo_latency = LatencyRecorder()
        self.filter_latency = LatencyRecorder()
        self.render_latency = LatencyRecorder()
        self.upload_latency = LatencyRecorder()
        self.db_latency = LatencyRecorder()
        self.pipeline_latency = LatencyRecorder()
        self.frames_total = 0
        self.frames_with_detection_total = 0
        self.frames_with_violation_total = 0
        self.events_saved_total = 0
        self.last_detection_count = 0
        self.last_violation_count = 0

    # -------------------- Dependency injection --------------------

    def set_clients(self, minio_client: MinIOClient, mongo_client: MongoDBClient):
        self.minio_client = minio_client
        self.mongo_client = mongo_client

    def set_model_loader(self, loader: YOLOModelLoader):
        self.model_loader = loader

    def set_zone_checker(self, zone_checker: NoParkingZoneChecker):
        self.zone_checker = zone_checker
        if hasattr(self, 'processor') and self.processor:
            self.processor.set_zone_checker(zone_checker)

    def flush_remaining(self):
        """No-op in batch mode (kept for interface compatibility)."""
        pass

    def get_runtime_metrics(self) -> Dict[str, Any]:
        with self.metrics_lock:
            return {
                "frames_total": self.frames_total,
                "frames_with_detection_total": self.frames_with_detection_total,
                "frames_with_violation_total": self.frames_with_violation_total,
                "events_saved_total": self.events_saved_total,
                "input_fps_10s": self.frame_counter.snapshot()["rate_per_sec"],
                "event_fps_10s": self.event_counter.snapshot()["rate_per_sec"],
                "last_detection_count": self.last_detection_count,
                "last_violation_count": self.last_violation_count,
                "yolo_latency": self.yolo_latency.snapshot(),
                "filter_latency": self.filter_latency.snapshot(),
                "render_latency": self.render_latency.snapshot(),
                "upload_latency": self.upload_latency.snapshot(),
                "db_latency": self.db_latency.snapshot(),
                "pipeline_latency": self.pipeline_latency.snapshot(),
            }

    def _get_dwell_tracker(self, source_id: str) -> DwellTracker:
        """Return (or create) a per-source DwellTracker."""
        with self._dwell_lock:
            tracker = self._dwell_trackers.get(source_id)
            if tracker is None:
                tracker = DwellTracker(dwell_threshold=self._dwell_threshold)
                self._dwell_trackers[source_id] = tracker
            return tracker

    # -------------------- Main entry --------------------

    def process_frame(self, frame_meta: FrameWithMetadata) -> None:
        """Whole-frame inference -> batch filter -> render -> upload -> save."""
        pipeline_started = time.perf_counter()
        self.frame_counter.add()
        with self.metrics_lock:
            self.frames_total += 1

        if not all([self.minio_client, self.mongo_client, self.model_loader, self.zone_checker]):
            logger.error("Service not fully initialised")
            return

        source_id = frame_meta.source_id
        image = frame_meta.image
        current_timestamp = time.time()

        # 1. YOLO inference (serialised)
        model = self.model_loader.get_model("vehicle")
        if model is None:
            logger.error("Vehicle model not loaded")
            return

        yolo_started = time.perf_counter()
        with self._inference_lock:
            raw_detections = YOLOInference.run_detection(model, image, conf_threshold=0.3)
        self.yolo_latency.record(time.perf_counter() - yolo_started)

        detections: List[Dict[str, Any]] = [
            {"class_name": cls, "confidence": float(conf), "bbox": bbox}
            for cls, conf, bbox in raw_detections
        ]
        with self.metrics_lock:
            self.last_detection_count = len(detections)
            if detections:
                self.frames_with_detection_total += 1

        if not detections:
            self.pipeline_latency.record(time.perf_counter() - pipeline_started)
            return

        # 2. Zone filter
        filter_started = time.perf_counter()
        in_zone = self.zone_checker.filter_violations_in_zones(detections, source_id)
        self.filter_latency.record(time.perf_counter() - filter_started)

        # 3. Dwell-time tracking: only vehicles that stay in the zone for
        #    N consecutive frames are treated as violations (NFR2.1).
        tracker = self._get_dwell_tracker(source_id)
        violations = tracker.update(in_zone)

        with self.metrics_lock:
            self.last_violation_count = len(violations)
            if violations:
                self.frames_with_violation_total += 1

        # If no dwell-triggered violations this frame, skip upload/persist.
        if not violations:
            self.pipeline_latency.record(time.perf_counter() - pipeline_started)
            return

        zones = self.zone_checker.get_zones_for_source(source_id)

        # 4. Render
        render_started = time.perf_counter()
        rendered = render_official_frame(
            image=image, all_detections=detections, violations=violations, zones=zones
        )
        self.render_latency.record(time.perf_counter() - render_started)

        # 5. Upload
        upload_started = time.perf_counter()
        image_url = self.minio_client.upload_frame(
            image_data=rendered, camera_id=source_id, timestamp=current_timestamp, event_type="violations"
        )
        self.upload_latency.record(time.perf_counter() - upload_started)

        if not image_url:
            self.pipeline_latency.record(time.perf_counter() - pipeline_started)
            logger.error("Upload failed for %s", source_id)
            return

        clean_url = image_url.split('?')[0] if '?X-Amz-' in image_url else image_url

        # 6. Persist
        db_started = time.perf_counter()
        ok = handle_parking_violation_events(
            minio_client=self.minio_client,
            mongo_client=self.mongo_client,
            image_url=clean_url,
            camera_id=source_id,
            timestamp=current_timestamp,
            frame_index=frame_meta.frame_index,
            detections=violations,
            zones=zones,
            lat_lng=self.lat_lng or None,
            location=self.location or None,
        )
        self.db_latency.record(time.perf_counter() - db_started)

        if ok:
            self.event_counter.add()
            with self.metrics_lock:
                self.events_saved_total += 1
            logger.info("Frame event saved: %d dwell violations / %d detections for %s",
                        len(violations), len(detections), source_id)
        else:
            logger.error("Failed to save frame events for %s", source_id)

        self.pipeline_latency.record(time.perf_counter() - pipeline_started)


# Module-level singleton (used by main.py's legacy initialisation path)
detection_service = ViolationDetectionService()
