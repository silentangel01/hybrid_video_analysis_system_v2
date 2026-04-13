"""
Per-stream runtime construction for RTSP processing.

Design:
  - AppResources: process-wide shared external resources/config
  - StreamRuntime: per-stream isolated execution context
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from ml_models.yolov8.model_loader import YOLOModelLoader
from backend.services.violation_detection import ViolationDetectionService
from backend.services.smoke_flame_detection import SmokeFlameDetectionService
from backend.services.common_space_detection import CommonSpaceDetectionService
from backend.utils.performance_metrics import (
    SlidingCounter,
    LatencyRecorder,
    get_thread_pool_queue_size,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppResources:
    """Process-wide shared resources that are safe to reuse."""

    minio: Any
    mongo: Any
    zone_checker: Any
    qwen_vl_client: Any
    smoke_service_ready: bool
    common_space_service_ready: bool
    common_space_interval_sec: float
    dwell_threshold: int = 5
    weights_dir: Optional[str] = None


class StreamExecutor:
    """Per-stream task executor. Keeps RTSP streams from sharing one global pool."""

    def __init__(self, stream_id: str, handlers: Dict[str, Any]):
        self.stream_id = stream_id
        self.handlers = handlers
        max_workers = max(2, len(handlers) * 2)
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"{stream_id}-task"
        )
        self.metrics_lock = threading.Lock()
        self.submitted_total = 0
        self.completed_total = 0
        self.failed_total = 0
        self.inflight_tasks = 0
        self.dispatch_counter = SlidingCounter(window_sec=10.0)
        self.completion_counter = SlidingCounter(window_sec=10.0)
        self.task_dispatch_counter = {
            task_name: SlidingCounter(window_sec=10.0)
            for task_name in handlers.keys()
        }
        self.task_latency = {
            task_name: LatencyRecorder()
            for task_name in handlers.keys()
        }

    def dispatch(self, frame_meta, tasks: List[str]):
        for task in tasks:
            handler = self.handlers.get(task)
            if handler is None:
                logger.warning(f"[{self.stream_id}] Missing handler for task: {task}")
                continue
            try:
                self.executor.submit(self._safe_process, handler, frame_meta, task)
                with self.metrics_lock:
                    self.submitted_total += 1
                    self.inflight_tasks += 1
                self.dispatch_counter.add()
                self.task_dispatch_counter[task].add()
            except Exception as e:
                logger.error(f"[{self.stream_id}] Failed to submit task {task}: {e}")

    def _safe_process(self, handler, frame_meta, task: str):
        started = time.perf_counter()
        failed = False
        try:
            handler.process_frame(frame_meta)
        except Exception as e:
            failed = True
            logger.error(f"[{self.stream_id}] Handler error in {task}: {e}")
        finally:
            elapsed = time.perf_counter() - started
            self.task_latency[task].record(elapsed)
            self.completion_counter.add()
            with self.metrics_lock:
                self.completed_total += 1
                self.inflight_tasks = max(0, self.inflight_tasks - 1)
                if failed:
                    self.failed_total += 1

    def shutdown(self):
        self.executor.shutdown(wait=False)

    def get_metrics(self) -> Dict[str, Any]:
        with self.metrics_lock:
            submitted_total = self.submitted_total
            completed_total = self.completed_total
            failed_total = self.failed_total
            inflight_tasks = self.inflight_tasks

        return {
            "max_workers": getattr(self.executor, "_max_workers", 0),
            "queue_size": get_thread_pool_queue_size(self.executor),
            "submitted_total": submitted_total,
            "completed_total": completed_total,
            "failed_total": failed_total,
            "inflight_tasks": inflight_tasks,
            "dispatch_fps_10s": self.dispatch_counter.snapshot()["rate_per_sec"],
            "completion_fps_10s": self.completion_counter.snapshot()["rate_per_sec"],
            "tasks": {
                task_name: {
                    "dispatch_fps_10s": self.task_dispatch_counter[task_name].snapshot()["rate_per_sec"],
                    "handler_call_latency": self.task_latency[task_name].snapshot(),
                }
                for task_name in self.handlers.keys()
            },
        }


@dataclass
class StreamRuntime:
    """Per-stream runtime state and handlers."""

    stream_id: str
    url: str
    camera_id: str
    tasks: List[str]
    model_loader: YOLOModelLoader
    handlers: Dict[str, Any]
    executor: StreamExecutor
    capture: Any = field(default=None, repr=False)
    started_event: Any = field(default_factory=threading.Event, repr=False)
    status: str = "connecting"
    created_at: float = field(default_factory=time.time)
    lat_lng: str = ""
    location: str = ""

    def dispatch_frame(self, frame_meta):
        self.executor.dispatch(frame_meta, self.tasks)

    def mark_capture_started(self):
        self.started_event.set()

    def get_metrics(self, capture_detail: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        task_metrics: Dict[str, Any] = {}
        for task_name, handler in self.handlers.items():
            getter = getattr(handler, "get_runtime_metrics", None)
            task_metrics[task_name] = getter() if callable(getter) else {}

        capture_metrics = {
            "status": (capture_detail or {}).get("status"),
            "capture_fps_10s": (capture_detail or {}).get("capture_fps_10s", 0.0),
            "emit_fps_10s": (capture_detail or {}).get("emit_fps_10s", 0.0),
            "batch_rate_10s": (capture_detail or {}).get("batch_rate_10s", 0.0),
            "frames_read_total": (capture_detail or {}).get("frames_read_total", 0),
            "frames_emitted_total": (capture_detail or {}).get("frames_emitted_total", 0),
            "batches_emitted_total": (capture_detail or {}).get("batches_emitted_total", 0),
            "disconnect_count": (capture_detail or {}).get("disconnect_count", 0),
            "open_failures_total": (capture_detail or {}).get("open_failures_total", 0),
            "consecutive_open_failures": (capture_detail or {}).get("consecutive_open_failures", 0),
            "last_batch_size": (capture_detail or {}).get("last_batch_size", 0),
        }

        return {
            "capture": capture_metrics,
            "executor": self.executor.get_metrics(),
            "tasks": task_metrics,
        }

    def get_bottleneck_hints(self, capture_detail: Optional[Dict[str, Any]] = None) -> List[str]:
        metrics = self.get_metrics(capture_detail)
        hints: List[str] = []

        capture = metrics["capture"]
        executor = metrics["executor"]
        tasks = metrics["tasks"]

        if capture.get("consecutive_open_failures", 0) > 0 or capture.get("status") == "reconnecting":
            hints.append("RTSP capture is unstable or reconnecting")

        if executor.get("queue_size", 0) > max(4, int(executor.get("max_workers", 0)) * 2):
            hints.append("Per-stream executor queue is backing up")

        parking = tasks.get("parking_violation", {})
        if parking.get("pipeline_latency", {}).get("recent_avg_ms", 0) > 350:
            hints.append("Parking pipeline latency is high")

        smoke = tasks.get("smoke_flame", {})
        if smoke.get("verification_queue_size", 0) > 4:
            hints.append("Smoke verification queue is backing up")
        if smoke.get("qwen_latency", {}).get("recent_avg_ms", 0) > 1500:
            hints.append("Smoke Qwen verification latency is high")

        common_space = tasks.get("common_space", {})
        if common_space.get("analysis_queue_size", 0) > 2:
            hints.append("Common-space analysis queue is backing up")

        return hints

    def stop(self):
        if self.capture:
            self.capture.stop_all()
        self.executor.shutdown()

        for task_name, handler in self.handlers.items():
            flush = getattr(handler, "flush_remaining", None)
            if callable(flush):
                try:
                    flush()
                except Exception as e:
                    logger.error(f"[{self.stream_id}] Failed to flush {task_name}: {e}")

        self.status = "stopped"


class StreamRuntimeFactory:
    """
    Build one isolated runtime per RTSP stream.

    Isolation boundary:
      - dedicated handler instances
      - dedicated thread pool
      - dedicated YOLOModelLoader / model instances
    Shared:
      - MongoDB, MinIO, zone config, Qwen client
    """

    VALID_TASKS = {"parking_violation", "smoke_flame", "common_space"}

    def __init__(self, resources: AppResources):
        self.resources = resources

    def create_runtime(
        self,
        stream_id: str,
        url: str,
        tasks: List[str],
        camera_id: str,
        created_at: Optional[float] = None,
        lat_lng: str = "",
        location: str = "",
    ) -> StreamRuntime:
        tasks = [t for t in tasks if t in self.VALID_TASKS]
        if not tasks:
            raise ValueError("No valid tasks provided for StreamRuntime")

        loader = self._build_model_loader(tasks)
        handlers = self._build_handlers(tasks, loader, lat_lng=lat_lng, location=location)
        executor = StreamExecutor(stream_id=stream_id, handlers=handlers)

        return StreamRuntime(
            stream_id=stream_id,
            url=url,
            camera_id=camera_id,
            tasks=tasks,
            model_loader=loader,
            handlers=handlers,
            executor=executor,
            created_at=created_at or time.time(),
            lat_lng=lat_lng,
            location=location,
        )

    def _build_model_loader(self, tasks: List[str]) -> YOLOModelLoader:
        if self.resources.weights_dir:
            loader = YOLOModelLoader(weights_dir=self.resources.weights_dir)
        else:
            loader = YOLOModelLoader()

        if "parking_violation" in tasks:
            if not loader.load_model("vehicle", "yolov8n.pt"):
                raise ValueError("Failed to load vehicle model for parking_violation")

        if "smoke_flame" in tasks:
            if not self.resources.smoke_service_ready:
                raise ValueError("smoke_flame task is not available")
            if not loader.load_model("smoke_flame", "smoke_flame.pt"):
                raise ValueError("Failed to load smoke_flame model")

        return loader

    def _build_handlers(self, tasks: List[str], loader: YOLOModelLoader, lat_lng: str = "", location: str = "") -> Dict[str, Any]:
        handlers: Dict[str, Any] = {}

        if "parking_violation" in tasks:
            parking_handler = ViolationDetectionService(
                dwell_threshold=self.resources.dwell_threshold,
            )
            parking_handler.set_clients(
                minio_client=self.resources.minio,
                mongo_client=self.resources.mongo,
            )
            parking_handler.set_model_loader(loader)
            parking_handler.set_zone_checker(self.resources.zone_checker)
            parking_handler.lat_lng = lat_lng
            parking_handler.location = location
            handlers["parking_violation"] = parking_handler

        if "smoke_flame" in tasks:
            if not self.resources.smoke_service_ready:
                raise ValueError("smoke_flame task is not available")
            smoke_handler = SmokeFlameDetectionService()
            smoke_handler.set_clients(self.resources.minio, self.resources.mongo)
            smoke_handler.set_model_loader(loader)
            if self.resources.qwen_vl_client:
                smoke_handler.set_qwen_vl_client(self.resources.qwen_vl_client)
            smoke_handler.lat_lng = lat_lng
            smoke_handler.location = location
            handlers["smoke_flame"] = smoke_handler

        if "common_space" in tasks:
            if not self.resources.common_space_service_ready:
                raise ValueError("common_space task is not available")
            common_space_handler = CommonSpaceDetectionService()
            common_space_handler.set_clients(self.resources.minio, self.resources.mongo)
            common_space_handler.set_qwen_vl_client(self.resources.qwen_vl_client)
            common_space_handler.set_sample_interval(
                int(self.resources.common_space_interval_sec)
            )
            handlers["common_space"] = common_space_handler

        return handlers
