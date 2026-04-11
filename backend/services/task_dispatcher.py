# backend/services/task_dispatcher.py
"""
TaskDispatcher - parallel frame fan-out to multiple detection services.

VLM does NOT block YOLO:
  - parking_violation runs synchronously in the pool thread (~50ms)
  - smoke_flame / common_space internally submit to their own async ThreadPools,
    so the dispatcher thread returns immediately.

LEGACY NOTE:
  - RTSP streams now use per-stream executors via StreamRuntime
  - this module remains for legacy/offline code paths
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class TaskDispatcher:
    TASK_SERVICE_MAP = {
        "parking_violation": "parking_service",
        "smoke_flame": "smoke_service",
        "common_space": "common_space_service",
    }

    def __init__(self, services: Dict[str, Any]):
        self.services = services
        self.executor = ThreadPoolExecutor(max_workers=12, thread_name_prefix="dispatch")

    def dispatch(self, stream_id: str, frame_meta, tasks: List[str]):
        """Fan-out a single frame to every requested detection service in parallel."""
        for task in tasks:
            svc_key = self.TASK_SERVICE_MAP.get(task)
            if not svc_key:
                logger.warning(f"[Dispatcher] Unknown task type: {task}")
                continue
            service = self.services.get(svc_key)
            if service is None:
                continue
            self.executor.submit(self._safe_process, service, frame_meta, task, stream_id)

    def _safe_process(self, service, frame_meta, task: str, stream_id: str):
        try:
            service.process_frame(frame_meta)
        except Exception as e:
            logger.error(f"[Dispatcher] Error running {task} on {stream_id}: {e}")

    def shutdown(self):
        self.executor.shutdown(wait=False)
        logger.info("[Dispatcher] Executor shut down.")
