# -*- coding: utf-8 -*-
# backend/services/smoke_flame_detection.py
"""
Smoke and Flame Detection Service

Two-stage pipeline:
  1. YOLOv8 preliminary detection (smoke_flame.pt)
  2. Qwen-VL API verification for positive cases
  3. Double confirmation -> save event
"""

import base64
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

from backend.services.event_generator import handle_frame_events
from backend.utils.frame_capture import FrameWithMetadata
from backend.utils.performance_metrics import (
    LatencyRecorder,
    SlidingCounter,
    get_thread_pool_queue_size,
)
from backend.utils.visualization import render_official_frame
from ml_models.yolov8.inference import YOLOInference
from ml_models.yolov8.model_loader import YOLOModelLoader
from storage.minio_client import MinIOClient
from storage.mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)


class SmokeFlameDetectionService:
    """Two-stage smoke/flame detection with YOLO + Qwen-VL verification."""

    def __init__(self):
        self.minio_client: Optional[MinIOClient] = None
        self.mongo_client: Optional[MongoDBClient] = None
        self.model_loader: Optional[YOLOModelLoader] = None
        self.yolo_model = None
        self.qwen_vl_client = None

        self.detection_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="smoke-detect")
        self.verification_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="smoke-verify")

        self.frame_skip = 2  # process 1 frame every 3 (capture already samples to ~1 fps)
        self.last_processed_frame: Dict[str, int] = {}
        self.detection_cache: Dict[str, Any] = {}

        # Metrics
        self.metrics_lock = threading.Lock()
        self.received_counter = SlidingCounter(window_sec=10.0)
        self.submitted_counter = SlidingCounter(window_sec=10.0)
        self.event_counter = SlidingCounter(window_sec=10.0)
        self.yolo_latency = LatencyRecorder()
        self.qwen_latency = LatencyRecorder()
        self.async_detection_latency = LatencyRecorder()
        self.save_latency = LatencyRecorder()
        self.frames_received_total = 0
        self.frames_skipped_total = 0
        self.frames_submitted_total = 0
        self.frames_with_candidates_total = 0
        self.frames_verified_total = 0
        self.events_saved_total = 0
        self.qwen_requests_total = 0
        self.last_candidate_count = 0
        self.last_verified_count = 0

    # -------------------- Dependency injection --------------------

    def set_clients(self, minio_client: MinIOClient, mongo_client: MongoDBClient):
        self.minio_client = minio_client
        self.mongo_client = mongo_client

    def set_model_loader(self, loader: YOLOModelLoader):
        self.model_loader = loader
        self.yolo_model = self.model_loader.get_model("smoke_flame")

    def set_qwen_vl_client(self, qwen_client):
        self.qwen_vl_client = qwen_client

    def set_frame_skip(self, skip_frames: int):
        self.frame_skip = skip_frames

    def get_runtime_metrics(self) -> Dict[str, Any]:
        with self.metrics_lock:
            return {
                "frame_skip": self.frame_skip,
                "frames_received_total": self.frames_received_total,
                "frames_skipped_total": self.frames_skipped_total,
                "frames_submitted_total": self.frames_submitted_total,
                "frames_with_candidates_total": self.frames_with_candidates_total,
                "frames_verified_total": self.frames_verified_total,
                "events_saved_total": self.events_saved_total,
                "qwen_requests_total": self.qwen_requests_total,
                "received_fps_10s": self.received_counter.snapshot()["rate_per_sec"],
                "submitted_fps_10s": self.submitted_counter.snapshot()["rate_per_sec"],
                "event_fps_10s": self.event_counter.snapshot()["rate_per_sec"],
                "last_candidate_count": self.last_candidate_count,
                "last_verified_count": self.last_verified_count,
                "detection_queue_size": get_thread_pool_queue_size(self.detection_pool),
                "verification_queue_size": get_thread_pool_queue_size(self.verification_pool),
                "yolo_latency": self.yolo_latency.snapshot(),
                "qwen_latency": self.qwen_latency.snapshot(),
                "async_detection_latency": self.async_detection_latency.snapshot(),
                "save_latency": self.save_latency.snapshot(),
            }

    # -------------------- Main pipeline --------------------

    def process_frame(self, frame_meta: FrameWithMetadata) -> None:
        """Entry point: decide whether to process this frame, then dispatch async."""
        self.received_counter.add()
        with self.metrics_lock:
            self.frames_received_total += 1

        if not all([self.minio_client, self.mongo_client, self.yolo_model]):
            logger.error("Service not fully initialised (minio=%s, mongo=%s, yolo=%s)",
                         self.minio_client is not None, self.mongo_client is not None,
                         self.yolo_model is not None)
            return

        source_id = frame_meta.source_id
        frame_index = frame_meta.frame_index

        if not self._should_process_frame(source_id, frame_index):
            with self.metrics_lock:
                self.frames_skipped_total += 1
            return

        self.last_processed_frame[source_id] = frame_index
        current_timestamp = time.time()

        self.submitted_counter.add()
        with self.metrics_lock:
            self.frames_submitted_total += 1

        future = self.detection_pool.submit(self._process_frame_async, frame_meta, current_timestamp)
        future.add_done_callback(lambda f: self._handle_processing_result(f, source_id, frame_index))

    def _should_process_frame(self, source_id: str, frame_index: int) -> bool:
        if source_id not in self.last_processed_frame:
            self.last_processed_frame[source_id] = -1
            return True
        return (frame_index - self.last_processed_frame[source_id]) > self.frame_skip

    def _process_frame_async(self, frame_meta: FrameWithMetadata, timestamp: float) -> Dict[str, Any]:
        """Full detection pipeline: YOLO -> filter -> Qwen-VL verify."""
        async_started = time.perf_counter()
        source_id = frame_meta.source_id
        empty_result = {"source_id": source_id, "detections": [], "timestamp": timestamp, "frame_meta": frame_meta}

        try:
            image = frame_meta.image

            # Stage 1: YOLO detection
            yolo_detections = self._yolo_detect(image)
            if not yolo_detections:
                with self.metrics_lock:
                    self.last_candidate_count = 0
                    self.last_verified_count = 0
                return empty_result

            with self.metrics_lock:
                self.frames_with_candidates_total += 1
                self.last_candidate_count = len(yolo_detections)

            # Stage 2: IoU-based dedup
            filtered = self._filter_duplicates(yolo_detections, image.shape)
            if not filtered:
                return empty_result

            # Stage 3: Qwen-VL verification
            verified = self._qwen_verify_parallel(image, filtered, source_id)
            if not verified:
                with self.metrics_lock:
                    self.last_verified_count = 0
                return empty_result

            with self.metrics_lock:
                self.frames_verified_total += 1
                self.last_verified_count = len(verified)

            return {"source_id": source_id, "detections": verified, "timestamp": timestamp, "frame_meta": frame_meta}

        except Exception as e:
            logger.error("Async frame processing failed: %s", e)
            return empty_result
        finally:
            self.async_detection_latency.record(time.perf_counter() - async_started)

    def _handle_processing_result(self, future, source_id: str, frame_index: int):
        try:
            result = future.result()
            if result["detections"]:
                self._save_events(result["frame_meta"], result["detections"], result["timestamp"])
        except Exception as e:
            logger.error("Error handling result for %s frame %d: %s", source_id, frame_index, e)

    # -------------------- YOLO detection --------------------

    def _yolo_detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLOv8 inference, filter tiny detections, return standardised dicts."""
        try:
            h, w = image.shape[:2]

            # Downsample only if very large
            if h > 1080 or w > 1920:
                scale = 1080 / h
                resized = cv2.resize(image, (int(w * scale), 1080))
            else:
                resized = image

            start = time.perf_counter()
            raw = YOLOInference.run_detection(self.yolo_model, resized, conf_threshold=0.10)
            self.yolo_latency.record(time.perf_counter() - start)

            detections = []
            for cls, conf, bbox in raw:
                if cls.lower() not in ("smoke", "fire", "flame"):
                    continue

                # Scale back if resized
                if resized is not image:
                    sx = w / resized.shape[1]
                    sy = h / resized.shape[0]
                    bbox = (bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy)

                x1, y1, x2, y2 = map(int, bbox)
                area_ratio = (x2 - x1) * (y2 - y1) / (h * w)
                if area_ratio < 0.0001:
                    continue

                detections.append({
                    "class_name": cls,
                    "confidence": float(conf),
                    "bbox": (x1, y1, x2, y2),
                    "detection_stage": "yolo_initial",
                    "area_ratio": area_ratio,
                })

            detections.sort(key=lambda d: d["confidence"], reverse=True)
            return detections

        except Exception as e:
            logger.error("YOLOv8 detection failed: %s", e)
            return []

    # -------------------- Duplicate filtering --------------------

    def _filter_duplicates(self, detections: List[Dict[str, Any]], image_shape: Tuple[int, ...]) -> List[Dict[str, Any]]:
        """Remove overlapping detections using IoU threshold."""
        if not detections:
            return []

        kept: List[Dict[str, Any]] = []
        used_bboxes: List[Tuple[int, int, int, int]] = []

        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox

            duplicate = False
            for ux1, uy1, ux2, uy2 in used_bboxes:
                ix1, iy1 = max(x1, ux1), max(y1, uy1)
                ix2, iy2 = min(x2, ux2), min(y2, uy2)
                if ix1 < ix2 and iy1 < iy2:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    min_area = min((x2 - x1) * (y2 - y1), (ux2 - ux1) * (uy2 - uy1))
                    if inter / min_area > 0.3:
                        duplicate = True
                        break

            if not duplicate:
                kept.append(det)
                used_bboxes.append(bbox)

        return kept

    # -------------------- Qwen-VL verification --------------------

    def _qwen_verify_parallel(
        self, image: np.ndarray, detections: List[Dict[str, Any]], source_id: str
    ) -> List[Dict[str, Any]]:
        """Submit parallel Qwen-VL verifications for each detection region."""
        if not self.qwen_vl_client:
            logger.warning("Qwen-VL client not available, using YOLO results directly")
            return detections

        to_verify = detections[:8]
        verified: List[Dict[str, Any]] = []
        futures = {}

        for i, det in enumerate(to_verify):
            cropped = self._crop_region(image, det["bbox"])
            if cropped is None:
                continue
            future = self.verification_pool.submit(self._single_verify, cropped, det, i, source_id)
            futures[future] = (det, i)

        for future in as_completed(futures):
            det, idx = futures[future]
            try:
                if future.result():
                    v = det.copy()
                    v["confidence"] = (det["confidence"] + 0.8) / 2
                    v["detection_stage"] = "qwen_verified"
                    verified.append(v)
            except Exception as e:
                logger.error("Verification failed for detection %d: %s", idx + 1, e)
                verified.append(det)  # conservative: keep on failure

        return verified

    def _single_verify(self, cropped: np.ndarray, det: Dict[str, Any], idx: int, source_id: str) -> bool:
        try:
            start = time.perf_counter()
            result = self.qwen_vl_client.verify_smoke_flame(cropped)
            self.qwen_latency.record(time.perf_counter() - start)
            with self.metrics_lock:
                self.qwen_requests_total += 1
            return result
        except Exception as e:
            logger.error("Verification error for detection %d: %s", idx + 1, e)
            return False

    def _crop_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Crop detection region with 20% border expansion, capped at 512px."""
        try:
            x1, y1, x2, y2 = bbox
            h, w = image.shape[:2]
            expand = 0.2
            bw, bh = x2 - x1, y2 - y1

            x1e = max(0, int(x1 - bw * expand))
            y1e = max(0, int(y1 - bh * expand))
            x2e = min(w, int(x2 + bw * expand))
            y2e = min(h, int(y2 + bh * expand))

            if x2e <= x1e or y2e <= y1e:
                return None

            cropped = image[y1e:y2e, x1e:x2e]
            if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
                return None

            max_dim = max(cropped.shape[:2])
            if max_dim > 512:
                scale = 512 / max_dim
                cropped = cv2.resize(cropped, (int(cropped.shape[1] * scale), int(cropped.shape[0] * scale)))

            return cropped
        except Exception as e:
            logger.error("Failed to crop detection region: %s", e)
            return None

    # -------------------- Event saving --------------------

    def _save_events(self, frame_meta: FrameWithMetadata, detections: List[Dict[str, Any]], timestamp: float):
        save_started = time.perf_counter()
        try:
            source_id = frame_meta.source_id
            image = frame_meta.image

            validated = self._validate_bboxes(detections, image.shape)

            rendered = render_official_frame(
                image=image, all_detections=validated, violations=validated, zones=None
            )

            image_url = self.minio_client.upload_frame(
                image_data=rendered, camera_id=source_id, timestamp=timestamp, event_type="smoke_flame"
            )
            if not image_url:
                logger.error("Failed to upload smoke/flame detection image")
                return

            clean_url = image_url.split('?')[0] if '?X-Amz-' in image_url else image_url

            ok = handle_frame_events(
                minio_client=self.minio_client,
                mongo_client=self.mongo_client,
                image_url=clean_url,
                camera_id=source_id,
                timestamp=timestamp,
                frame_index=frame_meta.frame_index,
                violations=validated,
                event_type_override="smoke_flame",
            )

            if ok:
                self.event_counter.add()
                with self.metrics_lock:
                    self.events_saved_total += 1
            else:
                logger.error("Failed to save smoke/flame events to database")

        except Exception as e:
            logger.error("Failed to save smoke/flame events: %s", e)
        finally:
            self.save_latency.record(time.perf_counter() - save_started)

    def _validate_bboxes(self, detections: List[Dict[str, Any]], image_shape: Tuple[int, ...]) -> List[Dict[str, Any]]:
        """Clamp bboxes to image bounds and drop invalid ones."""
        h, w = image_shape[:2]
        valid = []
        for det in detections:
            bbox = det.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = (max(0, min(int(v), dim)) for v, dim in zip(bbox, (w, h, w, h)))
            if x2 > x1 and y2 > y1:
                d = det.copy()
                d["bbox"] = (x1, y1, x2, y2)
                valid.append(d)
        return valid

    def flush_remaining(self):
        """Shutdown thread pools (called during graceful exit)."""
        self.detection_pool.shutdown(wait=True)
        self.verification_pool.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Qwen-VL API Client
# ---------------------------------------------------------------------------

class QwenVLAPIClient:
    """Multi-format Qwen-VL API client (DashScope / OpenAI-compatible / generic)."""

    def __init__(self, api_url: str, api_key: str, model_name: str = "qwen-vl-plus"):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.verify_prompt = "请仔细分析这张图片中是否有烟雾或火焰。只回答'是'或'否'，不要解释。"
        self.timeout = 20

    def verify_smoke_flame(self, image: np.ndarray) -> bool:
        """Return True if smoke/flame is detected in the image."""
        try:
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                logger.error("Failed to encode image for Qwen-VL")
                return False

            b64 = base64.b64encode(encoded).decode('utf-8')
            payload = self._build_payload(b64)

            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            resp = requests.post(self.api_url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()

            answer = self._extract_text(resp.json())
            return self._is_positive(answer)

        except requests.exceptions.Timeout:
            logger.error("Qwen-VL API request timeout")
            return False
        except requests.exceptions.RequestException as e:
            logger.error("Qwen-VL API request failed: %s", e)
            return False
        except Exception as e:
            logger.error("Qwen-VL verification failed: %s", e)
            return False

    def _build_payload(self, image_b64: str) -> Dict[str, Any]:
        """Build request payload for the detected API format."""
        image_data = f"data:image/jpeg;base64,{image_b64}"

        if "openai" in self.api_url or "v1/chat/completions" in self.api_url:
            return {
                "model": self.model_name,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_data}},
                    {"type": "text", "text": self.verify_prompt},
                ]}],
                "max_tokens": 10, "temperature": 0.1,
            }

        if "dashscope" in self.api_url:
            return {
                "model": self.model_name,
                "input": {"messages": [{"role": "user", "content": [
                    {"image": image_data},
                    {"text": self.verify_prompt},
                ]}]},
                "parameters": {"max_tokens": 10, "temperature": 0.1},
            }

        # Generic fallback
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": [
                {"type": "image", "image": image_data},
                {"type": "text", "text": self.verify_prompt},
            ]}],
            "max_tokens": 10, "temperature": 0.1,
        }

    def _extract_text(self, data: Any) -> str:
        """Recursively extract the answer text from various API response formats."""
        if isinstance(data, list):
            if not data:
                return ""
            first = data[0]
            if isinstance(first, dict):
                for key in ("content", "text", "message", "result"):
                    if key in first:
                        val = first[key]
                        return self._extract_text(val) if isinstance(val, list) else str(val).strip().lower()
            if isinstance(first, str):
                return " ".join(str(s).strip() for s in data).lower()
            return ""

        if not isinstance(data, dict):
            return str(data).strip().lower()

        # OpenAI format
        if "choices" in data:
            content = data["choices"][0].get("message", {}).get("content", "")
            return self._extract_text(content) if isinstance(content, list) else str(content).strip().lower()

        # DashScope format
        if "output" in data and "choices" in data["output"]:
            content = data["output"]["choices"][0].get("message", {}).get("content", "")
            return self._extract_text(content) if isinstance(content, list) else str(content).strip().lower()

        # Generic keys
        for key in ("content", "text", "result", "message"):
            if key in data:
                val = data[key]
                return self._extract_text(val) if isinstance(val, list) else str(val).strip().lower()

        logger.warning("Unrecognised API response format: %s", data)
        return str(data)

    @staticmethod
    def _is_positive(answer: str) -> bool:
        """Parse a yes/no answer in Chinese or English."""
        if not answer:
            return False
        cleaned = answer.strip().lower()
        yes_words = {"是", "有", "存在", "确认", "yes", "true", "对的", "正确"}
        no_words = {"否", "没有", "不存在", "未发现", "no", "false", "不是", "错误"}
        has_yes = any(w in cleaned for w in yes_words)
        has_no = any(w in cleaned for w in no_words)
        if has_yes and not has_no:
            return True
        return False


# Module-level singleton (used by main.py's legacy initialisation path)
smoke_flame_detection_service = SmokeFlameDetectionService()
