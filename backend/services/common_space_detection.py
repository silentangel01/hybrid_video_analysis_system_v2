# -*- coding: utf-8 -*-
# backend/services/common_space_detection.py
"""
Common Space Utilization Detection Service

Periodic frame sampling + Qwen-VL analysis for space utilization assessment.
"""

import base64
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

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
from storage.minio_client import MinIOClient
from storage.mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)


class CommonSpaceDetectionService:
    """Periodic sampling + Qwen-VL space utilisation analysis."""

    def __init__(self):
        self.minio_client: Optional[MinIOClient] = None
        self.mongo_client: Optional[MongoDBClient] = None
        self.qwen_vl_client = None

        self.sample_lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        self.analysis_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="common-space")

        self.sample_interval = 30  # seconds
        self.last_sample_time: Dict[str, float] = {}
        self.system_prompt = (
            "You are a professional public space analysis assistant. "
            "Please carefully observe the image and analyze the usage of public space."
        )
        self.user_prompt = (
            "Please analyze the public space usage in this image, including but not limited to: "
            "number of people, activity types, space occupancy rate, and any potential safety hazards. "
            "Provide a detailed analysis report."
        )

        # Location / dispatch metadata (set by StreamRuntimeFactory)
        self.area_code: str = ""
        self.group: str = ""

        # Metrics
        self.received_counter = SlidingCounter(window_sec=10.0)
        self.sampled_counter = SlidingCounter(window_sec=10.0)
        self.event_counter = SlidingCounter(window_sec=10.0)
        self.analysis_latency = LatencyRecorder()
        self.save_latency = LatencyRecorder()
        self.frames_received_total = 0
        self.frames_skipped_total = 0
        self.frames_sampled_total = 0
        self.events_saved_total = 0
        self.analysis_inflight = 0

    # -------------------- Dependency injection --------------------

    def set_clients(self, minio_client: MinIOClient, mongo_client: MongoDBClient):
        self.minio_client = minio_client
        self.mongo_client = mongo_client

    def set_qwen_vl_client(self, qwen_client):
        self.qwen_vl_client = qwen_client

    def set_sample_interval(self, interval_seconds: int):
        self.sample_interval = interval_seconds

    def set_prompts(self, system_prompt: str = None, user_prompt: str = None):
        if system_prompt:
            self.system_prompt = system_prompt
        if user_prompt:
            self.user_prompt = user_prompt

    def get_runtime_metrics(self) -> Dict[str, Any]:
        with self.metrics_lock:
            return {
                "sample_interval": self.sample_interval,
                "frames_received_total": self.frames_received_total,
                "frames_skipped_total": self.frames_skipped_total,
                "frames_sampled_total": self.frames_sampled_total,
                "events_saved_total": self.events_saved_total,
                "analysis_inflight": self.analysis_inflight,
                "received_fps_10s": self.received_counter.snapshot()["rate_per_sec"],
                "sampled_fps_10s": self.sampled_counter.snapshot()["rate_per_sec"],
                "event_fps_10s": self.event_counter.snapshot()["rate_per_sec"],
                "analysis_queue_size": get_thread_pool_queue_size(self.analysis_pool),
                "analysis_latency": self.analysis_latency.snapshot(),
                "save_latency": self.save_latency.snapshot(),
            }

    # -------------------- Main pipeline --------------------

    def process_frame(self, frame_meta: FrameWithMetadata) -> None:
        self.received_counter.add()
        with self.metrics_lock:
            self.frames_received_total += 1

        if not all([self.minio_client, self.mongo_client, self.qwen_vl_client]):
            logger.error("Common space service not fully initialised")
            return

        source_id = frame_meta.source_id
        current_time = time.time()

        if not self._reserve_sample_slot(source_id, current_time):
            with self.metrics_lock:
                self.frames_skipped_total += 1
            return

        self.sampled_counter.add()
        with self.metrics_lock:
            self.frames_sampled_total += 1

        future = self.analysis_pool.submit(self._analyse_frame, frame_meta, current_time)
        future.add_done_callback(lambda f: self._on_done(f, source_id))

    def _reserve_sample_slot(self, source_id: str, now: float) -> bool:
        """Atomically check and claim the next sample slot."""
        with self.sample_lock:
            last = self.last_sample_time.get(source_id)
            if last is not None and (now - last) < self.sample_interval:
                return False
            self.last_sample_time[source_id] = now
            return True

    def _on_done(self, future, source_id: str):
        try:
            future.result()
        except Exception as e:
            logger.error("Common space async worker crashed for %s: %s", source_id, e)

    # -------------------- Analysis --------------------

    def _analyse_frame(self, frame_meta: FrameWithMetadata, timestamp: float):
        started = time.perf_counter()
        with self.metrics_lock:
            self.analysis_inflight += 1

        try:
            source_id = frame_meta.source_id
            image = frame_meta.image

            analysis_text = self._call_qwen_vl(image)
            if not analysis_text:
                logger.warning("Qwen-VL analysis returned empty for %s", source_id)
                return

            summary = self._extract_summary(analysis_text)
            self._save_event(frame_meta, analysis_text, summary, timestamp)

        except Exception as e:
            logger.error("Common space analysis failed for %s: %s", frame_meta.source_id, e)
        finally:
            self.analysis_latency.record(time.perf_counter() - started)
            with self.metrics_lock:
                self.analysis_inflight = max(0, self.analysis_inflight - 1)

    def _call_qwen_vl(self, image: np.ndarray) -> Optional[str]:
        """Send image to Qwen-VL and return the analysis text."""
        try:
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return None

            b64 = base64.b64encode(encoded).decode('utf-8')
            prompt = f"{self.system_prompt}\n\n{self.user_prompt}"

            if hasattr(self.qwen_vl_client, 'chat_completion'):
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": self.user_prompt},
                    ]},
                ]
                resp = self.qwen_vl_client.chat_completion(messages=messages, max_tokens=500, temperature=0.3)
                if resp and "choices" in resp:
                    return resp["choices"][0]["message"]["content"]
                return None

            payload = self._build_payload(b64, prompt)
            headers = {"Authorization": f"Bearer {self.qwen_vl_client.api_key}", "Content-Type": "application/json"}
            resp = requests.post(self.qwen_vl_client.api_url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            return self.qwen_vl_client._extract_text(resp.json())

        except Exception as e:
            logger.error("Qwen-VL analysis call failed: %s", e)
            return None

    def _build_payload(self, image_b64: str, prompt: str) -> Dict[str, Any]:
        url = self.qwen_vl_client.api_url
        image_data = f"data:image/jpeg;base64,{image_b64}"

        if "openai" in url or "v1/chat/completions" in url:
            return {
                "model": self.qwen_vl_client.model_name,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_data}},
                    {"type": "text", "text": prompt},
                ]}],
                "max_tokens": 500, "temperature": 0.3,
            }
        if "dashscope" in url:
            return {
                "model": self.qwen_vl_client.model_name,
                "input": {"messages": [{"role": "user", "content": [
                    {"image": image_data}, {"text": prompt},
                ]}]},
                "parameters": {"max_tokens": 500, "temperature": 0.3},
            }
        return {
            "model": self.qwen_vl_client.model_name,
            "messages": [{"role": "user", "content": [
                {"type": "image", "image": image_data},
                {"type": "text", "text": prompt},
            ]}],
            "max_tokens": 500, "temperature": 0.3,
        }

    # -------------------- Summary extraction --------------------

    def _extract_summary(self, text: str) -> Dict[str, Any]:
        lower = text.lower()
        return {
            "estimated_people_count": self._extract_people_count(lower),
            "space_occupancy": self._estimate_occupancy(lower),
            "activity_types": self._extract_activities(lower),
            "safety_concerns": self._has_safety_concerns(lower),
            "keywords": self._extract_keywords(lower),
        }

    @staticmethod
    def _extract_people_count(text: str) -> int:
        for kw in ("人", "people", "persons", "person"):
            m = re.search(rf'(\d+)\s*{kw}', text)
            if m:
                return int(m.group(1))
        nums = re.findall(r'\b\d+\b', text)
        return int(nums[0]) if nums else 0

    @staticmethod
    def _estimate_occupancy(text: str) -> str:
        levels = {
            "high": ["拥挤", "人多", "high", "crowded", "busy"],
            "medium": ["适中", "一般", "medium", "moderate", "normal"],
            "low": ["空旷", "人少", "low", "empty", "sparse", "few"],
        }
        for level, words in levels.items():
            if any(w in text for w in words):
                return level
        return "unknown"

    @staticmethod
    def _extract_activities(text: str) -> List[str]:
        mapping = {
            "walking": ["行走", "走路", "walking", "walk", "pedestrian"],
            "sitting": ["坐着", "休息", "sitting", "sit"],
            "standing": ["站着", "站立", "standing", "stand"],
            "gathering": ["聚集", "gathering", "crowd", "group"],
            "working": ["工作", "办公", "working", "work"],
            "playing": ["玩耍", "playing", "play"],
            "talking": ["交谈", "聊天", "talking", "talk"],
            "waiting": ["等待", "等候", "waiting", "wait"],
        }
        found = [act for act, words in mapping.items() if any(w in text for w in words)]
        return found or ["unknown"]

    @staticmethod
    def _has_safety_concerns(text: str) -> bool:
        keywords = ["危险", "隐患", "不安全", "danger", "hazard", "unsafe", "emergency",
                     "blocked", "fire", "smoke", "accident"]
        return any(k in text for k in keywords)

    @staticmethod
    def _extract_keywords(text: str, limit: int = 10) -> List[str]:
        stop = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
                "have", "has", "had", "do", "does", "did", "this", "that", "it", "they"}
        words = [w for w in text.split() if w not in stop and len(w) > 2 and w.isalpha()]
        return list(set(words))[:limit]

    # -------------------- Event saving --------------------

    def _save_event(self, frame_meta: FrameWithMetadata, analysis_text: str,
                    summary: Dict[str, Any], timestamp: float):
        save_started = time.perf_counter()
        try:
            source_id = frame_meta.source_id

            image_url = self.minio_client.upload_frame(
                image_data=frame_meta.image, camera_id=source_id,
                timestamp=timestamp, event_type="common_space_analysis"
            )
            if not image_url:
                logger.error("Failed to upload common space image")
                return

            clean_url = image_url.split('?')[0] if '?X-Amz-' in image_url else image_url

            violations = [{
                "analysis_summary": summary,
                "full_analysis": analysis_text,
                "event_type": "common_space_utilization",
            }]

            ok = handle_frame_events(
                minio_client=self.minio_client,
                mongo_client=self.mongo_client,
                image_url=clean_url,
                camera_id=source_id,
                timestamp=timestamp,
                frame_index=frame_meta.frame_index,
                violations=violations,
                area_code=self.area_code or None,
                group=self.group or None,
            )

            if ok:
                self.event_counter.add()
                with self.metrics_lock:
                    self.events_saved_total += 1
            else:
                logger.error("Failed to save common space event for %s", source_id)

        except Exception as e:
            logger.error("Failed to save common space event: %s", e)
        finally:
            self.save_latency.record(time.perf_counter() - save_started)

    def flush_remaining(self):
        """Shutdown analysis pool (called during graceful exit)."""
        self.analysis_pool.shutdown(wait=True)


# Module-level singleton (used by main.py's legacy initialisation path)
common_space_detection_service = CommonSpaceDetectionService()
