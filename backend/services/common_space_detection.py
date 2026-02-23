# backend/services/common_space_detection.py
"""
Common Space Utilization Detection Service —— 公共空间利用率检测服务
Periodic frame sampling and Qwen-VL analysis for space utilization assessment.

公共空间利用率检测服务：
定时抽帧 + Qwen-VL API 分析空间使用情况
"""

import logging
import time
import base64
import requests
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np

from backend.utils.frame_capture import FrameWithMetadata
from storage.minio_client import MinIOClient
from storage.mongodb_client import MongoDBClient
from backend.services.event_generator import handle_frame_events

logger = logging.getLogger(__name__)


class CommonSpaceDetectionService:
    """公共空间利用率检测服务"""

    def __init__(self):
        self.minio_client = None
        self.mongo_client = None
        self.qwen_vl_client = None
        self.sample_interval = 30  # 默认采样间隔：30秒 | Default sampling interval: 30 seconds
        self.last_sample_time = {}  # 记录每个source_id的最后采样时间 | Track last sample time per source_id
        self.system_prompt = "You are a professional public space analysis assistant. Please carefully observe the image and analyze the usage of public space."
        self.user_prompt = "Please analyze the public space usage in this image, including but not limited to: number of people, activity types, space occupancy rate, and any potential safety hazards. Provide a detailed analysis report."

    # -------------------- Dependency Injection --------------------
    def set_clients(self, minio_client: MinIOClient, mongo_client: MongoDBClient):
        """Set storage clients"""
        self.minio_client = minio_client
        self.mongo_client = mongo_client

    def set_qwen_vl_client(self, qwen_client):
        """Set Qwen-VL client"""
        self.qwen_vl_client = qwen_client

    def set_sample_interval(self, interval_seconds: int):
        """Set sampling interval"""
        self.sample_interval = interval_seconds
        logger.info(f"🔄 Set common space sampling interval to {interval_seconds} seconds")

    def set_prompts(self, system_prompt: str = None, user_prompt: str = None):
        """Set prompts (support future customization)"""
        if system_prompt:
            self.system_prompt = system_prompt
        if user_prompt:
            self.user_prompt = user_prompt
        logger.info("🔄 Updated Qwen-VL prompts for common space analysis")

    # -------------------- Main Detection Pipeline --------------------
    def process_frame(self, frame_meta: FrameWithMetadata) -> None:
        """
        处理单帧的公共空间分析流程
        Process single frame for common space analysis pipeline

        Args:
            frame_meta: Frame metadata
        """
        if not all([self.minio_client, self.mongo_client, self.qwen_vl_client]):
            logger.error("❌ Common space service not fully initialized.")
            return

        source_id = frame_meta.source_id
        current_time = time.time()

        # 1. Sampling interval check
        if not self._should_sample_frame(source_id, current_time):
            return

        # 2. Update last sample time
        self.last_sample_time[source_id] = current_time

        image = frame_meta.image
        timestamp = current_time

        logger.info(f"🏢 Sampling frame for common space analysis: {source_id}")

        # 3. Async process analysis pipeline
        import threading
        thread = threading.Thread(
            target=self._process_frame_analysis,
            args=(frame_meta, timestamp),
            daemon=True
        )
        thread.start()

    def _should_sample_frame(self, source_id: str, current_time: float) -> bool:
        """
        判断是否应该采样当前帧
        Determine if current frame should be sampled
        """
        if source_id not in self.last_sample_time:
            self.last_sample_time[source_id] = 0
            return True

        last_sample = self.last_sample_time[source_id]
        time_since_last = current_time - last_sample

        return time_since_last >= self.sample_interval

    def _process_frame_analysis(self, frame_meta: FrameWithMetadata, timestamp: float):
        """
        处理帧分析 - 独立的线程执行
        Process frame analysis - executed in separate thread
        """
        try:
            source_id = frame_meta.source_id
            image = frame_meta.image

            # 1. Call Qwen-VL API for analysis
            analysis_result = self._analyze_with_qwen_vl(image)

            if not analysis_result:
                logger.warning(f"⚠️ Qwen-VL analysis failed for {source_id}")
                return

            logger.info(f"✅ Qwen-VL analysis completed for {source_id}")

            # 2. Create standardized event
            event_data = self._create_space_utilization_event(
                frame_meta, analysis_result, timestamp
            )

            # 3. Save event to storage
            self._save_analysis_event(frame_meta, event_data, timestamp)

        except Exception as e:
            logger.error(f"❌ Common space analysis failed for {frame_meta.source_id}: {e}")

    def _analyze_with_qwen_vl(self, image: np.ndarray) -> Optional[str]:
        """
        使用Qwen-VL分析公共空间使用情况
        Analyze common space utilization with Qwen-VL

        Args:
            image: Input image

        Returns:
            Optional[str]: Analysis result text
        """
        try:
            # Encode image to base64
            success, encoded_image = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                logger.error("❌ Failed to encode image for Qwen-VL")
                return None

            image_base64 = base64.b64encode(encoded_image).decode('utf-8')

            # Construct request payload based on API type
            if hasattr(self.qwen_vl_client, 'chat_completion'):
                # If client has chat_completion method (enhanced version)
                messages = [
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": self.user_prompt
                            }
                        ]
                    }
                ]

                response = self.qwen_vl_client.chat_completion(
                    messages=messages,
                    max_tokens=500,
                    temperature=0.3
                )

                if response and "choices" in response:
                    analysis_text = response["choices"][0]["message"]["content"]
                else:
                    logger.error("❌ Invalid response from Qwen-VL API")
                    return None

            else:
                # Use existing verify_smoke_flame style
                # First, prepare the combined prompt
                combined_prompt = f"{self.system_prompt}\n\n{self.user_prompt}"

                # Encode image and send request
                payload = self._build_request_payload(image_base64, combined_prompt)

                headers = {
                    "Authorization": f"Bearer {self.qwen_vl_client.api_key}",
                    "Content-Type": "application/json"
                }

                logger.debug(f"🔍 Sending request to Qwen-VL API: {self.qwen_vl_client.api_url}")
                response = requests.post(
                    self.qwen_vl_client.api_url,
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()

                result = response.json()
                analysis_text = self.qwen_vl_client._parse_response(result)

            logger.debug(f"🔍 Qwen-VL analysis result preview: {analysis_text[:200]}...")
            return analysis_text

        except Exception as e:
            logger.error(f"❌ Qwen-VL analysis failed: {e}")
            return None

    def _build_request_payload(self, image_base64: str, prompt: str) -> Dict[str, Any]:
        """
        构建API请求载荷
        Build API request payload

        Args:
            image_base64: Base64 encoded image
            prompt: Combined prompt text

        Returns:
            Dict: Request payload
        """
        api_url = self.qwen_vl_client.api_url

        # OpenAI compatible format
        if "openai" in api_url or "v1/chat/completions" in api_url:
            return {
                "model": self.qwen_vl_client.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.3
            }
        # DashScope format
        elif "dashscope" in api_url:
            return {
                "model": self.qwen_vl_client.model_name,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "image": f"data:image/jpeg;base64,{image_base64}"
                                },
                                {
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                },
                "parameters": {
                    "max_tokens": 500,
                    "temperature": 0.3
                }
            }
        # Generic format
        else:
            return {
                "model": self.qwen_vl_client.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{image_base64}"
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.3
            }

    def _create_space_utilization_event(self, frame_meta: FrameWithMetadata,
                                        analysis_result: str, timestamp: float) -> Dict[str, Any]:
        """
        创建公共空间利用率事件数据结构
        Create common space utilization event data structure

        Args:
            frame_meta: Frame metadata
            analysis_result: Analysis result text
            timestamp: Timestamp

        Returns:
            Dict: Event data
        """
        # Extract key information from analysis result
        summary = self._extract_summary_from_analysis(analysis_result)

        event_data = {
            "camera_id": frame_meta.source_id,
            "timestamp": timestamp,
            "frame_index": frame_meta.frame_index,
            "event_type": "common_space_utilization",
            "analysis_result": analysis_result,
            "summary": summary,
            "confidence": 1.0,  # Qwen-VL analysis always considered high confidence
            "detection_stage": "qwen_vl_analysis",
            "metadata": {
                "sample_interval": self.sample_interval,
                "system_prompt": self.system_prompt[:100] + "..." if len(
                    self.system_prompt) > 100 else self.system_prompt,
                "user_prompt": self.user_prompt[:100] + "..." if len(self.user_prompt) > 100 else self.user_prompt
            }
        }

        return event_data

    def _extract_summary_from_analysis(self, analysis_result: str) -> Dict[str, Any]:
        """
        从分析结果中提取结构化摘要
        Extract structured summary from analysis result

        Args:
            analysis_result: Complete analysis text

        Returns:
            Dict: Structured summary
        """
        analysis_lower = analysis_result.lower()

        summary = {
            "estimated_people_count": self._extract_number(analysis_lower, ["人", "people", "persons", "person"]),
            "space_occupancy": self._estimate_occupancy(analysis_lower),
            "activity_types": self._extract_activity_types(analysis_lower),
            "safety_concerns": self._check_safety_concerns(analysis_lower),
            "keywords": self._extract_keywords(analysis_lower)
        }

        return summary

    def _extract_number(self, text: str, keywords: list) -> int:
        """从文本中提取数字 | Extract number from text"""
        import re
        for keyword in keywords:
            pattern = rf'(\d+)\s*{keyword}'
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))

        # Also try to find any number in the text
        numbers = re.findall(r'\b\d+\b', text)
        if numbers:
            return int(numbers[0])

        return 0

    def _estimate_occupancy(self, text: str) -> str:
        """估计空间占用率 | Estimate space occupancy"""
        occupancy_keywords = {
            "high": ["拥挤", "拥挤的", "人多", "high", "crowded", "busy", "crowd", "busy"],
            "medium": ["适中", "适度", "一般", "medium", "moderate", "normal", "average"],
            "low": ["空旷", "稀少", "人少", "low", "empty", "sparse", "few", "quiet"]
        }

        for level, keywords in occupancy_keywords.items():
            if any(keyword in text for keyword in keywords):
                return level

        return "unknown"

    def _extract_activity_types(self, text: str) -> list:
        """提取活动类型 | Extract activity types"""
        activities = []
        activity_keywords = {
            "walking": ["行走", "走路", "步行", "walking", "walk", "pedestrian"],
            "sitting": ["坐着", "坐", "休息", "sitting", "sit", "seated"],
            "standing": ["站着", "站立", "standing", "stand"],
            "gathering": ["聚集", "集会", "聚集的", "gathering", "crowd", "group"],
            "working": ["工作", "办公", "working", "work", "office"],
            "playing": ["玩耍", "游戏", "playing", "play", "children"],
            "talking": ["交谈", "谈话", "聊天", "talking", "talk", "conversation"],
            "waiting": ["等待", "等候", "waiting", "wait", "queue"]
        }

        for activity, keywords in activity_keywords.items():
            if any(keyword in text for keyword in keywords):
                activities.append(activity)

        return activities if activities else ["unknown"]

    def _check_safety_concerns(self, text: str) -> bool:
        """检查是否存在安全隐患 | Check for safety concerns"""
        safety_keywords = ["危险", "隐患", "不安全", "拥挤过度", "堵塞", "紧急",
                           "危险", "danger", "hazard", "unsafe", "emergency", "blocked",
                           "fire", "smoke", "accident", "problem", "issue", "concern"]
        return any(keyword in text for keyword in safety_keywords)

    def _extract_keywords(self, text: str, max_keywords: int = 10) -> list:
        """提取关键词 | Extract keywords"""
        # Simple tokenization and filtering
        words = text.lower().split()

        # Remove common words and very short words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
                      "for", "of", "with", "by", "is", "are", "was", "were", "be",
                      "been", "being", "have", "has", "had", "do", "does", "did",
                      "this", "that", "these", "those", "it", "its", "they", "their"}

        keywords = [word for word in words
                    if word not in stop_words
                    and len(word) > 2
                    and not word.isdigit()
                    and word.isalpha()]

        return list(set(keywords))[:max_keywords]

    def _save_analysis_event(self, frame_meta: FrameWithMetadata,
                             event_data: Dict[str, Any], timestamp: float):
        """
        保存分析事件到存储
        Save analysis event to storage

        Args:
            frame_meta: Frame metadata
            event_data: Event data
            timestamp: Timestamp
        """
        try:
            source_id = frame_meta.source_id
            image = frame_meta.image

            # 1. Upload original image to MinIO
            image_url = self.minio_client.upload_frame(
                image_data=image,
                camera_id=source_id,
                timestamp=timestamp,
                event_type="common_space_analysis"
            )

            if not image_url:
                logger.error("❌ Failed to upload common space analysis image")
                return

            # Clean URL
            clean_url = image_url.split('?')[0] if '?X-Amz-' in image_url else image_url
            logger.info(f"📸 Common space analysis image URL: {clean_url}")

            # 2. Pass analysis result as violations field
            violations = [{
                "analysis_summary": event_data["summary"],
                "full_analysis": event_data["analysis_result"],
                "event_type": "common_space_utilization"
            }]

            # 3. Save to MongoDB
            ok = handle_frame_events(
                minio_client=self.minio_client,
                mongo_client=self.mongo_client,
                image_url=clean_url,
                camera_id=source_id,
                timestamp=timestamp,
                frame_index=frame_meta.frame_index,
                violations=violations
            )

            if ok:
                logger.info(f"✅ Common space analysis event saved: {source_id}")
                logger.info(f"   📋 Analysis summary: {event_data['summary']}")
            else:
                logger.error("❌ Failed to save common space analysis event to database")

        except Exception as e:
            logger.error(f"❌ Failed to save common space analysis event: {e}")

    def flush_remaining(self):
        """处理剩余帧（兼容性方法）| Process remaining frames (compatibility method)"""
        logger.info("🔄 Common space detection - Flush remaining called")


# -------------------- Global Instance --------------------
common_space_detection_service = CommonSpaceDetectionService()