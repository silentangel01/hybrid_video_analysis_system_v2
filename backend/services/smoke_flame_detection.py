# -*- coding: utf-8 -*-
# backend/services/smoke_flame_detection.py
"""
Smoke and Flame Detection Service —— 烟火检测服务
Two-stage detection pipeline:
1. YOLOv8 preliminary detection (smoke_flame.pt)
2. Qwen-VL API verification for positive cases
3. Double confirmation → save event

双阶段检测流水线：
1. YOLOv8 初步检测（smoke_flame.pt）
2. 阳性结果 → Qwen-VL API 验证
3. 双重确认 → 保存事件
"""

import logging
import time
import base64
import requests
import threading
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from ml_models.yolov8.inference import YOLOInference
from ml_models.yolov8.model_loader import YOLOModelLoader
from backend.services.event_generator import handle_frame_events
from backend.utils.visualization import render_official_frame
from storage.minio_client import MinIOClient
from storage.mongodb_client import MongoDBClient
from backend.utils.frame_capture import FrameWithMetadata
from backend.utils.performance_metrics import (
    SlidingCounter,
    LatencyRecorder,
    get_thread_pool_queue_size,
)

logger = logging.getLogger(__name__)


class SmokeFlameDetectionService:
    """烟火检测服务 - 双阶段验证"""

    def __init__(self):
        self.minio_client = None
        self.mongo_client = None
        self.model_loader = None
        self.yolo_model = None
        self.qwen_vl_client = None
        # LEGACY: one shared pool for both outer frame jobs and inner Qwen verification
        # could starve itself when _process_frame_async submitted nested work back into
        # the same executor.
        self.detection_pool = ThreadPoolExecutor(
            max_workers=6,
            thread_name_prefix="smoke-detect"
        )
        self.verification_pool = ThreadPoolExecutor(
            max_workers=8,
            thread_name_prefix="smoke-verify"
        )
        self.frame_skip = 23  # 跳帧设置，每24帧处理1帧 | Frame skipping, process 1 frame every 24 frames
        self.last_processed_frame = {}  # 记录每个source_id的最后处理帧 | Track last processed frame per source_id
        # LEGACY: self.thread_pool = ThreadPoolExecutor(max_workers=6)  # shared outer/inner pool
        self.detection_cache = {}  # 检测结果缓存 | Detection result cache
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

    # -------------------- 依赖注入 --------------------
    def set_clients(self, minio_client: MinIOClient, mongo_client: MongoDBClient):
        """Set storage clients"""
        self.minio_client = minio_client
        self.mongo_client = mongo_client

    def set_model_loader(self, loader: YOLOModelLoader):
        """Set model loader"""
        self.model_loader = loader
        self.yolo_model = self.model_loader.get_model("smoke_flame")

    def set_qwen_vl_client(self, qwen_client):
        """Set Qwen-VL client"""
        self.qwen_vl_client = qwen_client

    def set_frame_skip(self, skip_frames: int):
        """Set frame skipping"""
        self.frame_skip = skip_frames
        logger.debug(f"🔄 Set frame skip to {skip_frames} frames")

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

    # -------------------- 主检测流程 --------------------
    def process_frame(self, frame_meta: FrameWithMetadata) -> None:
        """
        处理单帧的烟火检测流程
        Process single frame for smoke/flame detection pipeline - optimized version

        Args:
            frame_meta: 帧元数据 | Frame metadata
        """
        self.received_counter.add()
        with self.metrics_lock:
            self.frames_received_total += 1

        if not all([self.minio_client, self.mongo_client, self.yolo_model]):
            logger.error(f"❌ Service not fully initialized. "
                         f"minio={self.minio_client is not None}, "
                         f"mongo={self.mongo_client is not None}, "
                         f"yolo={self.yolo_model is not None}")
            return

        source_id = frame_meta.source_id
        frame_index = frame_meta.frame_index

        # 1. 跳帧检测 - 大幅减少处理帧数 | Frame skipping - significantly reduce processed frames
        if not self._should_process_frame(source_id, frame_index):
            with self.metrics_lock:
                self.frames_skipped_total += 1
            return

        # 2. 更新最后处理帧 | Update last processed frame
        self.last_processed_frame[source_id] = frame_index

        image = frame_meta.image

        # 使用当前系统时间戳 | Use current system timestamp
        current_timestamp = time.time()

        logger.debug(f"🔥 Processing frame {frame_index} for {source_id}")

        # 3. 异步处理检测流程 | Async process detection pipeline
        # LEGACY: future = self.thread_pool.submit(self._process_frame_async, frame_meta, current_timestamp)
        self.submitted_counter.add()
        with self.metrics_lock:
            self.frames_submitted_total += 1
        future = self.detection_pool.submit(self._process_frame_async, frame_meta, current_timestamp)
        future.add_done_callback(lambda f: self._handle_processing_result(f, source_id, frame_index))

    def _should_process_frame(self, source_id: str, frame_index: int) -> bool:
        """
        Determine if current frame should be processed.
        """
        if source_id not in self.last_processed_frame:
            self.last_processed_frame[source_id] = -1
            return True

        last_processed = self.last_processed_frame[source_id]
        frames_since_last = frame_index - last_processed

        return frames_since_last > self.frame_skip

    def _process_frame_async(self, frame_meta: FrameWithMetadata, timestamp: float) -> Dict[str, Any]:
        """
        Async frame processing.
        """
        async_started = time.perf_counter()
        try:
            source_id = frame_meta.source_id
            image = frame_meta.image

            # 1. YOLOv8 初步检测 | YOLOv8 preliminary detection
            yolo_detections = self._yolo_detection_enhanced(image)

            if not yolo_detections:
                with self.metrics_lock:
                    self.last_candidate_count = 0
                    self.last_verified_count = 0
                return {"source_id": source_id, "detections": [], "timestamp": timestamp, "frame_meta": frame_meta}

            with self.metrics_lock:
                self.frames_with_candidates_total += 1
                self.last_candidate_count = len(yolo_detections)
            logger.debug(f"🔥 YOLOv8 detected {len(yolo_detections)} potential smoke/flame regions in {source_id}")

            # 2. 对检测结果进行分组和筛选 | Group and filter detection results
            filtered_detections = self._filter_and_group_detections(yolo_detections, image.shape)

            if not filtered_detections:
                logger.debug(f"🟡 No valid detections after filtering in {source_id}")
                return {"source_id": source_id, "detections": [], "timestamp": timestamp, "frame_meta": frame_meta}

            logger.debug(f"🔍 After filtering: {len(filtered_detections)} regions for Qwen-VL verification")

            # 3. Qwen-VL API 验证 - 并行处理多个检测区域 | Qwen-VL API verification - parallel processing
            verified_detections = self._qwen_vl_verification_parallel(image, filtered_detections, source_id)

            if not verified_detections:
                with self.metrics_lock:
                    self.last_verified_count = 0
                logger.debug(f"🟡 Qwen-VL rejected all YOLOv8 detections in {source_id}")
                return {"source_id": source_id, "detections": [], "timestamp": timestamp, "frame_meta": frame_meta}

            with self.metrics_lock:
                self.frames_verified_total += 1
                self.last_verified_count = len(verified_detections)
            logger.debug(f"✅ Qwen-VL verified {len(verified_detections)} smoke/flame events in {source_id}")

            return {
                "source_id": source_id,
                "detections": verified_detections,
                "timestamp": timestamp,
                "frame_meta": frame_meta
            }

        except Exception as e:
            logger.error(f"❌ Async frame processing failed: {e}")
            return {"source_id": frame_meta.source_id, "detections": [], "timestamp": timestamp,
                    "frame_meta": frame_meta}
        finally:
            self.async_detection_latency.record(time.perf_counter() - async_started)

    def _handle_processing_result(self, future, source_id: str, frame_index: int):
        """
        处理异步检测结果 | Handle async detection results
        """
        try:
            result = future.result()
            detections = result["detections"]

            if detections:
                # 保存检测事件 | Save detection events
                self._save_detection_events(result["frame_meta"], detections, result["timestamp"])

        except Exception as e:
            logger.error(f"❌ Error handling processing result for {source_id} frame {frame_index}: {e}")

    def _yolo_detection_enhanced(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        YOLOv8 烟火初步检测 - 增强版本，提高检测灵敏度
        YOLOv8 preliminary smoke/flame detection - enhanced version with higher sensitivity

        Args:
            image: 输入图像 | Input image

        Returns:
            List[Dict]: 检测结果列表 | List of detection results
        """
        try:
            # 保持原始分辨率以提高小目标检测能力 | Keep original resolution for better small object detection
            original_shape = image.shape

            # 只在图像很大时才降采样 | Only downsample if image is very large
            if original_shape[0] > 1080 or original_shape[1] > 1920:
                scale_factor = 1080 / original_shape[0]
                new_width = int(original_shape[1] * scale_factor)
                resized_image = cv2.resize(image, (new_width, 1080))
                logger.debug(f"🖼️ Resized image from {original_shape} to {resized_image.shape}")
            else:
                resized_image = image

            # 运行YOLOv8检测 - 使用更低的置信度阈值 | Run YOLOv8 detection - use lower confidence threshold
            start_time = time.perf_counter()
            raw_detections = YOLOInference.run_detection(
                self.yolo_model, resized_image, conf_threshold=0.10  # 降低阈值提高召回率 | Lower threshold for better recall
            )
            detection_time = time.perf_counter() - start_time
            self.yolo_latency.record(detection_time)
            logger.debug(f"⏱️ YOLOv8 detection time: {detection_time:.3f}s")

            # 转换为标准格式 | Convert to standard format
            detections = []
            for cls, conf, bbox in raw_detections:
                if cls.lower() in ["smoke", "fire", "flame"]:
                    # 如果调整了图像大小，需要调整边界框坐标 | If image was resized, adjust bbox coordinates
                    if resized_image is not image:
                        scale_x = original_shape[1] / resized_image.shape[1]
                        scale_y = original_shape[0] / resized_image.shape[0]
                        x1, y1, x2, y2 = bbox
                        bbox = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)

                    # 确保边界框坐标是整数 | Ensure bbox coordinates are integers
                    x1, y1, x2, y2 = map(int, bbox)
                    normalized_bbox = (x1, y1, x2, y2)

                    # 计算边界框面积 | Calculate bbox area
                    bbox_area = (x2 - x1) * (y2 - y1)
                    image_area = original_shape[0] * original_shape[1]
                    area_ratio = bbox_area / image_area

                    # 过滤掉太小的检测框 | Filter out too small detections
                    if area_ratio < 0.0001:  # 小于图像面积的0.01%
                        logger.debug(f"🔍 Skipped small detection: {area_ratio:.6f}")
                        continue

                    detections.append({
                        "class_name": cls,
                        "confidence": float(conf),
                        "bbox": normalized_bbox,
                        "detection_stage": "yolo_initial",
                        "area_ratio": area_ratio
                    })

            # 按置信度排序 | Sort by confidence
            detections.sort(key=lambda x: x["confidence"], reverse=True)

            logger.debug(f"🔥 YOLOv8 raw detections: {len(raw_detections)}, filtered: {len(detections)}")
            return detections

        except Exception as e:
            logger.error(f"❌ YOLOv8 detection failed: {e}")
            return []

    def _filter_and_group_detections(self, detections: List[Dict[str, Any]], image_shape: Tuple[int, int, int]) -> List[
        Dict[str, Any]]:
        """
        过滤和分组检测结果，避免重复检测同一区域
        Filter and group detection results to avoid duplicate detections in same area

        Args:
            detections: 原始检测结果 | Original detection results
            image_shape: 图像形状 | Image shape

        Returns:
            List[Dict]: 过滤后的检测结果 | Filtered detection results
        """
        if not detections:
            return []

        h, w = image_shape[:2]
        filtered_detections = []
        used_areas = []  # 记录已使用的区域 | Record used areas

        for detection in detections:
            try:
                bbox = detection["bbox"]
                x1, y1, x2, y2 = bbox

                # 计算当前检测框的中心点 | Calculate center point of current bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # 检查是否与已有检测框重叠 | Check if overlaps with existing detections
                is_duplicate = False
                for used_bbox in used_areas:
                    ux1, uy1, ux2, uy2 = used_bbox
                    # 计算IoU | Calculate IoU
                    inter_x1 = max(x1, ux1)
                    inter_y1 = max(y1, uy1)
                    inter_x2 = min(x2, ux2)
                    inter_y2 = min(y2, uy2)

                    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                        # 计算重叠面积 | Calculate overlap area
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        current_area = (x2 - x1) * (y2 - y1)
                        used_area = (ux2 - ux1) * (uy2 - uy1)

                        iou = inter_area / min(current_area, used_area)

                        # 如果IoU大于阈值，认为是重复检测 | If IoU > threshold, consider as duplicate
                        if iou > 0.3:
                            is_duplicate = True
                            logger.debug(f"🔍 Skipped duplicate detection with IoU: {iou:.3f}")
                            break

                if not is_duplicate:
                    filtered_detections.append(detection)
                    used_areas.append(bbox)

            except Exception as e:
                logger.error(f"❌ Error in detection filtering: {e}")
                continue

        logger.debug(f"🔍 Detection filtering: {len(detections)} -> {len(filtered_detections)}")
        return filtered_detections

    def _qwen_vl_verification_parallel(self, image: np.ndarray, yolo_detections: List[Dict[str, Any]],
                                       source_id: str) -> List[Dict[str, Any]]:
        """
        Qwen-VL API 双重验证 - 并行版本
        Qwen-VL API double verification - parallel version

        Args:
            image: 原始图像 | Original image
            yolo_detections: YOLOv8检测结果 | YOLOv8 detection results
            source_id: 源标识 | Source identifier

        Returns:
            List[Dict]: 验证通过的检测结果 | Verified detection results
        """
        if not self.qwen_vl_client:
            logger.warning("⚠️ Qwen-VL client not available, using YOLOv8 results directly")
            return yolo_detections

        # 增加最大并行验证数量 | Increase maximum parallel verifications
        max_parallel_verifications = 8  # 增加到8个并行验证
        detections_to_verify = yolo_detections[:max_parallel_verifications]

        verified_detections = []
        verification_futures = {}

        # 提交并行验证任务 | Submit parallel verification tasks
        for i, detection in enumerate(detections_to_verify):
            try:
                # 提取检测区域 | Extract detection region
                bbox = detection["bbox"]
                cropped_image = self._crop_detection_region(image, bbox)

                if cropped_image is None:
                    continue

                # 提交验证任务 | Submit verification task
                # LEGACY: future = self.thread_pool.submit(
                future = self.verification_pool.submit(
                    self._single_verification,
                    cropped_image, detection, i, source_id
                )
                verification_futures[future] = (detection, i)

            except Exception as e:
                logger.error(f"❌ Failed to submit verification for detection {i + 1}: {e}")
                # 验证失败时保守处理：保留YOLO结果 | Conservative approach: keep YOLO result on failure
                verified_detections.append(detection)

        # 收集验证结果 | Collect verification results
        for future in as_completed(verification_futures):
            detection, original_index = verification_futures[future]
            try:
                is_verified = future.result()
                if is_verified:
                    verified_detection = detection.copy()
                    verified_detection["confidence"] = (detection["confidence"] + 0.8) / 2
                    verified_detection["detection_stage"] = "qwen_verified"
                    verified_detections.append(verified_detection)
                    logger.debug(f"✅ Qwen-VL verified detection {original_index + 1}: {detection['class_name']}")
                else:
                    logger.debug(f"❌ Qwen-VL rejected detection {original_index + 1}: {detection['class_name']}")
            except Exception as e:
                logger.error(f"❌ Verification failed for detection {original_index + 1}: {e}")
                # 验证失败时保守处理：保留YOLO结果
                verified_detections.append(detection)

        return verified_detections

    def _single_verification(self, cropped_image: np.ndarray, detection: Dict[str, Any],
                             index: int, source_id: str) -> bool:
        """
        单个检测区域的验证 | Single detection region verification
        """
        try:
            start_time = time.perf_counter()
            is_verified = self.qwen_vl_client.verify_smoke_flame(cropped_image)
            verification_time = time.perf_counter() - start_time
            self.qwen_latency.record(verification_time)
            with self.metrics_lock:
                self.qwen_requests_total += 1
            logger.debug(f"⏱️ Qwen-VL verification {index + 1} time: {verification_time:.3f}s")
            return is_verified
        except Exception as e:
            logger.error(f"❌ Single verification failed for detection {index + 1}: {e}")
            return False

    def _crop_detection_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        裁剪检测区域用于Qwen-VL分析 - 增加边界扩展
        Crop detection region for Qwen-VL analysis - with border expansion

        Args:
            image: 原始图像 | Original image
            bbox: 边界框 (x1, y1, x2, y2) | Bounding box (x1, y1, x2, y2)

        Returns:
            Optional[np.ndarray]: 裁剪后的图像 | Cropped image
        """
        try:
            x1, y1, x2, y2 = bbox

            # 扩展边界框以包含更多上下文 | Expand bbox to include more context
            expand_ratio = 0.2  # 扩展20%
            width = x2 - x1
            height = y2 - y1

            x1_expanded = max(0, int(x1 - width * expand_ratio))
            y1_expanded = max(0, int(y1 - height * expand_ratio))
            x2_expanded = min(image.shape[1], int(x2 + width * expand_ratio))
            y2_expanded = min(image.shape[0], int(y2 + height * expand_ratio))

            # 确保坐标在图像范围内 | Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x1_expanded = max(0, x1_expanded)
            y1_expanded = max(0, y1_expanded)
            x2_expanded = min(w, x2_expanded)
            y2_expanded = min(h, y2_expanded)

            if x2_expanded <= x1_expanded or y2_expanded <= y1_expanded:
                return None

            cropped = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded]

            # 确保裁剪区域足够大 | Ensure cropped region is large enough
            if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
                return None

            # 限制最大尺寸以减少API传输时间 | Limit maximum size to reduce API transmission time
            max_size = 512
            if cropped.shape[0] > max_size or cropped.shape[1] > max_size:
                scale = max_size / max(cropped.shape[0], cropped.shape[1])
                new_width = int(cropped.shape[1] * scale)
                new_height = int(cropped.shape[0] * scale)
                cropped = cv2.resize(cropped, (new_width, new_height))

            return cropped

        except Exception as e:
            logger.error(f"❌ Failed to crop detection region: {e}")
            return None

    def _save_detection_events(self, frame_meta: FrameWithMetadata,
                               detections: List[Dict[str, Any]], timestamp: float):
        """
        保存检测事件到存储
        Save detection events to storage

        Args:
            frame_meta: 帧元数据 | Frame metadata
            detections: 检测结果 | Detection results
            timestamp: 时间戳 | Timestamp
        """
        save_started = time.perf_counter()
        try:
            source_id = frame_meta.source_id
            image = frame_meta.image

            # 确保检测结果的边界框格式正确 | Ensure bbox format is correct in detection results
            validated_detections = self._validate_detection_bboxes(detections, image.shape)

            # 渲染结果图像 | Render result image
            rendered = render_official_frame(
                image=image,
                all_detections=validated_detections,
                violations=validated_detections,
                zones=None
            )

            # 上传到MinIO | Upload to MinIO
            image_url = self.minio_client.upload_frame(
                image_data=rendered,
                camera_id=source_id,
                timestamp=timestamp,
                event_type="smoke_flame"
            )

            if not image_url:
                logger.error("❌ Failed to upload smoke/flame detection image")
                return

            # 清理URL | Clean URL
            clean_url = image_url.split('?')[0] if '?X-Amz-' in image_url else image_url
            logger.info(f"📸 Smoke/Flame image URL: {clean_url}")

            # 保存到MongoDB | Save to MongoDB
            ok = handle_frame_events(
                minio_client=self.minio_client,
                mongo_client=self.mongo_client,
                image_url=clean_url,
                camera_id=source_id,
                timestamp=timestamp,
                frame_index=frame_meta.frame_index,
                violations=validated_detections,
                event_type_override="smoke_flame"
            )

            if ok:
                self.event_counter.add()
                with self.metrics_lock:
                    self.events_saved_total += 1
                logger.debug(f"✅ Smoke/Flame events saved: {len(validated_detections)} detections")
            else:
                logger.error("❌ Failed to save smoke/flame events to database")

        except Exception as e:
            logger.error(f"❌ Failed to save smoke/flame detection events: {e}")
        finally:
            self.save_latency.record(time.perf_counter() - save_started)

    def _validate_detection_bboxes(self, detections: List[Dict[str, Any]], image_shape: Tuple[int, int, int]) -> List[
        Dict[str, Any]]:
        """
        验证和修复检测结果的边界框格式
        Validate and fix bbox format in detection results

        Args:
            detections: 检测结果列表 | List of detection results
            image_shape: 图像形状 (h, w, c) | Image shape (h, w, c)

        Returns:
            List[Dict]: 验证后的检测结果 | Validated detection results
        """
        validated_detections = []
        h, w = image_shape[:2]

        for detection in detections:
            try:
                bbox = detection["bbox"]

                # 检查边界框格式 | Check bbox format
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    # 确保所有坐标都是整数 | Ensure all coordinates are integers
                    x1, y1, x2, y2 = map(int, bbox)

                    # 确保坐标在合理范围内 | Ensure coordinates are within reasonable range
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))

                    # 确保边界框有效 | Ensure bbox is valid
                    if x2 > x1 and y2 > y1:
                        validated_detection = detection.copy()
                        validated_detection["bbox"] = (x1, y1, x2, y2)
                        validated_detections.append(validated_detection)
                    else:
                        logger.warning(f"⚠️ Invalid bbox skipped: {bbox}")
                else:
                    logger.warning(f"⚠️ Invalid bbox format: {bbox}, type: {type(bbox)}")

            except Exception as e:
                logger.error(f"❌ Error validating bbox {detection.get('bbox')}: {e}")
                # 跳过无效的检测 | Skip invalid detection

        if len(validated_detections) != len(detections):
            logger.warning(f"⚠️ Bbox validation: {len(detections)} -> {len(validated_detections)} valid detections")

        return validated_detections

    def flush_remaining(self):
        """处理剩余帧（兼容性方法）| Process remaining frames (compatibility method)"""
        logger.debug("🔄 Smoke/Flame detection - Flush remaining called")
        # 等待所有异步任务完成 | Wait for all async tasks to complete
        # LEGACY: self.thread_pool.shutdown(wait=True)
        self.detection_pool.shutdown(wait=True)
        self.verification_pool.shutdown(wait=True)


# -------------------- Qwen-VL API 客户端 --------------------
class QwenVLAPIClient:
    """Qwen-VL API 客户端 | Qwen-VL API Client"""

    def __init__(self, api_url: str, api_key: str, model_name: str = "qwen-vl-plus"):
        """
        初始化Qwen-VL API客户端
        Initialize Qwen-VL API client

        Args:
            api_url: API端点 | API endpoint (e.g., DashScope, OpenAI-compatible API)
            api_key: API密钥 | API key
            model_name: 模型名称 | Model name
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.verify_prompt = "请仔细分析这张图片中是否有烟雾或火焰。只回答'是'或'否'，不要解释。"
        self.timeout = 20  # 稍微增加超时时间 | Slightly increase timeout

    def verify_smoke_flame(self, image: np.ndarray) -> bool:
        """
        验证图像中是否有烟雾或火焰
        Verify if there is smoke or flame in the image

        Args:
            image: 输入图像 | Input image

        Returns:
            bool: 是否存在烟雾或火焰 | Whether smoke or flame exists
        """
        try:
            # 编码图像为base64 | Encode image to base64
            success, encoded_image = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])  # 适当提高质量
            if not success:
                logger.error("❌ Failed to encode image for Qwen-VL")
                return False

            image_base64 = base64.b64encode(encoded_image).decode('utf-8')

            # 构造请求载荷 | Construct request payload
            payload = self._build_request_payload(image_base64)

            # 发送请求 | Send request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            logger.debug(f"🔍 Sending request to Qwen-VL API: {self.api_url}")
            start_time = time.time()
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=self.timeout)
            response_time = time.time() - start_time
            logger.debug(f"⏱️ Qwen-VL API response time: {response_time:.3f}s")

            response.raise_for_status()

            # 解析响应 | Parse response
            result = response.json()
            logger.debug(f"🔍 Qwen-VL API raw response: {result}")

            # 解析回答 | Parse answer
            answer = self._parse_response(result)
            logger.debug(f"🔍 Qwen-VL parsed answer: '{answer}'")

            return self._parse_verification_result(answer)

        except requests.exceptions.Timeout:
            logger.error("❌ Qwen-VL API request timeout")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Qwen-VL API request failed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Qwen-VL verification failed: {e}")
            return False

    def _build_request_payload(self, image_base64: str) -> Dict[str, Any]:
        """
        构建API请求载荷
        Build API request payload

        Args:
            image_base64: base64编码的图像 | Base64 encoded image

        Returns:
            Dict: 请求载荷 | Request payload
        """
        # 方法1: OpenAI兼容格式（适用于大多数VL模型API）
        if "openai" in self.api_url or "v1/chat/completions" in self.api_url:
            return {
                "model": self.model_name,
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
                                "text": self.verify_prompt
                            }
                        ]
                    }
                ],
                "max_tokens": 10,
                "temperature": 0.1
            }
        # 方法2: DashScope格式（阿里云通义千问）
        elif "dashscope" in self.api_url:
            return {
                "model": self.model_name,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "image": f"data:image/jpeg;base64,{image_base64}"
                                },
                                {
                                    "text": self.verify_prompt
                                }
                            ]
                        }
                    ]
                },
                "parameters": {
                    "max_tokens": 10,
                    "temperature": 0.1
                }
            }
        # 方法3: 通用格式
        else:
            return {
                "model": self.model_name,
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
                                "text": self.verify_prompt
                            }
                        ]
                    }
                ],
                "max_tokens": 10,
                "temperature": 0.1
            }

    def _parse_response(self, response_data: Dict[str, Any]) -> str:
        """
        解析API响应 - 修复列表对象错误
        Parse API response - fix 'List' object has no attribute 'strip' error

        Args:
            response_data: API响应数据 | API response data

        Returns:
            str: 模型回答文本 | Model answer text
        """
        try:
            logger.debug(f"🔍 Raw response type: {type(response_data)}")
            logger.debug(
                f"🔍 Raw response keys: {response_data.keys() if isinstance(response_data, dict) else 'Not a dict'}")

            # 处理列表响应的情况
            if isinstance(response_data, list):
                logger.debug(f"🔍 Response is a list, length: {len(response_data)}")
                # 如果是列表，尝试提取第一个元素的文本内容
                if response_data:
                    first_item = response_data[0]
                    if isinstance(first_item, dict):
                        # 尝试从字典中提取文本内容
                        for content_key in ['content', 'text', 'message', 'result']:
                            if content_key in first_item:
                                content = first_item[content_key]
                                if isinstance(content, str):
                                    return content.strip().lower()
                                elif isinstance(content, list):
                                    # 如果内容还是列表，继续递归处理
                                    return self._parse_response(content)
                    # 如果是字符串列表，合并所有字符串
                    elif isinstance(first_item, str):
                        return " ".join([str(item).strip() for item in response_data]).lower()
                return ""

            # OpenAI兼容格式
            if "choices" in response_data:
                choice = response_data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    if isinstance(content, str):
                        return content.strip().lower()
                    elif isinstance(content, list):
                        # 如果content是列表，递归处理
                        return self._parse_response(content)

            # DashScope格式
            elif "output" in response_data and "choices" in response_data["output"]:
                choice = response_data["output"]["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    if isinstance(content, str):
                        return content.strip().lower()
                    elif isinstance(content, list):
                        return self._parse_response(content)

            # 尝试通用解析
            for key in ["content", "text", "result", "message"]:
                if key in response_data:
                    content = response_data[key]
                    if isinstance(content, str):
                        return content.strip().lower()
                    elif isinstance(content, list):
                        return self._parse_response(content)

            # 如果无法解析，返回原始响应的字符串表示用于调试
            logger.warning(f"⚠️ Unrecognized API response format: {response_data}")
            return str(response_data)

        except Exception as e:
            logger.error(f"❌ Failed to parse Qwen-VL response: {e}")
            logger.error(f"🔍 Problematic response data: {response_data}")
            return ""

    def _parse_verification_result(self, answer: str) -> bool:
        """
        解析验证结果 - 增强容错性
        Parse verification result - enhanced error tolerance

        Args:
            answer: 模型回答 | Model answer

        Returns:
            bool: 验证结果 | Verification result
        """
        if not answer:
            logger.warning("⚠️ Empty Qwen-VL answer, treating as negative")
            return False

        # 清理回答文本
        cleaned_answer = answer.strip().lower()

        # 中文肯定回答 | Chinese affirmative answers
        chinese_yes = any(
            word in cleaned_answer for word in ["是", "有", "存在", "确认", "yes", "true", "对的", "正确"])
        # 中文否定回答 | Chinese negative answers
        chinese_no = any(
            word in cleaned_answer for word in ["否", "没有", "不存在", "未发现", "no", "false", "不是", "错误"])

        if chinese_yes and not chinese_no:
            logger.debug(f"✅ Qwen-VL confirmed: '{cleaned_answer}'")
            return True
        elif chinese_no and not chinese_yes:
            logger.debug(f"❌ Qwen-VL rejected: '{cleaned_answer}'")
            return False
        else:
            # 模糊回答，保守处理为否定 | Ambiguous answer, conservatively treat as negative
            logger.warning(f"⚠️ Ambiguous Qwen-VL answer: '{cleaned_answer}', treating as negative")
            return False


# -------------------- 全局实例 --------------------
smoke_flame_detection_service = SmokeFlameDetectionService()
