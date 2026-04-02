# backend/services/violation_detection.py
"""
Violation Detection Service
1. 整帧 YOLO 推理
2. 批量过滤禁停区
3. 一张渲染图 → 一次上传
4. 一帧只写一条文档（含全部违规目标）
"""

from typing import List, Dict, Any
from backend.utils.frame_capture import FrameWithMetadata
from ml_models.yolov8.inference import YOLOInference
from backend.services.event_generator import handle_frame_events
from ml_models.yolov8.model_loader import YOLOModelLoader
from backend.services.parking_zone_checker import NoParkingZoneChecker
from backend.utils.visualization import render_official_frame
from storage.minio_client import MinIOClient
from storage.mongodb_client import MongoDBClient
import logging
import cv2
import numpy as np
import time
import threading
from datetime import datetime

logger = logging.getLogger(__name__)


class ViolationDetectionService:
    """整帧批量检测服务."""

    def __init__(self):
        self.minio_client = None
        self.mongo_client = None
        self.model_loader = None
        self.zone_checker = None
        # Ultralytics YOLO 对同一个模型实例的并发 predict 并不稳定，RTSP 多帧并发时需要串行化这一步。
        self._inference_lock = threading.Lock()

    # -------------------- 依赖注入 --------------------
    def set_clients(self, minio_client: MinIOClient, mongo_client: MongoDBClient):
        self.minio_client = minio_client
        self.mongo_client = mongo_client

    def set_model_loader(self, loader: YOLOModelLoader):
        self.model_loader = loader

    def set_zone_checker(self, zone_checker: NoParkingZoneChecker):
        self.zone_checker = zone_checker
        # 如果 processor 存在，也同步设置
        if hasattr(self, 'processor') and self.processor:
            self.processor.set_zone_checker(zone_checker)

    def flush_remaining(self):
        """处理剩余帧（兼容性方法）"""
        logger.info("🔄 Flush remaining called (no-op in batch mode)")
        # 在批量模式下不需要处理剩余帧
        pass

    # -------------------- 主入口 --------------------
    def process_frame(self, frame_meta: FrameWithMetadata) -> None:
        """整帧推理 → 批量过滤 → 一张图 → 一条文档."""
        if not all([self.minio_client, self.mongo_client, self.model_loader, self.zone_checker]):
            logger.error("❌ Service not fully initialized.")
            return

        source_id = frame_meta.source_id
        image = frame_meta.image

        logger.debug(f"🔍DEBUG: Backend source_id = '{source_id}' (type: {type(source_id)})")

        # 🔴 修复时间戳问题 - 使用当前系统时间
        current_timestamp = time.time()
        current_datetime = datetime.now()

        logger.debug(f"🕒 Frame timestamp - Original: {frame_meta.timestamp}, Corrected: {current_timestamp}")
        logger.debug(f"🕒 Frame datetime - {current_datetime}")

        # 1. 推理 → 强制转 dict（只转一次）
        model = self.model_loader.get_model("vehicle")
        if model is None:
            logger.error("❌ Vehicle model not loaded")
            return

        # 串行化同一个 vehicle 模型实例的推理，避免并发导致模型内部 fuse/bn 状态异常
        with self._inference_lock:
            raw_detections = YOLOInference.run_detection(model, image, conf_threshold=0.3)

        detections: List[Dict[str, Any]] = [
            {"class_name": cls, "confidence": float(conf), "bbox": bbox}
            for cls, conf, bbox in raw_detections
        ]
        logger.debug(f"🔒 CONVERTED: {len(detections)} detections")

        if not detections:
            logger.debug("🟢 No objects detected.")
            return

        # 2. 过滤
        logger.debug(f"🔒 BEFORE filter: {len(detections)} detections")
        violations = self.zone_checker.filter_violations_in_zones(detections, source_id)

        zones = self.zone_checker.get_zones_for_source(source_id)
        # ✅【DEBUG 3】打印关键状态
        logger.debug(f"🔍 DEBUG: Zones count = {len(zones)} | Violations count = {len(violations)}")
        if zones:
            logger.debug(f"🔍 DEBUG: First zone example = {zones[0][:3]}...")  # 打印前3个点

        logger.debug(f"🔒 AFTER filter: {len(violations)} violations")

        # 3. 一次性渲染 - 🔴 修复：传递所有检测目标和违规目标
        logger.debug(f"🖌️ Rendering {len(detections)} total detections, {len(violations)} violations")
        rendered = render_official_frame(
            image=image,
            all_detections=detections,  # 🔴 传递所有检测目标
            violations=violations,  # 🔴 传递违规目标
            zones=self.zone_checker.get_zones_for_source(source_id)
        )

        # 4. 一次性上传
        image_url = self.minio_client.upload_frame(
            image_data=rendered,
            camera_id=source_id,
            timestamp=current_timestamp,
            event_type="violations"
        )
        if not image_url:
            logger.error("❌ Upload failed.")
            return

        # 记录清理后的URL
        clean_url = image_url.split('?')[0] if '?X-Amz-' in image_url else image_url
        logger.info(f"📸 Image URL: {clean_url}")

        # 5. 一次性写库
        ok = handle_frame_events(
            minio_client=self.minio_client,
            mongo_client=self.mongo_client,
            image_url=clean_url,
            camera_id=source_id,
            timestamp=current_timestamp,
            frame_index=frame_meta.frame_index,
            violations=violations
        )
        if ok:
            logger.info(
                f"✅ Frame-level event saved: {len(violations)} violations out of {len(detections)} total detections")
        else:
            logger.error("❌ Failed to save frame events to database")


# -------------------- 全局实例 --------------------
detection_service = ViolationDetectionService()