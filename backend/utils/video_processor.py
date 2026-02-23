# backend/utils/video_processor.py (更新后的完整文件)
"""
视频处理器 - 更新版本支持公共空间分析
Video Processor - Updated version supporting common space analysis
"""

import os
import cv2
import time
import logging
from typing import List, Dict, Any, Optional
from backend.utils.frame_capture import FrameWithMetadata
from backend.services.violation_detection import detection_service as parking_detection_service
from backend.services.smoke_flame_detection import smoke_flame_detection_service
from backend.services.common_space_detection import common_space_detection_service  # ✅ 新增
from ml_models.yolov8.model_loader import YOLOModelLoader
from backend.services.parking_zone_checker import NoParkingZoneChecker
from storage.minio_client import MinIOClient
from storage.mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 统一视频处理器：支持多种检测类型（包括公共空间分析）
# ------------------------------------------------------------------
class UnifiedVideoProcessor:
    """
    统一视频处理器 - 支持电子围栏、烟火检测和公共空间分析
    Unified Video Processor - supports parking violation, smoke/flame detection, and common space analysis
    """

    def __init__(self):
        self.model_loader = None
        self.zone_checker = None
        self.minio_client = None
        self.mongo_client = None
        self.qwen_vl_client = None

    def initialize_services(
            self,
            model_loader: YOLOModelLoader,
            zone_checker: NoParkingZoneChecker,
            minio_client: MinIOClient,
            mongo_client: MongoDBClient,
            qwen_vl_client=None
    ):
        """初始化所有服务 | Initialize all services"""
        self.model_loader = model_loader
        self.zone_checker = zone_checker
        self.minio_client = minio_client
        self.mongo_client = mongo_client
        self.qwen_vl_client = qwen_vl_client

        # 初始化电子围栏检测服务
        parking_detection_service.set_clients(minio_client, mongo_client)
        parking_detection_service.set_model_loader(model_loader)
        parking_detection_service.set_zone_checker(zone_checker)

        # 初始化烟火检测服务
        smoke_flame_detection_service.set_clients(minio_client, mongo_client)
        smoke_flame_detection_service.set_model_loader(model_loader)
        if qwen_vl_client:
            smoke_flame_detection_service.set_qwen_vl_client(qwen_vl_client)

        # ✅ 新增：初始化公共空间分析服务
        common_space_detection_service.set_clients(minio_client, mongo_client)
        if qwen_vl_client:
            common_space_detection_service.set_qwen_vl_client(qwen_vl_client)
        # 设置采样间隔为30秒
        common_space_detection_service.set_sample_interval(30)

        logger.info("✅ Unified video processor services initialized")

    def process_video(
            self,
            video_path: str,
            detection_type: str = "parking_violation",
            frame_interval: float = 1.0
    ) -> None:
        """
        处理视频文件，支持多种检测类型
        Process video file with support for multiple detection types

        Args:
            video_path: 视频文件路径 | Video file path
            detection_type: 检测类型 | Detection type ("parking_violation", "smoke_flame", or "common_space")
            frame_interval: 采样间隔（秒）| Sampling interval (seconds)
        """
        if not self.model_loader or not self.minio_client or not self.mongo_client:
            logger.error("❌ Video processor not properly initialized")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"❌ Cannot open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        # 根据检测类型调整采样策略
        if detection_type == "common_space":
            # 公共空间分析使用固定时间间隔，而不是帧间隔
            frame_skip = 1  # 处理每一帧，由服务内部控制采样间隔
        else:
            frame_skip = int(fps * frame_interval) if frame_interval > 0 else 1

        source_id = os.path.basename(video_path)

        logger.info(f"🎬 Processing {source_id} | Type: {detection_type} | FPS: {fps:.1f}")

        frame_idx = 0
        last_time = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            now = time.time()

            # 时间间隔采样（对于公共空间分析，跳过时间间隔检查）
            if detection_type != "common_space" and frame_interval > 0 and now - last_time < frame_interval:
                continue
            last_time = now

            # 构造帧元数据
            frame_meta = FrameWithMetadata(
                image=frame,
                source_id=source_id,
                timestamp=now,  # 使用系统时间戳
                frame_index=frame_idx,
                original_time_str=time.strftime("%H:%M:%S", time.gmtime(now)),
                is_rtsp=False
            )

            # 根据检测类型调用对应的服务
            if detection_type == "parking_violation":
                self._process_parking_frame(frame_meta)
            elif detection_type == "smoke_flame":
                self._process_smoke_flame_frame(frame_meta)
            elif detection_type == "common_space":
                self._process_common_space_frame(frame_meta)
            else:
                logger.error(f"❌ Unknown detection type: {detection_type}")
                break

        cap.release()

        # 刷新剩余数据
        if detection_type == "parking_violation":
            parking_detection_service.flush_remaining()
        elif detection_type == "smoke_flame":
            smoke_flame_detection_service.flush_remaining()
        elif detection_type == "common_space":
            common_space_detection_service.flush_remaining()

        logger.info(f"✅ Finished processing {source_id} | Type: {detection_type} | Frames: {frame_idx}")

    def _process_parking_frame(self, frame_meta: FrameWithMetadata):
        """处理电子围栏检测帧 | Process parking violation frame"""
        try:
            parking_detection_service.process_frame(frame_meta)
        except Exception as e:
            logger.error(f"❌ Parking detection failed for frame {frame_meta.frame_index}: {e}")

    def _process_smoke_flame_frame(self, frame_meta: FrameWithMetadata):
        """处理烟火检测帧 | Process smoke/flame frame"""
        try:
            smoke_flame_detection_service.process_frame(frame_meta)
        except Exception as e:
            logger.error(f"❌ Smoke/flame detection failed for frame {frame_meta.frame_index}: {e}")

    def _process_common_space_frame(self, frame_meta: FrameWithMetadata):
        """✅ 新增：处理公共空间分析帧 | Process common space analysis frame"""
        try:
            common_space_detection_service.process_frame(frame_meta)
        except Exception as e:
            logger.error(f"❌ Common space analysis failed for frame {frame_meta.frame_index}: {e}")


# ------------------------------------------------------------------
# 单文件入口：向后兼容（更新支持公共空间分析）
# ------------------------------------------------------------------
def process_video_official(
        video_path: str,
        model_loader: YOLOModelLoader,
        zone_checker: NoParkingZoneChecker,
        frame_interval: float = 1.0,
        detection_type: str = "parking_violation",
        minio_client: MinIOClient = None,
        mongo_client: MongoDBClient = None,
        qwen_vl_client=None
) -> None:
    """
    以官方示例风格处理本地视频，支持多种检测类型（包括公共空间分析）：
    - 整帧推理
    - 批量处理
    - 一张图 → 一条文档

    Process local video in official style with multiple detection types support (including common space analysis)

    Parameters:
        video_path: 本地视频路径 | Local video path
        model_loader: YOLO 模型加载器 | YOLO model loader
        zone_checker: 禁停区检查器 | No parking zone checker
        frame_interval: 采样间隔（秒）| Sampling interval (seconds)
        detection_type: 检测类型 | Detection type ("parking_violation", "smoke_flame", or "common_space")
        minio_client: MinIO客户端（可选）| MinIO client (optional)
        mongo_client: MongoDB客户端（可选）| MongoDB client (optional)
        qwen_vl_client: Qwen-VL客户端（可选，烟火检测和公共空间分析需要）| Qwen-VL client (optional, required for smoke/flame and common space analysis)
    """
    # 创建统一处理器实例
    processor = UnifiedVideoProcessor()

    # 如果未提供存储客户端，尝试从现有服务获取
    if not minio_client or not mongo_client:
        try:
            from backend.config.database import init_clients
            minio_client, mongo_client = init_clients()
            logger.info("📦 Using default storage clients")
        except Exception as e:
            logger.error(f"❌ Failed to initialize storage clients: {e}")
            return

    # 初始化服务
    processor.initialize_services(
        model_loader=model_loader,
        zone_checker=zone_checker,
        minio_client=minio_client,
        mongo_client=mongo_client,
        qwen_vl_client=qwen_vl_client
    )

    # 处理视频
    processor.process_video(
        video_path=video_path,
        detection_type=detection_type,
        frame_interval=frame_interval
    )


# ------------------------------------------------------------------
# 文件夹批量处理（更新支持公共空间分析）
# ------------------------------------------------------------------
def process_video_folder(
        folder_path: str,
        model_loader: YOLOModelLoader,
        zone_checker: NoParkingZoneChecker,
        detection_type: str = "parking_violation",
        frame_interval: float = 1.0,
        minio_client: MinIOClient = None,
        mongo_client: MongoDBClient = None,
        qwen_vl_client=None
) -> None:
    """
    批量处理文件夹中的所有视频文件
    Batch process all video files in a folder

    Args:
        folder_path: 文件夹路径 | Folder path
        model_loader: YOLO 模型加载器 | YOLO model loader
        zone_checker: 禁停区检查器 | No parking zone checker
        detection_type: 检测类型 | Detection type
        frame_interval: 采样间隔 | Sampling interval
        minio_client: MinIO客户端 | MinIO client
        mongo_client: MongoDB客户端 | MongoDB client
        qwen_vl_client: Qwen-VL客户端 | Qwen-VL client
    """
    supported_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}

    if not os.path.exists(folder_path):
        logger.error(f"❌ Folder not found: {folder_path}")
        return

    # 创建统一处理器
    processor = UnifiedVideoProcessor()

    # 初始化存储客户端
    if not minio_client or not mongo_client:
        try:
            from backend.config.database import init_clients
            minio_client, mongo_client = init_clients()
        except Exception as e:
            logger.error(f"❌ Failed to initialize storage clients: {e}")
            return

    # 初始化服务
    processor.initialize_services(
        model_loader=model_loader,
        zone_checker=zone_checker,
        minio_client=minio_client,
        mongo_client=mongo_client,
        qwen_vl_client=qwen_vl_client
    )

    # 处理文件夹中的每个视频文件
    video_files = []
    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in supported_exts:
            video_files.append(os.path.join(folder_path, filename))

    if not video_files:
        logger.info(f"🟡 No supported video files found in: {folder_path}")
        return

    logger.info(f"📁 Processing {len(video_files)} videos from: {folder_path}")

    for video_path in video_files:
        try:
            processor.process_video(
                video_path=video_path,
                detection_type=detection_type,
                frame_interval=frame_interval
            )
        except Exception as e:
            logger.error(f"❌ Failed to process {video_path}: {e}")
            continue


# ------------------------------------------------------------------
# 全局统一处理器实例
# ------------------------------------------------------------------
unified_video_processor = UnifiedVideoProcessor()


# ------------------------------------------------------------------
# 工具函数：检测类型推断（更新支持公共空间分析）
# ------------------------------------------------------------------
def infer_detection_type_from_path(file_path: str, base_folder: str = "./uploads") -> str:
    """
    根据文件路径推断检测类型
    Infer detection type from file path

    Args:
        file_path: 文件路径 | File path
        base_folder: 基础文件夹路径 | Base folder path

    Returns:
        str: 检测类型 | Detection type ("parking_violation", "smoke_flame", or "common_space")
    """
    try:
        relative_path = os.path.relpath(file_path, base_folder)
        folder_name = relative_path.split(os.sep)[0]  # 获取第一级子文件夹名

        detection_type_map = {
            "parking": "parking_violation",
            "smoke_flame": "smoke_flame",
            "common_space": "common_space"  # ✅ 新增
        }

        return detection_type_map.get(folder_name, "parking_violation")  # 默认为电子围栏检测
    except ValueError:
        # 如果文件不在base_folder下，返回默认类型
        return "parking_violation"