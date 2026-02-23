# backend/main.py

import os
import sys
import time
import logging
from typing import Dict, Any

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 配置加载 | Load config
# ------------------------------------------------------------------
def load_config() -> Dict[str, Any]:
    """从环境变量读取配置，带默认值."""
    return {
        "minio_endpoint": os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        "minio_access_key": os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        "minio_secret_key": os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        "minio_bucket": os.getenv("MINIO_BUCKET", "video-events"),
        "mongo_uri": os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        "mongo_db_name": os.getenv("MONGO_DB_NAME", "video_analysis_db"),
        "upload_folder": os.getenv("UPLOAD_FOLDER", "./uploads"),
        "frame_interval_sec": float(os.getenv("FRAME_INTERVAL", "1.0")),
        "common_space_interval_sec": float(os.getenv("COMMON_SPACE_INTERVAL", "30.0")),
        "rtsp_urls": os.getenv("RTSP_URLS", "").split(","),
        "qwen_vl_api_url": os.getenv("QWEN_VL_API_URL", ""),
        "qwen_vl_api_key": os.getenv("QWEN_VL_API_KEY", ""),
        "qwen_vl_model_name": os.getenv("QWEN_VL_MODEL_NAME", "qwen-vl-plus"),
    }


# ------------------------------------------------------------------
# 服务初始化 | Service initialization
# ------------------------------------------------------------------
def initialize_services(cfg: Dict[str, Any]):
    """初始化所有服务 | Initialize all services"""

    # ----------- 1. 存储客户端 -----------
    from storage.minio_client import MinIOClient
    from storage.mongodb_client import MongoDBClient

    minio = MinIOClient(
        endpoint=cfg["minio_endpoint"],
        access_key=cfg["minio_access_key"],
        secret_key=cfg["minio_secret_key"],
        bucket_name=cfg["minio_bucket"]
    )
    mongo = MongoDBClient(mongo_uri=cfg["mongo_uri"], db_name=cfg["mongo_db_name"])

    # ----------- 2. 模型 & 检查器 -----------
    from ml_models.yolov8.model_loader import YOLOModelLoader
    from backend.services.parking_zone_checker import NoParkingZoneChecker

    loader = YOLOModelLoader()
    loader.load_model("vehicle", "yolov8n.pt")
    try:
        loader.load_model("smoke_flame", "smoke_flame.pt")
        logger.info("✅ Smoke/Flame model loaded")
    except Exception as e:
        logger.warning(f"⚠️ Smoke/Flame model not available: {e}")

    # 【关键修复】创建唯一的 zone_checker 实例
    zone_checker = NoParkingZoneChecker()
    logger.info(f"✅ Created single zone_checker instance (ID: {id(zone_checker)}) with zones: {list(zone_checker.zones.keys())}")

    # ----------- 3. 检测服务 -----------
    from backend.services.violation_detection import detection_service as parking_service
    from backend.services.smoke_flame_detection import smoke_flame_detection_service, QwenVLAPIClient
    from backend.services.common_space_detection import common_space_detection_service
    from backend.config.qwen_vl_config import qwen_vl_api_config

    # 【关键修复】电子围栏检测服务 - 注入同一个 zone_checker 实例
    parking_service.set_clients(minio_client=minio, mongo_client=mongo)
    parking_service.set_model_loader(loader)
    parking_service.set_zone_checker(zone_checker)  # ← 关键：注入实例
    logger.info("✅ Parking violation detection service ready.")

    # 烟火检测服务
    smoke_service_ready = False
    qwen_vl_client = None

    logger.info(f"🔍 Qwen-VL Config Check:")
    logger.info(f"   Config API URL: {qwen_vl_api_config.get_api_url()}")
    logger.info(f"   Config API Key configured: {bool(qwen_vl_api_config.get_api_key())}")
    logger.info(f"   Config Overall configured: {qwen_vl_api_config.is_configured()}")

    if qwen_vl_api_config.is_configured():
        try:
            qwen_vl_client = QwenVLAPIClient(
                api_url=qwen_vl_api_config.get_api_url(),
                api_key=qwen_vl_api_config.get_api_key(),
                model_name=qwen_vl_api_config.get_model_name()
            )
            smoke_flame_detection_service.set_clients(minio, mongo)
            smoke_flame_detection_service.set_model_loader(loader)
            smoke_flame_detection_service.set_qwen_vl_client(qwen_vl_client)
            smoke_service_ready = True
            logger.info("✅ Smoke/Flame detection service ready (using config file).")
        except Exception as e:
            logger.error(f"❌ Failed to initialize smoke/flame detection with config file: {e}")
    elif cfg["qwen_vl_api_url"] and cfg["qwen_vl_api_key"]:
        try:
            qwen_vl_client = QwenVLAPIClient(
                api_url=cfg["qwen_vl_api_url"],
                api_key=cfg["qwen_vl_api_key"],
                model_name=cfg["qwen_vl_model_name"]
            )
            smoke_flame_detection_service.set_clients(minio, mongo)
            smoke_flame_detection_service.set_model_loader(loader)
            smoke_flame_detection_service.set_qwen_vl_client(qwen_vl_client)
            smoke_service_ready = True
            logger.info("✅ Smoke/Flame detection service ready (using environment variables).")
        except Exception as e:
            logger.error(f"❌ Failed to initialize smoke/flame detection with environment variables: {e}")
    else:
        logger.warning(
            "⚠️ Qwen-VL API not configured in both config file and environment variables, smoke/flame detection disabled")

    # 公共空间分析服务
    common_space_service_ready = False
    if qwen_vl_client:
        try:
            common_space_detection_service.set_clients(minio, mongo)
            common_space_detection_service.set_qwen_vl_client(qwen_vl_client)
            common_space_detection_service.set_sample_interval(cfg["common_space_interval_sec"])
            common_space_service_ready = True
            logger.info(
                f"✅ Common space analysis service ready (sampling interval: {cfg['common_space_interval_sec']}s).")
        except Exception as e:
            logger.error(f"❌ Failed to initialize common space analysis service: {e}")
    else:
        logger.warning("⚠️ Qwen-VL client not available, common space analysis service disabled")

    return {
        "minio": minio,
        "mongo": mongo,
        "loader": loader,
        "zone_checker": zone_checker,  # ← 返回同一个实例
        "parking_service": parking_service,
        "smoke_service": smoke_flame_detection_service,
        "common_space_service": common_space_detection_service,
        "qwen_vl_client": qwen_vl_client,
        "smoke_service_ready": smoke_service_ready,
        "common_space_service_ready": common_space_service_ready
    }


# ------------------------------------------------------------------
# 更新多文件夹监控配置 | Update multi-folder monitoring configuration
# ------------------------------------------------------------------
def update_monitor_folders_config():
    """更新监控文件夹配置以包含公共空间分析"""
    try:
        from scripts.file_watcher import MONITOR_FOLDERS

        if "common_space" not in MONITOR_FOLDERS:
            MONITOR_FOLDERS["common_space"] = "common_space"
            logger.info("✅ Added common_space folder to MONITOR_FOLDERS configuration")

        return MONITOR_FOLDERS
    except Exception as e:
        logger.error(f"❌ Failed to update monitor folders config: {e}")
        return None


# ------------------------------------------------------------------
# 多文件夹监控启动 | Multi-folder monitoring startup
# ------------------------------------------------------------------
def start_multi_folder_monitoring(cfg: Dict[str, Any], services: Dict[str, Any]) -> bool:
    """
    尝试启动多文件夹监控（确保传入同一个 zone_checker 实例）
    """
    try:
        from scripts.file_watcher import start_multi_folder_watchdog

        MONITOR_FOLDERS = update_monitor_folders_config()
        if not MONITOR_FOLDERS:
            logger.error("❌ Failed to get monitor folders configuration")
            return None

        upload_folder = cfg["upload_folder"]
        for folder_name in MONITOR_FOLDERS.keys():
            folder_path = os.path.join(upload_folder, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            logger.info(f"📁 Created/verified folder: {folder_path}")

        # 【关键修复】传入同一个 zone_checker 实例
        observer = start_multi_folder_watchdog(
            base_folder=cfg["upload_folder"],
            model_loader=services["loader"],
            parking_detection_service=services["parking_service"],
            smoke_flame_detection_service=services["smoke_service"],
            zone_checker=services["zone_checker"],  # ← 关键：同一个实例
            frame_interval=cfg["frame_interval_sec"]
        )

        logger.info("🎯 Multi-folder monitoring started successfully!")
        logger.info("📂 Folder structure:")
        logger.info("   uploads/")
        logger.info("   ├── 🅿️ parking/        - 电子围栏检测视频")
        logger.info("   ├── 🔥 smoke_flame/    - 烟火检测视频")
        logger.info("   └── 🏢 common_space/   - 公共空间分析视频")

        if not services["smoke_service_ready"]:
            logger.warning("   ⚠️ 烟火检测服务未就绪，请检查Qwen-VL配置")
        if not services["common_space_service_ready"]:
            logger.warning("   ⚠️ 公共空间分析服务未就绪，请检查Qwen-VL配置")

        return observer

    except Exception as e:
        logger.warning(f"⚠️ Multi-folder monitoring not available: {e}")
        return None


# ------------------------------------------------------------------
# 单文件夹监控启动（向后兼容）| Single folder monitoring (backward compatible)
# ------------------------------------------------------------------
def start_single_folder_monitoring(cfg: Dict[str, Any], services: Dict[str, Any]):
    """启动单文件夹监控（向后兼容）"""
    try:
        from scripts.file_watcher import start_file_watchdog

        # 【关键修复】传入同一个 zone_checker 实例
        observer = start_file_watchdog(
            folder_path=cfg["upload_folder"],
            model_loader=services["loader"],
            detection_service=services["parking_service"],
            zone_checker=services["zone_checker"],  # ← 关键：同一个实例
            frame_interval=cfg["frame_interval_sec"]
        )

        logger.info(f"👀 Single folder monitoring started: {cfg['upload_folder']}")
        return observer

    except Exception as e:
        logger.error(f"❌ Single folder monitoring failed: {e}")
        return None


# ------------------------------------------------------------------
# 文件处理回调 | File processing callback
# ------------------------------------------------------------------
def create_file_processor(cfg: Dict[str, Any], services: Dict[str, Any]):
    """创建文件处理器"""

    def process_video_file(video_path: str):
        """处理视频文件"""
        try:
            from backend.utils.video_processor import infer_detection_type_from_path, process_video_official

            detection_type = infer_detection_type_from_path(video_path, cfg["upload_folder"])
            logger.info(f"🎬 Processing: {os.path.basename(video_path)} | Type: {detection_type}")

            qwen_client = None
            if detection_type in ["smoke_flame", "common_space"]:
                qwen_client = services["qwen_vl_client"]
                if not qwen_client:
                    logger.error(f"❌ Qwen-VL client not available for {detection_type} detection")
                    return

            process_video_official(
                video_path=video_path,
                model_loader=services["loader"],
                zone_checker=services["zone_checker"],  # ← 同一个实例
                frame_interval=cfg["frame_interval_sec"],
                detection_type=detection_type,
                minio_client=services["minio"],
                mongo_client=services["mongo"],
                qwen_vl_client=qwen_client
            )

        except Exception as e:
            logger.error(f"❌ Failed to process {video_path}: {e}")

    return process_video_file


# ------------------------------------------------------------------
# RTSP处理回调 | RTSP processing callback
# ------------------------------------------------------------------
def create_rtsp_callback(services: Dict[str, Any], rtsp_id: str, detection_type: str = "parking_violation"):
    """创建RTSP回调函数"""

    def rtsp_callback(source_id, frames):
        for frame_meta in frames:
            try:
                if detection_type == "parking_violation":
                    services["parking_service"].process_frame(frame_meta)
                elif detection_type == "smoke_flame":
                    if services["smoke_service_ready"]:
                        services["smoke_service"].process_frame(frame_meta)
                elif detection_type == "common_space":
                    if services["common_space_service_ready"]:
                        services["common_space_service"].process_frame(frame_meta)
            except Exception as e:
                logger.error(f"❌ RTSP processing error for {source_id}: {e}")

    return rtsp_callback


# ------------------------------------------------------------------
# 启动RTSP源 | Start RTSP sources
# ------------------------------------------------------------------
def start_rtsp_sources(cfg: Dict[str, Any], services: Dict[str, Any]):
    """启动多路RTSP源"""
    rtsp_count = 0
    rtsp_configs = []

    for idx, url_with_type in enumerate(cfg["rtsp_urls"]):
        if not url_with_type.strip():
            continue

        parts = url_with_type.strip().split("|")
        rtsp_url = parts[0].strip()
        if len(parts) > 1:
            detection_type = parts[1].strip()
        else:
            detection_type = "parking_violation"

        valid_types = ["parking_violation", "smoke_flame", "common_space"]
        if detection_type not in valid_types:
            logger.warning(f"⚠️ Invalid detection type '{detection_type}' for RTSP {idx}, using default")
            detection_type = "parking_violation"

        rtsp_configs.append({
            "idx": idx,
            "url": rtsp_url,
            "detection_type": detection_type
        })

    for config in rtsp_configs:
        try:
            from backend.utils.frame_capture import VideoFrameCapture

            rtsp_id = f"rtsp_{config['idx']}"
            detection_type = config["detection_type"]

            if detection_type == "smoke_flame" and not services["smoke_service_ready"]:
                logger.warning(f"⚠️ Smoke/flame detection not ready for RTSP {config['idx']}, skipping")
                continue
            elif detection_type == "common_space" and not services["common_space_service_ready"]:
                logger.warning(f"⚠️ Common space analysis not ready for RTSP {config['idx']}, skipping")
                continue

            cap = VideoFrameCapture()
            cap.register_batch_callback(create_rtsp_callback(services, rtsp_id, detection_type))
            cap.add_rtsp_source(
                source_id=rtsp_id,
                rtsp_url=config["url"],
                batch_size=8,
                batch_sec=1.0,
                reconnect_delay=5
            )

            rtsp_count += 1
            logger.info(f"📹 RTSP source {config['idx']} added: {config['url']} | Type: {detection_type}")

        except Exception as e:
            logger.error(f"❌ Failed to add RTSP source {config['idx']}: {e}")

    return rtsp_count


# ------------------------------------------------------------------
# 主函数 | Main entry
# ------------------------------------------------------------------
def main():
    logger.info("🚀 Starting Hybrid Video Analysis System...")
    cfg = load_config()

    try:
        # ----------- 1. 初始化所有服务 -----------
        logger.info("🔄 Initializing services...")
        services = initialize_services(cfg)

        # ----------- 2. 尝试启动多文件夹监控 -----------
        observer = start_multi_folder_monitoring(cfg, services)

        # ----------- 3. 如果多文件夹监控失败，回退到单文件夹监控 -----------
        if not observer:
            logger.info("🔄 Falling back to single folder monitoring...")
            observer = start_single_folder_monitoring(cfg, services)

            if not observer:
                logger.error("❌ Both multi-folder and single-folder monitoring failed")
                return

        # ----------- 4. 多路 RTSP（可选） -----------
        rtsp_count = start_rtsp_sources(cfg, services)

        if rtsp_count > 0:
            logger.info(f"✅ {rtsp_count} RTSP sources initialized")

        # ----------- 5. 系统状态报告 -----------
        logger.info("📊 System Status:")
        logger.info(f"   📁 Upload folder: {cfg['upload_folder']}")
        logger.info(f"   🅿️ Parking detection: ✅ Ready")
        logger.info(f"   🔥 Smoke/Flame detection: {'✅ Ready' if services['smoke_service_ready'] else '❌ Disabled'}")
        logger.info(
            f"   🏢 Common space analysis: {'✅ Ready' if services['common_space_service_ready'] else '❌ Disabled'}")
        if services['common_space_service_ready']:
            logger.info(f"   ⏱️ Common space interval: {cfg['common_space_interval_sec']}s")
        logger.info(f"   📹 RTSP sources: {rtsp_count}")
        logger.info(f"   ⏱️ Frame interval: {cfg['frame_interval_sec']}s")

        # ----------- 6. 创建uploads目录结构 -----------
        upload_folder = cfg["upload_folder"]
        os.makedirs(upload_folder, exist_ok=True)

        common_space_folder = os.path.join(upload_folder, "common_space")
        os.makedirs(common_space_folder, exist_ok=True)
        logger.info(f"📁 Created directory: {common_space_folder}")

        for folder_name in ["parking", "smoke_flame"]:
            folder_path = os.path.join(upload_folder, folder_name)
            os.makedirs(folder_path, exist_ok=True)

        # ----------- 7. 使用说明 -----------
        logger.info("📋 Usage Instructions:")
        logger.info("   1. Place videos for analysis in the following folders:")
        logger.info("      - 🅿️  uploads/parking/        : Parking violation detection")
        logger.info("      - 🔥  uploads/smoke_flame/    : Smoke/Flame detection")
        logger.info("      - 🏢  uploads/common_space/   : Public space analysis (new)")
        logger.info("   2. The system will automatically process uploaded videos")
        logger.info("   3. For common space analysis, frames are sampled every 30 seconds")
        logger.info("   4. Analysis results are saved to MinIO and MongoDB")

        # ----------- 8. 主线程保活 -----------
        logger.info("🎉 System started successfully! Press Ctrl+C to stop.")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("👋 Shutting down...")
            observer.stop()
            observer.join()

            services["parking_service"].flush_remaining()
            if services["smoke_service_ready"]:
                services["smoke_service"].flush_remaining()
            if services["common_space_service_ready"]:
                services["common_space_service"].flush_remaining()

            logger.info("✅ System stopped gracefully")

    except Exception as e:
        logger.error(f"❌ Main error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
