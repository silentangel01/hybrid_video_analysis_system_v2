# backend/main.py

import logging
import os
import sys
import threading
import time
from typing import Any, Dict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables with defaults."""
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
        "dwell_threshold": int(os.getenv("DWELL_THRESHOLD", "5")),
    }


# ------------------------------------------------------------------
# Service initialisation
# ------------------------------------------------------------------
def initialize_services(cfg: Dict[str, Any]):
    """Initialise all shared services and return them as a dict."""

    from storage.minio_client import MinIOClient
    from storage.mongodb_client import MongoDBClient

    minio = MinIOClient(
        endpoint=cfg["minio_endpoint"],
        access_key=cfg["minio_access_key"],
        secret_key=cfg["minio_secret_key"],
        bucket_name=cfg["minio_bucket"],
    )
    mongo = MongoDBClient(mongo_uri=cfg["mongo_uri"], db_name=cfg["mongo_db_name"])

    from ml_models.yolov8.model_loader import YOLOModelLoader
    from backend.services.parking_zone_checker import NoParkingZoneChecker
    from backend.services.stream_runtime import AppResources, StreamRuntimeFactory

    loader = YOLOModelLoader()
    loader.load_model("vehicle", "yolov8n.pt")
    try:
        loader.load_model("smoke_flame", "smoke_flame.pt")
        logger.info("Smoke/Flame model loaded")
    except Exception as e:
        logger.warning("Smoke/Flame model not available: %s", e)

    zone_checker = NoParkingZoneChecker()
    logger.info("Zone checker ready (zones: %s)", list(zone_checker.zones.keys()))

    # -- Detection services --
    from backend.services.violation_detection import detection_service as parking_service
    from backend.services.smoke_flame_detection import smoke_flame_detection_service, QwenVLAPIClient
    from backend.services.common_space_detection import common_space_detection_service
    from backend.config.qwen_vl_config import qwen_vl_api_config

    parking_service.set_clients(minio_client=minio, mongo_client=mongo)
    parking_service.set_model_loader(loader)
    parking_service.set_zone_checker(zone_checker)
    logger.info("Parking violation detection service ready")

    # Qwen-VL client (config file first, then env vars)
    smoke_service_ready = False
    qwen_vl_client = None

    qwen_sources = [
        (qwen_vl_api_config.is_configured(),
         lambda: (qwen_vl_api_config.get_api_url(), qwen_vl_api_config.get_api_key(), qwen_vl_api_config.get_model_name()),
         "config file"),
        (bool(cfg["qwen_vl_api_url"] and cfg["qwen_vl_api_key"]),
         lambda: (cfg["qwen_vl_api_url"], cfg["qwen_vl_api_key"], cfg["qwen_vl_model_name"]),
         "environment variables"),
    ]

    for available, get_params, source_label in qwen_sources:
        if not available or qwen_vl_client is not None:
            continue
        try:
            url, key, model = get_params()
            qwen_vl_client = QwenVLAPIClient(api_url=url, api_key=key, model_name=model)
            smoke_flame_detection_service.set_clients(minio, mongo)
            smoke_flame_detection_service.set_model_loader(loader)
            smoke_flame_detection_service.set_qwen_vl_client(qwen_vl_client)
            smoke_service_ready = True
            logger.info("Smoke/Flame detection service ready (source: %s)", source_label)
        except Exception as e:
            logger.error("Failed to init smoke/flame from %s: %s", source_label, e)

    if not qwen_vl_client:
        logger.warning("Qwen-VL API not configured; smoke/flame detection disabled")

    # Common space service
    common_space_service_ready = False
    if qwen_vl_client:
        try:
            common_space_detection_service.set_clients(minio, mongo)
            common_space_detection_service.set_qwen_vl_client(qwen_vl_client)
            common_space_detection_service.set_sample_interval(cfg["common_space_interval_sec"])
            common_space_service_ready = True
            logger.info("Common space analysis service ready (interval: %ss)", cfg["common_space_interval_sec"])
        except Exception as e:
            logger.error("Failed to init common space service: %s", e)
    else:
        logger.warning("Common space analysis disabled (no Qwen-VL client)")

    app_resources = AppResources(
        minio=minio,
        mongo=mongo,
        zone_checker=zone_checker,
        qwen_vl_client=qwen_vl_client,
        smoke_service_ready=smoke_service_ready,
        common_space_service_ready=common_space_service_ready,
        common_space_interval_sec=cfg["common_space_interval_sec"],
        dwell_threshold=cfg["dwell_threshold"],
        weights_dir=loader.weights_dir,
    )

    # -- Webhook service --
    from backend.services.webhook_service import WebhookService
    from backend.services.event_generator import set_webhook_service

    webhook_service = WebhookService(mongo)
    set_webhook_service(webhook_service)
    logger.info("Webhook notification service ready")

    return {
        "minio": minio,
        "mongo": mongo,
        "loader": loader,
        "zone_checker": zone_checker,
        "parking_service": parking_service,
        "smoke_service": smoke_flame_detection_service,
        "common_space_service": common_space_detection_service,
        "qwen_vl_client": qwen_vl_client,
        "smoke_service_ready": smoke_service_ready,
        "common_space_service_ready": common_space_service_ready,
        "app_resources": app_resources,
        "stream_runtime_factory": StreamRuntimeFactory(app_resources),
        "webhook_service": webhook_service,
    }


# ------------------------------------------------------------------
# Folder monitoring
# ------------------------------------------------------------------
def start_multi_folder_monitoring(cfg: Dict[str, Any], services: Dict[str, Any]):
    """Start watchdog-based multi-folder monitoring."""
    try:
        from scripts.file_watcher import start_multi_folder_watchdog, MONITOR_FOLDERS

        if "common_space" not in MONITOR_FOLDERS:
            MONITOR_FOLDERS["common_space"] = "common_space"

        upload_folder = cfg["upload_folder"]
        for name in MONITOR_FOLDERS:
            os.makedirs(os.path.join(upload_folder, name), exist_ok=True)

        observer = start_multi_folder_watchdog(
            base_folder=upload_folder,
            model_loader=services["loader"],
            parking_detection_service=services["parking_service"],
            smoke_flame_detection_service=services["smoke_service"],
            zone_checker=services["zone_checker"],
            frame_interval=cfg["frame_interval_sec"],
        )
        logger.info("Multi-folder monitoring started")
        return observer

    except Exception as e:
        logger.warning("Multi-folder monitoring not available: %s", e)
        return None


def start_single_folder_monitoring(cfg: Dict[str, Any], services: Dict[str, Any]):
    """Fallback: single-folder monitoring."""
    try:
        from scripts.file_watcher import start_file_watchdog

        observer = start_file_watchdog(
            folder_path=cfg["upload_folder"],
            model_loader=services["loader"],
            detection_service=services["parking_service"],
            zone_checker=services["zone_checker"],
            frame_interval=cfg["frame_interval_sec"],
        )
        logger.info("Single folder monitoring started: %s", cfg["upload_folder"])
        return observer

    except Exception as e:
        logger.error("Single folder monitoring failed: %s", e)
        return None


# ------------------------------------------------------------------
# StreamManager + Flask API
# ------------------------------------------------------------------
def create_stream_manager(services: Dict[str, Any]):
    from backend.services.stream_manager import StreamManager

    sm = StreamManager(
        runtime_factory=services["stream_runtime_factory"],
        mongo_client=services["mongo"],
    )
    logger.info("StreamManager created")
    return sm


def start_api_server(stream_manager, mongo_client, webhook_service, port: int = 5000):
    """Start Flask API in a daemon thread."""
    from flask import Flask
    from flask_cors import CORS
    from backend.api.stream_routes import stream_bp, init_stream_routes
    from backend.api.event_routes import event_bp, init_event_routes
    from backend.api.webhook_routes import webhook_bp, init_webhook_routes

    app = Flask(__name__)
    CORS(app)

    init_stream_routes(stream_manager)
    init_event_routes(mongo_client)
    init_webhook_routes(webhook_service)
    app.register_blueprint(stream_bp)
    app.register_blueprint(event_bp)
    app.register_blueprint(webhook_bp)

    def _run():
        logger.info("Flask API starting on port %d...", port)
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    t = threading.Thread(target=_run, daemon=True, name="flask-api")
    t.start()
    return t


# ------------------------------------------------------------------
# RTSP sources via StreamManager
# ------------------------------------------------------------------
def start_rtsp_sources(cfg: Dict[str, Any], services: Dict[str, Any], stream_manager):
    """Parse RTSP_URLS env var and register streams."""
    count = 0
    for idx, entry in enumerate(cfg["rtsp_urls"]):
        entry = entry.strip()
        if not entry:
            continue

        parts = entry.split("|")
        url = parts[0].strip()
        tasks = [t.strip() for t in parts[1].split("+")] if len(parts) > 1 else ["parking_violation"]
        camera_id = parts[2].strip() if len(parts) > 2 else ""

        valid = {"parking_violation", "smoke_flame", "common_space"}
        tasks = [t for t in tasks if t in valid] or ["parking_violation"]

        try:
            stream_manager.add_stream(url, tasks, camera_id=camera_id or None)
            count += 1
            logger.info("RTSP source %d added: %s | tasks=%s", idx, url, tasks)
        except Exception as e:
            logger.error("Failed to add RTSP source %d: %s", idx, e)

    return count


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    logger.info("Starting Hybrid Video Analysis System...")
    cfg = load_config()

    try:
        services = initialize_services(cfg)

        # File monitoring
        observer = start_multi_folder_monitoring(cfg, services)
        if not observer:
            logger.info("Falling back to single folder monitoring...")
            observer = start_single_folder_monitoring(cfg, services)
            if not observer:
                logger.error("All folder monitoring failed")
                return

        # Ensure upload sub-folders exist
        for name in ("parking", "smoke_flame", "common_space"):
            os.makedirs(os.path.join(cfg["upload_folder"], name), exist_ok=True)

        # StreamManager + API
        stream_manager = create_stream_manager(services)
        start_api_server(stream_manager, services["mongo"], services["webhook_service"], port=5000)

        # Optional RTSP sources from env
        rtsp_count = start_rtsp_sources(cfg, services, stream_manager)

        # Status summary
        logger.info("System ready | Parking: OK | Smoke/Flame: %s | CommonSpace: %s | RTSP: %d | API: :5000",
                     "OK" if services["smoke_service_ready"] else "OFF",
                     "OK" if services["common_space_service_ready"] else "OFF",
                     rtsp_count)

        # Block until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            stream_manager.stop_all()
            observer.stop()
            observer.join()

            services["parking_service"].flush_remaining()
            if services["smoke_service_ready"]:
                services["smoke_service"].flush_remaining()
            if services["common_space_service_ready"]:
                services["common_space_service"].flush_remaining()

            logger.info("System stopped gracefully")

    except Exception as e:
        logger.error("Main error: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
