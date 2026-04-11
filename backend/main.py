# backend/main.py

import os
import sys
import time
import logging
import threading
from typing import Dict, Any

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Config loading
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
    }


# ------------------------------------------------------------------
# Service initialization
# ------------------------------------------------------------------
def initialize_services(cfg: Dict[str, Any]):
    """Initialize all shared services."""

    # ----------- 1. Storage clients -----------
    from storage.minio_client import MinIOClient
    from storage.mongodb_client import MongoDBClient

    minio = MinIOClient(
        endpoint=cfg["minio_endpoint"],
        access_key=cfg["minio_access_key"],
        secret_key=cfg["minio_secret_key"],
        bucket_name=cfg["minio_bucket"]
    )
    mongo = MongoDBClient(mongo_uri=cfg["mongo_uri"], db_name=cfg["mongo_db_name"])

    # ----------- 2. Models and checkers -----------
    from ml_models.yolov8.model_loader import YOLOModelLoader
    from backend.services.parking_zone_checker import NoParkingZoneChecker
    from backend.services.stream_runtime import AppResources, StreamRuntimeFactory

    loader = YOLOModelLoader()
    loader.load_model("vehicle", "yolov8n.pt")
    try:
        loader.load_model("smoke_flame", "smoke_flame.pt")
        logger.info("Smoke/Flame model loaded")
    except Exception as e:
        logger.warning(f"Smoke/Flame model not available: {e}")

    # Keep one shared zone_checker instance for the whole process.
    zone_checker = NoParkingZoneChecker()
    logger.info(
        f"Created single zone_checker instance (ID: {id(zone_checker)}) "
        f"with zones: {list(zone_checker.zones.keys())}"
    )

    # ----------- 3. Detection services -----------
    from backend.services.violation_detection import detection_service as parking_service
    from backend.services.smoke_flame_detection import smoke_flame_detection_service, QwenVLAPIClient
    from backend.services.common_space_detection import common_space_detection_service
    from backend.config.qwen_vl_config import qwen_vl_api_config

    # Inject the shared zone_checker instance into the parking service.
    parking_service.set_clients(minio_client=minio, mongo_client=mongo)
    parking_service.set_model_loader(loader)
    parking_service.set_zone_checker(zone_checker)
    logger.info("Parking violation detection service ready.")

    # Smoke/flame detection service
    smoke_service_ready = False
    qwen_vl_client = None

    logger.debug("Qwen-VL config check:")
    logger.debug(f"   Config API URL: {qwen_vl_api_config.get_api_url()}")
    logger.debug(f"   Config API Key configured: {bool(qwen_vl_api_config.get_api_key())}")
    logger.debug(f"   Config Overall configured: {qwen_vl_api_config.is_configured()}")

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
            logger.info("Smoke/Flame detection service ready (using config file).")
        except Exception as e:
            logger.error(f"Failed to initialize smoke/flame detection with config file: {e}")
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
            logger.info("Smoke/Flame detection service ready (using environment variables).")
        except Exception as e:
            logger.error(f"Failed to initialize smoke/flame detection with environment variables: {e}")
    else:
        logger.warning(
            "Qwen-VL API is not configured in either the config file or environment variables; "
            "smoke/flame detection is disabled"
        )

    # Common space analysis service
    common_space_service_ready = False
    if qwen_vl_client:
        try:
            common_space_detection_service.set_clients(minio, mongo)
            common_space_detection_service.set_qwen_vl_client(qwen_vl_client)
            common_space_detection_service.set_sample_interval(cfg["common_space_interval_sec"])
            common_space_service_ready = True
            logger.info(
                f"Common space analysis service ready "
                f"(sampling interval: {cfg['common_space_interval_sec']}s)."
            )
        except Exception as e:
            logger.error(f"Failed to initialize common space analysis service: {e}")
    else:
        logger.warning("Qwen-VL client not available; common space analysis service disabled")

    app_resources = AppResources(
        minio=minio,
        mongo=mongo,
        zone_checker=zone_checker,
        qwen_vl_client=qwen_vl_client,
        smoke_service_ready=smoke_service_ready,
        common_space_service_ready=common_space_service_ready,
        common_space_interval_sec=cfg["common_space_interval_sec"],
        weights_dir=loader.weights_dir,
    )
    stream_runtime_factory = StreamRuntimeFactory(app_resources)

    return {
        "minio": minio,
        "mongo": mongo,
        "loader": loader,
        "zone_checker": zone_checker,  # shared instance
        "parking_service": parking_service,
        "smoke_service": smoke_flame_detection_service,
        "common_space_service": common_space_detection_service,
        "qwen_vl_client": qwen_vl_client,
        "smoke_service_ready": smoke_service_ready,
        "common_space_service_ready": common_space_service_ready,
        "app_resources": app_resources,
        "stream_runtime_factory": stream_runtime_factory,
    }


# ------------------------------------------------------------------
# Update multi-folder monitoring configuration
# ------------------------------------------------------------------
def update_monitor_folders_config():
    """Ensure the monitoring config includes the common_space folder."""
    try:
        from scripts.file_watcher import MONITOR_FOLDERS

        if "common_space" not in MONITOR_FOLDERS:
            MONITOR_FOLDERS["common_space"] = "common_space"
            logger.debug("Added common_space folder to MONITOR_FOLDERS configuration")

        return MONITOR_FOLDERS
    except Exception as e:
        logger.error(f"Failed to update monitor folders config: {e}")
        return None


# ------------------------------------------------------------------
# Multi-folder monitoring startup
# ------------------------------------------------------------------
def start_multi_folder_monitoring(cfg: Dict[str, Any], services: Dict[str, Any]) -> bool:
    """Start multi-folder monitoring with the shared zone_checker instance."""
    try:
        from scripts.file_watcher import start_multi_folder_watchdog

        MONITOR_FOLDERS = update_monitor_folders_config()
        if not MONITOR_FOLDERS:
            logger.error("Failed to get monitor folders configuration")
            return None

        upload_folder = cfg["upload_folder"]
        for folder_name in MONITOR_FOLDERS.keys():
            folder_path = os.path.join(upload_folder, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            logger.debug(f"Created/verified folder: {folder_path}")

        # Pass the shared zone_checker instance through.
        observer = start_multi_folder_watchdog(
            base_folder=cfg["upload_folder"],
            model_loader=services["loader"],
            parking_detection_service=services["parking_service"],
            smoke_flame_detection_service=services["smoke_service"],
            zone_checker=services["zone_checker"],
            frame_interval=cfg["frame_interval_sec"]
        )

        logger.info("Multi-folder monitoring started successfully")
        logger.info("Folder structure:")
        logger.info("   uploads/")
        logger.info("   - parking/      : parking_violation videos")
        logger.info("   - smoke_flame/  : smoke_flame videos")
        logger.info("   - common_space/ : common_space videos")

        if not services["smoke_service_ready"]:
            logger.warning("   Smoke/Flame detection service is not ready; check Qwen-VL configuration")
        if not services["common_space_service_ready"]:
            logger.warning("   Common space analysis service is not ready; check Qwen-VL configuration")

        return observer

    except Exception as e:
        logger.warning(f"Multi-folder monitoring not available: {e}")
        return None


# ------------------------------------------------------------------
# Single-folder monitoring (backward compatible)
# ------------------------------------------------------------------
def start_single_folder_monitoring(cfg: Dict[str, Any], services: Dict[str, Any]):
    """Start single-folder monitoring for backward compatibility."""
    try:
        from scripts.file_watcher import start_file_watchdog

        # Pass the shared zone_checker instance through.
        observer = start_file_watchdog(
            folder_path=cfg["upload_folder"],
            model_loader=services["loader"],
            detection_service=services["parking_service"],
            zone_checker=services["zone_checker"],
            frame_interval=cfg["frame_interval_sec"]
        )

        logger.info(f"Single folder monitoring started: {cfg['upload_folder']}")
        return observer

    except Exception as e:
        logger.error(f"Single folder monitoring failed: {e}")
        return None


# ------------------------------------------------------------------
# File processing callback
# ------------------------------------------------------------------
def create_file_processor(cfg: Dict[str, Any], services: Dict[str, Any]):
    """Create the file processing callback."""

    def process_video_file(video_path: str):
        """Process a local video file."""
        try:
            from backend.utils.video_processor import infer_detection_type_from_path, process_video_official

            detection_type = infer_detection_type_from_path(video_path, cfg["upload_folder"])
            logger.info(f"Processing: {os.path.basename(video_path)} | Type: {detection_type}")

            qwen_client = None
            if detection_type in ["smoke_flame", "common_space"]:
                qwen_client = services["qwen_vl_client"]
                if not qwen_client:
                    logger.error(f"Qwen-VL client not available for {detection_type} detection")
                    return

            process_video_official(
                video_path=video_path,
                model_loader=services["loader"],
                zone_checker=services["zone_checker"],
                frame_interval=cfg["frame_interval_sec"],
                detection_type=detection_type,
                minio_client=services["minio"],
                mongo_client=services["mongo"],
                qwen_vl_client=qwen_client
            )

        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")

    return process_video_file


# ------------------------------------------------------------------
# RTSP processing callback (legacy, kept for compatibility)
# ------------------------------------------------------------------
def create_rtsp_callback(services: Dict[str, Any], rtsp_id: str, detection_type: str = "parking_violation"):
    """Create a legacy RTSP callback."""

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
                logger.error(f"RTSP processing error for {source_id}: {e}")

    return rtsp_callback


# ------------------------------------------------------------------
# Create StreamManager instance
# ------------------------------------------------------------------
def create_stream_manager(services: Dict[str, Any]):
    """Create the StreamManager instance."""
    from backend.services.stream_manager import StreamManager

    # LEGACY:
    # stream_manager = StreamManager(services=services, mongo_client=services["mongo"])
    stream_manager = StreamManager(
        runtime_factory=services["stream_runtime_factory"],
        mongo_client=services["mongo"]
    )
    logger.info("StreamManager created")
    return stream_manager


# ------------------------------------------------------------------
# Flask API server (daemon thread)
# ------------------------------------------------------------------
def start_api_server(stream_manager, mongo_client, port: int = 5000):
    """Start the Flask API server in a daemon thread."""
    from flask import Flask
    from flask_cors import CORS
    from backend.api.stream_routes import stream_bp, init_stream_routes
    from backend.api.event_routes import event_bp, init_event_routes

    app = Flask(__name__)
    CORS(app)

    init_stream_routes(stream_manager)
    init_event_routes(mongo_client)
    app.register_blueprint(stream_bp)
    app.register_blueprint(event_bp)

    def _run():
        logger.info(f"Flask API starting on port {port}...")
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    t = threading.Thread(target=_run, daemon=True, name="flask-api")
    t.start()
    logger.info(f"Flask API thread started (port {port})")
    return t


# ------------------------------------------------------------------
# Start RTSP sources via StreamManager
# ------------------------------------------------------------------
def start_rtsp_sources(cfg: Dict[str, Any], services: Dict[str, Any], stream_manager=None):
    """Start RTSP sources, preferring StreamManager when available."""
    rtsp_count = 0

    for idx, url_with_type in enumerate(cfg["rtsp_urls"]):
        if not url_with_type.strip():
            continue

        parts = url_with_type.strip().split("|")
        rtsp_url = parts[0].strip()

        # New format: url|task1+task2  (e.g. rtsp://cam1|parking_violation+smoke_flame)
        # Legacy format: url|single_type
        if len(parts) > 1:
            type_str = parts[1].strip()
            tasks = [t.strip() for t in type_str.split("+") if t.strip()]
        else:
            tasks = ["parking_violation"]

        camera_id = parts[2].strip() if len(parts) > 2 else ""

        valid_types = ["parking_violation", "smoke_flame", "common_space"]
        tasks = [t for t in tasks if t in valid_types]
        if not tasks:
            logger.warning(f"No valid tasks for RTSP {idx}, using default")
            tasks = ["parking_violation"]

        # Use StreamManager if available
        if stream_manager:
            try:
                stream_id = stream_manager.add_stream(
                    rtsp_url,
                    tasks,
                    camera_id=camera_id or None
                )
                rtsp_count += 1
                logger.info(
                    f"RTSP source {idx} added via StreamManager: "
                    f"{rtsp_url} | camera_id={camera_id or '-'} | Tasks: {tasks}"
                )
            except Exception as e:
                logger.error(f"Failed to add RTSP source {idx} via StreamManager: {e}")
        else:
            # Legacy fallback (single task per stream)
            detection_type = tasks[0]
            try:
                from backend.utils.frame_capture import VideoFrameCapture

                rtsp_id = f"rtsp_{idx}"
                source_key = camera_id or rtsp_id

                if detection_type == "smoke_flame" and not services["smoke_service_ready"]:
                    logger.warning(f"Smoke/flame detection not ready for RTSP {idx}, skipping")
                    continue
                elif detection_type == "common_space" and not services["common_space_service_ready"]:
                    logger.warning(f"Common space analysis not ready for RTSP {idx}, skipping")
                    continue

                if detection_type == "parking_violation" and camera_id:
                    zones = services["zone_checker"].get_zones_for_source(camera_id)
                    if not zones:
                        logger.warning(
                            f"Legacy RTSP fallback mode found no zone for camera_id={camera_id}. "
                            f"Please use StreamManager/API path for GUI-assisted setup."
                        )
                        continue

                cap = VideoFrameCapture()
                cap.register_batch_callback(create_rtsp_callback(services, rtsp_id, detection_type))
                cap.add_rtsp_source(
                    source_id=source_key,
                    rtsp_url=rtsp_url,
                    batch_size=8,
                    batch_sec=1.0,
                    reconnect_delay=5
                )

                rtsp_count += 1
                logger.info(f"RTSP source {idx} added (legacy): {rtsp_url} | Type: {detection_type}")

            except Exception as e:
                logger.error(f"Failed to add RTSP source {idx}: {e}")

    return rtsp_count


# ------------------------------------------------------------------
# Main entry
# ------------------------------------------------------------------
def main():
    logger.info("Starting Hybrid Video Analysis System...")
    cfg = load_config()

    try:
        # ----------- 1. Initialize shared services -----------
        logger.info("Initializing services...")
        services = initialize_services(cfg)

        # ----------- 2. Try multi-folder monitoring first -----------
        observer = start_multi_folder_monitoring(cfg, services)

        # ----------- 3. Fall back to single-folder monitoring if needed -----------
        if not observer:
            logger.info("Falling back to single folder monitoring...")
            observer = start_single_folder_monitoring(cfg, services)

            if not observer:
                logger.error("Both multi-folder and single-folder monitoring failed")
                return

        # ----------- 4. StreamManager + Flask API -----------
        stream_manager = create_stream_manager(services)
        api_thread = start_api_server(stream_manager, services["mongo"], port=5000)

        # ----------- 5. Optional RTSP sources via StreamManager -----------
        rtsp_count = start_rtsp_sources(cfg, services, stream_manager=stream_manager)

        if rtsp_count > 0:
            logger.info(f"{rtsp_count} RTSP sources initialized")

        # ----------- 6. System status summary -----------
        logger.info("System Status:")
        logger.info(f"   Upload folder: {cfg['upload_folder']}")
        logger.info("   Parking detection: Ready")
        logger.info(
            f"   Smoke/Flame detection: "
            f"{'Ready' if services['smoke_service_ready'] else 'Disabled'}"
        )
        logger.info(
            f"   Common space analysis: "
            f"{'Ready' if services['common_space_service_ready'] else 'Disabled'}"
        )
        if services['common_space_service_ready']:
            logger.info(f"   Common space interval: {cfg['common_space_interval_sec']}s")
        logger.info(f"   RTSP sources: {rtsp_count}")
        logger.info(f"   Frame interval: {cfg['frame_interval_sec']}s")
        logger.info("   Flask API: http://0.0.0.0:5000/api/streams")

        # ----------- 7. Ensure upload folders exist -----------
        upload_folder = cfg["upload_folder"]
        os.makedirs(upload_folder, exist_ok=True)

        common_space_folder = os.path.join(upload_folder, "common_space")
        os.makedirs(common_space_folder, exist_ok=True)
        logger.info(f"Created directory: {common_space_folder}")

        for folder_name in ["parking", "smoke_flame"]:
            folder_path = os.path.join(upload_folder, folder_name)
            os.makedirs(folder_path, exist_ok=True)

        # ----------- 8. Usage notes -----------
        logger.info("Usage Instructions:")
        logger.info("   1. Place videos for analysis in the following folders:")
        logger.info("      - uploads/parking/        : Parking violation detection")
        logger.info("      - uploads/smoke_flame/    : Smoke/Flame detection")
        logger.info("      - uploads/common_space/   : Public space analysis")
        logger.info("   2. The system will automatically process uploaded videos")
        logger.info("   3. For common space analysis, frames are sampled every 30 seconds")
        logger.info("   4. Analysis results are saved to MinIO and MongoDB")

        # ----------- 9. Keep the main thread alive -----------
        logger.info("System started successfully. Press Ctrl+C to stop.")

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
        logger.error(f"Main error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
