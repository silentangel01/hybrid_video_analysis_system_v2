# backend/api/health_routes.py
"""Health check endpoint for HVAS (Phase 1.7)."""

import logging
import time

from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)

health_bp = Blueprint("health", __name__)

_mongo_client = None
_minio_client = None
_stream_manager = None
_start_time: float = time.time()


def init_health_routes(mongo_client, minio_client=None, stream_manager=None):
    global _mongo_client, _minio_client, _stream_manager
    _mongo_client = mongo_client
    _minio_client = minio_client
    _stream_manager = stream_manager


@health_bp.route("/api/health", methods=["GET"])
def health_check():
    # MongoDB
    mongo_status = "disconnected"
    if _mongo_client is not None:
        try:
            if _mongo_client.health_check():
                mongo_status = "connected"
        except Exception:
            pass

    # MinIO
    minio_status = "disconnected"
    if _minio_client is not None:
        try:
            client = getattr(_minio_client, "client", None)
            if client is not None:
                client.list_buckets()
                minio_status = "connected"
        except Exception:
            pass

    # Active streams & loaded models
    active_streams = 0
    models_loaded = []
    if _stream_manager is not None:
        try:
            streams = _stream_manager.get_streams()
            active_streams = len(streams)
            task_set = set()
            for s in streams:
                for t in s.get("tasks", []):
                    task_set.add(t)
            # Map task names to model names
            task_model_map = {
                "parking_violation": "vehicle",
                "smoke_flame": "smoke_flame",
                "common_space": "qwen_vl",
            }
            models_loaded = sorted(
                task_model_map[t] for t in task_set if t in task_model_map
            )
        except Exception:
            pass

    overall = "ok" if mongo_status == "connected" else "degraded"

    return jsonify({
        "status": overall,
        "mongodb": mongo_status,
        "minio": minio_status,
        "active_streams": active_streams,
        "models_loaded": models_loaded,
        "uptime_sec": round(time.time() - _start_time),
    }), 200
