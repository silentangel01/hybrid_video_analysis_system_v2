# backend/api/stream_routes.py
"""
Flask Blueprint for RTSP stream management REST API.

Endpoints:
  GET    /api/streams            - list all streams
  GET    /api/streams/<id>/metrics - get runtime metrics for one stream
  POST   /api/streams            - add a stream  {url, tasks, camera_id}
  DELETE /api/streams/<id>       - remove a stream
  PUT    /api/streams/<id>/tasks - update tasks   {tasks}
"""

import logging
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

stream_bp = Blueprint("streams", __name__)

# Will be set from main.py before the blueprint is registered
_stream_manager = None


def init_stream_routes(stream_manager):
    global _stream_manager
    _stream_manager = stream_manager


# ------------------------------------------------------------------
# GET /api/streams
# ------------------------------------------------------------------
@stream_bp.route("/api/streams", methods=["GET"])
def list_streams():
    if _stream_manager is None:
        return jsonify({"error": "StreamManager not initialised"}), 503
    return jsonify(_stream_manager.get_streams()), 200


@stream_bp.route("/api/streams/<stream_id>/metrics", methods=["GET"])
def get_stream_metrics(stream_id):
    if _stream_manager is None:
        return jsonify({"error": "StreamManager not initialised"}), 503

    data = _stream_manager.get_stream_metrics(stream_id)
    if data is None:
        return jsonify({"error": f"{stream_id} not found"}), 404
    return jsonify(data), 200


# ------------------------------------------------------------------
# POST /api/streams
# ------------------------------------------------------------------
@stream_bp.route("/api/streams", methods=["POST"])
def add_stream():
    if _stream_manager is None:
        return jsonify({"error": "StreamManager not initialised"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    url = data.get("url", "").strip()
    tasks = data.get("tasks", [])
    camera_id = data.get("camera_id", "").strip()

    if not url:
        return jsonify({"error": "url is required"}), 400
    if not tasks or not isinstance(tasks, list):
        return jsonify({"error": "tasks must be a non-empty list"}), 400

    try:
        stream_id = _stream_manager.add_stream(url, tasks, camera_id=camera_id or None)
        return jsonify({
            "stream_id": stream_id,
            "url": url,
            "tasks": tasks,
            "camera_id": camera_id
        }), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"[API] add_stream error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


# ------------------------------------------------------------------
# DELETE /api/streams/<id>
# ------------------------------------------------------------------
@stream_bp.route("/api/streams/<stream_id>", methods=["DELETE"])
def remove_stream(stream_id):
    if _stream_manager is None:
        return jsonify({"error": "StreamManager not initialised"}), 503

    ok = _stream_manager.remove_stream(stream_id)
    if ok:
        return jsonify({"message": f"{stream_id} removed"}), 200
    return jsonify({"error": f"{stream_id} not found"}), 404


# ------------------------------------------------------------------
# PUT /api/streams/<id>/tasks
# ------------------------------------------------------------------
@stream_bp.route("/api/streams/<stream_id>/tasks", methods=["PUT"])
def update_tasks(stream_id):
    if _stream_manager is None:
        return jsonify({"error": "StreamManager not initialised"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    tasks = data.get("tasks", [])
    if not tasks or not isinstance(tasks, list):
        return jsonify({"error": "tasks must be a non-empty list"}), 400

    ok = _stream_manager.update_tasks(stream_id, tasks)
    if ok:
        return jsonify({"stream_id": stream_id, "tasks": tasks}), 200
    return jsonify({"error": f"{stream_id} not found or invalid tasks"}), 404
