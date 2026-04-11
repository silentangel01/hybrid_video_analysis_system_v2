"""Flask Blueprint for event query REST API."""

from datetime import datetime
import logging
from typing import Any, Dict

from bson import ObjectId
from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

event_bp = Blueprint("events", __name__)

_mongo_client = None


def init_event_routes(mongo_client):
    global _mongo_client
    _mongo_client = mongo_client


def _require_mongo():
    if _mongo_client is None or getattr(_mongo_client, "collection", None) is None:
        return jsonify({"error": "MongoDB client not initialised"}), 503
    return None


def _parse_float(value, field_name: str):
    if value in (None, ""):
        return None, None
    try:
        return float(value), None
    except (TypeError, ValueError):
        return None, (jsonify({"error": f"{field_name} must be a number"}), 400)


def _parse_int(value, field_name: str, default: int, minimum: int = 0, maximum: int = 1000):
    if value in (None, ""):
        return default, None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None, (jsonify({"error": f"{field_name} must be an integer"}), 400)

    if parsed < minimum or parsed > maximum:
        return None, (jsonify({"error": f"{field_name} must be between {minimum} and {maximum}"}), 400)

    return parsed, None


def _serialize_value(value: Any):
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    return value


def _serialize_event(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _serialize_value(value) for key, value in doc.items()}


@event_bp.route("/api/events", methods=["GET"])
def list_events():
    err = _require_mongo()
    if err:
        return err

    camera_id = request.args.get("camera_id", "").strip() or None
    event_type = request.args.get("event_type", "").strip() or None
    detection_stage = request.args.get("detection_stage", "").strip() or None

    start_time, parse_err = _parse_float(request.args.get("start_time"), "start_time")
    if parse_err:
        return parse_err

    end_time, parse_err = _parse_float(request.args.get("end_time"), "end_time")
    if parse_err:
        return parse_err

    limit, parse_err = _parse_int(request.args.get("limit"), "limit", default=50, minimum=1, maximum=500)
    if parse_err:
        return parse_err

    skip, parse_err = _parse_int(request.args.get("skip"), "skip", default=0, minimum=0, maximum=100000)
    if parse_err:
        return parse_err

    events = _mongo_client.find_events(
        camera_id=camera_id,
        event_type=event_type,
        detection_stage=detection_stage,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        skip=skip
    )

    total_query: Dict[str, Any] = {}
    if camera_id:
        total_query["camera_id"] = camera_id
    if event_type:
        total_query["event_type"] = event_type
    if detection_stage:
        total_query["detection_stage"] = detection_stage
    if start_time is not None or end_time is not None:
        total_query["timestamp"] = {}
        if start_time is not None:
            total_query["timestamp"]["$gte"] = start_time
        if end_time is not None:
            total_query["timestamp"]["$lte"] = end_time

    total = _mongo_client.collection.count_documents(total_query)

    return jsonify({
        "items": [_serialize_event(event) for event in events],
        "pagination": {
            "limit": limit,
            "skip": skip,
            "returned": len(events),
            "total": total,
            "has_more": skip + len(events) < total,
        },
        "filters": {
            "camera_id": camera_id,
            "event_type": event_type,
            "detection_stage": detection_stage,
            "start_time": start_time,
            "end_time": end_time,
        }
    }), 200


@event_bp.route("/api/events/latest", methods=["GET"])
def latest_events():
    err = _require_mongo()
    if err:
        return err

    since, parse_err = _parse_float(request.args.get("since"), "since")
    if parse_err:
        return parse_err

    limit, parse_err = _parse_int(request.args.get("limit"), "limit", default=100, minimum=1, maximum=500)
    if parse_err:
        return parse_err

    camera_id = request.args.get("camera_id", "").strip() or None
    event_type = request.args.get("event_type", "").strip() or None

    query: Dict[str, Any] = {}
    if since is not None:
        query["timestamp"] = {"$gt": since}
    if camera_id:
        query["camera_id"] = camera_id
    if event_type:
        query["event_type"] = event_type

    cursor = _mongo_client.collection.find(query).sort("timestamp", 1).limit(limit)
    items = [_serialize_event(doc) for doc in cursor]
    newest_timestamp = max((item.get("timestamp", since or 0) for item in items), default=since)

    return jsonify({
        "items": items,
        "since": since,
        "next_since": newest_timestamp,
        "returned": len(items),
    }), 200


@event_bp.route("/api/events/stats", methods=["GET"])
def event_stats():
    err = _require_mongo()
    if err:
        return err

    stats = _mongo_client.get_event_statistics()
    return jsonify(_serialize_value(stats)), 200


@event_bp.route("/api/events/<event_id>", methods=["GET"])
def event_detail(event_id: str):
    err = _require_mongo()
    if err:
        return err

    try:
        object_id = ObjectId(event_id)
    except Exception:
        return jsonify({"error": "invalid event_id"}), 400

    doc = _mongo_client.collection.find_one({"_id": object_id})
    if doc is None:
        return jsonify({"error": "event not found"}), 404

    return jsonify(_serialize_event(doc)), 200
