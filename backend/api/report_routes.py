"""Flask Blueprint for common-space report APIs."""

from datetime import datetime
from typing import Any, Dict, Optional

from bson import ObjectId

from flask import Blueprint, jsonify, request

from backend.services.common_space_report_service import CommonSpaceReportService

report_bp = Blueprint("reports", __name__)

_stream_manager = None
_report_service: Optional[CommonSpaceReportService] = None
_report_llm_client = None


def init_report_routes(stream_manager, mongo_client, report_llm_client=None):
    global _stream_manager, _report_service, _report_llm_client
    _stream_manager = stream_manager
    _report_service = CommonSpaceReportService(mongo_client)
    _report_llm_client = report_llm_client


def _stream_payload(stream: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "stream_id": stream.get("stream_id"),
        "camera_id": stream.get("camera_id"),
        "url": stream.get("url"),
        "tasks": stream.get("tasks") or [],
        "status": stream.get("status"),
        "location": stream.get("location"),
        "lat_lng": stream.get("lat_lng"),
        "area_code": stream.get("area_code"),
        "group": stream.get("group"),
        "created_at": stream.get("created_at"),
    }


def _find_stream(stream_id: str, stream_url: str):
    if _stream_manager is None:
        return None

    return next(
        (
            item for item in _stream_manager.get_streams()
            if (stream_id and item.get("stream_id") == stream_id)
            or (stream_url and item.get("url") == stream_url)
        ),
        None,
    )


def _serialize_value(value: Any):
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    return value


def _parse_limit(raw_value, default: int = 5, minimum: int = 1, maximum: int = 50):
    if raw_value in (None, ""):
        return default, None
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return None, (jsonify({"success": False, "error": "limit must be an integer"}), 400)

    if parsed < minimum or parsed > maximum:
        return None, (
            jsonify({"success": False, "error": f"limit must be between {minimum} and {maximum}"}),
            400,
        )
    return parsed, None


@report_bp.route("/api/reports/common-space/streams", methods=["GET"])
def list_common_space_streams():
    if _stream_manager is None:
        return jsonify({"success": False, "error": "StreamManager not initialised"}), 503

    streams = [
        _stream_payload(stream)
        for stream in _stream_manager.get_streams()
        if "common_space" in (stream.get("tasks") or [])
    ]
    return jsonify({"success": True, "items": streams}), 200


@report_bp.route("/api/reports/common-space/history", methods=["GET"])
def list_common_space_report_history():
    if _report_service is None:
        return jsonify({"success": False, "error": "Report service not initialised"}), 503

    limit, parse_err = _parse_limit(request.args.get("limit"), default=5)
    if parse_err:
        return parse_err

    items = _report_service.list_saved_reports(limit=limit)
    return jsonify({
        "success": True,
        "items": [_serialize_value(item) for item in items],
    }), 200


@report_bp.route("/api/reports/common-space/generate", methods=["POST"])
def generate_common_space_report():
    if _stream_manager is None or _report_service is None:
        return jsonify({"success": False, "error": "Report service not initialised"}), 503

    data = request.get_json(silent=True) or {}
    stream_id = str(data.get("stream_id") or "").strip()
    stream_url = str(data.get("url") or "").strip()
    start_time = data.get("start_time")
    end_time = data.get("end_time")

    if not stream_id and not stream_url:
        return jsonify({"success": False, "error": "stream_id or url is required"}), 400
    if start_time in (None, "") or end_time in (None, ""):
        return jsonify({"success": False, "error": "start_time and end_time are required"}), 400

    try:
        start_time = float(start_time)
        end_time = float(end_time)
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "start_time and end_time must be numbers"}), 400

    stream = _find_stream(stream_id, stream_url)
    if stream is None:
        return jsonify({"success": False, "error": "stream not found"}), 404
    if "common_space" not in (stream.get("tasks") or []):
        return jsonify({"success": False, "error": "stream has no common_space task"}), 400

    report = _report_service.build_report(stream, start_time, end_time)
    saved_report_id = _report_service.save_generated_report(report)
    return jsonify({
        "success": True,
        "report": report,
        "saved_report_id": saved_report_id,
        "saved_report_kind": "rule",
    }), 200


@report_bp.route("/api/reports/common-space/generate-llm", methods=["POST"])
def generate_common_space_report_llm():
    if _stream_manager is None or _report_service is None:
        return jsonify({"success": False, "error": "Report service not initialised"}), 503
    if _report_llm_client is None:
        return jsonify({"success": False, "error": "Report LLM not configured"}), 503

    data = request.get_json(silent=True) or {}
    stream_id = str(data.get("stream_id") or "").strip()
    stream_url = str(data.get("url") or "").strip()
    start_time = data.get("start_time")
    end_time = data.get("end_time")
    language = str(data.get("language") or "zh-CN").strip()

    if not stream_id and not stream_url:
        return jsonify({"success": False, "error": "stream_id or url is required"}), 400
    if start_time in (None, "") or end_time in (None, ""):
        return jsonify({"success": False, "error": "start_time and end_time are required"}), 400

    try:
        start_time = float(start_time)
        end_time = float(end_time)
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "start_time and end_time must be numbers"}), 400

    stream = _find_stream(stream_id, stream_url)
    if stream is None:
        return jsonify({"success": False, "error": "stream not found"}), 404
    if "common_space" not in (stream.get("tasks") or []):
        return jsonify({"success": False, "error": "stream has no common_space task"}), 400

    report = _report_service.build_report(stream, start_time, end_time)

    try:
        llm_result = _report_llm_client.summarize_common_space_report(
            report,
            language=language,
        )
    except Exception as exc:
        return jsonify({
            "success": False,
            "error": f"Report LLM summarisation failed: {exc}",
            "report": report,
        }), 502

    saved_report_id = _report_service.save_generated_report(
        report,
        llm_summary=llm_result["summary"],
        llm_meta={
            "provider": llm_result["provider"],
            "model": llm_result["model"],
            "language": llm_result["language"],
        },
    )

    return jsonify({
        "success": True,
        "report": report,
        "llm_summary": llm_result["summary"],
        "llm_meta": {
            "provider": llm_result["provider"],
            "model": llm_result["model"],
            "language": llm_result["language"],
            "fallback": False,
        },
        "saved_report_id": saved_report_id,
        "saved_report_kind": "llm",
    }), 200
