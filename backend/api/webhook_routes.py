# backend/api/webhook_routes.py
"""
REST API for webhook configuration (FR6.3).

Endpoints:
  POST   /api/webhooks          — register a webhook
  GET    /api/webhooks          — list all webhooks
  DELETE /api/webhooks/<id>     — delete a webhook
"""

from flask import Blueprint, jsonify, request

webhook_bp = Blueprint("webhook", __name__)

_webhook_service = None


def init_webhook_routes(webhook_service):
    global _webhook_service
    _webhook_service = webhook_service


@webhook_bp.route("/api/webhooks", methods=["POST"])
def register_webhook():
    data = request.get_json(silent=True) or {}
    url = data.get("url")
    if not url:
        return jsonify({"error": "url is required"}), 400

    event_types = data.get("event_types")
    if event_types is not None and not isinstance(event_types, list):
        return jsonify({"error": "event_types must be a list"}), 400

    try:
        wh_id = _webhook_service.register(url, event_types)
        return jsonify({"id": wh_id, "url": url, "event_types": event_types or []}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@webhook_bp.route("/api/webhooks", methods=["GET"])
def list_webhooks():
    try:
        return jsonify(_webhook_service.list_all()), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@webhook_bp.route("/api/webhooks/<webhook_id>", methods=["DELETE"])
def delete_webhook(webhook_id):
    try:
        deleted = _webhook_service.delete(webhook_id)
        if deleted:
            return jsonify({"deleted": True}), 200
        return jsonify({"error": "not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
