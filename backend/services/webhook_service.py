# backend/services/webhook_service.py
"""
Webhook notification service (FR6.3).

Design:
  - Async POST via ThreadPoolExecutor(4), fire-and-forget
  - Webhook configs cached from MongoDB ``webhook_configs`` collection (60 s TTL)
  - Per-webhook: 5 s timeout, up to 2 retries
  - Optional ``event_types`` filter per webhook
"""

import hashlib
import hmac
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import urllib.request
import urllib.error

from backend.utils.performance_metrics import get_thread_pool_queue_size

logger = logging.getLogger(__name__)

_CACHE_TTL_SEC = 60.0
_REQUEST_TIMEOUT_SEC = 5
_MAX_RETRIES = 2
_WEBHOOK_SECRET: Optional[str] = os.environ.get("WEBHOOK_SECRET")


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class WebhookService:
    """Manages webhook configs and dispatches event notifications."""

    def __init__(self, mongo_client):
        self._mongo = mongo_client
        self._col = (
            mongo_client.db["webhook_configs"]
            if mongo_client is not None and mongo_client.db is not None
            else None
        )
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="webhook"
        )
        self._cache: Optional[List[Dict[str, Any]]] = None
        self._cache_ts: float = 0.0
        self._cache_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._enabled = _env_flag("ENABLE_WEBHOOKS", default=True)
        self._backpressure_enabled = _env_flag(
            "ENABLE_BACKPRESSURE_PROTECTION",
            default=False,
        )
        self._max_queue_size = max(1, int(os.environ.get("WEBHOOK_MAX_QUEUE", "100")))
        self._dropped_notifications_total = 0

        if self._enabled:
            logger.info("Webhook notifications enabled")
        else:
            logger.warning("Webhook notifications disabled by ENABLE_WEBHOOKS")
        if self._backpressure_enabled:
            logger.info("Webhook queue protection enabled (max_queue_size=%d)", self._max_queue_size)

    # ------------------------------------------------------------------
    # CRUD (called from API routes)
    # ------------------------------------------------------------------

    def register(self, url: str, event_types: Optional[List[str]] = None) -> str:
        """Register a new webhook. Returns its ``_id`` as a string."""
        if self._col is None:
            raise RuntimeError("MongoDB not available")
        doc = {"url": url, "event_types": event_types or [], "created_at": time.time()}
        result = self._col.insert_one(doc)
        self._invalidate_cache()
        logger.info("Webhook registered: %s (id=%s)", url, result.inserted_id)
        return str(result.inserted_id)

    def list_all(self) -> List[Dict[str, Any]]:
        if self._col is None:
            return []
        docs = list(self._col.find())
        for d in docs:
            d["_id"] = str(d["_id"])
        return docs

    def delete(self, webhook_id: str) -> bool:
        if self._col is None:
            return False
        from bson import ObjectId

        result = self._col.delete_one({"_id": ObjectId(webhook_id)})
        self._invalidate_cache()
        return result.deleted_count > 0

    # ------------------------------------------------------------------
    # Notification dispatch
    # ------------------------------------------------------------------

    def notify(self, event_data: Dict[str, Any]) -> None:
        """Fire-and-forget: POST *event_data* to all matching webhooks."""
        if not self._enabled:
            return

        configs = self._get_configs()
        if not configs:
            return

        event_type = event_data.get("event_type", "")
        for cfg in configs:
            # Filter by event_types if the webhook specifies any.
            allowed = cfg.get("event_types") or []
            if allowed and event_type not in allowed:
                continue
            url = cfg.get("url")
            if not url:
                continue
            if self._backpressure_enabled:
                queue_size = get_thread_pool_queue_size(self._executor)
                if queue_size >= self._max_queue_size:
                    with self._stats_lock:
                        self._dropped_notifications_total += 1
                        dropped_total = self._dropped_notifications_total
                    if dropped_total == 1 or dropped_total % 50 == 0:
                        logger.warning(
                            "Webhook queue full (%d >= %d); dropped %d notification(s)",
                            queue_size,
                            self._max_queue_size,
                            dropped_total,
                        )
                    continue
            self._executor.submit(self._post_with_retry, url, event_data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_configs(self) -> List[Dict[str, Any]]:
        with self._cache_lock:
            if self._cache is not None and (time.time() - self._cache_ts) < _CACHE_TTL_SEC:
                return self._cache

        try:
            configs = self.list_all()
        except Exception as e:
            logger.error("Failed to load webhook configs: %s", e)
            configs = []

        with self._cache_lock:
            self._cache = configs
            self._cache_ts = time.time()
        return configs

    def _invalidate_cache(self) -> None:
        with self._cache_lock:
            self._cache = None
            self._cache_ts = 0.0

    @staticmethod
    def _post_with_retry(url: str, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, default=str).encode("utf-8")
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        # Compute HMAC-SHA256 signature when a shared secret is configured.
        if _WEBHOOK_SECRET:
            sig = hmac.new(
                _WEBHOOK_SECRET.encode("utf-8"), data, hashlib.sha256
            ).hexdigest()
            headers["X-HVAS-Signature"] = sig
        for attempt in range(1 + _MAX_RETRIES):
            try:
                req = urllib.request.Request(
                    url, data=data, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_SEC) as resp:
                    logger.debug("Webhook POST %s -> %d", url, resp.status)
                    return
            except Exception as e:
                logger.warning(
                    "Webhook POST %s attempt %d failed: %s", url, attempt + 1, e
                )
        logger.error("Webhook POST %s exhausted retries", url)
