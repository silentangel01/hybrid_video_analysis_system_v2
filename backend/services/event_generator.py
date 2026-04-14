# backend/services/event_generator.py
"""
Event Generator -- 事件持久化模块

负责：
  1. 单事件写库 (handle_event_detected)
  2. 整帧聚合写库 (handle_frame_events)
  3. 各任务专用入口 (handle_parking_violation_events / handle_smoke_flame_events / handle_common_space_events)
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from backend.models.event import EventModel
from storage.minio_client import MinIOClient
from storage.mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)

# Module-level webhook service reference (set by main.py after init).
_webhook_service = None


def set_webhook_service(service) -> None:
    """Inject the WebhookService instance for event notifications."""
    global _webhook_service
    _webhook_service = service


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _clean_image_url(url: str) -> str:
    """Strip pre-signed query parameters from MinIO URLs."""
    return url.split('?')[0] if '?X-Amz-' in url else url


def _infer_frame_event_type(
    violations: List[Dict[str, Any]],
    event_type_override: Optional[str] = None,
    zones: Optional[List[List[tuple]]] = None,
) -> str:
    """Infer a single frame-level event type from aggregated detections."""
    if event_type_override:
        return event_type_override
    if zones:
        return "parking_violation"

    raw_types = {
        str(item.get("event_type") or item.get("class_name", "")).lower()
        for item in violations
        if item.get("event_type") or item.get("class_name")
    }

    if raw_types and raw_types.issubset({"smoke", "fire", "flame", "smoke_flame"}):
        return "smoke_flame"
    if len(raw_types) == 1:
        return raw_types.pop()
    return "unknown"


def _build_aggregate_objects(violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build compact per-object details for one frame-level event."""
    objects: List[Dict[str, Any]] = []
    for item in violations:
        obj: Dict[str, Any] = {}
        for key in ("class_name", "event_type", "bbox", "detection_stage"):
            if item.get(key) is not None:
                obj[key] = item[key]
        if item.get("confidence") is not None:
            try:
                obj["confidence"] = float(item["confidence"])
            except (ValueError, TypeError):
                obj["confidence"] = 0.0
        objects.append(obj)
    return objects


def _build_count_by_class(violations: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in violations:
        key = item.get("class_name") or item.get("event_type") or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _build_description(
    event_type: str,
    confidence: float,
    detection_stage: Optional[str],
    object_count: Optional[int],
    analysis_result: Optional[str],
    analysis_summary: Optional[Dict[str, Any]],
) -> str:
    """Auto-generate a human-readable description when none is supplied."""
    if event_type == "common_space_utilization":
        return _describe_common_space(analysis_result, analysis_summary)
    if event_type.lower() in ("smoke", "fire", "flame", "smoke_flame"):
        return _describe_smoke_flame(event_type, confidence, detection_stage, object_count)
    if event_type == "parking_violation":
        return _describe_parking(confidence, object_count)
    return f"{event_type.title()} detected (conf={confidence:.2f})"


def _describe_common_space(
    analysis_result: Optional[str],
    analysis_summary: Optional[Dict[str, Any]],
) -> str:
    if analysis_summary:
        people = analysis_summary.get("estimated_people_count", 0)
        occupancy = analysis_summary.get("space_occupancy", "unknown")
        activities = analysis_summary.get("activity_types", [])
        safety = analysis_summary.get("safety_concerns", False)

        parts = [f"Public space analysis: {people} people"]
        if occupancy != "unknown":
            parts.append(f"{occupancy} occupancy")
        if activities and activities != ["unknown"]:
            text = ", ".join(activities[:3])
            if len(activities) > 3:
                text += f" and {len(activities) - 3} more"
            parts.append(f"activities: {text}")
        if safety:
            parts.append("safety concerns identified")
        return ", ".join(parts)

    if analysis_result:
        period = analysis_result.find(".")
        comma = analysis_result.find(",")
        if 0 < period < 100:
            return analysis_result[: period + 1]
        if 0 < comma < 100:
            return analysis_result[:comma] + "..."
        return (analysis_result[:120] + "...") if len(analysis_result) > 120 else analysis_result

    return "Public space utilization analyzed"


def _describe_smoke_flame(
    event_type: str, confidence: float, detection_stage: Optional[str], object_count: Optional[int]
) -> str:
    stage = f" ({detection_stage})" if detection_stage else ""
    if object_count and object_count > 1:
        return f"Smoke/flame detected in {object_count} regions{stage} (max_conf={confidence:.2f})"
    label = "Smoke/flame" if event_type.lower() == "smoke_flame" else event_type.title()
    return f"{label} detected{stage} (conf={confidence:.2f})"


def _describe_parking(confidence: float, object_count: Optional[int]) -> str:
    if object_count and object_count > 1:
        return f"{object_count} vehicles in no-parking zone (max_conf={confidence:.2f})"
    return f"Vehicle in no-parking zone (conf={confidence:.2f})"


# ------------------------------------------------------------------
# Core: single-event persistence
# ------------------------------------------------------------------

def handle_event_detected(
    minio_client: MinIOClient,
    mongo_client: MongoDBClient,
    image_url: str,
    camera_id: str,
    timestamp: float,
    event_type: str,
    confidence: float = 0.0,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    frame_index: Optional[int] = None,
    description: Optional[str] = None,
    zone_polygon: Optional[List[Tuple[int, int]]] = None,
    detection_stage: Optional[str] = None,
    analysis_result: Optional[str] = None,
    analysis_summary: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    object_count: Optional[int] = None,
    objects: Optional[List[Dict[str, Any]]] = None,
    lat_lng: Optional[str] = None,
    location: Optional[str] = None,
    area_code: Optional[str] = None,
    group: Optional[str] = None,
) -> bool:
    """Persist a single event metadata document into MongoDB."""
    _ = minio_client  # kept in signature for caller compatibility

    try:
        clean_url = _clean_image_url(image_url)
        if not clean_url:
            logger.warning("Image URL is empty; event saved without visual evidence.")

        try:
            conf = float(confidence)
        except (ValueError, TypeError):
            logger.warning("Invalid confidence value: %s, using 0.0", confidence)
            conf = 0.0

        if description is None:
            description = _build_description(
                event_type, conf, detection_stage, object_count, analysis_result, analysis_summary
            )

        event_data: Dict[str, Any] = {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "event_type": event_type,
            "confidence": conf,
            "image_url": clean_url,
            "description": description,
            "processed_at": datetime.now(),
        }

        # Attach optional fields only when present.
        _optional = {
            "bbox": bbox,
            "object_count": object_count,
            "objects": objects,
            "frame_index": frame_index,
            "detection_stage": detection_stage,
            "zone_polygon": zone_polygon,
            "analysis_result": analysis_result,
            "analysis_summary": analysis_summary,
            "metadata": metadata,
            "lat_lng": lat_lng,
            "location": location,
            "area_code": area_code,
            "group": group,
        }
        for key, val in _optional.items():
            if val is not None:
                event_data[key] = val

        try:
            event = EventModel(**event_data)
        except Exception as e:
            logger.error("EventModel validation error: %s | data=%s", e, event_data)
            return False

        event_id = mongo_client.save_event(event)
        if event_id:
            logger.debug("Event saved: %s from %s (id=%s)", description[:50], camera_id, event_id)
            # Fire webhook notification (non-blocking)
            if _webhook_service is not None:
                try:
                    _webhook_service.notify({
                        "event_id": event_id,
                        "event_type": event_type,
                        "camera_id": camera_id,
                        "timestamp": timestamp,
                        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "confidence": conf,
                        "image_url": clean_url,
                        "description": description,
                        "object_count": object_count,
                        "lat_lng": lat_lng,
                        "location": location,
                        "area_code": area_code or "",
                        "group": group or "",
                    })
                except Exception as wh_err:
                    logger.warning("Webhook notification failed: %s", wh_err)
        else:
            logger.error("MongoDB save failed for camera_id=%s", camera_id)
        return bool(event_id)

    except Exception as e:
        logger.error("Unexpected error in handle_event_detected: %s", e, exc_info=True)
        return False


# ------------------------------------------------------------------
# Core: aggregated frame-level event
# ------------------------------------------------------------------

def handle_frame_events(
    minio_client: MinIOClient,
    mongo_client: MongoDBClient,
    image_url: str,
    camera_id: str,
    timestamp: float,
    frame_index: int,
    violations: List[Dict[str, Any]],
    zones: Optional[List[List[tuple]]] = None,
    event_type_override: Optional[str] = None,
    lat_lng: Optional[str] = None,
    location: Optional[str] = None,
    area_code: Optional[str] = None,
    group: Optional[str] = None,
) -> bool:
    """Save one aggregated event document for one frame."""
    if not violations:
        return True

    clean_url = _clean_image_url(image_url)
    current_event_type = _infer_frame_event_type(violations, event_type_override, zones)

    # Common space: pass through the first violation's analysis data.
    if current_event_type == "common_space_utilization":
        v = violations[0]
        meta = dict(v.get("metadata") or {})
        meta.update(processing_time=time.time(), source="common_space_analysis")
        return handle_event_detected(
            minio_client=minio_client,
            mongo_client=mongo_client,
            image_url=clean_url,
            camera_id=camera_id,
            timestamp=timestamp,
            event_type=current_event_type,
            confidence=v.get("confidence", 1.0),
            bbox=v.get("bbox"),
            frame_index=frame_index,
            detection_stage=v.get("detection_stage"),
            analysis_result=v.get("analysis_result"),
            analysis_summary=v.get("analysis_summary"),
            metadata=meta,
            lat_lng=lat_lng,
            location=location,
            area_code=area_code,
            group=group,
        )

    # Non-common-space: aggregate all detections into one document.
    objects = _build_aggregate_objects(violations)
    count_by_class = _build_count_by_class(violations)

    primary = max(violations, key=lambda item: float(item.get("confidence", 0.0)))
    conf = float(primary.get("confidence", 0.0))
    bbox = primary.get("bbox")
    detection_stage = primary.get("detection_stage")

    meta: Dict[str, Any] = {
        "object_classes": sorted(count_by_class.keys()),
        "object_count_by_class": count_by_class,
    }

    zone_polygon = None
    if zones and bbox and current_event_type == "parking_violation":
        meta["zone_count"] = len(zones)
        try:
            from backend.utils.bbox_utils import is_point_in_polygon

            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2
                for zone in zones:
                    if is_point_in_polygon((cx, cy), zone):
                        zone_polygon = zone
                        break
        except ImportError:
            logger.warning("bbox_utils not available, skipping zone polygon lookup")

    ok = handle_event_detected(
        minio_client=minio_client,
        mongo_client=mongo_client,
        image_url=clean_url,
        camera_id=camera_id,
        timestamp=timestamp,
        event_type=current_event_type,
        confidence=conf,
        bbox=bbox,
        frame_index=frame_index,
        zone_polygon=zone_polygon,
        detection_stage=detection_stage,
        metadata=meta,
        object_count=len(objects),
        objects=objects,
        lat_lng=lat_lng,
        location=location,
        area_code=area_code,
        group=group,
    )

    if ok:
        logger.debug("Saved aggregated event: type=%s, objects=%d", current_event_type, len(objects))
    else:
        logger.error("Failed to save aggregated event: type=%s, objects=%d", current_event_type, len(objects))
    return ok


# ------------------------------------------------------------------
# Task-specific convenience wrappers
# ------------------------------------------------------------------

def handle_smoke_flame_events(
    minio_client: MinIOClient,
    mongo_client: MongoDBClient,
    image_url: str,
    camera_id: str,
    timestamp: float,
    frame_index: int,
    detections: List[Dict[str, Any]],
    lat_lng: Optional[str] = None,
    location: Optional[str] = None,
    area_code: Optional[str] = None,
    group: Optional[str] = None,
) -> bool:
    """Smoke/flame event save entry point."""
    return handle_frame_events(
        minio_client=minio_client,
        mongo_client=mongo_client,
        image_url=image_url,
        camera_id=camera_id,
        timestamp=timestamp,
        frame_index=frame_index,
        violations=detections,
        event_type_override="smoke_flame",
        lat_lng=lat_lng,
        location=location,
        area_code=area_code,
        group=group,
    )


def handle_parking_violation_events(
    minio_client: MinIOClient,
    mongo_client: MongoDBClient,
    image_url: str,
    camera_id: str,
    timestamp: float,
    frame_index: int,
    detections: List[Dict[str, Any]],
    zones: List[List[tuple]],
    lat_lng: Optional[str] = None,
    location: Optional[str] = None,
    area_code: Optional[str] = None,
    group: Optional[str] = None,
) -> bool:
    """Parking violation event save entry point."""
    return handle_frame_events(
        minio_client=minio_client,
        mongo_client=mongo_client,
        image_url=image_url,
        camera_id=camera_id,
        timestamp=timestamp,
        frame_index=frame_index,
        violations=detections,
        zones=zones,
        event_type_override="parking_violation",
        lat_lng=lat_lng,
        location=location,
        area_code=area_code,
        group=group,
    )


def handle_common_space_events(
    minio_client: MinIOClient,
    mongo_client: MongoDBClient,
    image_url: str,
    camera_id: str,
    timestamp: float,
    frame_index: int,
    analysis_data: Dict[str, Any],
    sample_interval: Optional[int] = None,
    area_code: Optional[str] = None,
    group: Optional[str] = None,
) -> bool:
    """Common space analysis event save entry point."""
    if not analysis_data.get("analysis_result"):
        logger.error("Missing analysis_result in analysis_data")
        return False

    violation = {
        "event_type": "common_space_utilization",
        "analysis_result": analysis_data["analysis_result"],
        "analysis_summary": analysis_data.get("analysis_summary", {}),
        "detection_stage": "qwen_vl_analysis",
        "confidence": 1.0,
        "metadata": {
            "sample_interval": sample_interval,
            "processing_time": time.time(),
            "source": "common_space_analysis",
        },
    }

    return handle_frame_events(
        minio_client=minio_client,
        mongo_client=mongo_client,
        image_url=image_url,
        camera_id=camera_id,
        timestamp=timestamp,
        frame_index=frame_index,
        violations=[violation],
        event_type_override="common_space_utilization",
        area_code=area_code,
        group=group,
    )
