# backend/services/event_generator.py
"""
Event Generator —— 仅保存元数据
Annotated image 由上游 DetectionProcessor 提前上传；
本模块负责：
  1. 单目标事件写库 (handle_event_detected)
  2. 整帧批量事件写库 (handle_frame_events)
  支持：电子围栏检测 + 烟火检测 + 公共空间分析
"""

from typing import List, Dict, Any, Optional, Tuple
from storage.minio_client import MinIOClient
from storage.mongodb_client import MongoDBClient
from backend.models.event import EventModel
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


def _infer_frame_event_type(
        violations: List[Dict[str, Any]],
        event_type_override: Optional[str] = None,
        zones: Optional[List[List[tuple]]] = None
) -> str:
    """Infer a single frame-level event type from aggregated detections."""
    if event_type_override:
        return event_type_override

    raw_types: List[str] = []
    for item in violations:
        item_type = item.get("event_type") or item.get("class_name")
        if item_type:
            raw_types.append(str(item_type).lower())

    unique_types = set(raw_types)

    if zones:
        return "parking_violation"

    if unique_types and unique_types.issubset({"smoke", "fire", "flame", "smoke_flame"}):
        return "smoke_flame"

    if len(unique_types) == 1:
        return unique_types.pop()

    return "unknown"


def _build_aggregate_objects(violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep compact per-object details inside one frame-level event."""
    objects: List[Dict[str, Any]] = []

    for item in violations:
        obj: Dict[str, Any] = {}

        if item.get("class_name") is not None:
            obj["class_name"] = item["class_name"]
        if item.get("event_type") is not None:
            obj["event_type"] = item["event_type"]
        if item.get("confidence") is not None:
            try:
                obj["confidence"] = float(item["confidence"])
            except (ValueError, TypeError):
                obj["confidence"] = 0.0
        if item.get("bbox") is not None:
            obj["bbox"] = item["bbox"]
        if item.get("detection_stage") is not None:
            obj["detection_stage"] = item["detection_stage"]

        objects.append(obj)

    return objects


def _build_count_by_class(violations: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in violations:
        key = item.get("class_name") or item.get("event_type") or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


# ------------------------------------------------------------------
# LEGACY: old per-object event saving implementation
# Kept as a non-executable reference while the aggregated frame-event
# implementation below remains the only active implementation.
# ------------------------------------------------------------------
LEGACY_SINGLE_TARGET_IMPLEMENTATION = r'''
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
        # ✅ 新增：公共空间分析相关字段
        analysis_result: Optional[str] = None,
        analysis_summary: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        object_count: Optional[int] = None,
        objects: Optional[List[Dict[str, Any]]] = None
) -> bool:
    """
    将单个事件的元数据写入 MongoDB。
    Persist a SINGLE event metadata into MongoDB.
    支持：电子围栏检测 + 烟火检测 + 公共空间分析

    Parameters:
        minio_client: MinIO 客户端
        mongo_client: MongoDB 客户端实例
        image_url: 预上传的整帧渲染图 URL
        camera_id: 摄像头/视频源标识
        timestamp: 事件 Unix 时间戳（秒）
        event_type: 事件类型，如 'car', 'smoke', 'fire', 'common_space_utilization'
        confidence: 置信度 [0.0, 1.0]（默认0.0）
        bbox: 边界框坐标 (x1, y1, x2, y2)（可选，公共空间分析可能没有）
        frame_index: 帧号（可选）
        description: 描述文本（可选，如果为空则自动生成）
        zone_polygon: 禁停区多边形（仅电子围栏检测使用）
        detection_stage: 检测阶段（如 "yolo_initial", "qwen_verified", "qwen_vl_analysis"）
        analysis_result: AI生成的完整分析结果（公共空间分析使用）
        analysis_summary: 结构化分析摘要（公共空间分析使用）
        metadata: 额外元数据

    Returns:
        bool: 成功返回 True，否则 False
    """
    try:
        # 1. 清理URL（移除预签名参数）
        clean_image_url = image_url.split('?')[0] if '?X-Amz-' in image_url else image_url

        # 2. 校验图片链接
        if not clean_image_url:
            logger.warning("⚠️ Image URL is empty — event saved without visual evidence.")

        # 3. 确保confidence是float类型
        try:
            confidence_float = float(confidence)
        except (ValueError, TypeError):
            logger.warning(f"⚠️ Invalid confidence value: {confidence}, using 0.0")
            confidence_float = 0.0

        # 4. 根据事件类型生成描述（如果未提供）
        if description is None:
            if event_type == "common_space_utilization":
                # ✅ 改进：为公共空间分析生成更有意义的描述
                if analysis_summary:
                    # 从结构化摘要生成详细描述
                    people_count = analysis_summary.get("estimated_people_count", 0)
                    occupancy = analysis_summary.get("space_occupancy", "unknown")
                    activities = analysis_summary.get("activity_types", [])
                    safety = analysis_summary.get("safety_concerns", False)

                    # 构建详细描述
                    description_parts = [f"Public space analysis: {people_count} people"]

                    if occupancy != "unknown":
                        description_parts.append(f"{occupancy} occupancy")

                    if activities and len(activities) > 0 and activities != ["unknown"]:
                        activity_text = ", ".join(activities[:3])  # 最多显示3个活动类型
                        if len(activities) > 3:
                            activity_text += f" and {len(activities) - 3} more"
                        description_parts.append(f"activities: {activity_text}")

                    if safety:
                        description_parts.append("safety concerns identified")

                    description = ", ".join(description_parts)

                elif analysis_result:
                    # 如果没有summary，尝试从完整回答中提取关键信息
                    # 找到第一个句号或逗号作为截断点
                    period_idx = analysis_result.find('.')
                    comma_idx = analysis_result.find(',')

                    if period_idx > 0 and period_idx < 100:
                        # 使用第一个句子作为描述
                        description = analysis_result[:period_idx + 1]
                    elif comma_idx > 0 and comma_idx < 100:
                        # 使用第一个逗号前的部分
                        description = analysis_result[:comma_idx] + "..."
                    else:
                        # 截断前120个字符
                        description = analysis_result[:120] + "..." if len(analysis_result) > 120 else analysis_result
                else:
                    description = "Public space utilization analyzed"

            elif event_type.lower() in ['smoke', 'fire', 'flame']:
                # 烟火检测描述
                stage_info = f" ({detection_stage})" if detection_stage else ""
                description = f"{event_type.title()} detected{stage_info} (conf={confidence_float:.2f})"
            else:
                # 电子围栏检测描述
                description = f"{event_type.title()} in no-parking zone (conf={confidence_float:.2f})"
        # 5. 构造事件文档 - 只包含实际存在的字段
        event_data = {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "event_type": event_type,
            "confidence": confidence_float,
            "image_url": clean_image_url,
            "description": description,
            "processed_at": datetime.now()
        }

        # 可选字段：边界框（电子围栏和烟火检测有，公共空间分析没有）
        if bbox is not None:
            event_data["bbox"] = bbox

        # 可选字段：帧索引
        if frame_index is not None:
            event_data["frame_index"] = frame_index

        # 可选字段：检测阶段
        if detection_stage is not None:
            event_data["detection_stage"] = detection_stage

        # 可选字段：禁停区多边形（仅电子围栏检测）
        if zone_polygon is not None:
            event_data["zone_polygon"] = zone_polygon

        # ✅ 新增：公共空间分析字段
        if analysis_result is not None:
            event_data["analysis_result"] = analysis_result

        if analysis_summary is not None:
            event_data["analysis_summary"] = analysis_summary

        if metadata is not None:
            event_data["metadata"] = metadata

        # 6. 创建并验证事件模型
        try:
            event = EventModel(**event_data)
        except Exception as e:
            logger.error(f"❌ EventModel validation error: {e}")
            logger.error(f"   Event data: {event_data}")
            return False

        # 7. 落库
        success: bool = mongo_client.save_event(event)
        if success:
            logger.debug(f"✅ Event saved: {description[:50]}... from {camera_id}")

            # 如果是公共空间分析，记录更多信息
            if event_type == "common_space_utilization" and analysis_result:
                preview = analysis_result[:100] + "..." if len(analysis_result) > 100 else analysis_result
                logger.debug(f"   🤖 AI Analysis: {preview}")
        else:
            logger.error("❌ MongoDB save failed.")

        return success

    except Exception as e:
        logger.error(f"[Event Generator] Unexpected error: {e}", exc_info=True)
        return False
'''


# ------------------------------------------------------------------
# LEGACY: old per-object batch save implementation
# Kept as a non-executable reference while the aggregated frame-event
# implementation below remains the only active implementation.
# ------------------------------------------------------------------
LEGACY_BATCH_EVENT_IMPLEMENTATION = r'''
def handle_frame_events(
        minio_client: MinIOClient,
        mongo_client: MongoDBClient,
        image_url: str,
        camera_id: str,
        timestamp: float,
        frame_index: int,
        violations: List[Dict[str, Any]],
        zones: Optional[List[List[tuple]]] = None,
        event_type_override: Optional[str] = None
) -> bool:
    """
    一次性将整帧所有事件写入 MongoDB。
    Batch save ALL events within one frame.
    支持：电子围栏违规 + 烟火检测 + 公共空间分析

    Parameters:
        minio_client: MinIO 客户端
        mongo_client: MongoDB 客户端实例
        image_url: 预上传的整帧渲染图 URL
        camera_id: 源 ID
        timestamp: 帧时间戳
        frame_index: 帧号
        violations: 事件列表
        zones: 禁停区列表（可选，仅电子围栏检测使用）
        event_type_override: 覆盖事件类型（可选）

    Returns:
        bool: 全部成功返回 True，任一失败返回 False
    """
    if not violations:
        logger.debug("🟢 No events to save")
        return True

    # 清理URL
    clean_image_url = image_url.split('?')[0] if '?X-Amz-' in image_url else image_url
    logger.debug(f"🔄 Saving {len(violations)} events with image: {clean_image_url}")

    all_ok = True
    saved_count = 0

    for i, violation in enumerate(violations):
        try:
            # 判断事件类型
            violation_type = violation.get("event_type", "")

            # 使用覆盖类型（如果有），否则使用violation中的类型
            current_event_type = event_type_override or violation_type

            # 如果没有指定类型，尝试从class_name推断
            if not current_event_type and "class_name" in violation:
                current_event_type = violation["class_name"]

            # 默认事件类型
            if not current_event_type:
                current_event_type = "unknown"
                logger.warning(f"⚠️ No event type specified for violation {i}, using 'unknown'")

            logger.debug(f"📝 Processing event {i}: {current_event_type}")

            # 准备公共空间分析的特殊字段
            analysis_result = None
            analysis_summary = None
            metadata = None

            if current_event_type == "common_space_utilization":
                analysis_result = violation.get("analysis_result")
                analysis_summary = violation.get("analysis_summary")
                metadata = violation.get("metadata", {})

                # 添加公共空间分析的元数据
                if not metadata:
                    metadata = {}
                metadata["processing_time"] = time.time()
                metadata["source"] = "common_space_analysis"

            # 准备其他字段
            class_name = violation.get("class_name", current_event_type)
            confidence = violation.get("confidence", 1.0 if current_event_type == "common_space_utilization" else 0.0)
            bbox = violation.get("bbox")  # 可能为None
            detection_stage = violation.get("detection_stage")

            # 对于电子围栏检测，查找对应的禁停区
            zone_polygon = None
            if zones and bbox and current_event_type == "parking_violation":
                # 尝试导入几何工具函数
                try:
                    from backend.utils.bbox_utils import is_point_in_polygon

                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        center_x = (bbox[0] + bbox[2]) // 2
                        center_y = (bbox[1] + bbox[3]) // 2

                        for idx, zone in enumerate(zones):
                            if is_point_in_polygon((center_x, center_y), zone):
                                zone_polygon = zone
                                break
                except ImportError:
                    logger.warning("⚠️ bbox_utils module not available, skipping zone detection")

            # 保存事件
            ok = handle_event_detected(
                minio_client=minio_client,
                mongo_client=mongo_client,
                image_url=clean_image_url,
                camera_id=camera_id,
                timestamp=timestamp,
                event_type=current_event_type,
                confidence=confidence,
                bbox=bbox,  # 可能是None
                frame_index=frame_index,
                description=None,  # 让handle_event_detected自动生成
                zone_polygon=zone_polygon,
                detection_stage=detection_stage,
                analysis_result=analysis_result,
                analysis_summary=analysis_summary,
                metadata=metadata
            )

            if ok:
                saved_count += 1
            else:
                all_ok = False
                logger.error(f"❌ Failed to save event {i}: {current_event_type}")

        except Exception as e:
            logger.error(f"❌ Error processing event {i}: {e}", exc_info=True)
            all_ok = False

    if all_ok:
        logger.debug(f"✅ Successfully saved all {saved_count} events")
    else:
        logger.warning(f"⚠️ Saved {saved_count}/{len(violations)} events (some failed)")

    return all_ok
'''


# ------------------------------------------------------------------
# ③ 烟火检测专用落库 | Smoke/Flame specific save
# ------------------------------------------------------------------
def handle_smoke_flame_events(
        minio_client: MinIOClient,
        mongo_client: MongoDBClient,
        image_url: str,
        camera_id: str,
        timestamp: float,
        frame_index: int,
        detections: List[Dict[str, Any]]
) -> bool:
    """
    烟火检测专用事件保存
    Smoke/Flame specific event saving

    Parameters:
        minio_client: MinIO 客户端
        mongo_client: MongoDB 客户端实例
        image_url: 预上传的整帧渲染图 URL
        camera_id: 源 ID
        timestamp: 帧时间戳
        frame_index: 帧号
        detections: 烟火检测结果列表

    Returns:
        bool: 全部成功返回 True，任一失败返回 False
    """
    return handle_frame_events(
        minio_client=minio_client,
        mongo_client=mongo_client,
        image_url=image_url,
        camera_id=camera_id,
        timestamp=timestamp,
        frame_index=frame_index,
        violations=detections,
        event_type_override="smoke_flame"
    )


# ------------------------------------------------------------------
# ④ 公共空间分析专用落库 | Common space analysis specific save
# ------------------------------------------------------------------
def handle_common_space_events(
        minio_client: MinIOClient,
        mongo_client: MongoDBClient,
        image_url: str,
        camera_id: str,
        timestamp: float,
        frame_index: int,
        analysis_data: Dict[str, Any],
        sample_interval: Optional[int] = None
) -> bool:
    """
    公共空间分析专用事件保存
    Common space analysis specific event saving

    Parameters:
        minio_client: MinIO 客户端
        mongo_client: MongoDB 客户端实例
        image_url: 预上传的整帧渲染图 URL
        camera_id: 源 ID
        timestamp: 帧时间戳
        frame_index: 帧号
        analysis_data: 分析数据，必须包含 analysis_result 和 analysis_summary
        sample_interval: 采样间隔（秒）

    Returns:
        bool: 成功返回 True，否则 False
    """
    try:
        # 验证分析数据
        if not analysis_data.get("analysis_result"):
            logger.error("❌ Missing analysis_result in analysis_data")
            return False

        if not analysis_data.get("analysis_summary"):
            logger.warning("⚠️ Missing analysis_summary in analysis_data")

        # 构建violation数据（注意：公共空间分析没有bbox）
        violation = {
            "event_type": "common_space_utilization",
            "analysis_result": analysis_data["analysis_result"],
            "analysis_summary": analysis_data.get("analysis_summary", {}),
            "detection_stage": "qwen_vl_analysis",
            "confidence": 1.0,
            "metadata": {
                "sample_interval": sample_interval,
                "processing_time": time.time(),
                "source": "common_space_analysis"
            }
        }

        # 保存事件
        return handle_frame_events(
            minio_client=minio_client,
            mongo_client=mongo_client,
            image_url=image_url,
            camera_id=camera_id,
            timestamp=timestamp,
            frame_index=frame_index,
            violations=[violation],
            event_type_override="common_space_utilization"
        )

    except Exception as e:
        logger.error(f"❌ Error in handle_common_space_events: {e}", exc_info=True)
        return False


# ------------------------------------------------------------------
# ⑤ 电子围栏检测专用落库 | Parking violation specific save
# ------------------------------------------------------------------
def handle_parking_violation_events(
        minio_client: MinIOClient,
        mongo_client: MongoDBClient,
        image_url: str,
        camera_id: str,
        timestamp: float,
        frame_index: int,
        detections: List[Dict[str, Any]],
        zones: List[List[tuple]]
) -> bool:
    """
    电子围栏检测专用事件保存
    Parking violation specific event saving

    Parameters:
        minio_client: MinIO 客户端
        mongo_client: MongoDB 客户端实例
        image_url: 预上传的整帧渲染图 URL
        camera_id: 源 ID
        timestamp: 帧时间戳
        frame_index: 帧号
        detections: 检测结果列表
        zones: 禁停区列表

    Returns:
        bool: 全部成功返回 True，任一失败返回 False
    """
    return handle_frame_events(
        minio_client=minio_client,
        mongo_client=mongo_client,
        image_url=image_url,
        camera_id=camera_id,
        timestamp=timestamp,
        frame_index=frame_index,
        violations=detections,
        zones=zones,
        event_type_override="parking_violation"
    )


# ------------------------------------------------------------------
# ⑥ 当前生效实现 | Active aggregated frame-event implementation
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
        objects: Optional[List[Dict[str, Any]]] = None
) -> bool:
    """Persist a single event metadata document into MongoDB."""
    _ = minio_client

    try:
        clean_image_url = image_url.split('?')[0] if '?X-Amz-' in image_url else image_url

        if not clean_image_url:
            logger.warning("Image URL is empty; event will be saved without visual evidence.")

        try:
            confidence_float = float(confidence)
        except (ValueError, TypeError):
            logger.warning(f"Invalid confidence value: {confidence}, using 0.0")
            confidence_float = 0.0

        if description is None:
            if event_type == "common_space_utilization":
                if analysis_summary:
                    people_count = analysis_summary.get("estimated_people_count", 0)
                    occupancy = analysis_summary.get("space_occupancy", "unknown")
                    activities = analysis_summary.get("activity_types", [])
                    safety = analysis_summary.get("safety_concerns", False)

                    description_parts = [f"Public space analysis: {people_count} people"]
                    if occupancy != "unknown":
                        description_parts.append(f"{occupancy} occupancy")
                    if activities and activities != ["unknown"]:
                        activity_text = ", ".join(activities[:3])
                        if len(activities) > 3:
                            activity_text += f" and {len(activities) - 3} more"
                        description_parts.append(f"activities: {activity_text}")
                    if safety:
                        description_parts.append("safety concerns identified")
                    description = ", ".join(description_parts)
                elif analysis_result:
                    period_idx = analysis_result.find(".")
                    comma_idx = analysis_result.find(",")
                    if period_idx > 0 and period_idx < 100:
                        description = analysis_result[:period_idx + 1]
                    elif comma_idx > 0 and comma_idx < 100:
                        description = analysis_result[:comma_idx] + "..."
                    else:
                        description = analysis_result[:120] + "..." if len(analysis_result) > 120 else analysis_result
                else:
                    description = "Public space utilization analyzed"
            elif event_type.lower() in ["smoke", "fire", "flame", "smoke_flame"]:
                stage_info = f" ({detection_stage})" if detection_stage else ""
                if object_count and object_count > 1:
                    description = (
                        f"Smoke/flame detected in {object_count} regions"
                        f"{stage_info} (max_conf={confidence_float:.2f})"
                    )
                else:
                    label = "Smoke/flame" if event_type.lower() == "smoke_flame" else event_type.title()
                    description = f"{label} detected{stage_info} (conf={confidence_float:.2f})"
            elif event_type == "parking_violation":
                if object_count and object_count > 1:
                    description = (
                        f"{object_count} vehicles in no-parking zone "
                        f"(max_conf={confidence_float:.2f})"
                    )
                else:
                    description = f"Vehicle in no-parking zone (conf={confidence_float:.2f})"
            else:
                description = f"{event_type.title()} detected (conf={confidence_float:.2f})"

        event_data = {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "event_type": event_type,
            "confidence": confidence_float,
            "image_url": clean_image_url,
            "description": description,
            "processed_at": datetime.now()
        }

        if bbox is not None:
            event_data["bbox"] = bbox
        if object_count is not None:
            event_data["object_count"] = object_count
        if objects is not None:
            event_data["objects"] = objects
        if frame_index is not None:
            event_data["frame_index"] = frame_index
        if detection_stage is not None:
            event_data["detection_stage"] = detection_stage
        if zone_polygon is not None:
            event_data["zone_polygon"] = zone_polygon
        if analysis_result is not None:
            event_data["analysis_result"] = analysis_result
        if analysis_summary is not None:
            event_data["analysis_summary"] = analysis_summary
        if metadata is not None:
            event_data["metadata"] = metadata

        try:
            event = EventModel(**event_data)
        except Exception as e:
            logger.error(f"EventModel validation error: {e}")
            logger.error(f"   Event data: {event_data}")
            return False

        success: bool = mongo_client.save_event(event)
        if success:
            logger.debug(f"Event saved: {description[:50]}... from {camera_id}")
            if event_type == "common_space_utilization" and analysis_result:
                preview = analysis_result[:100] + "..." if len(analysis_result) > 100 else analysis_result
                logger.debug(f"AI Analysis: {preview}")
        else:
            logger.error("MongoDB save failed.")

        return success
    except Exception as e:
        logger.error(f"[Event Generator] Unexpected error: {e}", exc_info=True)
        return False


def handle_frame_events(
        minio_client: MinIOClient,
        mongo_client: MongoDBClient,
        image_url: str,
        camera_id: str,
        timestamp: float,
        frame_index: int,
        violations: List[Dict[str, Any]],
        zones: Optional[List[List[tuple]]] = None,
        event_type_override: Optional[str] = None
) -> bool:
    """Save one aggregated event for one frame."""
    if not violations:
        logger.debug("No events to save")
        return True

    clean_image_url = image_url.split('?')[0] if '?X-Amz-' in image_url else image_url
    current_event_type = _infer_frame_event_type(
        violations=violations,
        event_type_override=event_type_override,
        zones=zones
    )

    # LEGACY:
    # The old implementation looped over each violation and wrote one document
    # per object. This override writes one aggregated event per frame per task.

    if current_event_type == "common_space_utilization":
        violation = violations[0]
        metadata = violation.get("metadata", {}) or {}
        metadata["processing_time"] = time.time()
        metadata["source"] = "common_space_analysis"

        return handle_event_detected(
            minio_client=minio_client,
            mongo_client=mongo_client,
            image_url=clean_image_url,
            camera_id=camera_id,
            timestamp=timestamp,
            event_type=current_event_type,
            confidence=violation.get("confidence", 1.0),
            bbox=violation.get("bbox"),
            frame_index=frame_index,
            description=None,
            zone_polygon=None,
            detection_stage=violation.get("detection_stage"),
            analysis_result=violation.get("analysis_result"),
            analysis_summary=violation.get("analysis_summary"),
            metadata=metadata
        )

    objects = _build_aggregate_objects(violations)
    object_count = len(objects)
    count_by_class = _build_count_by_class(violations)

    primary_violation = max(
        violations,
        key=lambda item: float(item.get("confidence", 0.0))
    )
    confidence = float(primary_violation.get("confidence", 0.0))
    bbox = primary_violation.get("bbox")
    detection_stage = primary_violation.get("detection_stage")

    metadata: Dict[str, Any] = {
        "object_classes": sorted(count_by_class.keys()),
        "object_count_by_class": count_by_class,
    }

    zone_polygon = None
    if zones and bbox and current_event_type == "parking_violation":
        metadata["zone_count"] = len(zones)
        try:
            from backend.utils.bbox_utils import is_point_in_polygon

            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2

                for zone in zones:
                    if is_point_in_polygon((center_x, center_y), zone):
                        zone_polygon = zone
                        break
        except ImportError:
            logger.warning("bbox_utils module not available, skipping zone detection")

    ok = handle_event_detected(
        minio_client=minio_client,
        mongo_client=mongo_client,
        image_url=clean_image_url,
        camera_id=camera_id,
        timestamp=timestamp,
        event_type=current_event_type,
        confidence=confidence,
        bbox=bbox,
        frame_index=frame_index,
        description=None,
        zone_polygon=zone_polygon,
        detection_stage=detection_stage,
        metadata=metadata,
        object_count=object_count,
        objects=objects
    )

    if ok:
        logger.debug(
            f"Saved aggregated frame event: type={current_event_type}, object_count={object_count}"
        )
    else:
        logger.error(
            f"Failed to save aggregated frame event: type={current_event_type}, object_count={object_count}"
        )

    return ok


def get_event_type_description(event_type: str) -> str:
    """
    获取事件类型的描述
    Get description for event type

    Returns:
        str: 事件类型描述
    """
    descriptions = {
        "parking_violation": "Parking violation detected",
        "smoke_flame": "Smoke or flame detected",
        "common_space_utilization": "Public space utilization analysis",
        "car": "Vehicle detected",
        "person": "Person detected",
        "fire": "Fire detected",
        "smoke": "Smoke detected"
    }

    return descriptions.get(event_type, f"{event_type} detected")


def create_default_analysis_summary(analysis_result: str) -> Dict[str, Any]:
    """
    从分析结果创建默认的结构化摘要
    Create default structured summary from analysis result

    Returns:
        Dict: 结构化摘要
    """
    analysis_lower = analysis_result.lower()

    # 简单的关键词提取
    estimated_people = 0
    if "people" in analysis_lower or "person" in analysis_lower:
        # 尝试提取数字
        import re
        numbers = re.findall(r'\b\d+\b', analysis_lower)
        if numbers:
            estimated_people = int(numbers[0])

    # 估计空间占用率
    occupancy = "unknown"
    if any(word in analysis_lower for word in ["crowded", "busy", "many", "high"]):
        occupancy = "high"
    elif any(word in analysis_lower for word in ["moderate", "average", "normal"]):
        occupancy = "medium"
    elif any(word in analysis_lower for word in ["empty", "few", "low", "quiet"]):
        occupancy = "low"

    # 检查安全隐患
    safety_concerns = any(word in analysis_lower for word in
                          ["danger", "hazard", "unsafe", "emergency", "risk", "problem"])

    return {
        "estimated_people_count": estimated_people,
        "space_occupancy": occupancy,
        "activity_types": ["analyzed"],
        "safety_concerns": safety_concerns,
        "keywords": []
    }


# ------------------------------------------------------------------
# ⑦ 简化的事件保存函数 | Simplified event save function
# ------------------------------------------------------------------
def save_event_simple(
        mongo_client: MongoDBClient,
        camera_id: str,
        timestamp: float,
        event_type: str,
        image_url: str,
        description: Optional[str] = None,
        **kwargs
) -> bool:
    """
    简化的事件保存函数
    Simplified event save function

    Returns:
        bool: 成功返回 True，否则 False
    """
    try:
        event_data = {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "event_type": event_type,
            "image_url": image_url,
            "description": description or f"{event_type} event",
            "processed_at": datetime.now()
        }

        # 添加额外参数
        for key, value in kwargs.items():
            if value is not None:
                event_data[key] = value

        event = EventModel(**event_data)
        return mongo_client.save_event(event)

    except Exception as e:
        logger.error(f"❌ Error in save_event_simple: {e}")
        return False
