# backend/utils/visualization.py
"""
Visualization Utilities
  1. 批量绘制检测框
  2. 半透明填充 + 边缘描边禁停区
  3. 区分违规目标（红色）和非违规目标（绿色）
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

def draw_detection_box(
        image: np.ndarray,
        class_name: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int] = (0, 0, 255),
        is_violation: bool = False
) -> np.ndarray:
    """
    在图像上绘制单个检测框 + 类别/置信度文字
    Draw single detection box with label on image.

    Parameters:
        image: 输入图像 (H, W, 3)
        class_name: 类别名称
        confidence: 置信度 0-1
        bbox: (x1, y1, x2, y2)
        color: BGR 颜色
        is_violation: 是否为违规目标

    Returns:
        np.ndarray: 绘制后的图像（原地修改）
    """
    try:
        x1, y1, x2, y2 = bbox

        # 确保 confidence 是数值类型
        if isinstance(confidence, str):
            if confidence == 'confidence':
                conf_value = 0.5
            else:
                try:
                    conf_value = float(confidence)
                except ValueError:
                    conf_value = 0.5
        else:
            conf_value = float(confidence)

        # 创建标签 - 违规目标添加 [VIOLATION] 标识
        violation_tag = "[VIOLATION] " if is_violation else ""
        label = f"{violation_tag}{class_name} {conf_value:.2f}"

        # 画框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 画标签背景
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), color, -1)

        # 画文字
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return image
    except Exception as e:
        logger.error(f"❌ Error in draw_detection_box: {e}")
        return image


def draw_no_parking_zone(
        image: np.ndarray,
        polygon: List[Tuple[int, int]],
        color: Tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.3
) -> np.ndarray:
    """
    半透明填充 + 边缘描边禁停区
    Semi-transparent fill + edge line for no-parking zone.

    Parameters:
        image: 输入图像
        polygon: [(x, y), ...] 多边形顶点
        color: BGR 填充颜色，默认蓝色
        alpha: 透明度 0-1

    Returns:
        np.ndarray: 叠加后的图像
    """
    try:
        overlay = image.copy()
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))

        # 半透明填充
        cv2.fillPoly(overlay, [pts], color)

        # 边缘描边
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)

        # 叠加透明层
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    except Exception as e:
        logger.error(f"❌ Error in draw_no_parking_zone: {e}")
        return image


def render_official_frame(
        image: np.ndarray,
        all_detections: List[Dict[str, Any]],  # 🔴 改为所有检测目标
        violations: List[Dict[str, Any]],  # 🔴 违规目标
        zones: Optional[List[List[Tuple[int, int]]]] = None
) -> np.ndarray:
    """
    1. 画所有禁停区
    2. 画所有检测框（绿色 - 正常，红色 - 违规）
    3. 显示统计信息

    Parameters:
        image: 原始帧
        all_detections: 所有检测到的目标
        violations: 在禁停区内的违规目标
        zones: 所有禁停区多边形

    Returns:
        np.ndarray: 完全渲染后的图像
    """
    try:
        img = image.copy()

        logger.debug(f"🖌️ Rendering: {len(all_detections)} total detections, {len(violations)} violations")

        # 1. 画所有禁停区
        if zones:
            for zone in zones:
                img = draw_no_parking_zone(img, zone, color=(0, 0, 255), alpha=0.25)
                logger.debug(f"Drawing zone绘制违停区域: {zone}")

        # 2. 创建违规目标集合用于快速查找
        violation_set = set()
        for violation in violations:
            # 通过bbox来标识唯一目标（因为可能有多个相同位置的目标）
            bbox_key = tuple(violation.get('bbox', ()))
            violation_set.add(bbox_key)

        # 3. 画所有检测框
        normal_count = 0
        violation_count = 0

        for detection in all_detections:
            try:
                class_name = detection.get('class_name', 'unknown')
                confidence = detection.get('confidence', 0.0)
                bbox = detection.get('bbox', (0, 0, 0, 0))

                # 验证 bbox 是否有效
                if len(bbox) != 4:
                    continue

                # 检查是否为违规目标
                #bbox_key = tuple(bbox)
                #is_violation = bbox_key in violation_set

                is_violation = detection in violations

                color = (0, 0, 255) if is_violation else (0, 255, 0)

                # 绘制检测框
                img = draw_detection_box(
                    image=img,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox,
                    color=color,
                    is_violation=is_violation
                )

                # 统计
                if is_violation:
                    violation_count += 1
                else:
                    normal_count += 1

            except Exception as e:
                logger.error(f"❌ Error rendering detection: {e}")
                continue

        # 4. 帧级文字提示
        cv2.putText(img, f"Total: {len(all_detections)}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(img, f"Normal: {normal_count}", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(img, f"Violations: {violation_count}", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        logger.debug(f"✅ Frame rendered: {normal_count} normal, {violation_count} violations")
        return img

    except Exception as e:
        logger.error(f"❌ Error in render_official_frame: {e}")
        return image


def render_debug_frame(
        image: np.ndarray,
        all_detections: List[Dict[str, Any]],
        violations: List[Dict[str, Any]],
        zones: Optional[List[List[Tuple[int, int]]]] = None
) -> np.ndarray:
    """
    调试用渲染：显示所有检测和禁停区
    Debug rendering: show all detections and zones
    """
    try:
        img = image.copy()

        # 1. 画所有禁停区
        if zones:
            for zone in zones:
                img = draw_no_parking_zone(img, zone, color=(0, 0, 255), alpha=0.25)

        # 2. 画所有非违规检测框（绿色）
        for detection in all_detections:
            try:
                class_name = detection.get('class_name', 'unknown')
                confidence = detection.get('confidence', 0.0)
                bbox = detection.get('bbox', (0, 0, 0, 0))

                if len(bbox) == 4:
                    img = draw_detection_box(
                        img, class_name, confidence, bbox, color=(0, 255, 0)  # 绿色
                    )
            except Exception as e:
                logger.error(f"❌ Error rendering debug detection: {e}")
                continue

        # 3. 画违规框（红色）
        for violation in violations:
            try:
                class_name = violation.get('class_name', 'unknown')
                confidence = violation.get('confidence', 0.0)
                bbox = violation.get('bbox', (0, 0, 0, 0))

                if len(bbox) == 4:
                    img = draw_detection_box(
                        img, class_name, confidence, bbox, color=(0, 0, 255), is_violation=True  # 红色 + 违规标识
                    )
            except Exception as e:
                logger.error(f"❌ Error rendering debug violation: {e}")
                continue

        # 4. 统计信息
        cv2.putText(img, f"Total Detections: {len(all_detections)}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(img, f"Violations: {len(violations)}", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        return img

    except Exception as e:
        logger.error(f"❌ Error in render_debug_frame: {e}")
        return image