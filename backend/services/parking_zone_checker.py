# backend/services/parking_zone_checker.py
"""
No-Parking Zone Checker —— 官方示例适配版
支持：1. 整帧多框一次性过滤  2. 单框中心点判断
配置依旧读取 JSON，兼容通配符。
"""

import cv2
import numpy as np
import json
import os
import fnmatch
import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

CONFIG_FILE = "no_parking_config.json"

# 默认落库配置 | default fallback zones
DEFAULT_ZONES: Dict[str, List[List[Tuple[int, int]]]] = {
    "camera_parking_lot": [
        [(100, 150), (300, 100), (400, 200), (200, 300)]   # tuple
    ]
}


# ------------------------------------------------------------------
# 加载配置 | Load zones from JSON
# ------------------------------------------------------------------
def load_zones_from_file(config_path: str = CONFIG_FILE) -> Dict[str, List[List[Tuple[int, int]]]]:
    """
    总是返回 Dict[str, List[List[Tuple]]]，不会 None。
    Always return Dict[str, List[List[Tuple]]], never None.
    """
    if not os.path.exists(config_path):
        logger.info(f"Config file '{config_path}' not found. Creating with default zones.")
        try:
            os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
            serializable = {k: [[list(pt) for pt in poly] for poly in zones]
                            for k, zones in DEFAULT_ZONES.items()}
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
            logger.info(f"Default config saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to create config file: {e}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        zones = {}
        for src_id, zone_list in data.items():
            clean_zones = []
            for poly in zone_list:
                # 只接受 [[x,y], ...] 格式
                if isinstance(poly, list) and len(poly) >= 3:
                    pts = [tuple(p) for p in poly if isinstance(p, list) and len(p) == 2]
                    if len(pts) >= 3:
                        clean_zones.append(pts)
            zones[src_id] = clean_zones
        return zones

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {config_path}: {e}")
        return {}


# ------------------------------------------------------------------
# 核心 Checker 类
# ------------------------------------------------------------------
class NoParkingZoneChecker:
    def __init__(self, config_path: str = CONFIG_FILE):
        self.zones = load_zones_from_file(config_path)

    # ----------------------------------------------------------
    # 整帧过滤 —— 官方示例用法
    # ----------------------------------------------------------
    def filter_violations_in_zones(
        self,
        violations: List[Dict[str, Any]],
        source_id: str
    ) -> List[Dict[str, Any]]:
        """
        一次性过滤「整帧违规列表」，返回「中心点在禁停区」的子列表。
        Batch-filter whole-frame violations; return sub-list whose center is inside any zone.

        Parameters:
            violations: 整帧检测列表，每项含 bbox 等
            source_id: 摄像头/视频源标识

        Returns:
            List[Dict]: 在禁停区内的违规项
        """
        logger.info(f"🔍 ENTER: first 2 items = {violations[:2]}")
        zones = self.get_zones_for_source(source_id)
        if not zones:
            logger.debug(f"No zone defined for {source_id}")
            return []

        valid = []
        for v in violations:
            # 强制类型检查
            assert isinstance(v, dict), f"Expected dict, got {type(v)}: {v}"
            x1, y1, x2, y2 = v["bbox"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            # 只需判断任意一个区域
            for poly in zones:
                if cv2.pointPolygonTest(np.array(poly, dtype=np.float32), (cx, cy), False) >= 0:
                    valid.append(v)
                    break
        logger.info(f"Zone check: {len(valid)}/{len(violations)} cars inside no-parking zone.")
        logger.info(f"🔍 EXIT:  {len(valid)}/{len(violations)} items, first 2 = {valid[:2]}")
        return valid

    # ----------------------------------------------------------
    # 单框判断 —— 保留兼容
    # ----------------------------------------------------------
    def is_center_in_zone(
        self,
        bbox: Tuple[int, int, int, int],
        source_id: str,
        frame_shape: Tuple[int, int] = None
    ) -> bool:
        """
        单框中心点是否在禁停区。
        Single-box center check (kept for compatibility).
        """
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        zones = self.get_zones_for_source(source_id)

        for poly in zones:
            if cv2.pointPolygonTest(np.array(poly, dtype=np.float32), (cx, cy), False) >= 0:
                return True
        return False

    # ----------------------------------------------------------
    # 通用：获取区域列表
    # ----------------------------------------------------------
    def get_zones_for_source(self, source_id: str) -> List[List[Tuple[int, int]]]:
        """增强匹配：尝试原始ID、basename、无扩展名三种形式"""
        # 标准化：移除路径，保留扩展名（与GUI保存逻辑对齐）
        clean_key = os.path.basename(source_id)
        logger.info(f"🔍 DEBUG: Requested source_id = '{source_id}'")

        # 优先匹配带扩展名的键（GUI保存格式）
        if clean_key in self.zones:
            logger.info(f"✅ Matched config key (with ext): '{clean_key}'")
            return self.zones[clean_key]

        # 备用：尝试无扩展名匹配
        no_ext_key = os.path.splitext(clean_key)[0]
        if no_ext_key in self.zones:
            logger.info(f"✅ Matched config key (no ext): '{no_ext_key}'")
            return self.zones[no_ext_key]

        logger.warning(
            f"⚠️ No zones for '{source_id}'. Tried: ['{clean_key}', '{no_ext_key}']. "
            f"Available keys: {list(self.zones.keys())}"
        )
        return []