# backend/services/detection_processor.py
"""
DetectionProcessor —— 毕设简化版（官方逻辑）
1. 整帧推理结果一次性画完
2. 一帧只写 1 条文档（含目标数组）
3. 不再逐目标存 EventModel，去掉 frame_id
"""

import cv2
import logging
from typing import List, Dict, Any
from backend.models.event import EventModel   # 如不用可换成 dict
from storage.minio_client import MinIOClient
from storage.mongodb_client import MongoDBClient
from backend.utils.frame_capture import FrameWithMetadata
from backend.utils.visualization import draw_detection_box, draw_no_parking_zone

logger = logging.getLogger(__name__)


class DetectionProcessor:
    """官方同款：整帧多框 → 一张图 → 一条记录"""

    def __init__(self, minio_client: MinIOClient, mongo_client: MongoDBClient):
        self.minio = minio_client
        self.db = mongo_client
        self.zone_checker = None   # 外部注入

    # -------------------- 唯一公开入口 --------------------
    def process_detections(self,
                           frame_meta: FrameWithMetadata,
                           detections: List[Dict[str, Any]]) -> None:
        """
        主入口 | Main Entry
        Parameters:
            frame_meta: 帧元数据（含图像、时间戳、摄像头ID）
            detections: YOLOv8 返回的整帧检测结果
        Returns:
            None
        Purpose:
            过滤违规目标 → 一次性画完所有框+禁停区 → 上传 → 写一条文档
        Business Logic:
            1. 只保留「car」且中心在禁停区的目标
            2. 若有违规，绘制所有框+禁停区
            3. 上传渲染图到 MinIO
            4. 把「整帧目标数组」塞到一条 EventModel 落 Mongo
               （若后续想加 frame_id，在此补充即可）
        """
        # 1. 过滤违规
        violations = [
            det for det in detections
            if det.get("class_name") == "car"
            and self.zone_checker
            and self.zone_checker.is_center_in_zone(
                det["bbox"], frame_meta.source_id, frame_meta.image.shape[:2])
        ]
        if not violations:
            return

        # 2. 一次性画图
        rendered = self._render_frame(frame_meta.image, violations, frame_meta.source_id)

        # 3. 上传 MinIO
        url = self.minio.upload_frame(
            image_data=rendered,
            camera_id=frame_meta.source_id,
            timestamp=frame_meta.timestamp,
            event_type="violations"
        )
        if not url:
            logger.warning("⚠️ Upload failed, skip save.")
            return

        # 4. 写库：一条文档概括整帧
        doc = EventModel(
            camera_id=frame_meta.source_id,
            timestamp=frame_meta.timestamp,
            event_type="car",               # 代表「整车违规帧」
            confidence=max(v["confidence"] for v in violations),  # 取最高置信度
            bbox=self._merge_bboxes(violations),  # 可给整帧大框，也可留空
            image_url=url,
            frame_index=frame_meta.frame_index,
            description=f"Total {len(violations)} car(s) in no-parking zone"
            # frame_id=...   # 如后续需要，在此添加
        )
        ok = self.db.save_event(doc)
        logger.info(f"✅ Saved frame-level event, cars={len(violations)}")

    # -------------------- 内部工具 --------------------
    def _render_frame(self, image: "ndarray", violations: List[Dict], source_id: str):
        """一次性画禁停区 + 所有框"""
        img = image.copy()
        # 禁停区
        if self.zone_checker:
            zones = self.zone_checker.get_zones_for_source(source_id)
            for zone in zones:
                img = draw_no_parking_zone(img, zone)
        # 检测框
        for v in violations:
            draw_detection_box(img, "car", v["confidence"], v["bbox"])
        cv2.putText(img, f"Cars: {len(violations)}", (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        return img

    @staticmethod
    def _merge_bboxes(violations: List[Dict]) -> tuple:
        """合并所有 bbox 为整帧大框，也可直接返回 (0,0,0,0)"""
        x1 = min(v["bbox"][0] for v in violations)
        y1 = min(v["bbox"][1] for v in violations)
        x2 = max(v["bbox"][2] for v in violations)
        y2 = max(v["bbox"][3] for v in violations)
        return (x1, y1, x2, y2)