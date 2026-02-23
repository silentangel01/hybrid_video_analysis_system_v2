# backend/models/event.py
# 后端 / 数据模型 / 事件定义文件
# Backend data-model definition for video-analysis events

from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from pydantic import BaseModel, Field


class EventModel(BaseModel):
    """
    Data model for video-analysis events.
    Stored in MongoDB and references the rendered image in MinIO.
    视频分析事件的数据模型，存入 MongoDB，并关联 MinIO 中的渲染图。

    支持的事件类型：
    - 电子围栏检测 (parking_violation): 车辆违规停放
    - 烟火检测 (smoke_flame): 烟雾或火焰检测
    - 公共空间分析 (common_space_utilization): 公共空间使用情况分析
    """

    # ==================== 必需字段 ====================
    camera_id: str
    # 摄像头唯一标识，例如 'camera_front_gate'
    # Unique camera identifier, e.g. 'camera_front_gate'

    timestamp: float
    # 事件发生时的 Unix 时间戳（秒）
    # Unix timestamp (seconds) when the event occurred

    event_type: str
    # 事件类型：'parking_violation', 'smoke_flame', 'common_space_utilization', 'car', 'smoke', 'fire' 等
    # Event type: 'parking_violation', 'smoke_flame', 'common_space_utilization', 'car', 'smoke', 'fire', etc.

    image_url: str
    # MinIO 中已渲染图片的访问 URL（含所有框 + 禁停区）
    # Public URL of the rendered image in MinIO (with all bboxes & no-parking zones)

    # ==================== 可选字段 ====================
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # 模型置信度 [0.0, 1.0]，默认为 0.0
    # Model confidence score in range [0.0, 1.0], default 0.0

    bbox: Optional[Tuple[int, int, int, int]] = None
    # 单个边界框 (x1, y1, x2, y2) - ✅ 修复：改为可选字段
    # Single bounding box (top-left x, top-left y, bottom-right x, bottom-right y) - ✅ FIXED: Changed to optional

    frame_index: Optional[int] = None
    # 事件在视频流中的帧号，便于回溯
    # Frame number in the stream for quick rewind

    description: Optional[str] = None
    # 人类可读的事件描述，备用
    # Human-readable description, reserved field

    detection_stage: Optional[str] = None
    # 检测阶段标识，例如 'yolo_initial', 'qwen_verified', 'qwen_vl_analysis'
    # Detection stage identifier, e.g. 'yolo_initial', 'qwen_verified', 'qwen_vl_analysis'

    zone_polygon: Optional[List[Tuple[int, int]]] = None
    # 禁停区多边形坐标（仅电子围栏检测使用）
    # No-parking zone polygon coordinates (only for parking violation detection)

    # ==================== 新增：公共空间分析字段 ====================
    analysis_result: Optional[str] = None
    # 完整的AI分析结果文本（仅公共空间分析使用）
    # Complete AI analysis result text (only for common space analysis)

    analysis_summary: Optional[Dict[str, Any]] = None
    # 结构化分析摘要（仅公共空间分析使用）
    # Structured analysis summary (only for common space analysis)

    metadata: Optional[Dict[str, Any]] = None
    # 额外元数据，可存储采样间隔、提示词等
    # Additional metadata, can store sample interval, prompts, etc.

    # ==================== 系统字段 ====================
    created_at: datetime = Field(default_factory=datetime.utcnow)
    # 文档创建时间，默认当前 UTC 时间
    # Document creation time, defaults to current UTC time

    processed_at: Optional[datetime] = None

    # 事件处理完成时间
    # Event processing completion time

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "camera_id": "camera_front_gate",
                    "timestamp": 1672531200.0,
                    "event_type": "parking_violation",
                    "image_url": "http://localhost:9000/video-events/camera_front_gate/2023/01/01/event_123456.jpg",
                    "confidence": 0.85,
                    "bbox": [100, 100, 200, 200],
                    "frame_index": 1234,
                    "description": "Car in no-parking zone",
                    "detection_stage": "yolo_initial",
                    "zone_polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
                    "created_at": "2023-01-01T00:00:00Z"
                },
                {
                    "camera_id": "indoor_camera_001",
                    "timestamp": 1672531200.0,
                    "event_type": "smoke_flame",
                    "image_url": "http://localhost:9000/video-events/indoor_camera_001/2023/01/01/event_123457.jpg",
                    "confidence": 0.92,
                    "bbox": [150, 150, 250, 250],
                    "frame_index": 5678,
                    "description": "Smoke detected (qwen_verified)",
                    "detection_stage": "qwen_verified",
                    "created_at": "2023-01-01T00:00:00Z"
                },
                {
                    "camera_id": "public_space_camera",
                    "timestamp": 1672531200.0,
                    "event_type": "common_space_utilization",
                    "image_url": "http://localhost:9000/video-events/public_space_camera/2023/01/01/event_123458.jpg",
                    "confidence": 1.0,
                    "description": "Public space analysis: 15 people, high occupancy",
                    "detection_stage": "qwen_vl_analysis",
                    "analysis_result": "This image shows a crowded public space with approximately 15 people engaged in various activities...",
                    "analysis_summary": {
                        "estimated_people_count": 15,
                        "space_occupancy": "high",
                        "activity_types": ["walking", "sitting", "talking"],
                        "safety_concerns": False,
                        "keywords": ["crowded", "public", "space", "people", "activities"]
                    },
                    "metadata": {
                        "sample_interval": 30,
                        "system_prompt": "You are a professional public space analysis assistant...",
                        "user_prompt": "Please analyze the public space usage in this image...",
                        "processing_time": 2.5
                    },
                    "created_at": "2023-01-01T00:00:00Z"
                }
            ]
        }