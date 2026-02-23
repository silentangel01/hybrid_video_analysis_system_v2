# backend/config/no_parking_zones.py
# 后端/配置/禁停区域文件
# Backend configuration file for no-parking zones

"""
Configuration for no-parking zones (electronic fences).
Each camera can have its own polygon(s).
Coordinates are in (x, y) format relative to video resolution.
"""
"""
禁停区域（电子围栏）配置文件。
每个摄像头可拥有专属的一个或多个多边形。
坐标以视频分辨率为基准，采用 (x, y) 格式。
"""

NO_PARKING_ZONES = {
    # 顶级字典：键为摄像头ID或通配符，值为多边形列表
    # Top-level dict: key = camera ID or wildcard, value = list of polygons

    # For uploaded videos, use filename or dynamic assignment
    # 针对上传的视频文件，可用通配符或运行时动态匹配
    "upload_video_*.mp4": [
        [(120, 90), (250, 80), (260, 180), (130, 190)]  # 默认多边形，供上传视频使用
        # Default polygon for uploaded videos
    ]
}
