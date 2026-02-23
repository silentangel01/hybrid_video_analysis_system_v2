# scripts/file_watcher.py
"""
File Watcher Module —— 多文件夹监控版本
监控上传目录的不同子文件夹 → 调用对应的检测服务
- uploads/parking/ → 电子围栏检测
- uploads/smoke_flame/ → 烟火检测
- uploads/common_space/ → 公共空间分析
同时保持向后兼容性
"""

import sys
import os
import subprocess
import time
import logging
from tkinter import messagebox

import cv2
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from backend.services.parking_zone_checker import load_zones_from_file
NO_PARKING_CONFIG_PATH = "no_parking_config.json"
SUPPORTED_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}

# 定义监控的子文件夹和对应的检测类型
MONITOR_FOLDERS = {
    "parking": "parking_violation",
    "smoke_flame": "smoke_flame",
    "common_space": "common_space"
}

# -------------------- 工具函数 --------------------
def is_file_locked(filepath: str) -> bool:
    try:
        buffer = os.open(filepath, os.O_RDONLY)
        os.close(buffer)
        return False
    except OSError:
        return True


def wait_for_file_ready(filepath: str, timeout: float = 30.0, interval: float = 0.5) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        if not os.path.exists(filepath):
            time.sleep(interval)
            continue
        if is_file_locked(filepath):
            time.sleep(interval)
            continue
        size1 = os.path.getsize(filepath)
        time.sleep(0.2)
        size2 = os.path.getsize(filepath)
        if size1 == size2:
            return True
    return False


def create_monitor_folders(base_folder: str):
    """创建监控的子文件夹"""
    for folder_name in MONITOR_FOLDERS.keys():
        folder_path = os.path.join(base_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"📁 Created monitor folder: {folder_path}")


# -------------------- 多文件夹事件处理器 --------------------
class MultiFolderVideoHandler(FileSystemEventHandler):
    def __init__(
            self,
            base_folder: str,
            model_loader,
            parking_detection_service,
            smoke_flame_detection_service,
            zone_checker,
            frame_interval: float = 1.0
    ):
        self.base_folder = base_folder
        self.model_loader = model_loader
        self.parking_detection_service = parking_detection_service
        self.smoke_flame_detection_service = smoke_flame_detection_service
        self.zone_checker = zone_checker
        self.frame_interval = frame_interval

    def on_created(self, event):
        if event.is_directory:
            return

        video_path = os.path.abspath(event.src_path)
        ext = os.path.splitext(video_path)[1].lower()
        if ext not in SUPPORTED_VIDEO_EXTS:
            return

        if not wait_for_file_ready(video_path, timeout=30.0):
            logger.error(f"[Error] Timeout waiting for file: {video_path}")
            return

        # 确定文件在哪个子文件夹中，从而选择对应的检测服务
        detection_type = self._get_detection_type_from_path(video_path)
        if not detection_type:
            logger.warning(f"⚠️ File not in monitored subfolder: {video_path}")
            return

        self.process_video(video_path, os.path.basename(video_path), detection_type)

    def _get_detection_type_from_path(self, file_path: str) -> Optional[str]:
        """
        根据文件路径确定检测类型
        Determine detection type based on file path
        """
        try:
            relative_path = os.path.relpath(file_path, self.base_folder)
            folder_name = relative_path.split(os.sep)[0]  # 获取第一级子文件夹名

            return MONITOR_FOLDERS.get(folder_name)
        except ValueError:
            # 如果文件不在base_folder下，返回None
            return None

    def process_video(self, video_path: str, source_id: str, detection_type: str):
        """
        根据检测类型处理视频
        Process video based on detection type
        """
        logger.info(f"🎬 Processing: {source_id} | Type: {detection_type}")

        if detection_type == "parking_violation":
            self._process_parking_violation(video_path, source_id)
        elif detection_type == "smoke_flame":
            self._process_smoke_flame(video_path, source_id)
        elif detection_type == "common_space":  # ✅ 新增
            self._process_common_space(video_path, source_id)
        else:
            logger.error(f"❌ Unknown detection type: {detection_type}")

    def _process_parking_violation(self, video_path: str, source_id: str):
        """
        处理电子围栏检测
        Process parking violation detection
        """
        if source_id not in self.zone_checker.zones:
            logger.warning(f"⚠️ NO EXACT CONFIG FOUND for '{source_id}'! Launching GUI to define zone...")

            gui_script = os.path.join(os.path.dirname(__file__), "draw_fence_gui.py")
            if not os.path.exists(gui_script):
                gui_script = "draw_fence_gui.py"  # Fallback

            cmd = [
                sys.executable,
                gui_script,
                "--video", video_path,
                "--test-mode"
            ]

            try:
                logger.info(f"🎨 LAUNCHING GUI: {' '.join(cmd)}")
                # 阻塞等待GUI完成（用户点击"Finish & Exit"）
                subprocess.run(cmd, check=True)

                # 重新加载配置（覆盖整个 zones 字典）
                logger.info(f"🔄 Reloading zone config from: {NO_PARKING_CONFIG_PATH}")
                new_zones = load_zones_from_file(NO_PARKING_CONFIG_PATH)

                # 更新 file_watcher 中的 zone_checker
                self.zone_checker.zones = new_zones

                # 【关键修复】同步更新 detection_service 中的 zone_checker
                if (hasattr(self.parking_detection_service, 'zone_checker') and
                        self.parking_detection_service.zone_checker):
                    self.parking_detection_service.zone_checker.zones = new_zones
                    logger.info("✅ Synced zone_checker to detection_service")
                else:
                    logger.warning("⚠️ detection_service.zone_checker not found, may cause inconsistency")

                # 二次验证：GUI是否真的保存了配置
                if source_id not in self.zone_checker.zones:
                    logger.error(f"❌ GUI exited but NO zone saved for '{source_id}'. SKIPPING video.")
                    if messagebox:
                        messagebox.showerror("Error", f"Zone not saved for {source_id}. Video skipped.")
                    return
                else:
                    logger.info(f"✅ Zone successfully configured for '{source_id}' via GUI. Resuming processing...")
            except subprocess.CalledProcessError:
                logger.error(f"❌ GUI closed without saving (user canceled). SKIPPING video '{source_id}'.")
                return
            except Exception as e:
                logger.error(f"❌ Failed to launch GUI: {e}. SKIPPING video.")
                return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"[Error] Cannot open: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        sample_frames = max(1, int(fps * self.frame_interval))
        logger.info(f"🅿️ Parking detection: {source_id} | FPS: {fps:.1f} | Sample every {sample_frames} frames")

        frame_idx = 0
        last_time = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            now = time.time()

            # 时间间隔采样
            if self.frame_interval > 0 and now - last_time < self.frame_interval:
                continue
            last_time = now

            # 构造整帧元数据
            from backend.utils.frame_capture import FrameWithMetadata
            frame_meta = FrameWithMetadata(
                image=frame,
                source_id=source_id,
                timestamp=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                frame_index=frame_idx,
                original_time_str=time.strftime("%H:%M:%S", time.gmtime(now)),
                is_rtsp=False
            )

            # 官方检测服务：整帧 → 一张图 → 一条文档
            self.parking_detection_service.process_frame(frame_meta)

        cap.release()
        self.parking_detection_service.flush_remaining()
        logger.info(f"✅ Finished parking detection: {source_id}")

    def _process_smoke_flame(self, video_path: str, source_id: str):
        """
        处理烟火检测
        Process smoke/flame detection
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"[Error] Cannot open: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        sample_frames = max(1, int(fps * self.frame_interval))
        logger.info(f"🔥 Smoke/Flame detection: {source_id} | FPS: {fps:.1f} | Sample every {sample_frames} frames")

        frame_idx = 0
        last_time = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            now = time.time()

            # 时间间隔采样
            if self.frame_interval > 0 and now - last_time < self.frame_interval:
                continue
            last_time = now

            # 构造整帧元数据
            from backend.utils.frame_capture import FrameWithMetadata
            frame_meta = FrameWithMetadata(
                image=frame,
                source_id=source_id,
                timestamp=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                frame_index=frame_idx,
                original_time_str=time.strftime("%H:%M:%S", time.gmtime(now)),
                is_rtsp=False
            )

            # 烟火检测服务
            self.smoke_flame_detection_service.process_frame(frame_meta)

        cap.release()
        self.smoke_flame_detection_service.flush_remaining()
        logger.info(f"✅ Finished smoke/flame detection: {source_id}")

    # ✅ 新增：公共空间处理方法
    def _process_common_space(self, video_path: str, source_id: str):
        """
        处理公共空间分析
        Process common space analysis
        """
        # 导入公共空间检测服务
        from backend.services.common_space_detection import common_space_detection_service

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"[Error] Cannot open: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        logger.info(f"🏢 Common space analysis: {source_id} | FPS: {fps:.1f} | Sampling interval: 30s")

        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            now = time.time()

            # 构造整帧元数据
            from backend.utils.frame_capture import FrameWithMetadata
            frame_meta = FrameWithMetadata(
                image=frame,
                source_id=source_id,
                timestamp=now,
                frame_index=frame_idx,
                original_time_str=time.strftime("%H:%M:%S", time.gmtime(now)),
                is_rtsp=False
            )

            # 公共空间分析服务（内部会控制采样间隔）
            common_space_detection_service.process_frame(frame_meta)

        cap.release()
        common_space_detection_service.flush_remaining()
        logger.info(f"✅ Finished common space analysis: {source_id}")


# -------------------- 向后兼容的事件处理器 --------------------
class LegacyVideoHandler(FileSystemEventHandler):
    def __init__(self, model_loader, detection_service, zone_checker, frame_interval):
        self.model_loader = model_loader
        self.detection_service = detection_service
        self.zone_checker = zone_checker
        self.frame_interval = frame_interval

    def on_created(self, event):
        if event.is_directory:
            return

        video_path = os.path.abspath(event.src_path)
        ext = os.path.splitext(video_path)[1].lower()
        if ext not in SUPPORTED_VIDEO_EXTS:
            return

        if not wait_for_file_ready(video_path, timeout=30.0):
            logger.error(f"[Error] Timeout waiting for file: {video_path}")
            return

        self.process_video(video_path, os.path.basename(video_path))

    def process_video(self, video_path: str, source_id: str):
        """
        原有的电子围栏处理逻辑
        Original parking violation processing logic
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"[Error] Cannot open: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        sample_frames = max(1, int(fps * self.frame_interval))
        logger.info(f"▶️ Processing: {source_id} | FPS: {fps:.1f} | Sample every {sample_frames} frames")

        frame_idx = 0
        last_time = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            now = time.time()

            # 时间间隔采样
            if self.frame_interval > 0 and now - last_time < self.frame_interval:
                continue
            last_time = now

            # 构造整帧元数据
            from backend.utils.frame_capture import FrameWithMetadata
            frame_meta = FrameWithMetadata(
                image=frame,
                source_id=source_id,
                timestamp=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                frame_index=frame_idx,
                original_time_str=time.strftime("%H:%M:%S", time.gmtime(now)),
                is_rtsp=False
            )

            # 官方检测服务：整帧 → 一张图 → 一条文档
            self.detection_service.process_frame(frame_meta)

        cap.release()
        self.detection_service.flush_remaining()
        logger.info(f"✅ Finished: {source_id}")


# -------------------- 启动函数 --------------------
def start_multi_folder_watchdog(
        base_folder: str,
        model_loader,
        parking_detection_service,
        smoke_flame_detection_service,
        zone_checker,
        frame_interval: float = 1.0
) -> Observer:
    """
    启动多文件夹监控（新功能）
    Start multi-folder watchdog (new feature)

    Args:
        base_folder: 基础监控文件夹
        model_loader: 模型加载器
        parking_detection_service: 电子围栏检测服务
        smoke_flame_detection_service: 烟火检测服务
        zone_checker: 区域检查器
        frame_interval: 帧间隔
    """
    base_folder = os.path.abspath(base_folder)
    os.makedirs(base_folder, exist_ok=True)

    # 创建监控的子文件夹
    create_monitor_folders(base_folder)

    handler = MultiFolderVideoHandler(
        base_folder=base_folder,
        model_loader=model_loader,
        parking_detection_service=parking_detection_service,
        smoke_flame_detection_service=smoke_flame_detection_service,
        zone_checker=zone_checker,
        frame_interval=frame_interval
    )

    observer = Observer()

    # 为每个子文件夹注册监控
    for folder_name in MONITOR_FOLDERS.keys():
        folder_path = os.path.join(base_folder, folder_name)
        observer.schedule(handler, folder_path, recursive=False)
        logger.info(f"👀 Watching subfolder: {folder_path}")

    observer.start()
    logger.info(f"🎯 Multi-folder watchdog started: {base_folder}")
    logger.info("📂 Monitored folders:")
    for folder_name, detection_type in MONITOR_FOLDERS.items():
        logger.info(f"   - {folder_name}/ → {detection_type}")

    return observer


def start_file_watchdog(
        folder_path: str,
        model_loader,
        detection_service,
        zone_checker,
        frame_interval: float = 1.0
) -> Observer:
    """
    向后兼容的启动函数（只监控单个文件夹，用于电子围栏检测）
    Backward compatible startup function (monitors single folder for parking detection)
    """
    folder_path = os.path.abspath(folder_path)
    os.makedirs(folder_path, exist_ok=True)

    handler = LegacyVideoHandler(
        model_loader=model_loader,
        detection_service=detection_service,
        zone_checker=zone_checker,
        frame_interval=frame_interval
    )

    observer = Observer()
    observer.schedule(handler, folder_path, recursive=False)
    observer.start()
    logger.info(f"👀 Watching (legacy): {folder_path} | Interval: {frame_interval}s")

    return observer


# -------------------- 主启动函数（新增） --------------------
def start_detection_system(
        uploads_folder: str = "./uploads",
        model_loader=None,
        parking_detection_service=None,
        smoke_flame_detection_service=None,
        zone_checker=None,
        frame_interval: float = 1.0,
        use_multi_folder: bool = True
) -> Observer:
    """
    统一的检测系统启动函数
    Unified detection system startup function

    Args:
        uploads_folder: 上传文件夹路径
        model_loader: 模型加载器
        parking_detection_service: 电子围栏检测服务
        smoke_flame_detection_service: 烟火检测服务
        zone_checker: 区域检查器
        frame_interval: 帧间隔
        use_multi_folder: 是否使用多文件夹模式
    """
    if use_multi_folder and parking_detection_service and smoke_flame_detection_service:
        # 使用多文件夹模式
        logger.info("🚀 Starting multi-folder detection system...")
        return start_multi_folder_watchdog(
            base_folder=uploads_folder,
            model_loader=model_loader,
            parking_detection_service=parking_detection_service,
            smoke_flame_detection_service=smoke_flame_detection_service,
            zone_checker=zone_checker,
            frame_interval=frame_interval
        )
    else:
        # 使用向后兼容的单文件夹模式
        logger.info("🚀 Starting legacy single-folder detection system...")
        return start_file_watchdog(
            folder_path=uploads_folder,
            model_loader=model_loader,
            detection_service=parking_detection_service,
            zone_checker=zone_checker,
            frame_interval=frame_interval
        )


# -------------------- 直接启动支持（保持原有功能） --------------------
if __name__ == "__main__":
    """
    直接启动时，使用原有的单文件夹模式（向后兼容）
    When started directly, use legacy single-folder mode (backward compatible)
    """
    import sys
    from backend.config.database import init_clients
    from ml_models.yolov8.model_loader import YOLOModelLoader
    from backend.services.violation_detection import detection_service
    from backend.services.parking_zone_checker import zone_checker, load_zones_from_file

    # 初始化服务（原有的初始化逻辑）
    minio_client, mongo_client = init_clients()
    model_loader = YOLOModelLoader()
    detection_service.set_clients(minio_client, mongo_client)
    detection_service.set_model_loader(model_loader)
    detection_service.set_zone_checker(zone_checker)

    # 启动文件监控（单文件夹模式）
    observer = start_file_watchdog(
        folder_path="./uploads",
        model_loader=model_loader,
        detection_service=detection_service,
        zone_checker=zone_checker,
        frame_interval=1.0
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()
        logger.info("✅ System stopped")