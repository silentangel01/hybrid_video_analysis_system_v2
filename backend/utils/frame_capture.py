# backend/utils/frame_capture.py
"""
Frame Capture Utility —— 官方示例风格（批量消费）
支持：
 1. RTSP 多路并发（线程）
 2. 本地视频文件（可循环）
 3. 帧采样（时间间隔 or 帧跳过）
 4. 批量回调 → 上游一次性推理（官方同款）
 5. 无自动存图，仅推送 FrameWithMetadata 列表
"""

import cv2
import time
import logging
import threading
import queue
import os
from typing import List, Optional, Callable, Dict, Any
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------- 数据载体 --------------------
@dataclass
class FrameWithMetadata:
    """Frame + context，供下游批量检测."""
    image: Any  # OpenCV 图像 (np.ndarray)
    source_id: str  # 源唯一标识
    timestamp: float  # Unix 秒（带毫秒）- 🔴 修复：始终使用系统时间
    frame_index: int  # 本源内序号
    original_time_str: str  # 人类可读时间
    is_rtsp: bool  # True=实时流，False=文件


# -------------------- 批量缓存器 --------------------
class FrameBuffer:
    """缓存指定时长/数量的帧，然后一次性推给上游."""

    def __init__(self, source_id: str, batch_size: int = 1, batch_sec: float = 1.0):
        self.source_id = source_id
        self.batch_size = batch_size
        self.batch_sec = batch_sec
        self.buffer: List[FrameWithMetadata] = []
        self.last_push = time.time()

    def add(self, frame: FrameWithMetadata) -> Optional[List[FrameWithMetadata]]:
        """添加帧；满足条件时返回整批，否则 None."""
        self.buffer.append(frame)
        meet_size = len(self.buffer) >= self.batch_size
        meet_time = (frame.timestamp - self.last_push) >= self.batch_sec
        if meet_size or meet_time:
            self.last_push = frame.timestamp
            batch, self.buffer = self.buffer[:], []
            return batch
        return None


# -------------------- 统一捕获管理器 --------------------
class VideoFrameCapture:
    """
    官方示例风格：
    每路源独立线程 → 帧缓存 → 批量回调 → 上游一次性推理
    """

    def __init__(self):
        self.sources: Dict[str, threading.Thread] = {}
        self.running = True
        # 回调函数：batch_callback(source_id, frame_list)
        self._callback: Optional[Callable[[str, List[FrameWithMetadata]], None]] = None

    def register_batch_callback(self, callback: Callable[[str, List[FrameWithMetadata]], None]):
        """注册批量回调，官方示例一次性推理入口."""
        self._callback = callback

    # -------------------- 添加源 --------------------
    def add_rtsp_source(
            self,
            source_id: str,
            rtsp_url: str,
            batch_size: int = 1,
            batch_sec: float = 1.0,
            reconnect_delay: int = 5
    ):
        """添加 RTSP 源，支持批量推送."""
        if source_id in self.sources:
            logger.warning(f"[{source_id}] Already exists.")
            return
        thread = threading.Thread(
            target=self._rtsp_loop,
            args=(source_id, rtsp_url, batch_size, batch_sec, reconnect_delay),
            daemon=True
        )
        self.sources[source_id] = thread
        thread.start()

    def add_local_video_source(
            self,
            source_id: str,
            video_path: str,
            batch_size: int = 1,
            batch_sec: float = 1.0,
            loop_play: bool = True
    ):
        """添加本地视频文件源，支持批量推送."""
        if not os.path.exists(video_path):
            logger.error(f"[{source_id}] File not found: {video_path}")
            return
        if source_id in self.sources:
            logger.warning(f"[{source_id}] Already exists.")
            return
        thread = threading.Thread(
            target=self._local_loop,
            args=(source_id, video_path, batch_size, batch_sec, loop_play),
            daemon=True
        )
        self.sources[source_id] = thread
        thread.start()

    # -------------------- RTSP 线程 --------------------
    def _rtsp_loop(
            self,
            source_id: str,
            url: str,
            batch_size: int,
            batch_sec: float,
            reconnect_delay: int
    ):
        """RTSP 捕获 + 批量推送."""
        buffer = FrameBuffer(source_id, batch_size, batch_sec)
        frame_idx = 0

        logger.info(f"[{source_id}] RTSP capture started: {url}")

        while self.running:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                logger.error(f"[{source_id}] RTSP open failed, retry in {reconnect_delay}s")
                time.sleep(reconnect_delay)
                continue

            logger.info(f"[{source_id}] RTSP connected.")
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                # 🔴 修复：RTSP 使用当前系统时间，而不是视频时间戳
                current_time = time.time()
                current_dt = datetime.fromtimestamp(current_time)

                meta = FrameWithMetadata(
                    image=frame,
                    source_id=source_id,
                    timestamp=current_time,  # 🔴 使用当前系统时间
                    frame_index=frame_idx,
                    original_time_str=current_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    is_rtsp=True
                )

                # 调试日志：显示时间戳信息
                if frame_idx % 30 == 1:  # 每30帧记录一次
                    logger.debug(f"[{source_id}] Frame {frame_idx} - "
                                 f"System time: {current_dt} | "
                                 f"Timestamp: {current_time}")

                batch = buffer.add(meta)
                if batch and self._callback:
                    self._callback(source_id, batch)

            cap.release()
            logger.warning(f"[{source_id}] RTSP disconnected, reconnecting...")
            time.sleep(reconnect_delay)

    # -------------------- 本地文件线程 --------------------
    def _local_loop(
            self,
            source_id: str,
            path: str,
            batch_size: int,
            batch_sec: float,
            loop_play: bool
    ):
        """本地文件捕获 + 批量推送."""
        buffer = FrameBuffer(source_id, batch_size, batch_sec)

        logger.info(f"[{source_id}] Local video capture started: {path}")

        while self.running:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                logger.error(f"[{source_id}] Cannot open file: {path}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            logger.info(f"[{source_id}] Video info: FPS={fps:.1f}, Frames={total_frames}, Duration={duration:.1f}s")

            frame_idx = 0
            start_system_time = time.time()  # 🔴 记录开始时的系统时间

            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                # 🔴 修复：本地视频使用相对时间 + 系统基准时间
                video_pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                # 方法1：使用系统时间作为基准（推荐）
                current_system_time = start_system_time + video_pos_sec
                current_dt = datetime.fromtimestamp(current_system_time)

                # 方法2：或者直接使用当前系统时间（更简单）
                # current_system_time = time.time()
                # current_dt = datetime.now()

                meta = FrameWithMetadata(
                    image=frame,
                    source_id=source_id,
                    timestamp=current_system_time,  # 🔴 使用系统相关时间
                    frame_index=frame_idx,
                    original_time_str=current_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    is_rtsp=False
                )

                # 调试日志：显示时间戳信息
                if frame_idx % 100 == 1:  # 每100帧记录一次
                    logger.debug(f"[{source_id}] Frame {frame_idx}/{total_frames} - "
                                 f"Video time: {video_pos_sec:.1f}s - "
                                 f"System time: {current_dt} | "
                                 f"Timestamp: {current_system_time}")

                batch = buffer.add(meta)
                if batch and self._callback:
                    self._callback(source_id, batch)

                # 模拟实时播放速度（如果不需要实时，可以注释掉）
                if batch_sec <= 0.1:
                    time.sleep(1.0 / fps)

            cap.release()

            # 调试：显示循环信息
            logger.info(f"[{source_id}] Video finished at frame {frame_idx}")

            if not loop_play:
                logger.info(f"[{source_id}] Single play completed.")
                break

            logger.info(f"[{source_id}] Local video looping...")
            time.sleep(1)  # 循环间隔

    # -------------------- 时间戳调试方法 --------------------
    def debug_timestamp_issue(self, source_id: str):
        """调试特定源的时间戳问题"""
        if source_id not in self.sources:
            logger.error(f"[{source_id}] Source not found for debugging")
            return

        current_time = time.time()
        current_dt = datetime.now()

        logger.info(f"=== 时间戳调试 [{source_id}] ===")
        logger.info(f"当前系统时间: {current_dt}")
        logger.info(f"当前Unix时间戳: {current_time}")
        logger.info(f"转换为日期: {datetime.fromtimestamp(current_time)}")

        # 检查是否是未来时间戳
        test_timestamp = 1761401166.298111  # 你的问题时间戳
        if test_timestamp > current_time:
            logger.warning(f"❌ 检测到未来时间戳: {test_timestamp}")
            logger.warning(f"❌ 对应日期: {datetime.fromtimestamp(test_timestamp)}")
        else:
            logger.info(f"✅ 时间戳 {test_timestamp} 是过去时间")

    # -------------------- 优雅停机 --------------------
    def stop_all(self):
        """停止所有捕获线程."""
        self.running = False
        for tid, t in self.sources.items():
            if t.is_alive():
                t.join(timeout=2.0)
                logger.info(f"🛑 Stopped: {tid}")
        logger.info("🛑 All capture threads stopped.")

    # -------------------- 状态检查 --------------------
    def get_source_status(self) -> Dict[str, str]:
        """获取所有源的状态"""
        status = {}
        for source_id, thread in self.sources.items():
            status[source_id] = "alive" if thread.is_alive() else "dead"
        return status


# -------------------- 快速测试 --------------------
if __name__ == "__main__":
    def demo_batch_callback(src_id: str, frames: List[FrameWithMetadata]):
        """示例：批量打印，可替换为 DetectionService.process_batch(frames)"""
        if frames:
            first_frame = frames[0]
            last_frame = frames[-1]
            time_span = last_frame.timestamp - first_frame.timestamp

            logger.info(f"📦 Batch from {src_id}: {len(frames)} frames, "
                        f"time span {time_span:.2f}s, "
                        f"first: {datetime.fromtimestamp(first_frame.timestamp)}, "
                        f"last: {datetime.fromtimestamp(last_frame.timestamp)}")


    cap = VideoFrameCapture()
    cap.register_batch_callback(demo_batch_callback)

    # 本地演示视频（循环）
    test_video_path = "./sample_videos/fire_test.mp4"
    if os.path.exists(test_video_path):
        cap.add_local_video_source(
            source_id="demo_video",
            video_path=test_video_path,
            batch_size=8,  # 一次 8 帧
            batch_sec=1.0,  # 或 1 秒
            loop_play=True
        )
    else:
        logger.warning(f"Test video not found: {test_video_path}")
        # 使用默认摄像头测试
        logger.info("Using default camera for testing...")
        cap.add_rtsp_source(
            source_id="demo_camera",
            rtsp_url="0",  # 默认摄像头
            batch_size=4,
            batch_sec=0.5
        )

    try:
        # 运行一段时间后检查状态
        time.sleep(5)
        status = cap.get_source_status()
        logger.info(f"📊 Source status: {status}")

        # 持续运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 Stopping capture...")
        cap.stop_all()