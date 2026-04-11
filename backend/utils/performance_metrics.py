"""Lightweight performance metrics for RTSP pipeline diagnostics."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Tuple


class SlidingCounter:
    """Count events in a recent time window and expose an approximate rate."""

    def __init__(self, window_sec: float = 10.0):
        self.window_sec = float(window_sec)
        self.lock = threading.Lock()
        self.total = 0
        self.window_total = 0
        self.points: Deque[Tuple[float, int]] = deque()

    def _trim(self, now: float):
        cutoff = now - self.window_sec
        while self.points and self.points[0][0] < cutoff:
            _, count = self.points.popleft()
            self.window_total -= count

    def add(self, count: int = 1, now: float | None = None):
        now = time.time() if now is None else now
        count = int(count)
        with self.lock:
            self.total += count
            self.window_total += count
            self.points.append((now, count))
            self._trim(now)

    def snapshot(self, now: float | None = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        with self.lock:
            self._trim(now)
            if self.points:
                span = max(now - self.points[0][0], 1e-6)
                effective_window = min(self.window_sec, span)
                rate_per_sec = self.window_total / max(effective_window, 1e-6)
            else:
                rate_per_sec = 0.0

            return {
                "window_sec": self.window_sec,
                "total": self.total,
                "recent_count": self.window_total,
                "rate_per_sec": round(rate_per_sec, 2),
            }


class LatencyRecorder:
    """Track average and recent latency without external dependencies."""

    def __init__(self, recent_size: int = 50):
        self.lock = threading.Lock()
        self.count = 0
        self.total_ms = 0.0
        self.max_ms = 0.0
        self.last_ms = 0.0
        self.recent_ms: Deque[float] = deque(maxlen=recent_size)

    def record(self, seconds: float):
        ms = max(float(seconds) * 1000.0, 0.0)
        with self.lock:
            self.count += 1
            self.total_ms += ms
            self.max_ms = max(self.max_ms, ms)
            self.last_ms = ms
            self.recent_ms.append(ms)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            recent_avg = (sum(self.recent_ms) / len(self.recent_ms)) if self.recent_ms else 0.0
            avg_ms = (self.total_ms / self.count) if self.count else 0.0
            return {
                "count": self.count,
                "avg_ms": round(avg_ms, 2),
                "recent_avg_ms": round(recent_avg, 2),
                "last_ms": round(self.last_ms, 2),
                "max_ms": round(self.max_ms, 2),
            }


def get_thread_pool_queue_size(pool: Any) -> int:
    """Read the pending queue size of ThreadPoolExecutor when available."""
    work_queue = getattr(pool, "_work_queue", None)
    if work_queue is None:
        return 0
    try:
        return work_queue.qsize()
    except Exception:
        return 0
