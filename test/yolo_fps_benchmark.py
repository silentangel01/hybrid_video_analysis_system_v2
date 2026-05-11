#!/usr/bin/env python
"""Measure maximum average FPS for a project YOLO model.

The benchmark runs the same project inference wrapper used by the backend:
ml_models.yolov8.inference.YOLOInference.run_detection.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "ml_models" / "yolov8" / "weights"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


def add_project_root() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def resolve_weight(weight: str) -> Path:
    candidate = Path(weight)
    if not candidate.is_absolute():
        candidate = DEFAULT_WEIGHTS_DIR / weight
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Weight file not found: {candidate}")
    if candidate.stat().st_size <= 0:
        raise RuntimeError(f"Weight file is empty: {candidate}")
    return candidate


def load_model(weight_path: Path, device: Optional[str]):
    add_project_root()
    from ml_models.yolov8.model_loader import YOLOModelLoader

    loader = YOLOModelLoader(weights_dir=str(weight_path.parent), device=device)
    if not loader.load_model("benchmark", weight_path.name):
        raise RuntimeError(f"Failed to load YOLO model: {weight_path}")
    return loader.get_model("benchmark")


def run_detection(model, image: np.ndarray, conf: float, iou: float, device: Optional[str]):
    add_project_root()
    from ml_models.yolov8.inference import YOLOInference

    return YOLOInference.run_detection(
        model,
        image,
        conf_threshold=conf,
        iou_threshold=iou,
        device=device,
    )


def sync_cuda(device: Optional[str]) -> None:
    if not device or "cuda" not in device.lower():
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def cuda_info(device: Optional[str]) -> dict:
    info = {"requested_device": device or "model-default"}
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            index = 0
            info["cuda_device_name"] = torch.cuda.get_device_name(index)
            info["cuda_memory_total_mb"] = round(
                torch.cuda.get_device_properties(index).total_memory / 1024 / 1024,
                2,
            )
            info["cuda_max_memory_allocated_mb"] = round(
                torch.cuda.max_memory_allocated(index) / 1024 / 1024,
                2,
            )
    except Exception as exc:
        info["torch_probe_error"] = str(exc)
    return info


def synthetic_frame(shape_text: str) -> np.ndarray:
    parts = [int(part.strip()) for part in shape_text.lower().split("x") if part.strip()]
    if len(parts) != 2:
        raise ValueError("--synthetic-shape must be WIDTHxHEIGHT, for example 1280x720")
    width, height = parts
    return np.zeros((height, width, 3), dtype=np.uint8)


def collect_source_files(source: Path) -> Tuple[List[Path], List[Path]]:
    if source.is_file():
        suffix = source.suffix.lower()
        if suffix in IMAGE_EXTS:
            return [source], []
        if suffix in VIDEO_EXTS:
            return [], [source]
        raise ValueError(f"Unsupported source file type: {source}")

    if not source.is_dir():
        raise FileNotFoundError(f"Source path not found: {source}")

    images: List[Path] = []
    videos: List[Path] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTS:
            images.append(path)
        elif suffix in VIDEO_EXTS:
            videos.append(path)
    return images, videos


def load_frames(source: Optional[str], sample_every: int, limit: int, synthetic_shape_text: str) -> List[np.ndarray]:
    if not source:
        return [synthetic_frame(synthetic_shape_text)]

    path = Path(source)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()

    frames: List[np.ndarray] = []
    images, videos = collect_source_files(path)
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is not None:
            frames.append(image)
        if len(frames) >= limit:
            return frames

    sample_every = max(1, int(sample_every))
    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        frame_index = 0
        try:
            while len(frames) < limit:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1
                if frame_index % sample_every == 0:
                    frames.append(frame)
        finally:
            cap.release()
        if len(frames) >= limit:
            break

    if not frames:
        raise RuntimeError(f"No readable benchmark frames found in: {path}")
    return frames


def make_output_path(output: Optional[str]) -> Path:
    if output:
        path = Path(output)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = PROJECT_ROOT / "test" / "results" / f"yolo_fps_benchmark_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark YOLO detection-only throughput.")
    parser.add_argument("--source", default="", help="Optional image/video file or directory. Empty uses a blank frame.")
    parser.add_argument("--weights", default="yolov8n.pt", help="Weight filename or absolute path.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold.")
    parser.add_argument("--device", default="", help="Optional YOLO device: cpu, cuda, cuda:0.")
    parser.add_argument("--duration", type=float, default=30.0, help="Measured duration in seconds.")
    parser.add_argument("--warmup", type=float, default=5.0, help="Warmup duration in seconds.")
    parser.add_argument("--sample-every", type=int, default=30, help="For videos, preload every Nth frame.")
    parser.add_argument("--preload-limit", type=int, default=256, help="Max frames to preload into memory.")
    parser.add_argument("--synthetic-shape", default="1280x720", help="Synthetic WIDTHxHEIGHT when --source is empty.")
    parser.add_argument("--output", default="", help="Output JSON path. Defaults to test/results.")
    args = parser.parse_args()

    weight_path = resolve_weight(args.weights)
    device = args.device.strip() or None
    model = load_model(weight_path, device)
    frames = load_frames(args.source or None, args.sample_every, max(1, args.preload_limit), args.synthetic_shape)

    frame_index = 0
    warmup_end = time.perf_counter() + max(0.0, args.warmup)
    while time.perf_counter() < warmup_end:
        frame = frames[frame_index % len(frames)]
        run_detection(model, frame, args.conf, args.iou, device)
        frame_index += 1

    try:
        import torch

        if device and "cuda" in device.lower() and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    measured_end = time.perf_counter() + max(0.1, args.duration)
    latencies_ms: List[float] = []
    detections_per_frame: List[int] = []
    measured_frames = 0
    measured_started = time.perf_counter()

    while time.perf_counter() < measured_end:
        frame = frames[frame_index % len(frames)]
        sync_cuda(device)
        started = time.perf_counter()
        detections = run_detection(model, frame, args.conf, args.iou, device)
        sync_cuda(device)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        detections_per_frame.append(len(detections))
        measured_frames += 1
        frame_index += 1

    elapsed_sec = time.perf_counter() - measured_started
    output_path = make_output_path(args.output)
    result = {
        "schema_version": 1,
        "test_type": "yolo_detection_fps_benchmark",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": args.source or f"synthetic:{args.synthetic_shape}",
        "weights": str(weight_path),
        "parameters": {
            "conf": args.conf,
            "iou": args.iou,
            "device": device or "model-default",
            "duration_sec": args.duration,
            "warmup_sec": args.warmup,
            "preloaded_frames": len(frames),
        },
        "summary": {
            "frames_processed": measured_frames,
            "elapsed_sec": round(elapsed_sec, 3),
            "avg_fps": round(measured_frames / elapsed_sec, 3) if elapsed_sec > 0 else 0.0,
            "latency_avg_ms": round(statistics.mean(latencies_ms), 3) if latencies_ms else 0.0,
            "latency_p50_ms": round(percentile(latencies_ms, 0.50), 3),
            "latency_p95_ms": round(percentile(latencies_ms, 0.95), 3),
            "latency_p99_ms": round(percentile(latencies_ms, 0.99), 3),
            "detections_per_frame_avg": round(statistics.mean(detections_per_frame), 3)
            if detections_per_frame
            else 0.0,
        },
        "device_info": cuda_info(device),
    }

    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output_path), **result["summary"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
