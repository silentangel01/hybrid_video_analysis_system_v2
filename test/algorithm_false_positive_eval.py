#!/usr/bin/env python
"""Negative-set false-positive evaluation for project YOLO models.

This script intentionally uses the existing project inference wrapper:
ml_models.yolov8.inference.YOLOInference.run_detection.

Example:
  python test/algorithm_false_positive_eval.py ^
    --source test/data/negative_smoke ^
    --weights smoke_flame.pt ^
    --classes smoke,fire,flame ^
    --conf 0.10
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

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


def parse_classes(raw: str) -> Optional[set[str]]:
    items = [item.strip().lower() for item in raw.split(",") if item.strip()]
    return set(items) if items else None


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


def discover_files(source: Path) -> Tuple[List[Path], List[Path]]:
    if source.is_file():
        suffix = source.suffix.lower()
        if suffix in IMAGE_EXTS:
            return [source], []
        if suffix in VIDEO_EXTS:
            return [], [source]
        raise ValueError(f"Unsupported source file type: {source}")

    if not source.is_dir():
        raise FileNotFoundError(f"Source path not found: {source}")

    image_files: List[Path] = []
    video_files: List[Path] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTS:
            image_files.append(path)
        elif suffix in VIDEO_EXTS:
            video_files.append(path)
    return image_files, video_files


def iter_frames(
    source: Path,
    sample_every: int,
    max_frames: Optional[int],
) -> Iterator[Tuple[str, np.ndarray]]:
    emitted = 0
    image_files, video_files = discover_files(source)

    for path in image_files:
        image = cv2.imread(str(path))
        if image is None:
            continue
        yield str(path), image
        emitted += 1
        if max_frames is not None and emitted >= max_frames:
            return

    sample_every = max(1, int(sample_every))
    for path in video_files:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            continue
        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1
                if frame_index % sample_every != 0:
                    continue
                yield f"{path}#frame={frame_index}", frame
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    return
        finally:
            cap.release()


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


def make_output_path(output: Optional[str]) -> Path:
    if output:
        path = Path(output)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = PROJECT_ROOT / "test" / "results" / f"false_positive_eval_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate false positives on a negative-only image/video set. "
            "Every detection in the provided source is counted as a false positive."
        )
    )
    parser.add_argument("--source", required=True, help="Negative image/video file or directory.")
    parser.add_argument("--weights", default="yolov8n.pt", help="Weight filename or absolute path.")
    parser.add_argument("--classes", default="", help="Comma-separated class filter; empty means all classes.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold.")
    parser.add_argument("--device", default="", help="Optional YOLO device: cpu, cuda, cuda:0.")
    parser.add_argument("--sample-every", type=int, default=30, help="For videos, evaluate every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional max frames/images to evaluate.")
    parser.add_argument("--top-k", type=int, default=25, help="Store top-K highest-confidence false positives.")
    parser.add_argument("--output", default="", help="Output JSON path. Defaults to test/results.")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    source = source.resolve()
    weight_path = resolve_weight(args.weights)
    device = args.device.strip() or None
    class_filter = parse_classes(args.classes)
    max_frames = args.max_frames if args.max_frames > 0 else None

    model = load_model(weight_path, device)

    frames_total = 0
    fp_frames_total = 0
    fp_detections_total = 0
    detections_by_class: Counter[str] = Counter()
    confidences: List[float] = []
    latencies_ms: List[float] = []
    examples: List[Dict[str, object]] = []

    started = time.perf_counter()
    for frame_label, image in iter_frames(source, args.sample_every, max_frames):
        frames_total += 1
        infer_started = time.perf_counter()
        raw_detections = run_detection(model, image, args.conf, args.iou, device)
        latencies_ms.append((time.perf_counter() - infer_started) * 1000.0)

        filtered = []
        for class_name, confidence, bbox in raw_detections:
            normalized = str(class_name).lower()
            if class_filter is not None and normalized not in class_filter:
                continue
            filtered.append((class_name, float(confidence), bbox))

        if filtered:
            fp_frames_total += 1
            fp_detections_total += len(filtered)

        for class_name, confidence, bbox in filtered:
            class_name = str(class_name)
            detections_by_class[class_name] += 1
            confidences.append(float(confidence))
            examples.append(
                {
                    "source": frame_label,
                    "class_name": class_name,
                    "confidence": round(float(confidence), 4),
                    "bbox": [int(v) for v in bbox],
                }
            )

    elapsed_sec = time.perf_counter() - started
    if frames_total == 0:
        raise RuntimeError(f"No readable frames/images found under: {source}")

    examples.sort(key=lambda item: float(item["confidence"]), reverse=True)
    output_path = make_output_path(args.output)

    false_positive_frame_rate = fp_frames_total / frames_total
    result = {
        "schema_version": 1,
        "test_type": "negative_set_false_positive_eval",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(source),
        "weights": str(weight_path),
        "parameters": {
            "conf": args.conf,
            "iou": args.iou,
            "device": device or "model-default",
            "classes": sorted(class_filter) if class_filter else "all",
            "sample_every": args.sample_every,
            "max_frames": max_frames,
        },
        "summary": {
            "frames_total": frames_total,
            "frames_with_false_positive": fp_frames_total,
            "false_positive_detections_total": fp_detections_total,
            "false_positive_frame_rate": round(false_positive_frame_rate, 6),
            "false_positives_per_1000_frames": round(fp_detections_total / frames_total * 1000.0, 3),
            "mean_false_positive_confidence": round(statistics.mean(confidences), 6) if confidences else 0.0,
            "p95_false_positive_confidence": round(percentile(confidences, 0.95), 6),
            "detection_only_avg_fps": round(frames_total / elapsed_sec, 3) if elapsed_sec > 0 else 0.0,
            "latency_avg_ms": round(statistics.mean(latencies_ms), 3) if latencies_ms else 0.0,
            "latency_p50_ms": round(percentile(latencies_ms, 0.50), 3),
            "latency_p95_ms": round(percentile(latencies_ms, 0.95), 3),
        },
        "detections_by_class": dict(detections_by_class),
        "top_false_positive_examples": examples[: max(0, args.top_k)],
    }

    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output_path), **result["summary"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
