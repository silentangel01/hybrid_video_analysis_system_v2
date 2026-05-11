#!/usr/bin/env python
"""Compare YOLO-only vs YOLO+Qwen-VL verification on negative video/images.

This is meant to answer a narrow question:
  "On the same negative samples, how many YOLO false positives remain after
   Qwen-VL yes/no verification?"

It uses existing project components where possible:
  - YOLOModelLoader
  - YOLOInference.run_detection
  - SmokeFlameDetectionService crop/dedup helpers
  - QwenVLAPIClient payload/response parsing helpers
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "ml_models" / "yolov8" / "weights"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


def add_project_root() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False, encoding="utf-8")
    except Exception:
        return


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


def parse_classes(raw: str) -> Optional[set[str]]:
    items = [item.strip().lower() for item in raw.split(",") if item.strip()]
    return set(items) if items else None


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
    if not loader.load_model("qwen_ab_eval", weight_path.name):
        raise RuntimeError(f"Failed to load YOLO model: {weight_path}")
    return loader.get_model("qwen_ab_eval")


def build_smoke_service_helpers():
    add_project_root()
    from backend.services.smoke_flame_detection import SmokeFlameDetectionService

    return SmokeFlameDetectionService()


def build_qwen_client(args):
    add_project_root()
    load_dotenv_if_available()

    from backend.config.qwen_vl_config import qwen_vl_api_config
    from backend.services.smoke_flame_detection import QwenVLAPIClient

    api_url = args.qwen_api_url or os.getenv("QWEN_VL_API_URL") or qwen_vl_api_config.get_api_url()
    api_key = args.qwen_api_key or os.getenv("QWEN_VL_API_KEY") or qwen_vl_api_config.get_api_key()
    model_name = args.qwen_model or os.getenv("QWEN_VL_MODEL_NAME") or qwen_vl_api_config.get_model_name()
    timeout = args.qwen_timeout or int(os.getenv("QWEN_VL_TIMEOUT", "0") or 0) or qwen_vl_api_config.get_timeout()

    if not api_url or not api_key:
        raise RuntimeError(
            "Qwen-VL is not configured. Set QWEN_VL_API_URL/QWEN_VL_API_KEY, "
            "or pass --qwen-api-url and --qwen-api-key."
        )

    client = QwenVLAPIClient(api_url=api_url, api_key=api_key, model_name=model_name)
    client.timeout = int(timeout)
    client.verify_prompt = args.qwen_prompt
    return client, {"api_url": api_url, "model_name": model_name, "timeout": int(timeout)}


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


def run_yolo(model, image: np.ndarray, conf: float, iou: float, device: Optional[str]):
    add_project_root()
    from ml_models.yolov8.inference import YOLOInference

    return YOLOInference.run_detection(
        model,
        image,
        conf_threshold=conf,
        iou_threshold=iou,
        device=device,
    )


def normalize_detections(
    raw_detections,
    image_shape: Tuple[int, ...],
    class_filter: Optional[set[str]],
    min_area_ratio: float,
) -> List[Dict[str, Any]]:
    h, w = image_shape[:2]
    results: List[Dict[str, Any]] = []
    for class_name, confidence, bbox in raw_detections:
        normalized = str(class_name).lower()
        if class_filter is not None and normalized not in class_filter:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        area_ratio = max(0, x2 - x1) * max(0, y2 - y1) / max(1, h * w)
        if area_ratio < min_area_ratio:
            continue
        results.append(
            {
                "class_name": str(class_name),
                "confidence": float(confidence),
                "bbox": (x1, y1, x2, y2),
                "detection_stage": "yolo_initial",
                "area_ratio": area_ratio,
            }
        )
    results.sort(key=lambda item: item["confidence"], reverse=True)
    return results


def make_output_dir(output_dir: Optional[str]) -> Path:
    if output_dir:
        path = Path(output_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = PROJECT_ROOT / "test" / "results" / f"qwen_necessity_eval_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_crop(crop_dir: Path, prefix: str, crop: np.ndarray) -> Optional[str]:
    crop_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in prefix)[:160]
    path = crop_dir / f"{safe_name}.jpg"
    ok = cv2.imwrite(str(path), crop)
    return str(path) if ok else None


def verify_with_qwen(client, crop: np.ndarray) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return {
                "positive": False,
                "answer": "",
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "error": "failed_to_encode_crop",
            }

        image_b64 = base64.b64encode(encoded).decode("utf-8")
        payload = client._build_payload(image_b64)
        headers = {
            "Authorization": f"Bearer {client.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(client.api_url, json=payload, headers=headers, timeout=client.timeout)
        response.raise_for_status()
        answer = client._extract_text(response.json())
        positive = bool(client._is_positive(answer))
        return {
            "positive": positive,
            "answer": answer,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "error": "",
        }
    except Exception as exc:
        return {
            "positive": False,
            "answer": "",
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "error": str(exc),
        }


def summarize_binary_negative_mode(prefix: str, frame_count: int, frame_positive_count: int, detection_count: int):
    return {
        f"{prefix}_frames_with_false_positive": frame_positive_count,
        f"{prefix}_false_positive_detections_total": detection_count,
        f"{prefix}_false_positive_frame_rate": round(frame_positive_count / frame_count, 6)
        if frame_count
        else 0.0,
        f"{prefix}_false_positives_per_1000_frames": round(detection_count / frame_count * 1000.0, 3)
        if frame_count
        else 0.0,
    }


def reduction(before: int, after: int) -> Dict[str, Any]:
    removed = before - after
    return {
        "absolute_reduction": removed,
        "relative_reduction_rate": round(removed / before, 6) if before > 0 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare YOLO-only false positives with YOLO+Qwen-VL verification on negative samples."
    )
    parser.add_argument("--source", required=True, help="Negative image/video file or directory.")
    parser.add_argument("--weights", default="smoke_flame.pt", help="Weight filename or absolute path.")
    parser.add_argument("--classes", default="smoke,fire,flame", help="Comma-separated YOLO class filter.")
    parser.add_argument("--conf", type=float, default=0.10, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold.")
    parser.add_argument("--device", default="", help="Optional YOLO device: cpu, cuda, cuda:0.")
    parser.add_argument("--sample-every", type=int, default=30, help="For videos, evaluate every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional max frames/images to evaluate.")
    parser.add_argument("--min-area-ratio", type=float, default=0.0001, help="Drop tiny YOLO boxes below this frame ratio.")
    parser.add_argument("--qwen-max-candidates-per-frame", type=int, default=8, help="Mirror production top-N Qwen verification.")
    parser.add_argument("--qwen-workers", type=int, default=2, help="Concurrent Qwen verification calls.")
    parser.add_argument("--qwen-api-url", default="", help="Override Qwen-VL API URL.")
    parser.add_argument("--qwen-api-key", default="", help="Override Qwen-VL API key. Never written to result files.")
    parser.add_argument("--qwen-model", default="", help="Override Qwen-VL model name.")
    parser.add_argument("--qwen-timeout", type=int, default=0, help="Override Qwen-VL timeout seconds.")
    parser.add_argument(
        "--qwen-prompt",
        default=(
            "Look carefully at this cropped detection region. "
            "Is there real smoke or real flame/fire visible? "
            "Answer only yes or no."
        ),
        help="Qwen verification prompt. Keep the answer constrained to yes/no.",
    )
    parser.add_argument("--top-k", type=int, default=25, help="Store top-K examples for each group.")
    parser.add_argument("--save-crops", action="store_true", help="Save candidate crops under the result directory.")
    parser.add_argument("--output-dir", default="", help="Output directory. Defaults to test/results.")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    source = source.resolve()

    weight_path = resolve_weight(args.weights)
    device = args.device.strip() or None
    class_filter = parse_classes(args.classes)
    max_frames = args.max_frames if args.max_frames > 0 else None
    output_dir = make_output_dir(args.output_dir or None)
    crop_dir = output_dir / "crops"

    model = load_model(weight_path, device)
    helper_service = build_smoke_service_helpers()
    qwen_client, qwen_public_config = build_qwen_client(args)

    frames_total = 0
    yolo_positive_frames = 0
    qwen_positive_frames = 0
    yolo_fp_detections_total = 0
    qwen_fp_detections_total = 0
    yolo_latency_ms: List[float] = []
    qwen_latency_ms: List[float] = []
    yolo_class_counts: Counter[str] = Counter()
    qwen_answer_counts: Counter[str] = Counter()
    qwen_errors: List[str] = []
    yolo_positive_examples: List[Dict[str, Any]] = []
    qwen_positive_examples: List[Dict[str, Any]] = []
    qwen_rejected_examples: List[Dict[str, Any]] = []
    per_frame_rows: List[Dict[str, Any]] = []

    started = time.perf_counter()
    for frame_label, image in iter_frames(source, args.sample_every, max_frames):
        frames_total += 1

        yolo_started = time.perf_counter()
        raw = run_yolo(model, image, args.conf, args.iou, device)
        yolo_latency_ms.append((time.perf_counter() - yolo_started) * 1000.0)

        candidates = normalize_detections(raw, image.shape, class_filter, args.min_area_ratio)
        candidates = helper_service._filter_duplicates(candidates, image.shape)
        candidates.sort(key=lambda item: item["confidence"], reverse=True)

        yolo_count = len(candidates)
        yolo_fp_detections_total += yolo_count
        if yolo_count:
            yolo_positive_frames += 1

        for index, det in enumerate(candidates):
            yolo_class_counts[det["class_name"]] += 1
            yolo_positive_examples.append(
                {
                    "source": frame_label,
                    "candidate_index": index,
                    "class_name": det["class_name"],
                    "confidence": round(float(det["confidence"]), 4),
                    "bbox": [int(v) for v in det["bbox"]],
                }
            )

        verify_candidates = candidates[: max(0, args.qwen_max_candidates_per_frame)]
        verification_results: List[Dict[str, Any]] = []

        def verify_one(item):
            index, det = item
            crop = helper_service._crop_region(image, det["bbox"])
            if crop is None:
                return index, det, {
                    "positive": False,
                    "answer": "",
                    "latency_ms": 0.0,
                    "error": "invalid_crop",
                }, None
            result = verify_with_qwen(qwen_client, crop)
            return index, det, result, crop

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.qwen_workers)) as pool:
            futures = [pool.submit(verify_one, item) for item in enumerate(verify_candidates)]
            for future in concurrent.futures.as_completed(futures):
                index, det, result, crop = future.result()
                qwen_latency_ms.append(float(result.get("latency_ms") or 0.0))
                answer = str(result.get("answer") or "").strip().lower()
                error = str(result.get("error") or "")
                if answer:
                    qwen_answer_counts[answer] += 1
                if error:
                    qwen_errors.append(error)

                example = {
                    "source": frame_label,
                    "candidate_index": index,
                    "class_name": det["class_name"],
                    "confidence": round(float(det["confidence"]), 4),
                    "bbox": [int(v) for v in det["bbox"]],
                    "qwen_positive": bool(result.get("positive")),
                    "qwen_answer": answer,
                    "qwen_latency_ms": result.get("latency_ms"),
                    "qwen_error": error,
                }
                if args.save_crops and crop is not None:
                    example["crop_path"] = save_crop(
                        crop_dir,
                        f"frame_{frames_total}_candidate_{index}_{det['class_name']}_{det['confidence']:.3f}",
                        crop,
                    )
                verification_results.append(example)

        qwen_positives = [item for item in verification_results if item["qwen_positive"]]
        qwen_rejected = [item for item in verification_results if not item["qwen_positive"]]
        qwen_count = len(qwen_positives)
        qwen_fp_detections_total += qwen_count
        if qwen_count:
            qwen_positive_frames += 1
        qwen_positive_examples.extend(qwen_positives)
        qwen_rejected_examples.extend(qwen_rejected)

        per_frame_rows.append(
            {
                "source": frame_label,
                "yolo_candidate_count": yolo_count,
                "qwen_verified_positive_count": qwen_count,
                "qwen_rejected_count": len(qwen_rejected),
            }
        )

    elapsed_sec = time.perf_counter() - started
    if frames_total == 0:
        raise RuntimeError(f"No readable frames/images found under: {source}")

    yolo_positive_examples.sort(key=lambda item: float(item["confidence"]), reverse=True)
    qwen_positive_examples.sort(key=lambda item: float(item["confidence"]), reverse=True)
    qwen_rejected_examples.sort(key=lambda item: float(item["confidence"]), reverse=True)

    summary = {
        "frames_total": frames_total,
        **summarize_binary_negative_mode(
            "yolo_only",
            frames_total,
            yolo_positive_frames,
            yolo_fp_detections_total,
        ),
        **summarize_binary_negative_mode(
            "yolo_qwen",
            frames_total,
            qwen_positive_frames,
            qwen_fp_detections_total,
        ),
        "detection_reduction": reduction(yolo_fp_detections_total, qwen_fp_detections_total),
        "frame_rate_reduction": reduction(yolo_positive_frames, qwen_positive_frames),
        "yolo_latency_avg_ms": round(statistics.mean(yolo_latency_ms), 3) if yolo_latency_ms else 0.0,
        "yolo_latency_p95_ms": round(percentile(yolo_latency_ms, 0.95), 3),
        "qwen_request_latency_avg_ms": round(statistics.mean(qwen_latency_ms), 3) if qwen_latency_ms else 0.0,
        "qwen_request_latency_p95_ms": round(percentile(qwen_latency_ms, 0.95), 3),
        "qwen_requests_total": len(qwen_latency_ms),
        "qwen_errors_total": len(qwen_errors),
        "elapsed_sec": round(elapsed_sec, 3),
    }

    result = {
        "schema_version": 1,
        "test_type": "negative_set_yolo_vs_yolo_qwen_necessity_eval",
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
            "min_area_ratio": args.min_area_ratio,
            "qwen_max_candidates_per_frame": args.qwen_max_candidates_per_frame,
            "qwen_workers": args.qwen_workers,
            "save_crops": args.save_crops,
        },
        "qwen_config": qwen_public_config,
        "summary": summary,
        "yolo_detections_by_class": dict(yolo_class_counts),
        "qwen_answer_counts": dict(qwen_answer_counts),
        "qwen_error_examples": qwen_errors[:10],
        "per_frame": per_frame_rows,
        "top_yolo_false_positive_examples": yolo_positive_examples[: max(0, args.top_k)],
        "top_qwen_remaining_false_positive_examples": qwen_positive_examples[: max(0, args.top_k)],
        "top_qwen_rejected_examples": qwen_rejected_examples[: max(0, args.top_k)],
    }

    summary_path = output_dir / "summary.json"
    details_path = output_dir / "details.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    details_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
