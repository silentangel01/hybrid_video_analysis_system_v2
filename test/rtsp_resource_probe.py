#!/usr/bin/env python
"""Probe multi-RTSP runtime metrics plus host CPU/GPU usage.

This script does not modify backend code. It talks to the existing REST API:
  GET  /api/streams
  GET  /api/streams/<stream_id>/metrics
  POST /api/streams
  DELETE /api/streams/<stream_id>

CPU/memory data uses optional psutil when available. GPU data uses nvidia-smi
when the NVIDIA CLI is present.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def import_psutil():
    try:
        import psutil  # type: ignore

        return psutil
    except Exception:
        return None


def request_json(method: str, url: str, body: Optional[dict] = None, timeout: float = 10.0):
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc


def api_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def add_stream(base_url: str, rtsp_url: str, tasks: List[str], camera_id: str) -> str:
    payload = {"url": rtsp_url, "tasks": tasks, "camera_id": camera_id}
    data = request_json("POST", api_url(base_url, "/api/streams"), payload)
    return str(data["stream_id"])


def delete_stream(base_url: str, stream_id: str) -> None:
    request_json("DELETE", api_url(base_url, f"/api/streams/{stream_id}"))


def get_streams(base_url: str) -> List[dict]:
    data = request_json("GET", api_url(base_url, "/api/streams"))
    return data if isinstance(data, list) else []


def get_stream_metrics(base_url: str, stream_id: str) -> dict:
    data = request_json("GET", api_url(base_url, f"/api/streams/{stream_id}/metrics"))
    return data if isinstance(data, dict) else {}


def find_backend_pid(psutil_module) -> Optional[int]:
    if psutil_module is None:
        return None
    markers = ("backend.main", "backend\\main.py", "backend/main.py")
    for proc in psutil_module.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if any(marker in cmdline for marker in markers):
                return int(proc.info["pid"])
        except Exception:
            continue
    return None


def query_gpu() -> List[Dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=3, check=True)
    except Exception:
        return []

    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 8:
            continue
        rows.append(
            {
                "index": int(float(parts[0])),
                "name": parts[1],
                "gpu_util_percent": float(parts[2]),
                "memory_util_percent": float(parts[3]),
                "memory_used_mb": float(parts[4]),
                "memory_total_mb": float(parts[5]),
                "temperature_c": float(parts[6]),
                "power_draw_w": safe_float(parts[7]),
            }
        )
    return rows


def safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def collect_resource_sample(psutil_module, backend_pid: Optional[int]) -> Dict[str, Any]:
    sample: Dict[str, Any] = {"system": {}, "process": {}, "gpu": query_gpu(), "notes": []}
    if psutil_module is None:
        sample["notes"].append("psutil not installed; CPU and memory probes are limited.")
        return sample

    try:
        memory = psutil_module.virtual_memory()
        sample["system"] = {
            "cpu_percent": psutil_module.cpu_percent(interval=None),
            "memory_percent": memory.percent,
            "memory_used_mb": round(memory.used / 1024 / 1024, 2),
            "memory_total_mb": round(memory.total / 1024 / 1024, 2),
        }
    except Exception as exc:
        sample["notes"].append(f"system probe failed: {exc}")

    if backend_pid is not None:
        try:
            proc = psutil_module.Process(int(backend_pid))
            mem = proc.memory_info()
            sample["process"] = {
                "pid": int(backend_pid),
                "cpu_percent": proc.cpu_percent(interval=None),
                "rss_mb": round(mem.rss / 1024 / 1024, 2),
                "num_threads": proc.num_threads(),
            }
        except Exception as exc:
            sample["notes"].append(f"process probe failed for pid={backend_pid}: {exc}")
    return sample


def nested_get(data: dict, path: List[str], default=None):
    current = data
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def compact_stream_row(stream: dict) -> Dict[str, Any]:
    metrics = stream.get("metrics", {})
    capture = metrics.get("capture", {})
    executor = metrics.get("executor", {})
    tasks = metrics.get("tasks", {})
    parking = tasks.get("parking_violation", {})
    smoke = tasks.get("smoke_flame", {})
    return {
        "stream_id": stream.get("stream_id"),
        "status": stream.get("status"),
        "capture_fps_10s": capture.get("capture_fps_10s"),
        "emit_fps_10s": capture.get("emit_fps_10s"),
        "executor_queue_size": executor.get("queue_size"),
        "executor_inflight_tasks": executor.get("inflight_tasks"),
        "parking_yolo_recent_avg_ms": nested_get(parking, ["yolo_latency", "recent_avg_ms"], 0),
        "parking_pipeline_recent_avg_ms": nested_get(parking, ["pipeline_latency", "recent_avg_ms"], 0),
        "smoke_yolo_recent_avg_ms": nested_get(smoke, ["yolo_latency", "recent_avg_ms"], 0),
        "smoke_qwen_recent_avg_ms": nested_get(smoke, ["qwen_latency", "recent_avg_ms"], 0),
        "smoke_detection_queue_size": smoke.get("detection_queue_size"),
        "smoke_verification_queue_size": smoke.get("verification_queue_size"),
    }


def avg(values: List[float]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 3)


def maximum(values: List[float]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return round(max(clean), 3)


def summarize(samples: List[dict]) -> Dict[str, Any]:
    system_cpu = [nested_get(s, ["resources", "system", "cpu_percent"]) for s in samples]
    process_cpu = [nested_get(s, ["resources", "process", "cpu_percent"]) for s in samples]
    gpu_util = []
    gpu_mem = []
    for sample in samples:
        for gpu in nested_get(sample, ["resources", "gpu"], []):
            gpu_util.append(gpu.get("gpu_util_percent"))
            gpu_mem.append(gpu.get("memory_used_mb"))

    per_stream: Dict[str, Dict[str, List[float]]] = {}
    for sample in samples:
        for row in sample.get("stream_rows", []):
            sid = str(row.get("stream_id"))
            bucket = per_stream.setdefault(sid, {})
            for key, value in row.items():
                if key in {"stream_id", "status"}:
                    continue
                bucket.setdefault(key, []).append(value)

    return {
        "sample_count": len(samples),
        "system_cpu_percent_avg": avg(system_cpu),
        "system_cpu_percent_max": maximum(system_cpu),
        "backend_process_cpu_percent_avg": avg(process_cpu),
        "backend_process_cpu_percent_max": maximum(process_cpu),
        "gpu_util_percent_avg": avg(gpu_util),
        "gpu_util_percent_max": maximum(gpu_util),
        "gpu_memory_used_mb_max": maximum(gpu_mem),
        "streams": {
            sid: {
                f"{key}_avg": avg(values)
                for key, values in metrics.items()
                if avg(values) is not None
            }
            | {
                f"{key}_max": maximum(values)
                for key, values in metrics.items()
                if maximum(values) is not None
            }
            for sid, metrics in per_stream.items()
        },
    }


def write_csv(path: Path, samples: List[dict]) -> None:
    rows = []
    for sample in samples:
        timestamp = sample["timestamp"]
        resources = sample.get("resources", {})
        system = resources.get("system", {})
        process = resources.get("process", {})
        gpu_list = resources.get("gpu", [])
        first_gpu = gpu_list[0] if gpu_list else {}
        for stream_row in sample.get("stream_rows", []):
            rows.append(
                {
                    "timestamp": timestamp,
                    "system_cpu_percent": system.get("cpu_percent"),
                    "system_memory_percent": system.get("memory_percent"),
                    "backend_process_cpu_percent": process.get("cpu_percent"),
                    "backend_process_rss_mb": process.get("rss_mb"),
                    "gpu_util_percent": first_gpu.get("gpu_util_percent"),
                    "gpu_memory_used_mb": first_gpu.get("memory_used_mb"),
                    **stream_row,
                }
            )
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_output_dir(output_dir: Optional[str]) -> Path:
    if output_dir:
        path = Path(output_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = PROJECT_ROOT / "test" / "results" / f"rtsp_resource_probe_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect multi-RTSP runtime, CPU, and GPU metrics.")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000", help="Backend API base URL.")
    parser.add_argument("--rtsp-url", action="append", default=[], help="RTSP URL to create. Repeat for N streams.")
    parser.add_argument("--tasks", default="smoke_flame", help="Comma-separated task list for created streams.")
    parser.add_argument("--camera-id-prefix", default="perf_cam", help="Camera ID prefix for created streams.")
    parser.add_argument("--duration", type=float, default=60.0, help="Probe duration in seconds.")
    parser.add_argument("--interval", type=float, default=5.0, help="Sampling interval in seconds.")
    parser.add_argument("--backend-pid", default="", help="Backend PID, or 'auto' when psutil is installed.")
    parser.add_argument("--no-create", action="store_true", help="Only monitor currently running streams.")
    parser.add_argument("--keep-streams", action="store_true", help="Do not delete streams created by this probe.")
    parser.add_argument("--output-dir", default="", help="Output directory. Defaults to test/results.")
    args = parser.parse_args()

    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    if not tasks:
        raise ValueError("--tasks must contain at least one task")

    psutil_module = import_psutil()
    backend_pid: Optional[int] = None
    if args.backend_pid.strip().lower() == "auto":
        backend_pid = find_backend_pid(psutil_module)
    elif args.backend_pid.strip():
        backend_pid = int(args.backend_pid)

    if psutil_module is not None:
        psutil_module.cpu_percent(interval=None)
        if backend_pid is not None:
            try:
                psutil_module.Process(backend_pid).cpu_percent(interval=None)
            except Exception:
                pass

    output_dir = make_output_dir(args.output_dir or None)
    created_stream_ids: List[str] = []
    samples: List[dict] = []

    try:
        if not args.no_create:
            for index, url in enumerate(args.rtsp_url, start=1):
                sid = add_stream(
                    args.base_url,
                    url,
                    tasks,
                    f"{args.camera_id_prefix}_{index}",
                )
                created_stream_ids.append(sid)
                print(f"created stream: {sid}")

        deadline = time.perf_counter() + max(1.0, args.duration)
        jsonl_path = output_dir / "samples.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as jsonl:
            while time.perf_counter() < deadline:
                streams = get_streams(args.base_url)
                if created_stream_ids:
                    stream_ids = set(created_stream_ids)
                    streams = [stream for stream in streams if stream.get("stream_id") in stream_ids]

                detailed_streams = []
                for stream in streams:
                    sid = stream.get("stream_id")
                    if sid:
                        detail = get_stream_metrics(args.base_url, str(sid))
                        detailed_streams.append(detail or stream)

                sample = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "resources": collect_resource_sample(psutil_module, backend_pid),
                    "streams": detailed_streams,
                    "stream_rows": [compact_stream_row(stream) for stream in detailed_streams],
                }
                samples.append(sample)
                jsonl.write(json.dumps(sample, ensure_ascii=False) + "\n")
                jsonl.flush()
                print(json.dumps({"timestamp": sample["timestamp"], "stream_rows": sample["stream_rows"]}))
                time.sleep(max(0.5, args.interval))
    finally:
        if created_stream_ids and not args.keep_streams:
            for sid in created_stream_ids:
                try:
                    delete_stream(args.base_url, sid)
                    print(f"deleted stream: {sid}")
                except Exception as exc:
                    print(f"failed to delete stream {sid}: {exc}", file=sys.stderr)

    summary = summarize(samples)
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "samples.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_path, samples)
    print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
