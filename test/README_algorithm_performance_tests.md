# 算法与性能测试设计

## 已确认的现有接口

- YOLO 原始推理入口：`ml_models/yolov8/inference.py` 的 `YOLOInference.run_detection(model, image, conf_threshold, iou_threshold, device)`。
- YOLO 模型加载入口：`ml_models/yolov8/model_loader.py` 的 `YOLOModelLoader`，支持 `YOLO_DEVICE=cpu/cuda/cuda:0`。
- RTSP/本地视频采集指标：`backend/utils/frame_capture.py` 已记录 `capture_fps_10s`、`emit_fps_10s`、`frames_read_total`、`frames_emitted_total`、`frames_skipped_total`、断流/重连计数。
- 流运行时指标接口：`GET /api/streams` 和 `GET /api/streams/<stream_id>/metrics` 已暴露每条流的采集、任务执行器、队列和各任务指标。
- 停车违规任务指标：`ViolationDetectionService.get_runtime_metrics()` 已有 `input_fps_10s`、`yolo_latency`、`filter_latency`、`pipeline_latency`、检测/违规/事件计数。
- 烟火任务指标：`SmokeFlameDetectionService.get_runtime_metrics()` 已有 `received_fps_10s`、`submitted_fps_10s`、`yolo_latency`、`qwen_latency`、检测队列、复核队列、候选帧/事件计数。
- 公共空间任务指标：`CommonSpaceDetectionService.get_runtime_metrics()` 已有采样 FPS、分析队列、Qwen 分析延迟和事件计数。

## 当前缺口

- 没有标注数据集、标注格式或算法评测 API，因此无法在业务代码内直接计算标准 precision/recall/FPR。当前脚本先支持“负样本集误报率”：负样本中每一个检测结果都计为 false positive。
- 没有服务端 CPU/GPU 占用接口。`rtsp_resource_probe.py` 会在测试侧用可选 `psutil` 和 `nvidia-smi` 采集，不需要改后端。
- 没有“YOLO 最大平均 FPS”的后端接口。`yolo_fps_benchmark.py` 直接调用项目现有 YOLO 推理封装，测 detection-only 吞吐。

## 测试 1：误报率/假阳性

目标：用一批确定“不应出现目标”的负样本图片或视频帧，统计模型误报情况。

建议数据：

- 烟火模型：无烟、无火、夜间灯光、雾气、云、反光、厨房/车灯等容易误判的负样本。
- 车辆模型：无车区域、人、自行车、广告牌、树影、反光玻璃、空停车场等负样本。

示例命令：

```powershell
.\venv\Scripts\python.exe test\algorithm_false_positive_eval.py `
  --source test\data\negative_smoke `
  --weights smoke_flame.pt `
  --classes smoke,fire,flame `
  --conf 0.10
```

输出指标：

- `false_positive_frame_rate`：出现至少一个误报的帧占比。
- `false_positives_per_1000_frames`：每 1000 帧误报检测框数。
- `detections_by_class`：各类别误报数量。
- `top_false_positive_examples`：高置信度误报样本，便于人工复核。

## 测试 2：YOLO 最大平均 FPS

目标：排除 RTSP、数据库、MinIO、Qwen 等影响，单独测 YOLO 推理上限。

示例命令：

```powershell
.\venv\Scripts\python.exe test\yolo_fps_benchmark.py `
  --source test\data\benchmark_frames `
  --weights yolov8n.pt `
  --device cpu `
  --duration 30 `
  --warmup 5
```

如需测 GPU：

```powershell
.\venv\Scripts\python.exe test\yolo_fps_benchmark.py `
  --source test\data\benchmark_frames `
  --weights yolov8n.pt `
  --device cuda `
  --duration 60 `
  --warmup 10
```

输出指标：

- `avg_fps`：检测路径最大平均 FPS。
- `latency_avg_ms`、`latency_p50_ms`、`latency_p95_ms`、`latency_p99_ms`。
- `device_info`：Torch/CUDA 可用性、GPU 名称、峰值显存等。

## 测试 3：YOLO-only vs YOLO+Qwen-VL 必要性

目标：在同一个负样本视频/图片集上，对比纯 YOLO 初筛误报，以及经过 Qwen-VL yes/no 复核后仍保留的误报。

示例命令：

```powershell
.\venv\Scripts\python.exe test\qwen_necessity_eval.py `
  --source test\data\negative_smoke `
  --weights smoke_flame.pt `
  --classes smoke,fire,flame `
  --conf 0.10 `
  --sample-every 30 `
  --qwen-workers 2
```

如果希望保存每个候选框裁剪图，便于人工观察 Qwen 判断对象：

```powershell
.\venv\Scripts\python.exe test\qwen_necessity_eval.py `
  --source test\data\negative_smoke `
  --weights smoke_flame.pt `
  --classes smoke,fire,flame `
  --conf 0.10 `
  --sample-every 30 `
  --save-crops
```

输出目录类似 `test\results\qwen_necessity_eval_YYYYMMDD_HHMMSS`，重点看：

- `summary.json`：A/B 汇总。
- `details.json`：每帧、每个候选框、Qwen 回答和示例。
- `yolo_only_false_positive_frame_rate`：纯 YOLO 误报帧率。
- `yolo_qwen_false_positive_frame_rate`：Qwen 复核后仍误报的帧率。
- `detection_reduction.relative_reduction_rate`：Qwen 对误报框数量的削减比例。
- `frame_rate_reduction.relative_reduction_rate`：Qwen 对误报帧数量的削减比例。
- `qwen_errors_total`：Qwen 调用错误数；如果大于 0，先检查配置/网络，再解释结果。

## 测试 4：多 RTSP 流资源占用

目标：在 1/2/4/8 条 RTSP 流下，持续采集后端 API 指标、系统 CPU、进程内存、GPU 利用率。

前置条件：

- 后端已启动并监听 `http://127.0.0.1:5000`。
- RTSP 推流源已经可播放。
- 如果要采 CPU/内存，建议安装 `psutil`；如果要采 GPU，机器需有 `nvidia-smi`。

示例命令：

```powershell
.\venv\Scripts\python.exe test\rtsp_resource_probe.py `
  --base-url http://127.0.0.1:5000 `
  --rtsp-url rtsp://127.0.0.1:8554/cam1 `
  --rtsp-url rtsp://127.0.0.1:8554/cam2 `
  --tasks smoke_flame `
  --duration 120 `
  --interval 5 `
  --backend-pid auto
```

说明：

- 默认任务用 `smoke_flame`，避免 `parking_violation` 在没有禁停区配置时触发画线 GUI。
- 如果测试停车违规任务，请先准备好对应 `camera_id` 的禁停区配置，再将 `--tasks parking_violation` 或 `--tasks parking_violation,smoke_flame`。
- 脚本默认会删除它创建的流；需要保留时加 `--keep-streams`。

输出文件：

- `samples.jsonl`：每次采样的原始 API/资源指标。
- `samples.csv`：便于画图的扁平表。
- `summary.json`：CPU/GPU 峰值与均值、每条流平均 FPS/延迟/队列等汇总。

## 建议的验收阈值

阈值需要结合硬件、模型和场景确定，建议先用本目录脚本跑一次基线，再固化为项目验收标准：

- 负样本误报帧率：烟火模型建议先观察 `conf=0.10` 和 `conf=0.25` 两档，最终目标可设为核心负样本集 `< 5%`。
- YOLO detection-only FPS：记录 CPU 和 GPU 两组基线，后续模型或阈值调整不应低于基线的 `90%`。
- 多 RTSP 稳定性：持续 10-30 分钟，`executor_queue_size`、`smoke_detection_queue_size`、`smoke_verification_queue_size` 不应持续增长。
- 端到端延迟：关注 `/api/streams/<id>/metrics` 中 `yolo_latency.recent_avg_ms` 与 `pipeline_latency.recent_avg_ms`，按实际业务 SLA 固化阈值。

## 下一步建议

1. 建立固定评测集：`test/data/negative_*`、`test/data/positive_*`，并保存样本来源、采集时间、场景说明。
2. 如果要计算标准 precision/recall/F1，需要补一份标注规范，例如 COCO JSON 或 YOLO txt；之后可在 `test` 目录继续扩展成有标注的评测脚本。
3. 如果希望前端或 API 直接展示 CPU/GPU，占用率需要新增后端资源监控模块和接口；这一步会改业务代码，所以本次先不动。
