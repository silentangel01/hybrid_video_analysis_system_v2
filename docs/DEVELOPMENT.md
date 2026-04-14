# HVAS 开发文档 (Development Guide)

## 项目概述

**HVAS** (Hybrid Video Analysis System) — 混合视频分析系统，基于 YOLOv8 + Qwen-VL 大模型的智能视频监控分析平台。

- **后端**: Python Flask, 端口 5000
- **前端**: Vue 3 + Vite, 开发端口 5173, 生产端口 8080
- **数据库**: MongoDB (端口 27017)
- **对象存储**: MinIO (API 端口 9000, 控制台 9001)
- **项目路径**: `D:\hybrid_video_analysis_system_v2`

---

## 项目结构

```
hybrid_video_analysis_system_v2/
├── backend/
│   ├── main.py                        # 应用入口
│   ├── api/
│   │   ├── stream_routes.py           # 流管理 API
│   │   ├── event_routes.py            # 事件查询 API
│   │   ├── webhook_routes.py          # Webhook 配置 API
│   │   └── health_routes.py           # 健康检查 API
│   ├── services/
│   │   ├── stream_manager.py          # RTSP 流生命周期管理
│   │   ├── stream_runtime.py          # 单流检测流水线
│   │   ├── violation_detection.py     # 违停检测 (YOLOv8 + 区域判定)
│   │   ├── smoke_flame_detection.py   # 烟火检测 (YOLOv8 + Qwen-VL 二阶段)
│   │   ├── common_space_detection.py  # 公共空间分析 (Qwen-VL)
│   │   ├── event_generator.py         # 事件创建 + Webhook 分发
│   │   ├── webhook_service.py         # Webhook 通知 (HMAC 签名)
│   │   ├── dwell_tracker.py           # 车辆停留时间跟踪
│   │   └── parking_zone_checker.py    # 禁停区域多边形检测
│   ├── models/
│   │   └── event.py                   # EventModel (Pydantic)
│   ├── config/
│   │   └── qwen_vl_config.py          # Qwen-VL API 配置
│   └── utils/
│       └── frame_capture.py           # 视频帧提取
├── frontend/
│   ├── src/
│   │   ├── views/
│   │   │   ├── Dashboard.vue          # 实时仪表盘
│   │   │   ├── EventList.vue          # 事件列表查询
│   │   │   ├── StreamManager.vue      # 流管理界面
│   │   │   └── UploadVideo.vue        # 本地视频上传
│   │   ├── composables/
│   │   │   └── useFireAlert.js        # 火警报警状态管理
│   │   └── router/index.js            # 路由配置
│   ├── server.js                      # Express.js 生产代理服务器
│   └── package.json
├── ml_models/yolov8/
│   ├── model_loader.py                # YOLOv8 模型加载
│   └── inference.py                   # 推理 + 后处理
├── storage/
│   ├── mongodb_client.py              # MongoDB 事件存储
│   └── minio_client.py                # MinIO 图片存储
├── scripts/
│   ├── seed_demo_data.py              # 生成演示数据
│   └── draw_fence_gui.py              # 禁停区域画线工具
├── tests/
│   ├── test_detection_logic.py        # 检测逻辑测试
│   ├── test_storage_integration.py    # 存储集成测试
│   └── test_video_ingestion.py        # 视频采集测试
├── docs/
├── docker-compose.yml
└── requirements.txt
```

---

## 快速启动

### 1. 启动基础设施

```bash
cd D:\hybrid_video_analysis_system_v2
docker-compose up -d    # 启动 MongoDB + MinIO
```

### 2. 安装 Python 依赖

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 配置环境变量

创建 `.env` 文件:

```bash
# 存储
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=video_analysis_db
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=video-events

# 检测参数
FRAME_INTERVAL=1.0
COMMON_SPACE_INTERVAL=30.0
DWELL_THRESHOLD=5

# Qwen-VL (可选, 不配置则禁用烟火二阶段验证和公共空间分析)
QWEN_VL_API_URL=https://dashscope.aliyuncs.com/api/v1/services/aigc/video-understanding/video-generation
QWEN_VL_API_KEY=sk-xxxxxxxx
QWEN_VL_MODEL_NAME=qwen-vl-plus
QWEN_VL_TIMEOUT=30

# Webhook
WEBHOOK_SECRET=hvas-mubs-shared-secret

# 演示模式
DEMO_MODE=true
```

### 4. 启动后端

```bash
cd backend
python main.py    # 启动在 http://localhost:5000
```

### 5. 启动前端

```bash
cd frontend
npm install
npm run dev       # 开发模式 http://localhost:5173
# 或
npm run build && npm run start-server  # 生产模式 http://localhost:8080
```

### 6. 验证

```bash
curl http://localhost:5000/api/health
```

---

## API 接口

### 流管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/streams` | 获取所有活跃流 |
| POST | `/api/streams` | 添加 RTSP 流 |
| DELETE | `/api/streams/<stream_id>` | 移除流 |
| PUT | `/api/streams/<stream_id>/tasks` | 更新检测任务 |
| GET | `/api/streams/<stream_id>/metrics` | 获取运行指标 |

**添加流 (POST /api/streams)**:
```json
{
  "url": "rtsp://camera-ip:554/stream",
  "tasks": ["parking_violation", "smoke_flame"],
  "camera_id": "east_gate_01",
  "lat_lng": "31.2304, 121.4737",
  "location": "东门入口",
  "area_code": "east_district",
  "group": "fire_team"
}
```

### 事件查询

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/events` | 查询事件列表 (支持分页/过滤) |
| GET | `/api/events/latest` | 获取最新事件 (流模式) |
| GET | `/api/events/stats` | 获取聚合统计 |
| GET | `/api/events/<event_id>` | 获取事件详情 |
| PATCH | `/api/events/<event_id>/status` | 更新事件状态 (MUBS 反馈) |
| POST | `/api/events/mock` | 创建模拟事件 (需 DEMO_MODE=true) |

**查询参数 (GET /api/events)**:
- `camera_id` — 按摄像头过滤
- `event_type` — `parking_violation` / `smoke_flame` / `common_space_utilization`
- `since_id` — 增量同步 (返回 _id > since_id 的事件)
- `start_time` / `end_time` — 时间范围 (Unix 时间戳)
- `limit` (1-500, 默认 50) / `skip` — 分页

**事件状态反馈 (PATCH /api/events/<id>/status)**:
```json
{
  "status": "resolved",
  "handled_by": "fieldworker_zhang",
  "handle_note": "火已扑灭，无损失",
  "handle_image_url": "http://..."
}
```
状态值: `pending`, `dispatched`, `processing`, `resolved`, `rejected`

### Webhook 配置

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/webhooks` | 注册 Webhook |
| GET | `/api/webhooks` | 获取所有 Webhook |
| DELETE | `/api/webhooks/<webhook_id>` | 注销 Webhook |

### Webhook 推送 Payload

HVAS 检测到事件后自动向注册的 URL 推送:

```json
{
  "event_id": "6625a3f1b2c4d5e6f7890123",
  "event_type": "smoke_flame",
  "camera_id": "east_gate_01",
  "timestamp": 1714000000.0,
  "created_at": "2026-04-25T10:00:00Z",
  "confidence": 0.85,
  "image_url": "http://localhost:9000/video-events/...",
  "description": "Smoke/flame detected (conf=0.85)",
  "object_count": 2,
  "lat_lng": "31.2304, 121.4737",
  "location": "East Gate Entrance",
  "area_code": "east_district",
  "group": "fire_team"
}
```

签名: `X-HVAS-Signature` 请求头包含 HMAC-SHA256 十六进制签名 (基于请求体和共享密钥)。

### 健康检查

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 系统状态 |

```json
{
  "status": "ok",
  "mongodb": "connected",
  "minio": "connected",
  "active_streams": 3,
  "models_loaded": ["vehicle", "smoke_flame"],
  "uptime_sec": 3600
}
```

---

## 检测流水线架构

```
RTSP 流 / 本地视频
    ↓
StreamRuntime (帧提取)
    ↓
┌───────────────────────┬──────────────────────┐
│  违停检测              │  烟火检测              │
│  YOLOv8 车辆检测       │  YOLOv8 烟火初检       │
│  + 禁停区域多边形判定   │  + Qwen-VL 二阶段验证   │
│  + 停留时间跟踪        │                       │
└───────────────────────┴──────────────────────┘
                   ↓
           公共空间分析 (Qwen-VL, 30s 周期)
                   ↓
           EventGenerator → MongoDB + MinIO
                   ↓
           WebhookService → MUBS / 其他消费方
```

---

## 环境变量一览

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `MONGO_URI` | 是 | - | MongoDB 连接串 |
| `MONGO_DB_NAME` | 是 | - | 数据库名 |
| `MINIO_ENDPOINT` | 是 | - | MinIO 地址 |
| `MINIO_ACCESS_KEY` | 是 | - | MinIO 访问密钥 |
| `MINIO_SECRET_KEY` | 是 | - | MinIO 秘密密钥 |
| `MINIO_BUCKET` | 是 | - | MinIO 存储桶 |
| `FRAME_INTERVAL` | 否 | 1.0 | 帧采集间隔 (秒) |
| `COMMON_SPACE_INTERVAL` | 否 | 30.0 | 公共空间分析间隔 (秒) |
| `DWELL_THRESHOLD` | 否 | 5 | 停留判定阈值 (秒) |
| `QWEN_VL_API_URL` | 否 | - | Qwen-VL API 地址 |
| `QWEN_VL_API_KEY` | 否 | - | Qwen-VL API 密钥 |
| `WEBHOOK_SECRET` | 否 | - | HMAC 签名密钥 |
| `DEMO_MODE` | 否 | false | 启用模拟事件接口 |
