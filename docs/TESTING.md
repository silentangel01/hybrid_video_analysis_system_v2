# HVAS 测试文档 (Testing Guide)

## 自动化测试文件

```
tests/
├── test_detection_logic.py        # 检测逻辑单元测试 (待实现)
├── test_storage_integration.py    # 存储集成测试 (待实现)
└── test_video_ingestion.py        # 视频采集测试 (待实现)
```

> 注: 上述测试文件为占位文件，尚未编写具体用例。当前以手动 API 测试为主。

## 运行测试

```bash
cd D:\hybrid_video_analysis_system_v2

# 运行全部测试
python -m pytest tests/ -v

# 运行单个测试文件
python -m pytest tests/test_detection_logic.py -v

# 带覆盖率报告
python -m pytest tests/ -v --cov=backend --cov-report=term-missing
```

---

## 手动 API 测试

### 前置条件

```bash
# 确保 Docker 服务已启动 (MongoDB + MinIO)
cd D:\hybrid_video_analysis_system_v2
docker-compose up -d

# 启动后端
cd backend && python main.py
```

后端启动在 `http://localhost:5000`。

---

### 1. 健康检查 (GET /api/health)

```bash
curl http://localhost:5000/api/health
```

期望响应:
```json
{
  "status": "ok",
  "mongodb": "connected",
  "minio": "connected",
  "active_streams": 0,
  "models_loaded": [],
  "uptime_sec": 42
}
```

验证点:
- `status` 为 `ok` (MongoDB 已连接) 或 `degraded` (MongoDB 断开)
- `mongodb` / `minio` 反映实际连接状态
- `active_streams` 为当前活跃流数量
- `models_loaded` 列出已加载的模型 (如 `vehicle`, `smoke_flame`, `qwen_vl`)
- `uptime_sec` 为服务运行秒数

### 2. 添加 RTSP 流 (POST /api/streams)

```bash
curl -X POST http://localhost:5000/api/streams \
  -H "Content-Type: application/json" \
  -d '{
    "url": "rtsp://192.168.1.100:554/stream",
    "tasks": ["smoke_flame"],
    "camera_id": "test_cam_01",
    "lat_lng": "31.2304, 121.4737",
    "location": "East Gate Entrance",
    "area_code": "east_district",
    "group": "fire_team"
  }'
```

期望: `200`, 返回流信息含 `stream_id`

### 3. 查看活跃流 (GET /api/streams)

```bash
curl http://localhost:5000/api/streams
```

期望: 返回数组，包含上一步添加的流，含 `camera_id`, `tasks`, `lat_lng`, `location`, `area_code`, `group` 字段

### 4. 更新流检测任务 (PUT /api/streams/{id}/tasks)

```bash
STREAM_ID="<上一步返回的 stream_id>"
curl -X PUT http://localhost:5000/api/streams/$STREAM_ID/tasks \
  -H "Content-Type: application/json" \
  -d '{"tasks": ["smoke_flame", "parking_violation"]}'
```

期望: `200`, 流的 tasks 更新为两个检测任务

### 5. 获取流运行指标 (GET /api/streams/{id}/metrics)

```bash
curl http://localhost:5000/api/streams/$STREAM_ID/metrics
```

期望: 返回帧率、处理延迟等运行指标

### 6. 移除流 (DELETE /api/streams/{id})

```bash
curl -X DELETE http://localhost:5000/api/streams/$STREAM_ID
```

期望: `200`, 流被移除，后续 GET /api/streams 不再包含该流

---

### 7. 创建模拟事件 (POST /api/events/mock)

需要环境变量 `DEMO_MODE=true`。

```bash
# 指定事件类型
curl -X POST http://localhost:5000/api/events/mock \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "smoke_flame",
    "camera_id": "demo_camera_01",
    "area_code": "east_district",
    "group": "fire_team",
    "location": "East Gate"
  }'
```

期望: `201`
```json
{"message": "Mock event created", "event_type": "smoke_flame", "camera_id": "demo_camera_01"}
```

支持的 event_type: `smoke_flame`, `parking_violation`, `common_space_utilization`

DEMO_MODE 未启用时:
```bash
# 不设置 DEMO_MODE 环境变量
curl -X POST http://localhost:5000/api/events/mock \
  -H "Content-Type: application/json" \
  -d '{"event_type": "smoke_flame"}'
```

期望: `403 {"error": "DEMO_MODE is not enabled"}`

### 8. 查询事件列表 (GET /api/events)

```bash
# 查询所有事件 (默认 limit=50)
curl "http://localhost:5000/api/events"

# 按类型过滤
curl "http://localhost:5000/api/events?event_type=smoke_flame&limit=5"

# 按摄像头过滤
curl "http://localhost:5000/api/events?camera_id=demo_camera_01"

# 按时间范围过滤 (Unix 时间戳)
curl "http://localhost:5000/api/events?start_time=1714000000&end_time=1714100000"

# 分页
curl "http://localhost:5000/api/events?limit=10&skip=20"
```

期望: 返回分页结构
```json
{
  "items": [...],
  "pagination": {
    "limit": 50,
    "skip": 0,
    "returned": 10,
    "total": 80,
    "has_more": true
  },
  "filters": {
    "camera_id": null,
    "event_type": null,
    "detection_stage": null,
    "start_time": null,
    "end_time": null
  }
}
```

### 9. 增量同步 (GET /api/events?since_id=...)

```bash
# 获取第一批事件
RESPONSE=$(curl -s "http://localhost:5000/api/events?limit=5")
# 取最后一条的 _id
LAST_ID=$(echo $RESPONSE | python -c "import sys,json; items=json.load(sys.stdin)['items']; print(items[-1]['_id'] if items else '')")

# 增量获取后续事件
curl "http://localhost:5000/api/events?since_id=$LAST_ID&limit=10"
```

期望:
- 返回 `_id > since_id` 的事件
- 按 `_id` 升序排列 (时间顺序)
- 可与 `camera_id`, `event_type` 等过滤器组合使用

无效 since_id 测试:
```bash
curl "http://localhost:5000/api/events?since_id=invalid_id"
```

期望: `400 {"error": "invalid since_id"}`

### 10. 获取事件详情 (GET /api/events/{id})

```bash
EVENT_ID="<事件的 _id>"
curl http://localhost:5000/api/events/$EVENT_ID
```

期望: `200`, 返回完整事件文档

无效 ID 测试:
```bash
curl http://localhost:5000/api/events/invalid_id
```

期望: `400 {"error": "invalid event_id"}`

不存在的 ID 测试:
```bash
curl http://localhost:5000/api/events/000000000000000000000000
```

期望: `404 {"error": "event not found"}`

### 11. 获取最新事件 — 流模式 (GET /api/events/latest)

```bash
# 获取最新事件
curl "http://localhost:5000/api/events/latest?limit=10"

# 基于时间戳增量获取
curl "http://localhost:5000/api/events/latest?since=1714000000.0&limit=50"

# 按类型过滤
curl "http://localhost:5000/api/events/latest?event_type=smoke_flame&since=1714000000.0"
```

期望:
```json
{
  "items": [...],
  "since": 1714000000.0,
  "next_since": 1714050000.0,
  "returned": 10
}
```

`next_since` 可用于下次轮询的 `since` 参数。

### 12. 更新事件状态 (PATCH /api/events/{id}/status)

```bash
EVENT_ID="<事件的 _id>"

# 标记为已派遣
curl -X PATCH http://localhost:5000/api/events/$EVENT_ID/status \
  -H "Content-Type: application/json" \
  -d '{"status": "dispatched"}'

# 标记为处理中
curl -X PATCH http://localhost:5000/api/events/$EVENT_ID/status \
  -H "Content-Type: application/json" \
  -d '{"status": "processing"}'

# 标记为已解决 (含处置信息)
curl -X PATCH http://localhost:5000/api/events/$EVENT_ID/status \
  -H "Content-Type: application/json" \
  -d '{
    "status": "resolved",
    "handled_by": "worker_zhang",
    "handle_note": "Fire extinguished, no damage",
    "handle_image_url": "http://example.com/photo.jpg"
  }'
```

期望: `200`, 返回更新后的事件文档，包含 `status`, `handled_at`, `handled_by`, `handle_note`, `handle_image_url` 字段

合法状态值: `pending`, `dispatched`, `processing`, `resolved`, `rejected`

非法状态测试:
```bash
curl -X PATCH http://localhost:5000/api/events/$EVENT_ID/status \
  -H "Content-Type: application/json" \
  -d '{"status": "invalid_status"}'
```

期望: `400 {"error": "status must be one of: dispatched, pending, processing, rejected, resolved"}`

缺少 status 字段:
```bash
curl -X PATCH http://localhost:5000/api/events/$EVENT_ID/status \
  -H "Content-Type: application/json" \
  -d '{"handled_by": "someone"}'
```

期望: `400 {"error": "status is required"}`

不存在的事件:
```bash
curl -X PATCH http://localhost:5000/api/events/000000000000000000000000/status \
  -H "Content-Type: application/json" \
  -d '{"status": "resolved"}'
```

期望: `404 {"error": "event not found"}`

### 13. 聚合统计 (GET /api/events/stats)

```bash
curl http://localhost:5000/api/events/stats
```

期望: 返回按事件类型、摄像头等维度的聚合统计数据

---

### 14. 注册 Webhook (POST /api/webhooks)

```bash
# 注册接收所有事件类型
curl -X POST http://localhost:5000/api/webhooks \
  -H "Content-Type: application/json" \
  -d '{"url": "http://localhost:8090/api/v1/hvas/webhook"}'

# 注册仅接收特定事件类型
curl -X POST http://localhost:5000/api/webhooks \
  -H "Content-Type: application/json" \
  -d '{
    "url": "http://example.com/hook",
    "event_types": ["smoke_flame", "parking_violation"]
  }'
```

期望: `201`
```json
{"id": "<webhook_id>", "url": "http://...", "event_types": [...]}
```

缺少 URL:
```bash
curl -X POST http://localhost:5000/api/webhooks \
  -H "Content-Type: application/json" \
  -d '{}'
```

期望: `400 {"error": "url is required"}`

### 15. 查看已注册 Webhook (GET /api/webhooks)

```bash
curl http://localhost:5000/api/webhooks
```

期望: 返回所有已注册 webhook 的数组

### 16. 删除 Webhook (DELETE /api/webhooks/{id})

```bash
WEBHOOK_ID="<webhook_id>"
curl -X DELETE http://localhost:5000/api/webhooks/$WEBHOOK_ID
```

期望: `200 {"deleted": true}`

不存在的 ID:
```bash
curl -X DELETE http://localhost:5000/api/webhooks/000000000000000000000000
```

期望: `404 {"error": "not found"}`

---

## Webhook 签名验证测试

当 `WEBHOOK_SECRET` 环境变量已设置时，HVAS 推送 Webhook 会在 `X-HVAS-Signature` 请求头中附带 HMAC-SHA256 签名。

### Python 验证脚本

```python
import hmac
import hashlib
import json
import requests

secret = "hvas-mubs-shared-secret"
payload = {
    "event_id": "test-001",
    "event_type": "smoke_flame",
    "camera_id": "cam-01",
    "timestamp": 1714000000.0,
    "created_at": "2026-04-25T10:00:00Z",
    "confidence": 0.85,
    "description": "Test event"
}

body = json.dumps(payload).encode("utf-8")
signature = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()

# 发送到 MUBS 验证签名
resp = requests.post(
    "http://localhost:8090/api/v1/hvas/webhook",
    data=body,
    headers={
        "Content-Type": "application/json",
        "X-HVAS-Signature": signature
    }
)
print(resp.status_code, resp.json())
```

期望: `201 {"status": "created", "ticket_id": "...", "assigned_team": "fire_team"}`

---

## Webhook 推送 Payload 格式

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

特性:
- 签名: `X-HVAS-Signature` 请求头 (HMAC-SHA256 hex, 需设置 `WEBHOOK_SECRET`)
- 异步推送: ThreadPoolExecutor 4 线程
- 超时: 5 秒/请求
- 重试: 最多 2 次
- 事件类型过滤: 按 webhook 注册时的 `event_types` 过滤

---

## 演示数据生成

```bash
# 使用种子脚本生成 80 条历史事件 (需 MongoDB + MinIO 运行)
cd D:\hybrid_video_analysis_system_v2
python scripts/seed_demo_data.py
```

脚本功能:
- 插入 80 条历史事件到 MongoDB，时间跨度 7 天
- 覆盖 3 种事件类型 (smoke_flame ~30%, parking_violation ~35%, common_space_utilization ~35%)
- 5 个预设摄像头 (east_gate_01, west_gate_02, north_plaza_03, south_lobby_04, warehouse_05)
- 每条事件上传 16x16 占位图片到 MinIO
- 包含完整的 location 元数据 (lat_lng, location, area_code, group)

环境变量 (均有默认值):
- `MONGO_URI` — 默认 `mongodb://localhost:27017`
- `MONGO_DB` — 默认 `hvas`
- `MINIO_ENDPOINT` — 默认 `localhost:9000`
- `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` — 默认 `minioadmin`
- `MINIO_BUCKET` — 默认 `video-events`

---

## 端到端集成测试 (HVAS → MUBS)

### 前置条件
- HVAS 后端运行在 `localhost:5000` (DEMO_MODE=true, WEBHOOK_SECRET=hvas-mubs-shared-secret)
- MUBS 后端运行在 `localhost:8090` (HVAS_WEBHOOK_SECRET=hvas-mubs-shared-secret)
- MongoDB: HVAS 用 27017, MUBS 用 27018

### 步骤

```bash
# 1. 在 HVAS 注册 MUBS Webhook
curl -X POST http://localhost:5000/api/webhooks \
  -H "Content-Type: application/json" \
  -d '{"url": "http://localhost:8090/api/v1/hvas/webhook"}'

# 2. 在 HVAS 创建模拟事件
curl -X POST http://localhost:5000/api/events/mock \
  -H "Content-Type: application/json" \
  -d '{"event_type": "smoke_flame", "camera_id": "demo_cam", "area_code": "east_district", "group": "fire_team"}'

# 3. 在 MUBS 检查工单是否自动创建
TOKEN=$(curl -s -X POST http://localhost:8090/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' | python -c "import sys,json; print(json.load(sys.stdin)['token'])")

curl http://localhost:8090/api/tickets?size=1 \
  -H "Authorization: Bearer $TOKEN"

# 4. 在 MUBS 更新工单状态为 RESOLVED
TICKET_ID="<工单 ID>"
curl -X PATCH http://localhost:8090/api/tickets/$TICKET_ID/status \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"status": "RESOLVED", "note": "已处理"}'

# 5. 在 HVAS 检查事件状态是否被回传更新
EVENT_ID="<HVAS 事件 ID>"
curl http://localhost:5000/api/events/$EVENT_ID
```

期望:
- 步骤 3: MUBS 工单列表出现新工单, 状态 `DISPATCHED`, 团队 `fire_team`
- 步骤 5: HVAS 事件 `status` 变为 `resolved`, 含 `handled_at` 时间戳
