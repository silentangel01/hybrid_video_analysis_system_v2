# HVAS 测试文档 (Testing Guide)

## 测试文件

```
tests/
├── test_detection_logic.py        # 检测逻辑单元测试
├── test_storage_integration.py    # 存储集成测试
└── test_video_ingestion.py        # 视频采集测试
```

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

## 测试覆盖范围

### test_detection_logic.py — 检测逻辑
- YOLOv8 推理结果解析
- 禁停区域多边形点判定
- 烟火置信度阈值
- 事件生成正确性

### test_storage_integration.py — 存储集成
- MongoDB 事件写入和查询
- MongoDB 索引创建
- MinIO 图片上传和下载
- 存储连通性

### test_video_ingestion.py — 视频采集
- RTSP 流帧提取
- 本地视频文件帧提取
- 帧率控制逻辑
- 流水线端到端

---

## 手动 API 测试

### 前置条件

```bash
# 确保 Docker 服务已启动
docker-compose up -d

# 启动后端
cd backend && python main.py
```

### 1. 健康检查

```bash
curl http://localhost:5000/api/health
```

期望响应:
```json
{
  "status": "ok",
  "mongodb": "connected",
  "minio": "connected"
}
```

### 2. 添加 RTSP 流

```bash
curl -X POST http://localhost:5000/api/streams \
  -H "Content-Type: application/json" \
  -d '{
    "url": "rtsp://192.168.1.100:554/stream",
    "tasks": ["smoke_flame"],
    "camera_id": "test_cam_01",
    "area_code": "east_district",
    "group": "fire_team"
  }'
```

### 3. 创建模拟事件 (需 DEMO_MODE=true)

```bash
curl -X POST http://localhost:5000/api/events/mock \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "smoke_flame",
    "camera_id": "demo_camera_01",
    "area_code": "east_district"
  }'
```

### 4. 查询事件

```bash
# 查询所有事件
curl "http://localhost:5000/api/events?limit=10"

# 按类型过滤
curl "http://localhost:5000/api/events?event_type=smoke_flame&limit=5"

# 增量同步 (获取某 ID 之后的新事件)
curl "http://localhost:5000/api/events?since_id=6625a3f1b2c4d5e6f7890123"
```

### 5. 更新事件状态 (MUBS 反馈)

```bash
curl -X PATCH http://localhost:5000/api/events/<event_id>/status \
  -H "Content-Type: application/json" \
  -d '{
    "status": "resolved",
    "handled_by": "worker_zhang",
    "handle_note": "已处理"
  }'
```

### 6. 注册 Webhook

```bash
curl -X POST http://localhost:5000/api/webhooks \
  -H "Content-Type: application/json" \
  -d '{
    "url": "http://localhost:8090/api/v1/hvas/webhook",
    "event_types": ["smoke_flame", "parking_violation"]
  }'
```

### 7. 聚合统计

```bash
curl http://localhost:5000/api/events/stats
```

---

## Webhook 签名验证测试

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

## 演示数据生成

```bash
# 使用种子脚本生成演示事件
python scripts/seed_demo_data.py

# 批量加载测试数据
python scripts/load_test_events.py
```
