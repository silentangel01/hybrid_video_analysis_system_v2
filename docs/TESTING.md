# HVAS Testing Guide

## 1. Scope

This document defines the testing approach for HVAS, including automated test placeholders, manual API verification, webhook signature validation, demo data generation, and end-to-end HVAS-to-MUBS integration tests.

The current practical testing flow relies primarily on manual API tests and integration checks. Automated test files may exist as placeholders and should be completed as the implementation stabilizes.

---

## 2. Automated Test Layout

Expected automated test structure:

```text
tests/
|-- test_detection_logic.py        # Unit tests for detection logic
|-- test_storage_integration.py    # MongoDB and MinIO integration tests
`-- test_video_ingestion.py        # Video ingestion tests
```

Additional algorithm and performance test scripts may be maintained under `test/` when they are designed as executable evaluation tools rather than pytest test cases.

Recommended automated test categories:

| Category | Purpose |
|---|---|
| Unit tests | Validate pure detection logic, polygon filtering, dwell tracking, and event formatting. |
| Integration tests | Validate MongoDB, MinIO, webhook delivery, and API route behavior. |
| Video ingestion tests | Validate local video upload, RTSP frame capture, and sampling behavior. |
| Algorithm evaluation | Measure false positives, frame rate, latency, and model verification effectiveness. |
| Resource tests | Measure CPU, GPU, memory, and queue pressure under one or more RTSP streams. |

---

## 3. Running Automated Tests

Run all pytest tests:

```bash
python -m pytest tests/ -v
```

Run a single test file:

```bash
python -m pytest tests/test_detection_logic.py -v
```

Run tests with coverage:

```bash
python -m pytest tests/ -v --cov=backend --cov-report=term-missing
```

When algorithm and performance scripts are located under `test/`, run them according to the script-level README or command-line help.

---

## 4. Manual API Testing

### 4.1 Prerequisites

Start infrastructure:

```bash
docker compose up -d
```

Activate the backend virtual environment and start the backend:

```bash
venv\Scripts\activate
python backend/main.py
```

The backend should be available at:

```text
http://localhost:5000
```

---

## 5. Health Check

Request:

```bash
curl http://localhost:5000/api/health
```

Expected response:

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

Validation points:

- `status` is `ok` when MongoDB is connected.
- `status` is `degraded` when core dependencies are unavailable.
- `mongodb` reflects the actual MongoDB connection state.
- `minio` reflects the actual MinIO connection state.
- `active_streams` equals the current number of active streams.
- `models_loaded` lists the loaded model families, such as `vehicle`, `smoke_flame`, or `qwen_vl`.
- `uptime_sec` increases while the backend process remains active.

---

## 6. Stream Management Tests

### 6.1 Add an RTSP Stream

Request:

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

Expected result:

- HTTP `200`.
- Response contains stream metadata and a `stream_id`.
- The stream begins processing if the RTSP source is reachable.

### 6.2 List Active Streams

Request:

```bash
curl http://localhost:5000/api/streams
```

Expected result:

- Response is an array.
- The stream added in the previous step is present.
- The response includes `camera_id`, `tasks`, `lat_lng`, `location`, `area_code`, and `group`.

### 6.3 Update Stream Tasks

Request:

```bash
STREAM_ID="<stream_id returned by POST /api/streams>"

curl -X PUT http://localhost:5000/api/streams/$STREAM_ID/tasks \
  -H "Content-Type: application/json" \
  -d '{"tasks": ["smoke_flame", "parking_violation"]}'
```

Expected result:

- HTTP `200`.
- The stream task list is updated to include both tasks.

### 6.4 Retrieve Stream Metrics

Request:

```bash
curl http://localhost:5000/api/streams/$STREAM_ID/metrics
```

Expected result:

- Response includes runtime metrics such as frame rates, processing latency, event counters, queue state, and task-specific metrics when available.

### 6.5 Remove a Stream

Request:

```bash
curl -X DELETE http://localhost:5000/api/streams/$STREAM_ID
```

Expected result:

- HTTP `200`.
- The stream is removed.
- Subsequent `GET /api/streams` responses no longer include the removed stream.

---

## 7. Mock Event Tests

Mock event creation requires:

```bash
DEMO_MODE=true
```

### 7.1 Create a Mock Event

Request:

```bash
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

Expected response:

```json
{
  "message": "Mock event created",
  "event_type": "smoke_flame",
  "camera_id": "demo_camera_01"
}
```

Expected status:

```text
201
```

Supported event types:

- `smoke_flame`
- `parking_violation`
- `common_space_utilization`

### 7.2 Mock Event When Demo Mode Is Disabled

Request:

```bash
curl -X POST http://localhost:5000/api/events/mock \
  -H "Content-Type: application/json" \
  -d '{"event_type": "smoke_flame"}'
```

Expected response:

```json
{
  "error": "DEMO_MODE is not enabled"
}
```

Expected status:

```text
403
```

---

## 8. Event Query Tests

### 8.1 Query Events

Request examples:

```bash
curl "http://localhost:5000/api/events"
curl "http://localhost:5000/api/events?event_type=smoke_flame&limit=5"
curl "http://localhost:5000/api/events?camera_id=demo_camera_01"
curl "http://localhost:5000/api/events?start_time=1714000000&end_time=1714100000"
curl "http://localhost:5000/api/events?limit=10&skip=20"
```

Expected response shape:

```json
{
  "items": [],
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

Validation points:

- `items` contains event documents.
- Pagination fields reflect the request parameters.
- Filters in the response match the submitted query.

### 8.2 Incremental Synchronization

Request sequence:

```bash
RESPONSE=$(curl -s "http://localhost:5000/api/events?limit=5")
LAST_ID=$(echo $RESPONSE | python -c "import sys,json; items=json.load(sys.stdin)['items']; print(items[-1]['_id'] if items else '')")
curl "http://localhost:5000/api/events?since_id=$LAST_ID&limit=10"
```

Expected result:

- Response contains events with `_id > since_id`.
- Results are ordered for incremental consumption.
- `since_id` can be combined with filters such as `camera_id` and `event_type`.

Invalid `since_id` request:

```bash
curl "http://localhost:5000/api/events?since_id=invalid_id"
```

Expected response:

```json
{
  "error": "invalid since_id"
}
```

Expected status:

```text
400
```

### 8.3 Retrieve Event Detail

Request:

```bash
EVENT_ID="<event _id>"
curl http://localhost:5000/api/events/$EVENT_ID
```

Expected result:

- HTTP `200`.
- Response contains the complete event document.

Invalid ID request:

```bash
curl http://localhost:5000/api/events/invalid_id
```

Expected response:

```json
{
  "error": "invalid event_id"
}
```

Expected status:

```text
400
```

Nonexistent ID request:

```bash
curl http://localhost:5000/api/events/000000000000000000000000
```

Expected response:

```json
{
  "error": "event not found"
}
```

Expected status:

```text
404
```

### 8.4 Retrieve Latest Events

Request examples:

```bash
curl "http://localhost:5000/api/events/latest?limit=10"
curl "http://localhost:5000/api/events/latest?since=1714000000.0&limit=50"
curl "http://localhost:5000/api/events/latest?event_type=smoke_flame&since=1714000000.0"
```

Expected response shape:

```json
{
  "items": [],
  "since": 1714000000.0,
  "next_since": 1714050000.0,
  "returned": 10
}
```

Validation point:

- `next_since` can be used as the `since` value for the next polling request.

---

## 9. Event Status Tests

### 9.1 Update Status

Request examples:

```bash
EVENT_ID="<event _id>"

curl -X PATCH http://localhost:5000/api/events/$EVENT_ID/status \
  -H "Content-Type: application/json" \
  -d '{"status": "dispatched"}'

curl -X PATCH http://localhost:5000/api/events/$EVENT_ID/status \
  -H "Content-Type: application/json" \
  -d '{"status": "processing"}'

curl -X PATCH http://localhost:5000/api/events/$EVENT_ID/status \
  -H "Content-Type: application/json" \
  -d '{
    "status": "resolved",
    "handled_by": "worker_zhang",
    "handle_note": "Fire extinguished, no damage",
    "handle_image_url": "http://example.com/photo.jpg"
  }'
```

Expected result:

- HTTP `200`.
- Response contains the updated event document.
- Response includes `status`.
- For handled events, response includes `handled_at`, `handled_by`, `handle_note`, and `handle_image_url` when provided.

Valid status values:

- `pending`
- `dispatched`
- `processing`
- `resolved`
- `rejected`

### 9.2 Invalid Status

Request:

```bash
curl -X PATCH http://localhost:5000/api/events/$EVENT_ID/status \
  -H "Content-Type: application/json" \
  -d '{"status": "invalid_status"}'
```

Expected response:

```json
{
  "error": "status must be one of: dispatched, pending, processing, rejected, resolved"
}
```

Expected status:

```text
400
```

### 9.3 Missing Status

Request:

```bash
curl -X PATCH http://localhost:5000/api/events/$EVENT_ID/status \
  -H "Content-Type: application/json" \
  -d '{"handled_by": "someone"}'
```

Expected response:

```json
{
  "error": "status is required"
}
```

Expected status:

```text
400
```

### 9.4 Nonexistent Event

Request:

```bash
curl -X PATCH http://localhost:5000/api/events/000000000000000000000000/status \
  -H "Content-Type: application/json" \
  -d '{"status": "resolved"}'
```

Expected response:

```json
{
  "error": "event not found"
}
```

Expected status:

```text
404
```

---

## 10. Event Statistics Test

Request:

```bash
curl http://localhost:5000/api/events/stats
```

Expected result:

- Response contains aggregate statistics grouped by event type, camera, or other supported dimensions.
- Counts match the events currently stored in MongoDB.

---

## 11. Webhook Registration Tests

### 11.1 Register a Webhook

Register a webhook for all event types:

```bash
curl -X POST http://localhost:5000/api/webhooks \
  -H "Content-Type: application/json" \
  -d '{"url": "http://localhost:8090/api/v1/hvas/webhook"}'
```

Register a webhook for selected event types:

```bash
curl -X POST http://localhost:5000/api/webhooks \
  -H "Content-Type: application/json" \
  -d '{
    "url": "http://example.com/hook",
    "event_types": ["smoke_flame", "parking_violation"]
  }'
```

Expected response:

```json
{
  "id": "<webhook_id>",
  "url": "http://example.com/hook",
  "event_types": ["smoke_flame", "parking_violation"]
}
```

Expected status:

```text
201
```

### 11.2 Missing URL

Request:

```bash
curl -X POST http://localhost:5000/api/webhooks \
  -H "Content-Type: application/json" \
  -d '{}'
```

Expected response:

```json
{
  "error": "url is required"
}
```

Expected status:

```text
400
```

### 11.3 List Webhooks

Request:

```bash
curl http://localhost:5000/api/webhooks
```

Expected result:

- Response contains all registered webhook endpoints.

### 11.4 Delete a Webhook

Request:

```bash
WEBHOOK_ID="<webhook_id>"
curl -X DELETE http://localhost:5000/api/webhooks/$WEBHOOK_ID
```

Expected response:

```json
{
  "deleted": true
}
```

Expected status:

```text
200
```

Nonexistent webhook request:

```bash
curl -X DELETE http://localhost:5000/api/webhooks/000000000000000000000000
```

Expected response:

```json
{
  "error": "not found"
}
```

Expected status:

```text
404
```

---

## 12. Webhook Signature Verification

When `WEBHOOK_SECRET` is configured, HVAS signs webhook payloads with HMAC-SHA256 and sends the result in the `X-HVAS-Signature` header.

### 12.1 Verification Script

```python
import hashlib
import hmac
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
    "description": "Test event",
}

body = json.dumps(payload).encode("utf-8")
signature = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()

response = requests.post(
    "http://localhost:8090/api/v1/hvas/webhook",
    data=body,
    headers={
        "Content-Type": "application/json",
        "X-HVAS-Signature": signature,
    },
)

print(response.status_code, response.json())
```

Expected MUBS response:

```json
{
  "status": "created",
  "ticket_id": "...",
  "assigned_team": "fire_team"
}
```

Expected status:

```text
201
```

### 12.2 Webhook Payload Format

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

Webhook delivery characteristics:

- Signature header: `X-HVAS-Signature`.
- Signature algorithm: HMAC-SHA256 hexadecimal digest.
- Dispatch model: asynchronous thread pool.
- Request timeout: five seconds.
- Retry policy: up to two retries.
- Event filtering: based on `event_types` configured during webhook registration.

---

## 13. Demo Data Generation

Generate historical demo events:

```bash
python scripts/seed_demo_data.py
```

Script responsibilities:

- Insert 80 historical events into MongoDB.
- Distribute timestamps across the past seven days.
- Cover `smoke_flame`, `parking_violation`, and `common_space_utilization`.
- Use predefined cameras such as `east_gate_01`, `west_gate_02`, `north_plaza_03`, `south_lobby_04`, and `warehouse_05`.
- Upload placeholder evidence images to MinIO.
- Include location metadata such as `lat_lng`, `location`, `area_code`, and `group`.

Default environment variables:

| Variable | Default |
|---|---|
| `MONGO_URI` | `mongodb://localhost:27017` |
| `MONGO_DB` | `hvas` |
| `MINIO_ENDPOINT` | `localhost:9000` |
| `MINIO_ACCESS_KEY` | `minioadmin` |
| `MINIO_SECRET_KEY` | `minioadmin` |
| `MINIO_BUCKET` | `video-events` |

---

## 14. End-to-End HVAS-to-MUBS Test

### 14.1 Prerequisites

- HVAS backend is running at `http://localhost:5000`.
- HVAS uses `DEMO_MODE=true`.
- HVAS uses `WEBHOOK_SECRET=hvas-mubs-shared-secret`.
- MUBS backend is running at `http://localhost:8090`.
- MUBS uses `HVAS_WEBHOOK_SECRET=hvas-mubs-shared-secret`.
- HVAS MongoDB uses port `27017`.
- MUBS storage uses its configured database port.

### 14.2 Test Steps

Register the MUBS webhook in HVAS:

```bash
curl -X POST http://localhost:5000/api/webhooks \
  -H "Content-Type: application/json" \
  -d '{"url": "http://localhost:8090/api/v1/hvas/webhook"}'
```

Create a mock event in HVAS:

```bash
curl -X POST http://localhost:5000/api/events/mock \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "smoke_flame",
    "camera_id": "demo_cam",
    "area_code": "east_district",
    "group": "fire_team"
  }'
```

Authenticate with MUBS:

```bash
TOKEN=$(curl -s -X POST http://localhost:8090/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['token'])")
```

Check whether a MUBS ticket was created:

```bash
curl http://localhost:8090/api/tickets?size=1 \
  -H "Authorization: Bearer $TOKEN"
```

Update the MUBS ticket status:

```bash
TICKET_ID="<ticket id>"

curl -X PATCH http://localhost:8090/api/tickets/$TICKET_ID/status \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"status": "RESOLVED", "note": "Handled"}'
```

Verify the HVAS event state:

```bash
EVENT_ID="<HVAS event id>"
curl http://localhost:5000/api/events/$EVENT_ID
```

### 14.3 Expected Results

- MUBS receives the HVAS webhook.
- MUBS creates one ticket for the event.
- Duplicate webhook deliveries do not create duplicate tickets.
- The ticket is dispatched to `fire_team` when the dispatch rule matches.
- The ticket status can be updated to `RESOLVED`.
- HVAS event status reflects the handling feedback when feedback integration is enabled.

---

## 15. Test Completion Checklist

- [ ] Infrastructure starts successfully.
- [ ] Backend health check returns expected dependency states.
- [ ] Stream creation, listing, task update, metrics, and deletion work.
- [ ] Mock event creation works when demo mode is enabled.
- [ ] Event query filters and pagination work.
- [ ] Incremental synchronization works with `since_id`.
- [ ] Event status update validates allowed values.
- [ ] Webhook registration and deletion work.
- [ ] Webhook signature verification succeeds with the configured secret.
- [ ] Demo data script creates events and evidence.
- [ ] HVAS-to-MUBS end-to-end flow creates tickets without duplicates.
