# HVAS Development Guide

## 1. Project Overview

HVAS, the Hybrid Video Analysis System, is an intelligent video monitoring and event-analysis platform based on YOLOv8, Qwen-VL, MongoDB, MinIO, and a Vue-based management interface.

The system supports live RTSP streams and uploaded video files. It performs multiple video-analysis tasks, stores event metadata and visual evidence, exposes REST APIs for operational use, and can forward detected events to upstream business systems through signed webhooks.

### 1.1 Core Capabilities

- Parking violation detection based on vehicle detection, no-parking-zone filtering, and dwell-time tracking.
- Smoke and flame detection using a two-stage pipeline: YOLO candidate detection followed by optional Qwen-VL verification.
- Common-space utilization analysis using Qwen-VL at a configurable sampling interval.
- Event persistence through MongoDB and MinIO.
- Stream management, event search, status feedback, health checks, and webhook integration through REST APIs.
- Frontend dashboards for stream operations, event review, report generation, and local video upload.

### 1.2 Technology Stack

| Layer | Technology | Default Port |
|---|---|---|
| Backend API | Python Flask | 5000 |
| Frontend development server | Vue 3 + Vite | 5173 |
| Frontend production server | Express.js | 8080 |
| Database | MongoDB | 27017 |
| Object storage | MinIO | 9000 API, 9001 Console |
| Object detection | YOLOv8 | N/A |
| Vision-language verification | Qwen-VL | External API |

### 1.3 Repository Root

Use the repository root as the working directory for service startup, dependency installation, and testing:

```bash
hybrid_video_analysis_system_v2/
```

---

## 2. Repository Structure

```text
hybrid_video_analysis_system_v2/
|-- backend/
|   |-- main.py                         # Application entry point
|   |-- api/
|   |   |-- stream_routes.py            # Stream management API
|   |   |-- event_routes.py             # Event query and status API
|   |   |-- webhook_routes.py           # Webhook registration API
|   |   |-- report_routes.py            # Report generation API
|   |   `-- health_routes.py            # Health check API
|   |-- services/
|   |   |-- stream_manager.py           # RTSP stream lifecycle management
|   |   |-- stream_runtime.py           # Per-stream task runtime
|   |   |-- violation_detection.py      # Parking violation pipeline
|   |   |-- smoke_flame_detection.py    # Smoke/flame pipeline
|   |   |-- common_space_detection.py   # Common-space analysis pipeline
|   |   |-- event_generator.py          # Event persistence and webhook dispatch
|   |   |-- webhook_service.py          # Webhook delivery with HMAC signing
|   |   |-- dwell_tracker.py            # Vehicle dwell-time tracking
|   |   `-- parking_zone_checker.py     # No-parking-zone polygon filtering
|   |-- models/
|   |   `-- event.py                    # EventModel schema
|   |-- config/
|   |   |-- qwen_vl_config.py           # Qwen-VL configuration
|   |   `-- qwen_report_config.py       # Report LLM configuration
|   `-- utils/
|       |-- frame_capture.py            # Video frame extraction
|       |-- visualization.py            # Frame rendering utilities
|       `-- performance_metrics.py      # Runtime metrics helpers
|-- frontend/
|   |-- src/
|   |   |-- views/
|   |   |   |-- Dashboard.vue           # Runtime dashboard
|   |   |   |-- EventList.vue           # Event list and filtering
|   |   |   |-- ReportsView.vue         # Report UI
|   |   |   |-- StreamManager.vue       # Stream management UI
|   |   |   `-- UploadVideo.vue         # Local video upload UI
|   |   |-- composables/
|   |   `-- router/
|   |-- server.js                       # Production proxy server
|   `-- package.json
|-- ml_models/
|   `-- yolov8/
|       |-- model_loader.py             # YOLO model loading
|       `-- inference.py                # YOLO inference wrapper
|-- storage/
|   |-- mongodb_client.py               # MongoDB event storage
|   `-- minio_client.py                 # MinIO image storage
|-- scripts/
|   |-- seed_demo_data.py               # Demo data generation
|   |-- draw_fence_gui.py               # No-parking-zone drawing tool
|   `-- file_watcher.py                 # Folder-based video ingestion
|-- test/                               # Algorithm and performance test scripts
|-- tests/                              # Legacy automated test location, if present
|-- docs/
|-- docker-compose.yml
`-- requirements.txt
```

---

## 3. Quick Start

### 3.1 Start Infrastructure

Start MongoDB and MinIO from the repository root:

```bash
docker compose up -d
```

Confirm that the services are listening:

```bash
docker compose ps
```

### 3.2 Install Backend Dependencies

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3.3 Configure Environment Variables

Create a `.env` file in the repository root.

```bash
# Storage
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=video_analysis_db
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=video-events

# Detection runtime
FRAME_INTERVAL=1.0
RTSP_SAMPLE_INTERVAL=0.5
PARKING_RTSP_SAMPLE_INTERVAL=0.5
SMOKE_RTSP_SAMPLE_INTERVAL=0.2
COMMON_SPACE_INTERVAL=30.0
DWELL_THRESHOLD=5

# Queue and backpressure controls
ENABLE_BACKPRESSURE_PROTECTION=false
STREAM_EXECUTOR_MAX_QUEUE=24
SMOKE_DETECTION_MAX_QUEUE=8
SMOKE_VERIFICATION_MAX_QUEUE=12
COMMON_SPACE_MAX_QUEUE=4

# Qwen-VL. If omitted, smoke/flame second-stage verification and
# common-space analysis are disabled.
QWEN_VL_API_URL=https://dashscope.aliyuncs.com/api/v1/services/aigc/video-understanding/video-generation
QWEN_VL_API_KEY=sk-xxxxxxxx
QWEN_VL_MODEL_NAME=qwen-vl-plus
QWEN_VL_TIMEOUT=30

# Report generation LLM, optional.
QWEN_REPORT_API_URL=
QWEN_REPORT_API_KEY=
QWEN_REPORT_MODEL_NAME=qwen-plus
QWEN_REPORT_TIMEOUT=30
QWEN_REPORT_TEMPERATURE=0.2
QWEN_REPORT_MAX_TOKENS=700

# Webhook
WEBHOOK_SECRET=hvas-mubs-shared-secret

# Demo mode
DEMO_MODE=true
```

### 3.4 Start the Backend

Run from the repository root:

```bash
python backend/main.py
```

The backend listens on:

```text
http://localhost:5000
```

### 3.5 Start the Frontend

Development mode:

```bash
cd frontend
npm install
npm run dev
```

Development URL:

```text
http://localhost:5173
```

Production mode:

```bash
cd frontend
npm install
npm run build
npm run start-server
```

Production URL:

```text
http://localhost:8080
```

### 3.6 Verify Startup

```bash
curl http://localhost:5000/api/health
```

Expected healthy response:

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

---

## 4. Backend Service Architecture

### 4.1 Runtime Flow

```text
RTSP stream or local video
        |
        v
Frame capture and sampling
        |
        v
StreamRuntime
        |
        +--> Parking violation task
        |       |-- YOLOv8 vehicle detection
        |       |-- No-parking-zone polygon filtering
        |       |-- Dwell-time tracking
        |       `-- Render evidence frame
        |
        +--> Smoke/flame task
        |       |-- YOLOv8 smoke/flame candidate detection
        |       |-- Optional Qwen-VL verification
        |       `-- Render evidence frame
        |
        `--> Common-space task
                `-- Qwen-VL common-space analysis
        |
        v
EventGenerator
        |
        +--> MinIO evidence image upload
        +--> MongoDB event metadata persistence
        `--> Webhook notification dispatch
```

### 4.2 Parking Violation Pipeline

The parking pipeline performs whole-frame YOLO inference, filters detected vehicles against configured no-parking polygons, tracks consecutive in-zone dwell frames, and saves an event only when the dwell threshold is reached.

Main modules:

- `backend/services/violation_detection.py`
- `backend/services/dwell_tracker.py`
- `backend/services/parking_zone_checker.py`
- `backend/utils/visualization.py`

### 4.3 Smoke and Flame Pipeline

The smoke/flame pipeline uses YOLO as a fast first-stage detector. When Qwen-VL is configured, candidate regions are verified by a second-stage visual-language model before an event is persisted.

Main modules:

- `backend/services/smoke_flame_detection.py`
- `ml_models/yolov8/model_loader.py`
- `ml_models/yolov8/inference.py`
- `backend/services/event_generator.py`

### 4.4 Common-Space Analysis Pipeline

The common-space task samples frames at a lower frequency and sends them to Qwen-VL for scene-level analysis. It is intended for periodic utilization and activity summaries rather than high-frequency object detection.

Main module:

- `backend/services/common_space_detection.py`

### 4.5 Storage Responsibilities

| Component | Responsibility |
|---|---|
| MongoDB | Stores event metadata, status fields, timestamps, object summaries, and integration metadata. |
| MinIO | Stores rendered evidence images and exposes object URLs for the frontend and downstream systems. |

The event document stores a URL reference to MinIO evidence rather than embedding image bytes.

---

## 5. API Reference

### 5.1 Stream Management

| Method | Path | Description |
|---|---|---|
| GET | `/api/streams` | List active streams. |
| POST | `/api/streams` | Add an RTSP stream. |
| DELETE | `/api/streams/<stream_id>` | Remove a stream. |
| PUT | `/api/streams/<stream_id>/tasks` | Update task assignments for a stream. |
| GET | `/api/streams/<stream_id>/metrics` | Retrieve runtime metrics for a stream. |

Example request:

```json
{
  "url": "rtsp://camera-ip:554/stream",
  "tasks": ["parking_violation", "smoke_flame"],
  "camera_id": "east_gate_01",
  "lat_lng": "31.2304, 121.4737",
  "location": "East Gate Entrance",
  "area_code": "east_district",
  "group": "fire_team"
}
```

Supported task names:

- `parking_violation`
- `smoke_flame`
- `common_space`

### 5.2 Event Query

| Method | Path | Description |
|---|---|---|
| GET | `/api/events` | Query events with pagination and filters. |
| GET | `/api/events/latest` | Retrieve recent events for polling. |
| GET | `/api/events/stats` | Retrieve aggregate event statistics. |
| GET | `/api/events/<event_id>` | Retrieve one event by ID. |
| PATCH | `/api/events/<event_id>/status` | Update event handling status. |
| POST | `/api/events/mock` | Create a mock event when `DEMO_MODE=true`. |

Supported query parameters for `GET /api/events`:

| Parameter | Description |
|---|---|
| `camera_id` | Filter by camera ID. |
| `event_type` | Filter by event type. |
| `detection_stage` | Filter by detection stage. |
| `since_id` | Incremental synchronization filter. Returns records with `_id > since_id`. |
| `start_time` | Unix timestamp lower bound. |
| `end_time` | Unix timestamp upper bound. |
| `limit` | Page size. Valid range: 1 to 500. Default: 50. |
| `skip` | Number of records to skip. |

Supported event types:

- `parking_violation`
- `smoke_flame`
- `common_space_utilization`

### 5.3 Event Status Feedback

Endpoint:

```text
PATCH /api/events/<event_id>/status
```

Request body:

```json
{
  "status": "resolved",
  "handled_by": "fieldworker_zhang",
  "handle_note": "Fire extinguished, no damage observed.",
  "handle_image_url": "http://example.com/photo.jpg"
}
```

Valid status values:

- `pending`
- `dispatched`
- `processing`
- `resolved`
- `rejected`

### 5.4 Webhook Configuration

| Method | Path | Description |
|---|---|---|
| POST | `/api/webhooks` | Register a webhook endpoint. |
| GET | `/api/webhooks` | List registered webhooks. |
| DELETE | `/api/webhooks/<webhook_id>` | Delete a webhook endpoint. |

Webhook registration request:

```json
{
  "url": "http://localhost:8090/api/v1/hvas/webhook",
  "event_types": ["smoke_flame", "parking_violation"]
}
```

When `event_types` is omitted, the webhook receives all event types.

### 5.5 Webhook Payload

HVAS dispatches this payload when an event is persisted:

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

If `WEBHOOK_SECRET` is configured, HVAS signs the raw request body with HMAC-SHA256 and sends the hexadecimal digest in the `X-HVAS-Signature` header.

### 5.6 Health Check

Endpoint:

```text
GET /api/health
```

Response:

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

`status` is `ok` when MongoDB is connected. It is `degraded` when core dependencies are unavailable.

---

## 6. Environment Variable Reference

| Variable | Required | Default | Description |
|---|---:|---|---|
| `MONGO_URI` | Yes | `mongodb://localhost:27017` | MongoDB connection URI. |
| `MONGO_DB_NAME` | Yes | `video_analysis_db` | MongoDB database name. |
| `MINIO_ENDPOINT` | Yes | `localhost:9000` | MinIO API endpoint. |
| `MINIO_ACCESS_KEY` | Yes | `minioadmin` | MinIO access key. |
| `MINIO_SECRET_KEY` | Yes | `minioadmin` | MinIO secret key. |
| `MINIO_BUCKET` | Yes | `video-events` | MinIO bucket for evidence images. |
| `FRAME_INTERVAL` | No | `1.0` | Local video frame sampling interval, in seconds. |
| `RTSP_SAMPLE_INTERVAL` | No | `0.5` | Legacy RTSP sampling interval, in seconds. |
| `PARKING_RTSP_SAMPLE_INTERVAL` | No | `RTSP_SAMPLE_INTERVAL` | Parking task sampling interval. |
| `SMOKE_RTSP_SAMPLE_INTERVAL` | No | `0.2` | Smoke/flame task sampling interval. |
| `COMMON_SPACE_INTERVAL` | No | `30.0` | Common-space analysis interval. |
| `DWELL_THRESHOLD` | No | `5` | Number of consecutive in-zone frames required for a parking violation. |
| `YOLO_DEVICE` | No | Auto | YOLO inference device, for example `cpu`, `cuda`, or `cuda:0`. |
| `ENABLE_BACKPRESSURE_PROTECTION` | No | `false` | Enables bounded task queues for stream processing. |
| `STREAM_EXECUTOR_MAX_QUEUE` | No | `24` | Maximum stream executor queue size. |
| `SMOKE_DETECTION_MAX_QUEUE` | No | `8` | Maximum smoke detection queue size. |
| `SMOKE_VERIFICATION_MAX_QUEUE` | No | `12` | Maximum smoke verification queue size. |
| `COMMON_SPACE_MAX_QUEUE` | No | `4` | Maximum common-space analysis queue size. |
| `QWEN_VL_API_URL` | No | Empty | Qwen-VL API endpoint. |
| `QWEN_VL_API_KEY` | No | Empty | Qwen-VL API key. |
| `QWEN_VL_MODEL_NAME` | No | `qwen-vl-plus` | Qwen-VL model name. |
| `QWEN_REPORT_API_URL` | No | `QWEN_VL_API_URL` | Report LLM endpoint. |
| `QWEN_REPORT_API_KEY` | No | `QWEN_VL_API_KEY` | Report LLM API key. |
| `QWEN_REPORT_MODEL_NAME` | No | `qwen-plus` | Report LLM model name. |
| `WEBHOOK_SECRET` | No | Empty | Shared secret for webhook signing. |
| `DEMO_MODE` | No | `false` | Enables mock event creation. |

---

## 7. Operational Notes

### 7.1 Model Loading

The backend loads the vehicle model by default. The smoke/flame model is loaded when the corresponding model file is available. If Qwen-VL is not configured, smoke/flame second-stage verification and common-space analysis are disabled, while other backend functions remain available.

### 7.2 Stream Sampling

Sampling intervals are task-specific. Parking, smoke/flame, and common-space tasks can use different frame sampling rates on the same RTSP source. This allows high-frequency detection tasks and lower-frequency scene analysis to coexist without applying one global sampling interval to all tasks.

### 7.3 Backpressure

Backpressure protection should be enabled for multi-stream RTSP workloads or when Qwen-VL latency is high. When enabled, bounded queues prevent unbounded memory growth and preserve service stability under temporary load spikes.

### 7.4 Evidence Storage

Events should always reference immutable evidence URLs. Evidence images are rendered before upload so that bounding boxes, task annotations, and zone overlays are available during review.

### 7.5 Integration Contract

Downstream systems should use `event_id` as the stable deduplication key. Webhook consumers should verify `X-HVAS-Signature` when a shared secret is configured and should treat webhook delivery as at-least-once.

---

## 8. Common Commands

Start infrastructure:

```bash
docker compose up -d
```

Start backend:

```bash
venv\Scripts\activate
python backend/main.py
```

Start frontend in development mode:

```bash
cd frontend
npm run dev
```

Run backend health check:

```bash
curl http://localhost:5000/api/health
```

List streams:

```bash
curl http://localhost:5000/api/streams
```

Query latest events:

```bash
curl "http://localhost:5000/api/events/latest?limit=10"
```

Generate demo events:

```bash
python scripts/seed_demo_data.py
```
