# HVAS + MUBS Implementation Plan

## Overview

This plan covers two workstreams:
1. **HVAS-side changes** — Prepare HVAS APIs for MUBS integration
2. **MUBS development** — Build the upstream business system (Kotlin Spring Boot)

Target: **2026-05 final demo**

---

## Phase 1: HVAS-Side API Hardening (HVAS repo)

> Goal: Make HVAS ready to be consumed by MUBS

### 1.1 Webhook Payload Standardization

**Problem:** Current webhook payload lacks `event_id`, making dedup and mapping impossible for MUBS.

**Changes:**
- `backend/services/event_generator.py` — After `mongo_client.save_event()`, include the MongoDB `_id` (as string) in the webhook payload as `event_id`
- Add `created_at` (ISO 8601) to webhook payload
- Add `area_code` and `group` fields to webhook payload (from stream config)

**Webhook payload spec:**
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

### 1.2 Webhook Signature

**Changes:**
- Add `WEBHOOK_SECRET` env var (shared secret with MUBS)
- Compute `HMAC-SHA256(secret, request_body)` and send as `X-HVAS-Signature` header
- `backend/services/webhook_service.py` — Add signature computation in `_dispatch_one()`

### 1.3 Stream Config: area_code & group Fields

**Changes (same pattern as lat_lng/location):**
- `backend/services/stream_runtime.py` — Add `area_code: str = ""` and `group: str = ""` to `StreamRuntime`
- `backend/services/stream_runtime.py` — Pass through in `StreamRuntimeFactory.create_runtime()` and `_build_handlers()`
- `backend/services/stream_manager.py` — Accept in `add_stream()`, persist, restore, include in `get_streams()`
- `backend/api/stream_routes.py` — Extract from POST body
- `backend/services/event_generator.py` — Pass through to event docs and webhook payload
- `backend/services/violation_detection.py` / `smoke_flame_detection.py` — Store and forward
- `frontend/src/views/StreamManager.vue` — Add form inputs for Area Code and Group

### 1.4 Event Status Feedback API

**New endpoint:** `PATCH /api/events/<event_id>/status`

**Changes:**
- `backend/api/event_routes.py` — New route
- Request body:
  ```json
  {
    "status": "resolved",
    "handled_by": "fieldworker_zhang",
    "handle_note": "Fire extinguished",
    "handle_image_url": "http://..."
  }
  ```
- Updates the event document in MongoDB with these fields + `handled_at` timestamp
- Returns 200 with updated event, or 404

### 1.5 Incremental Sync Support

**Changes:**
- `backend/api/event_routes.py` — Add `since_id` query parameter to `GET /api/events`
- When provided, filter: `{"_id": {"$gt": ObjectId(since_id)}}`
- Keeps existing `start_time` / `end_time` / `event_type` filters

### 1.6 Demo Mode: Mock Event Trigger

**New endpoint:** `POST /api/events/mock`

**Changes:**
- `backend/api/event_routes.py` — New route
- Request body:
  ```json
  {
    "event_type": "smoke_flame",
    "camera_id": "east_gate_01"
  }
  ```
- Generates a synthetic event with current timestamp, placeholder image, realistic description
- Saves to MongoDB and triggers webhook (same as real events)
- Protected: only works when `DEMO_MODE=true` env var is set

### 1.7 Health Check Endpoint

**New endpoint:** `GET /api/health`

**Changes:**
- `backend/api/stream_routes.py` or new `health_routes.py`
- Returns:
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

### 1.8 Seed Data Script

**New file:** `scripts/seed_demo_data.py`

- Inserts 50-100 historical events into MongoDB spanning the past 7 days
- Covers all event types (smoke_flame, parking_violation, common_space_utilization)
- Uses multiple camera_ids with realistic timestamps
- Uploads placeholder images to MinIO

---

## Phase 2: MUBS Backend (Kotlin Spring Boot, separate repo)

> Goal: Core API server for ticket management, dispatch, and notifications

### 2.1 Project Scaffolding
- Spring Boot 3.x + Kotlin
- Dependencies: Spring Web, Spring Security (JWT), Spring Data JPA/MongoDB, Spring Mail
- PostgreSQL (or MongoDB) database
- Docker Compose service definition

### 2.2 Data Models
- **Ticket** — Core work order entity
  ```
  id, hvas_event_id (unique), event_type, camera_id, area_code, group,
  status (PENDING/DISPATCHED/PROCESSING/RESOLVED),
  assigned_team, assigned_worker,
  hvas_timestamp, created_at, dispatched_at, notified_at, accepted_at, resolved_at,
  evidence_url, handle_note, handle_image_url,
  timeline: [{action, actor, timestamp, note}]
  ```
- **User** — Pre-seeded accounts (admin, dispatcher, fieldworker)
- **DispatchRule** — event_type + area_code → team mapping
- **NotificationLog** — channel, recipient, status, sent_at

### 2.3 HVAS Webhook Receiver
- `POST /api/v1/hvas/webhook`
- Verify `X-HVAS-Signature` header
- Deduplicate by `event_id` (reject if ticket already exists)
- Create Ticket with status=PENDING
- Trigger dispatch rule engine

### 2.4 Dispatch Rule Engine
- On ticket creation → match rules by (event_type, area_code)
- Auto-assign to team → update status to DISPATCHED
- Record `dispatched_at` in timeline
- Trigger notification service
- Scheduled job: check DISPATCHED tickets older than 30min → auto-return to PENDING

### 2.5 Authentication
- `POST /api/auth/login` — username/password → JWT
- JWT filter on all `/api/**` routes (except login and health)
- Pre-seeded users: admin, dispatcher, fieldworker (with roles)

### 2.6 Ticket CRUD API
```
GET    /api/tickets              — List with filters (status, event_type, team)
GET    /api/tickets/{id}         — Detail with full timeline
PATCH  /api/tickets/{id}/status  — Update status (dispatch, accept, resolve, return)
GET    /api/tickets/stats        — Aggregated statistics
```

### 2.7 HVAS Polling Fallback
- Scheduled task (every 60s) calls `GET HVAS_URL/api/events?since_id=<last>`
- Creates tickets for any events not yet ingested
- Disabled by default, enabled via config flag

### 2.8 Notification Service
- Email: Spring Mail + SMTP (provider TBD)
- H5 push: WebSocket (STOMP over SockJS) for real-time, or short-polling fallback
- On dispatch → send email to team lead + push to assigned workers
- Log all notifications in NotificationLog table

---

## Phase 3: MUBS Web Dashboard (Vue 3 + TypeScript)

> Goal: Dispatcher management interface

### 3.1 Project Setup
- Vue 3 + TypeScript + Vite
- UI framework: Element Plus or Ant Design Vue
- Chart library: ECharts

### 3.2 Pages
- **Login** — JWT auth form
- **Dashboard** — Today/week event counts, event type pie chart, avg response time chart
- **Ticket List** — Filterable table (status, type, team), bulk actions
- **Ticket Detail** — Evidence preview, full timeline, manual dispatch/return actions
- **Dispatch Rules** — CRUD for event_type + area_code → team mapping
- **User Management** — List pre-seeded users (admin only)

### 3.3 Real-time Updates
- WebSocket connection for new ticket notifications
- Auto-refresh ticket list when new events arrive
- Toast/badge for incoming alerts

---

## Phase 4: MUBS H5 Mobile (Vue 3 + Vant UI)

> Goal: Field worker task handling interface

### 4.1 Project Setup
- Vue 3 + Vant UI
- Responsive design for mobile browsers (including WeChat built-in browser)
- Shared API layer with Web dashboard

### 4.2 Pages
- **Login** — Simple form (fieldworker credentials)
- **Task List** — Pending tasks assigned to current worker's team
- **Task Detail** — HVAS evidence image, event info, location/coordinates
- **Handle Task** — Upload photo (camera/gallery), add note, submit → Resolved
- **History** — Past resolved tasks

### 4.3 Key UX Requirements
- Complete a task in ≤ 3 steps: Open task → Upload photo → Submit
- One-tap jump from notification to task detail
- Offline-tolerant: queue submissions if network is unstable

---

## Phase 5: Notification Integration

> Goal: Email + H5 push notifications

### 5.1 Email Notification
- SMTP provider setup (TBD — SendGrid / 163 / QQ Enterprise)
- HTML email template: event type, time, location, evidence thumbnail, action link
- Sent on ticket dispatch to team lead / dispatcher

### 5.2 H5 Real-time Push
- WebSocket (STOMP) endpoint on MUBS backend
- H5 client auto-connects on login
- Push events: new task assigned, task timeout warning, system alerts
- Fallback: 10s polling if WebSocket unavailable

---

## Phase 6: Demo Preparation

> Goal: Ensure smooth demo experience

### 6.1 Seed Data
- Run `scripts/seed_demo_data.py` to populate HVAS with 7 days of history
- Run MUBS seed script to create matching tickets with various statuses
- Ensure dashboard charts are not empty

### 6.2 Demo Mode
- HVAS: `DEMO_MODE=true` enables `POST /api/events/mock`
- MUBS: Demo panel to trigger mock events without real cameras
- Pre-configured streams with sample videos (already in `video/` directory)

### 6.3 Demo Script Rehearsal
- **Scenario 1 (Fire):** HVAS detects flame → MUBS receives alert → Auto-dispatch to fire team → H5 notification → Field worker marks "extinguished" → Dashboard shows "Resolved"
- **Scenario 2 (Parking):** HVAS detects violation → MUBS receives alert → Auto-dispatch to traffic team → Email notification → Dispatcher views evidence → Marks "handled"

### 6.4 Stability Checklist
- [ ] HVAS streams stable for 30+ minutes without crash
- [ ] MUBS webhook receiver handles rapid event bursts
- [ ] H5 page loads in < 3s on mobile
- [ ] Email delivery confirmed with chosen SMTP provider
- [ ] Fallback: mock events work if cameras fail during demo

---

## Timeline Estimate

| Phase | Scope | Dependencies |
|-------|-------|-------------|
| **Phase 1** | HVAS API hardening | None — start immediately |
| **Phase 2** | MUBS backend | Phase 1 (webhook spec) |
| **Phase 3** | MUBS Web dashboard | Phase 2 (ticket API) |
| **Phase 4** | MUBS H5 mobile | Phase 2 (ticket API) |
| **Phase 5** | Notifications | Phase 2 + SMTP confirmation |
| **Phase 6** | Demo prep | All phases |

Phase 3 and Phase 4 can be developed in parallel once Phase 2 API is stable.

---

## Files Changed per Phase (HVAS repo)

| Phase 1 Task | Files |
|-------------|-------|
| 1.1 Webhook payload | `event_generator.py`, `webhook_service.py` |
| 1.2 Webhook signature | `webhook_service.py`, add `WEBHOOK_SECRET` env |
| 1.3 area_code/group | `stream_runtime.py`, `stream_manager.py`, `stream_routes.py`, `event_generator.py`, `violation_detection.py`, `smoke_flame_detection.py`, `StreamManager.vue` |
| 1.4 Event status API | `event_routes.py` |
| 1.5 Incremental sync | `event_routes.py` |
| 1.6 Demo mode | `event_routes.py`, add `DEMO_MODE` env |
| 1.7 Health check | New `health_routes.py` or `stream_routes.py` |
| 1.8 Seed data | New `scripts/seed_demo_data.py` |
