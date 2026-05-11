# HVAS and MUBS Implementation Plan

## 1. Overview

This plan defines the implementation scope for integrating HVAS, the Hybrid Video Analysis System, with MUBS, the upstream management and business system. The work is divided into HVAS API hardening, MUBS backend development, web dashboard development, mobile task handling, notification integration, and final demonstration preparation.

Target milestone: final demonstration in May 2026.

### 1.1 Workstreams

| Workstream | Scope | Repository |
|---|---|---|
| HVAS API hardening | Prepare HVAS APIs, event payloads, status feedback, and demo support for MUBS integration. | HVAS repository |
| MUBS backend | Build the core ticket, dispatch, authentication, and notification APIs. | Separate MUBS repository |
| MUBS web dashboard | Build the dispatcher-facing management dashboard. | Separate MUBS repository |
| MUBS H5 mobile client | Build the field-worker task handling interface. | Separate MUBS repository |
| Demo preparation | Prepare seed data, mock event flows, and stability verification. | Both systems |

---

## 2. Phase 1: HVAS API Hardening

Goal: make HVAS ready to be consumed by MUBS through stable APIs, complete event payloads, signed webhooks, and reliable demo utilities.

### 2.1 Webhook Payload Standardization

Problem: downstream systems require a stable event identifier and integration metadata for deduplication, routing, and ticket mapping.

Required changes:

- Include the MongoDB `_id` value as `event_id` after event persistence.
- Add `created_at` in ISO 8601 format.
- Add `area_code` and `group` fields from stream configuration.
- Preserve existing event details such as event type, camera ID, confidence, location, evidence URL, and object count.

Webhook payload specification:

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

Primary files:

- `backend/services/event_generator.py`
- `backend/services/webhook_service.py`

### 2.2 Webhook Signature

Problem: MUBS must verify that inbound event notifications were sent by HVAS and were not modified in transit.

Required changes:

- Add the `WEBHOOK_SECRET` environment variable.
- Compute `HMAC-SHA256(secret, request_body)` on the raw request body.
- Send the hexadecimal digest in the `X-HVAS-Signature` request header.
- Keep webhook signing optional so local development remains possible without a shared secret.

Primary file:

- `backend/services/webhook_service.py`

### 2.3 Stream Configuration Metadata

Problem: MUBS requires routing metadata to map events to operational areas and responsible teams.

Required changes:

- Add `area_code` and `group` to stream configuration.
- Preserve the same pass-through pattern used for `lat_lng` and `location`.
- Include the fields in stream creation, persistence, restoration, runtime handler construction, event documents, and webhook payloads.
- Expose the fields in the frontend stream management form.

Primary files:

- `backend/services/stream_runtime.py`
- `backend/services/stream_manager.py`
- `backend/api/stream_routes.py`
- `backend/services/event_generator.py`
- `backend/services/violation_detection.py`
- `backend/services/smoke_flame_detection.py`
- `frontend/src/views/StreamManager.vue`

### 2.4 Event Status Feedback API

Problem: downstream handling results must be written back to HVAS so event review screens can display the operational status.

Endpoint:

```text
PATCH /api/events/<event_id>/status
```

Request body:

```json
{
  "status": "resolved",
  "handled_by": "fieldworker_zhang",
  "handle_note": "Fire extinguished",
  "handle_image_url": "http://example.com/photo.jpg"
}
```

Required behavior:

- Update the event document with status fields.
- Add a `handled_at` timestamp when handling data is submitted.
- Return the updated event document on success.
- Return `400` for invalid status values or malformed IDs.
- Return `404` when the event does not exist.

Primary file:

- `backend/api/event_routes.py`

### 2.5 Incremental Event Synchronization

Problem: MUBS needs a polling fallback if webhook delivery is unavailable.

Required changes:

- Add the `since_id` query parameter to `GET /api/events`.
- When `since_id` is provided, filter events with `_id > ObjectId(since_id)`.
- Keep existing filters such as `camera_id`, `event_type`, `start_time`, and `end_time`.
- Return results in ascending `_id` order for incremental consumption.

Primary file:

- `backend/api/event_routes.py`

### 2.6 Demo Mode: Mock Event Trigger

Problem: demonstrations require repeatable event generation without depending on camera availability.

Endpoint:

```text
POST /api/events/mock
```

Request body:

```json
{
  "event_type": "smoke_flame",
  "camera_id": "east_gate_01",
  "area_code": "east_district",
  "group": "fire_team"
}
```

Required behavior:

- Generate a synthetic event with a realistic description and current timestamp.
- Save the event to MongoDB.
- Upload or reference placeholder evidence in MinIO.
- Trigger the same webhook flow used by real events.
- Enable the endpoint only when `DEMO_MODE=true`.

Primary file:

- `backend/api/event_routes.py`

### 2.7 Health Check Endpoint

Problem: operators and integration tests need a single endpoint to verify system readiness.

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

Primary file:

- `backend/api/health_routes.py`

### 2.8 Seed Data Script

Problem: dashboard and integration demonstrations require historical data.

Required behavior:

- Insert 50 to 100 historical events into MongoDB.
- Spread timestamps across the past seven days.
- Cover `smoke_flame`, `parking_violation`, and `common_space_utilization`.
- Use multiple camera IDs and realistic location metadata.
- Upload placeholder evidence images to MinIO.

Primary file:

- `scripts/seed_demo_data.py`

---

## 3. Phase 2: MUBS Backend

Goal: build the core backend for ticket management, dispatching, authentication, HVAS ingestion, and notifications.

### 3.1 Project Scaffolding

Recommended stack:

- Kotlin
- Spring Boot 3.x
- Spring Web
- Spring Security with JWT
- Spring Data JPA or Spring Data MongoDB
- Spring Mail
- PostgreSQL or MongoDB
- Docker Compose for local infrastructure

### 3.2 Data Models

#### Ticket

Core work-order entity created from HVAS events.

```text
id
hvas_event_id
event_type
camera_id
area_code
group
status
assigned_team
assigned_worker
hvas_timestamp
created_at
dispatched_at
notified_at
accepted_at
resolved_at
evidence_url
handle_note
handle_image_url
timeline
```

Status values:

- `PENDING`
- `DISPATCHED`
- `PROCESSING`
- `RESOLVED`
- `REJECTED`

#### User

Pre-seeded accounts for administration, dispatching, and field work.

Recommended roles:

- `ADMIN`
- `DISPATCHER`
- `FIELDWORKER`

#### DispatchRule

Maps an event type and area code to an operational team.

```text
event_type + area_code -> team
```

#### NotificationLog

Stores notification attempts, channels, recipients, delivery status, and timestamps.

### 3.3 HVAS Webhook Receiver

Endpoint:

```text
POST /api/v1/hvas/webhook
```

Required behavior:

- Verify the `X-HVAS-Signature` header when the shared secret is configured.
- Reject invalid signatures.
- Deduplicate events by `event_id`.
- Create a ticket with status `PENDING`.
- Trigger the dispatch rule engine.
- Return the created ticket identifier and assigned team.

### 3.4 Dispatch Rule Engine

Required behavior:

- Match new tickets by `(event_type, area_code)`.
- Assign the matched operational team.
- Update the ticket status to `DISPATCHED`.
- Record `dispatched_at` and append a timeline entry.
- Trigger notifications for the assigned team.
- Run a scheduled job that returns stale `DISPATCHED` tickets to `PENDING` after a configured timeout.

### 3.5 Authentication

Required endpoints and behavior:

- `POST /api/auth/login` accepts username and password.
- Successful login returns a JWT.
- All `/api/**` routes require JWT authentication except login and health endpoints.
- Role-based access controls should separate administrator, dispatcher, and field-worker operations.

Pre-seeded users:

- `admin`
- `dispatcher`
- `fieldworker`

### 3.6 Ticket API

```text
GET    /api/tickets
GET    /api/tickets/{id}
PATCH  /api/tickets/{id}/status
GET    /api/tickets/stats
```

Expected capabilities:

- List tickets with filters for status, event type, area code, team, and assignee.
- Retrieve ticket details with full timeline.
- Update ticket status for dispatch, acceptance, resolution, rejection, or return.
- Return dashboard statistics.

### 3.7 HVAS Polling Fallback

Required behavior:

- Run a scheduled task every 60 seconds when enabled.
- Call `GET <HVAS_URL>/api/events?since_id=<last_seen_id>`.
- Create tickets for events not yet ingested.
- Update the last seen ID after successful processing.
- Keep this mechanism disabled by default and enable it through configuration.

### 3.8 Notification Service

Notification channels:

- Email through Spring Mail and SMTP.
- H5 push through WebSocket or short-polling fallback.

Required behavior:

- Send notifications when tickets are dispatched.
- Notify team leads and assigned field workers.
- Store each notification attempt in `NotificationLog`.
- Support retry or failure marking for transient delivery errors.

---

## 4. Phase 3: MUBS Web Dashboard

Goal: provide a dispatcher-facing management interface for event handling and operational monitoring.

### 4.1 Project Setup

Recommended stack:

- Vue 3
- TypeScript
- Vite
- Element Plus or Ant Design Vue
- ECharts

### 4.2 Pages

| Page | Purpose |
|---|---|
| Login | Authenticate dashboard users. |
| Dashboard | Display daily and weekly event counts, event type distribution, response time, and unresolved ticket statistics. |
| Ticket List | Provide filtering, sorting, pagination, and bulk operations. |
| Ticket Detail | Show evidence, event metadata, full timeline, and manual dispatch or return actions. |
| Dispatch Rules | Manage event-type and area-code routing rules. |
| User Management | Manage pre-seeded or configured users. |

### 4.3 Real-Time Updates

Required behavior:

- Connect to the backend WebSocket channel after login.
- Display new ticket notifications.
- Refresh ticket lists when new events arrive.
- Show visible badges or toast notifications for urgent alerts.

---

## 5. Phase 4: MUBS H5 Mobile Client

Goal: provide a mobile task-handling interface for field workers.

### 5.1 Project Setup

Recommended stack:

- Vue 3
- Vant UI
- Shared API layer with the web dashboard
- Responsive layout for mobile browsers, including embedded WebView environments

### 5.2 Pages

| Page | Purpose |
|---|---|
| Login | Authenticate field workers. |
| Task List | Display pending and active tasks assigned to the current worker or team. |
| Task Detail | Show HVAS evidence image, event information, location, and coordinates. |
| Handle Task | Upload handling photo, enter notes, and submit resolution. |
| History | Display resolved or rejected historical tasks. |

### 5.3 User Experience Requirements

- A field worker should complete a task in no more than three main steps: open task, upload evidence, submit.
- Notifications should deep-link directly to task detail.
- The client should tolerate unstable networks by queuing submissions or clearly indicating retry status.

---

## 6. Phase 5: Notification Integration

Goal: provide reliable event notifications through email and H5 push channels.

### 6.1 Email Notification

Required work:

- Select and configure an SMTP provider.
- Create an HTML email template containing event type, time, location, evidence thumbnail, and action link.
- Send emails to team leads or dispatchers when tickets are dispatched.
- Log delivery attempts and failures.

Candidate providers:

- SendGrid
- 163 Mail
- QQ Enterprise Mail
- Internal SMTP service

### 6.2 H5 Real-Time Push

Required work:

- Add a WebSocket endpoint to the MUBS backend.
- Connect the H5 client after login.
- Push new task assignments, timeout warnings, and system alerts.
- Fall back to polling when WebSocket is unavailable.

---

## 7. Phase 6: Demonstration Preparation

Goal: ensure the final demonstration is repeatable and resilient.

### 7.1 Seed Data

Required work:

- Run `scripts/seed_demo_data.py` to populate HVAS with seven days of historical events.
- Run the MUBS seed script to create matching tickets across multiple statuses.
- Confirm that dashboard charts are populated.

### 7.2 Demo Mode

Required setup:

- Enable `DEMO_MODE=true` in HVAS.
- Use `POST /api/events/mock` to trigger events without live cameras.
- Provide a MUBS demo panel or test utility for controlled event scenarios.
- Prepare sample videos for stream-based demonstrations.

### 7.3 Demonstration Scenarios

#### Scenario 1: Smoke or Flame Event

1. HVAS detects smoke or flame.
2. HVAS saves evidence and sends a signed webhook.
3. MUBS receives the alert and creates a ticket.
4. The dispatch engine assigns the ticket to the fire team.
5. The H5 client receives a task notification.
6. The field worker marks the task as resolved.
7. The dashboard displays the resolved status.

#### Scenario 2: Parking Violation Event

1. HVAS detects a vehicle in a no-parking zone after the dwell threshold is reached.
2. HVAS saves evidence and sends a signed webhook.
3. MUBS receives the alert and creates a ticket.
4. The dispatch engine assigns the ticket to the traffic or facility team.
5. The dispatcher reviews evidence and updates handling status.
6. HVAS receives or exposes the updated status for event review.

### 7.4 Stability Checklist

- [ ] HVAS streams remain stable for at least 30 minutes.
- [ ] HVAS event storage and MinIO evidence upload are healthy.
- [ ] MUBS webhook receiver handles rapid event bursts without duplicate tickets.
- [ ] MUBS polling fallback can ingest missed events.
- [ ] H5 pages load within three seconds on target mobile devices.
- [ ] Email delivery is confirmed with the selected SMTP provider.
- [ ] Mock events work when live cameras are unavailable.
- [ ] Operators have credentials and scenario scripts before the demonstration.

---

## 8. Timeline Estimate

| Phase | Scope | Dependencies |
|---|---|---|
| Phase 1 | HVAS API hardening | None |
| Phase 2 | MUBS backend | Phase 1 webhook and event API contract |
| Phase 3 | MUBS web dashboard | Phase 2 ticket API |
| Phase 4 | MUBS H5 mobile client | Phase 2 ticket API |
| Phase 5 | Notification integration | Phase 2 and SMTP provider selection |
| Phase 6 | Demonstration preparation | All preceding phases |

Phase 3 and Phase 4 can proceed in parallel once the Phase 2 API contract is stable.

---

## 9. HVAS File Impact Summary

| Phase 1 Task | Files |
|---|---|
| Webhook payload standardization | `backend/services/event_generator.py`, `backend/services/webhook_service.py` |
| Webhook signature | `backend/services/webhook_service.py`, environment configuration |
| Stream metadata | `backend/services/stream_runtime.py`, `backend/services/stream_manager.py`, `backend/api/stream_routes.py`, `backend/services/event_generator.py`, `backend/services/violation_detection.py`, `backend/services/smoke_flame_detection.py`, `frontend/src/views/StreamManager.vue` |
| Event status API | `backend/api/event_routes.py` |
| Incremental synchronization | `backend/api/event_routes.py` |
| Demo mode | `backend/api/event_routes.py`, environment configuration |
| Health check | `backend/api/health_routes.py` |
| Seed data | `scripts/seed_demo_data.py` |

---

## 10. Integration Principles

- HVAS remains the source of truth for detection events and evidence URLs.
- MUBS remains the source of truth for dispatch workflow and field handling.
- `event_id` is the cross-system deduplication key.
- Webhook delivery should be treated as at-least-once.
- Polling fallback should be idempotent.
- Event evidence URLs should remain stable after ticket creation.
- Handling feedback should preserve auditability through status timestamps and timeline entries.
