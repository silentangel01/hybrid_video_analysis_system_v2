#!/usr/bin/env python3
"""
Seed Demo Data (Phase 1.8)

Insert 50-100 historical events into MongoDB spanning the past 7 days.
Covers all event types with multiple camera_ids and realistic timestamps.
Uploads 1x1 placeholder JPEG to MinIO for each event.

Usage:
    python -m scripts.seed_demo_data          # from project root
    python scripts/seed_demo_data.py          # direct invocation

Environment variables (all have defaults):
    MONGO_URI          mongodb://localhost:27017
    MINIO_ENDPOINT     localhost:9000
    MINIO_ACCESS_KEY   minioadmin
    MINIO_SECRET_KEY   minioadmin
    MINIO_BUCKET       video-events
"""

import io
import os
import random
import struct
import sys
import time
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `storage.*` / `backend.*` resolve.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from storage.mongodb_client import MongoDBClient  # noqa: E402
from storage.minio_client import MinIOClient       # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "hvas")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "video-events")

TOTAL_EVENTS = 80  # target count

# Camera presets — (camera_id, lat_lng, location, area_code, group)
CAMERAS = [
    ("east_gate_01", "31.2304, 121.4737", "East Gate Entrance", "east_district", "fire_team"),
    ("west_gate_02", "31.2290, 121.4680", "West Gate Parking Lot", "west_district", "traffic_team"),
    ("north_plaza_03", "31.2320, 121.4710", "North Plaza", "north_district", "security_team"),
    ("south_lobby_04", "31.2280, 121.4750", "South Lobby", "south_district", "fire_team"),
    ("warehouse_05", "31.2310, 121.4660", "Warehouse B", "east_district", "fire_team"),
]

# Event type templates
TEMPLATES = {
    "smoke_flame": {
        "weight": 30,
        "confidence_range": (0.65, 0.95),
        "detection_stages": ["yolo_initial", "qwen_verified"],
        "descriptions": [
            "Smoke detected (conf={conf:.2f})",
            "Flame detected (conf={conf:.2f})",
            "Smoke/flame detected in {count} regions (max_conf={conf:.2f})",
        ],
    },
    "parking_violation": {
        "weight": 35,
        "confidence_range": (0.70, 0.98),
        "detection_stages": ["yolo_initial"],
        "descriptions": [
            "Vehicle in no-parking zone (conf={conf:.2f})",
            "{count} vehicles in no-parking zone (max_conf={conf:.2f})",
        ],
    },
    "common_space_utilization": {
        "weight": 35,
        "confidence_range": (1.0, 1.0),
        "detection_stages": ["qwen_vl_analysis"],
        "descriptions": [
            "Public space analysis: {people} people, {occupancy} occupancy",
            "Public space analysis: {people} people, activities: walking, sitting",
        ],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_placeholder_jpeg() -> bytes:
    """Minimal valid 1x1 white JPEG (approx 631 bytes)."""
    # Smallest valid JFIF: SOI + APP0 + DQT + SOF0 + DHT + SOS + data + EOI
    # For simplicity, use a pre-built tiny JPEG via struct.
    # Actually, let's just create a small colored JPEG with raw bytes.
    # A simpler approach: use a known minimal JPEG.
    try:
        import numpy as np
        import cv2
        # 16x16 solid colour placeholder
        colour = random.choice([
            (40, 40, 200),   # red-ish (BGR)
            (30, 180, 240),  # orange-ish
            (200, 200, 50),  # cyan-ish
        ])
        img = np.full((16, 16, 3), colour, dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()
    except ImportError:
        # Fallback: minimal valid JPEG (1x1 white pixel)
        return (
            b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
            b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t'
            b'\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a'
            b'\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342'
            b'\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00'
            b'\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00'
            b'\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b'
            b'\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05'
            b'\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A'
            b'\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br'
            b'\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghij'
            b'stuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97'
            b'\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5'
            b'\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3'
            b'\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9'
            b'\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa'
            b'\xff\xda\x00\x08\x01\x01\x00\x00?\x00T\xdb\x9e\xa7\xa3@\x1f\xff\xd9'
        )


def _random_timestamp_in_past_7_days() -> float:
    """Return a random Unix timestamp within the past 7 days."""
    now = time.time()
    offset = random.uniform(0, 7 * 24 * 3600)
    return now - offset


def _pick_event_type() -> str:
    types = list(TEMPLATES.keys())
    weights = [TEMPLATES[t]["weight"] for t in types]
    return random.choices(types, weights=weights, k=1)[0]


def _build_event(event_type: str, camera: tuple, ts: float, image_url: str) -> dict:
    camera_id, lat_lng, location, area_code, group = camera
    tpl = TEMPLATES[event_type]
    conf = round(random.uniform(*tpl["confidence_range"]), 2)
    stage = random.choice(tpl["detection_stages"])
    count = random.randint(1, 4)
    people = random.randint(3, 30)
    occupancy = random.choice(["low", "moderate", "high"])
    desc = random.choice(tpl["descriptions"]).format(
        conf=conf, count=count, people=people, occupancy=occupancy,
    )

    doc = {
        "camera_id": camera_id,
        "timestamp": ts,
        "event_type": event_type,
        "confidence": conf,
        "image_url": image_url,
        "description": desc,
        "detection_stage": stage,
        "object_count": count if event_type != "common_space_utilization" else None,
        "lat_lng": lat_lng,
        "location": location,
        "area_code": area_code,
        "group": group,
        "created_at": datetime.utcfromtimestamp(ts),
        "processed_at": datetime.utcfromtimestamp(ts + random.uniform(0.1, 2.0)),
    }

    if event_type == "common_space_utilization":
        doc["analysis_summary"] = {
            "estimated_people_count": people,
            "space_occupancy": occupancy,
            "activity_types": random.sample(
                ["walking", "sitting", "talking", "standing", "running"], k=random.randint(1, 3)
            ),
            "safety_concerns": random.random() < 0.15,
        }

    # Remove None values
    return {k: v for k, v in doc.items() if v is not None}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Connecting to MongoDB: {MONGO_URI} / {MONGO_DB}")
    mongo = MongoDBClient(MONGO_URI, MONGO_DB)
    if not mongo.health_check():
        print("ERROR: Cannot connect to MongoDB")
        sys.exit(1)

    print(f"Connecting to MinIO: {MINIO_ENDPOINT} / {MINIO_BUCKET}")
    minio = MinIOClient(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        bucket_name=MINIO_BUCKET,
        secure=False,
    )
    if minio.client is None:
        print("ERROR: Cannot connect to MinIO")
        sys.exit(1)

    placeholder = _make_placeholder_jpeg()
    print(f"Placeholder image size: {len(placeholder)} bytes")

    inserted = 0
    for i in range(TOTAL_EVENTS):
        event_type = _pick_event_type()
        camera = random.choice(CAMERAS)
        ts = _random_timestamp_in_past_7_days()

        # Upload placeholder to MinIO
        image_url = minio.upload_frame(
            image_data=placeholder,
            camera_id=camera[0],
            timestamp=ts,
            event_type=event_type,
        )
        if not image_url:
            image_url = f"http://{MINIO_ENDPOINT}/{MINIO_BUCKET}/mock/placeholder_{i}.jpg"

        doc = _build_event(event_type, camera, ts, image_url)
        try:
            mongo.collection.insert_one(doc)
            inserted += 1
        except Exception as e:
            print(f"  Failed to insert event {i}: {e}")

        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{TOTAL_EVENTS}")

    print(f"\nDone! Inserted {inserted}/{TOTAL_EVENTS} events into MongoDB.")
    print(f"  Database: {MONGO_DB}")
    print(f"  Collection: events")

    # Quick summary
    pipeline = [
        {"$group": {"_id": "$event_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    stats = list(mongo.collection.aggregate(pipeline))
    for s in stats:
        print(f"  {s['_id']}: {s['count']}")

    mongo.close()


if __name__ == "__main__":
    main()
