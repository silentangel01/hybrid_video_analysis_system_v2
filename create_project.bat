@echo off
REM ======================================================
REM Hybrid Video Analysis System - Project Structure Generator (Windows)
REM Author: Your Name
REM Usage: Double-click or run in CMD to auto-create full folder/file structure.
REM All code comments are in English as requested.
REM ======================================================

REM === CONFIGURE YOUR PROJECT ROOT PATH HERE ===
SET PROJECT_ROOT=E:\Project Code

REM === DO NOT MODIFY BELOW THIS LINE ===
echo [INFO] Creating project at: %PROJECT_ROOT%
mkdir "%PROJECT_ROOT%" 2>nul

cd /d "%PROJECT_ROOT%"

REM --- Create core directories ---
mkdir backend backend\api backend\services backend\models backend\config backend\utils
mkdir ml_models ml_models\yolov8 ml_models\yolov8\weights
mkdir storage
mkdir frontend frontend\public frontend\src frontend\src\components frontend\src\views frontend\src\api frontend\src\router frontend\src\store
mkdir tests
mkdir scripts
mkdir docs

REM --- Create empty files with placeholder content (English comments) ---

REM Backend main & API routes
type nul > backend\main.py
type nul > backend\api\__init__.py
type nul > backend\api\event_routes.py
type nul > backend\api\alert_routes.py

REM Backend services
type nul > backend\services\__init__.py
type nul > backend\services\video_ingestion.py
type nul > backend\services\violation_detection.py
type nul > backend\services\common_space_detection.py
type nul > backend\services\smoke_flame_detection.py
type nul > backend\services\event_generator.py

REM Backend models
type nul > backend\models\__init__.py
type nul > backend\models\event.py
type nul > backend\models\alert.py

REM Backend config
type nul > backend\config\__init__.py
type nul > backend\config\settings.py
type nul > backend\config\database.py

REM Backend utils (including our frame_capture.py)
echo """Frame capture utility for RTSP and local video sources.""" > backend\utils\frame_capture.py
echo """Utilities for OpenCV frame processing and bounding box operations.""" > backend\utils\bbox_utils.py

REM ML Models (YOLOv8)
echo """Load YOLOv8 models from weights directory.""" > ml_models\yolov8\model_loader.py
echo """Run inference using loaded YOLOv8 models.""" > ml_models\yolov8\inference.py
type nul > ml_models\yolov8\weights\vehicle_det.pt
type nul > ml_models\yolov8\weights\smoke_flame.pt

REM Storage layer
echo """MongoDB client wrapper for event metadata storage.""" > storage\mongodb_client.py
echo """MinIO client wrapper for conditional image upload after detection.""" > storage\minio_client.py

REM Frontend structure (Vue/React placeholders)
type nul > frontend\src\main.js
type nul > frontend\src\App.vue
type nul > frontend\src\components\AlertList.vue
type nul > frontend\src\components\EventFilter.vue
type nul > frontend\src\views\Dashboard.vue
type nul > frontend\src\views\History.vue
type nul > frontend\src\api\backend_api.js
type nul > frontend\src\router\index.js
type nul > frontend\src\store\index.js

REM Tests
type nul > tests\test_video_ingestion.py
type nul > tests\test_detection_logic.py
type nul > tests\test_storage_integration.py

REM Scripts
echo """Initialize MinIO bucket for storing event images.""" > scripts\setup_minio_bucket.py
echo """Insert sample test events into MongoDB.""" > scripts\load_test_events.py

REM Docs
echo "# System Architecture" > docs\architecture.md
echo "openapi: 3.0.0" > docs\api_spec.yaml
echo "# User Manual" > docs\user_manual.md

REM Root level files
echo ultralytics^
fastapi^
uvicorn^
pymongo^
minio^
opencv-python-headless^
python-dotenv^
> requirements.txt

echo HOST=0.0.0.0^
PORT=8000^
MONGO_URI=mongodb://localhost:27017^
MINIO_ENDPOINT=localhost:9000^
MINIO_ACCESS_KEY=minioadmin^
MINIO_SECRET_KEY=minioadmin^
YOLOV8_VEHICLE_MODEL=./ml_models/yolov8/weights/vehicle_det.pt^
> .env.example

echo version: '3.8'^

services:^
  backend:^
    build: .^
    ports:^
      - "8000:8000"^
    depends_on:^
      - mongo^
      - minio^
    environment:^
      - MONGO_URI=mongodb://mongo:27017^
      - MINIO_ENDPOINT=minio:9000^
  mongo:^
    image: mongo:latest^
    ports:^
      - "27017:27017"^
  minio:^
    image: minio/minio^
    command: server /data --console-address ":9001"^
    ports:^
      - "9000:9000"^
      - "9001:9001"^
    environment:^
      - MINIO_ROOT_USER=minioadmin^
      - MINIO_ROOT_PASSWORD=minioadmin^
> docker-compose.yml

echo # Hybrid Video Analysis System^

This project detects parking violations, smoke/flame, and common space usage from multi-source video streams.^

## Setup^

```bash^
pip install -r requirements.txt^
```^

## Run^

```bash^
python backend/main.py^
```^
> README.md

REM --- Completion message ---
echo.
echo =====================================================================
echo ✅ Project structure successfully generated at:
echo %PROJECT_ROOT%
echo.
echo 📁 Folders created, 📄 files initialized with English-comment placeholders.
echo 🚀 You can now start coding your modules!
echo =====================================================================
pause