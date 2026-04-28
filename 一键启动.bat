@echo off
chcp 65001
title HVAS Launcher

set MONGO_CONTAINER=hybrid_video_analysis_system_v2-mongo-1
set MINIO_CONTAINER=hybrid_video_analysis_system_v2-minio-1

echo ========== Start Docker and Backend ==========
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
timeout /t 10 /nobreak >nul
docker start %MONGO_CONTAINER%
docker start %MINIO_CONTAINER%

start "Backend" cmd /k "cd /d %~dp0 && call .\venv\Scripts\activate.bat && set YOLO_DEVICE=cuda && set WEBHOOK_SECRET=hwas-mubs-shared-secret && python .\backend\main.py"
timeout /t 3 >nul

echo ========== Start Frontend ==========
start "Frontend Dev" cmd /k "cd /d %~dp0frontend && npm run dev"
start "Upload Service" cmd /k "cd /d %~dp0frontend && npm run start-server"

echo ========== Frontend windows launched ==========
pause
