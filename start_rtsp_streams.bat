@echo off
REM Push 3 video files to MediaMTX as RTSP streams
REM Streams will be available at:
REM   rtsp://127.0.0.1:8554/common_space
REM   rtsp://127.0.0.1:8554/digital_fence
REM   rtsp://127.0.0.1:8554/fire_smoke

set FFMPEG=D:\ffmpeg-8.1-full_build\bin\ffmpeg.exe
set VIDEO_DIR=D:\hybrid_video_analysis_system_v2\video
set MEDIAMTX=D:\mediamtx_v1.17.1_windows_amd64\mediamtx.exe
set MEDIAMTX_YML=D:\mediamtx_v1.17.1_windows_amd64\mediamtx.yml

REM Start MediaMTX RTSP server first
echo Starting MediaMTX RTSP server...
start "MediaMTX" cmd /c "%MEDIAMTX% %MEDIAMTX_YML%"
timeout /t 2 /nobreak >nul

echo Starting RTSP streams...
echo.
echo Stream 1: rtsp://127.0.0.1:8554/common_space
echo Stream 2: rtsp://127.0.0.1:8554/digital_fence
echo Stream 3: rtsp://127.0.0.1:8554/fire_smoke
echo.

start "RTSP - common_space" cmd /c "%FFMPEG% -re -stream_loop -1 -i %VIDEO_DIR%\common_space.mp4 -c:v copy -an -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/common_space"

start "RTSP - digital_fence" cmd /c "%FFMPEG% -re -stream_loop -1 -i %VIDEO_DIR%\digital_fence.mp4 -c:v copy -an -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/digital_fence"

start "RTSP - fire_smoke" cmd /c "%FFMPEG% -re -stream_loop -1 -i %VIDEO_DIR%\fire_smoke.mp4 -c:v copy -an -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/fire_smoke"

echo.
echo All streams started. Press any key to exit this window...
pause >nul
