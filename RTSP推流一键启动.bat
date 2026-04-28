@echo off
chcp 65001 >nul 2>&1
title RTSP 推流服务启动器

echo.
echo ======================================================
echo          RTSP 推流服务 自动启动
echo ======================================================
echo.

:: 1. 切换到 MediaMTX 目录
cd /d "D:\Brower_Download\mediamtx_v1.17.0_windows_amd64"
if %errorlevel% neq 0 (
    echo [错误] MediaMTX 目录不存在，请检查路径！
    pause & exit /b
)

echo [1/2] 正在启动 MediaMTX 服务器...
:: 使用 start 启动，标题为 MediaMTX_Server
start "MediaMTX_Server" cmd /k mediamtx.exe
timeout /t 3 /nobreak >nul

echo [2/2] 正在启动 FFmpeg 推流...
:: 设置 FFmpeg 路径 (请确认此路径是否正确，截图中看起来路径重复了两层文件夹)
set "FFMPEG_PATH=D:\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"

if not exist "%FFMPEG_PATH%" (
    echo [错误] 找不到 ffmpeg.exe，请检查路径是否正确。
    echo 当前检测路径: %FFMPEG_PATH%
    pause & exit /b
)

:: === 修改重点在这里 ===
:: 去掉了 cmd /k，直接使用 start 调用 exe，这样能避免路径解析的语法错误
:: 确保输入文件路径如果有空格也要加引号
start "FFmpeg_Stream" "%FFMPEG_PATH%" -re -stream_loop -1 -i "C:\Users\JunLing\Desktop\video\fire_smoke.mp4" -c:v libx264 -preset ultrafast -tune zerolatency -f rtsp -rtsp_transport tcp rtsp://localhost:8554/mystream

echo.
echo ✅ 服务窗口已独立弹出，请勿关闭此主窗口！
echo    (如果是黑框一闪而过，请检查 ffmpeg 路径或视频路径)
pause
exit /b
