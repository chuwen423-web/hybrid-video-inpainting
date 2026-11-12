@echo off
rem Week 4: de-flicker / de-halo / per-object inpainting on clip_A, clip_C, clip_D (uses Week 3 frames and SAM 2 masks).
chcp 65001 >nul
cd /d "%~dp0"
set PY=..\week2_mask_tracking\.venv\Scripts\python.exe
if not exist "..\week3_video_pipeline\work\clip_A\masks" (
    echo Week 3 masks not found: running the Week 3 pipeline for clip_A clip_C clip_D first...
    pushd ..\week3_video_pipeline
    "..\week2_mask_tracking\.venv\Scripts\python.exe" pipeline.py clip_A clip_C clip_D
    popd
)
set PYTHONIOENCODING=utf-8
"%PY%" temporal.py %*
pause
