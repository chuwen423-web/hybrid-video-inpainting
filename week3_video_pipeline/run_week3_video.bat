@echo off
rem Week 3 video pipeline: detection -> SAM 2 -> per-frame removal (Telea / LaMa) -> evaluation.
rem Uses the Week 2 environment (GPU PyTorch + SAM 2). clip_C replacement needs ComfyUI: see README.md.
rem Usage: run_week3_video.bat            all clips
rem        run_week3_video.bat clip_A     one clip
chcp 65001 >nul
cd /d "%~dp0"
set PY=..\week2_mask_tracking\.venv\Scripts\python.exe
if not exist "%PY%" (
    echo [error] Week 2 environment not found. Run ..\week2_mask_tracking\run_week2.bat once first.
    pause & exit /b 1
)
set PYTHONIOENCODING=utf-8
"%PY%" pipeline.py %* || (pause & exit /b 1)
"%PY%" evaluate_video.py
pause
