@echo off
rem Week 4 demo UI (local only): http://127.0.0.1:8766 . Keep this window open while presenting.
chcp 65001 >nul
cd /d "%~dp0"
set PY=..\week2_mask_tracking\.venv\Scripts\python.exe
if not exist "%PY%" (
    echo [error] week2_mask_tracking\.venv not found. Run week2_mask_tracking\run_week2.bat once first.
    pause & exit /b 1
)
if not exist "results\clip_A_w4.json" (
    echo First run: processing clip_A, clip_C, clip_D ^(about 10 minutes^)...
    "%PY%" temporal.py
)
set PYTHONIOENCODING=utf-8
"%PY%" demo_app.py
pause
