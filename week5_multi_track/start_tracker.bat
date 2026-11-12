@echo off
rem Multi-Object Tracker (local only): http://127.0.0.1:8768 . Keep this window open while using it.
chcp 65001 >nul
cd /d "%~dp0"
set PY=..\week2_mask_tracking\.venv\Scripts\python.exe
if not exist "%PY%" (
    echo [error] week2_mask_tracking\.venv not found. Run week2_mask_tracking\run_week2.bat once first.
    pause & exit /b 1
)
set PYTHONIOENCODING=utf-8
set PYTHONWARNINGS=ignore
"%PY%" track_app.py
pause
