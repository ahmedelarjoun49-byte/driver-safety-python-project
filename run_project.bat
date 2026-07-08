@echo off
echo ===================================================
echo 🔥 LAUNCHING DRIVESAFE-AI CORE MONITORING ENGINE...
echo ===================================================

:: Check if virtual environment directory exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment 'venv' not found!
    echo Please ensure it is created in this root folder.
    pause
    exit /b
)

:: Activate environment, set package directory path, and execute
call venv\Scripts\activate.bat
set PYTHONPATH=ai-driver-safety
python -m driver_safety.__main__ run --config ai-driver-safety/configs/default.yaml

pause