@echo off
setlocal
title Vision Monitor

echo =======================================================
echo          Starting Vision Monitor Development Mode
echo =======================================================
echo.

cd /d "%~dp0"

set "BACKEND_DIR="
if exist "%~dp0backend\.venv\Scripts\python.exe" (
    set "BACKEND_DIR=%~dp0backend"
) else if exist "%~dp0vision-ai-backend\.venv\Scripts\python.exe" (
    set "BACKEND_DIR=%~dp0vision-ai-backend"
) else (
    echo [ERROR] Backend virtual environment is missing.
    echo Run setup.bat once, then double-click run.bat again.
    pause
    exit /b 1
)

"%BACKEND_DIR%\.venv\Scripts\python.exe" -c "import uvicorn, fastapi, ultralytics, torch, cv2" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Required backend packages are missing or not working.
    echo Run setup.bat to repair the Python environment, then run run.bat again.
    pause
    exit /b 1
)

set "FRONTEND_DIR="
if exist "%~dp0frontend\package.json" (
    set "FRONTEND_DIR=%~dp0frontend"
) else if exist "%~dp0vision-ai-frontend\package.json" (
    set "FRONTEND_DIR=%~dp0vision-ai-frontend"
) else (
    echo [ERROR] Could not find frontend\package.json.
    pause
    exit /b 1
)

if exist "%LOCALAPPDATA%\Programs\node-v20.18.0-win-x64\npm.cmd" (
    set "PATH=%LOCALAPPDATA%\Programs\node-v20.18.0-win-x64;%PATH%"
)
if exist "%ProgramFiles%\nodejs\npm.cmd" (
    set "PATH=%ProgramFiles%\nodejs;%PATH%"
)
if exist "%ProgramFiles(x86)%\nodejs\npm.cmd" (
    set "PATH=%ProgramFiles(x86)%\nodejs;%PATH%"
)

where npm.cmd >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm.cmd was not found.
    echo Install Node.js or run setup.bat to install the bundled Node.js runtime.
    pause
    exit /b 1
)

if not exist "%FRONTEND_DIR%\node_modules\.bin\vite.cmd" (
    echo [ERROR] Frontend dependencies are missing.
    echo Run setup.bat once to install them, then run run.bat again.
    pause
    exit /b 1
)

echo [1/2] Starting Vision Monitor Backend API (Port 8000)...
start "Vision Monitor - Backend API" /d "%BACKEND_DIR%" cmd /k ".venv\Scripts\python.exe -m uvicorn app.main:app --reload --reload-dir app --host 127.0.0.1 --port 8000 --no-use-colors"
if errorlevel 1 (
    echo [ERROR] Could not start the backend console.
    pause
    exit /b 1
)

echo [2/2] Starting Vision Monitor Frontend UI (Port 5173)...
start "Vision Monitor - Frontend UI" /d "%FRONTEND_DIR%" cmd /k "npm.cmd run dev"
if errorlevel 1 (
    echo [ERROR] Could not start the frontend console.
    pause
    exit /b 1
)

echo.
echo =======================================================
echo [SUCCESS] Development server windows were opened.
echo.
echo   • Frontend Application : http://localhost:5173
echo   • Backend Health Check : http://localhost:8000/health
echo   • API Documentation    : http://localhost:8000/docs
echo.
echo Wait for "Local: http://localhost:5173/" in the frontend console.
echo Keep both console windows open while working.
echo =======================================================
echo.
pause