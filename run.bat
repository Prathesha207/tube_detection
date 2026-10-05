@echo off
setlocal enabledelayedexpansion
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
) else if exist "%~dp0backend" (
    set "BACKEND_DIR=%~dp0backend"
) else (
    set "BACKEND_DIR=%~dp0vision-ai-backend"
)

if not exist "!BACKEND_DIR!\.venv\Scripts\python.exe" (
    echo =======================================================
    echo [INFO] First-time setup detected: .venv is missing.
    echo Running automated setup to configure Python, CUDA, and dependencies...
    echo =======================================================
    echo.
    call "%~dp0setup.bat"
    if !ERRORLEVEL! NEQ 0 (
        echo.
        echo [ERROR] Setup failed. Please check the logs above.
        pause
        exit /b 1
    )
)

set "FRONTEND_DIR="
if exist "%~dp0frontend\package.json" (
    set "FRONTEND_DIR=%~dp0frontend"
) else (
    set "FRONTEND_DIR=%~dp0vision-ai-frontend"
)

if not exist "!FRONTEND_DIR!\node_modules" (
    echo =======================================================
    echo [INFO] Frontend node_modules missing. Installing dependencies...
    echo =======================================================
    pushd "!FRONTEND_DIR!"
    call npm install --include=optional
    popd
)

echo [1/2] Starting Vision Monitor Backend API (Port 8000)...
start "Vision Monitor - Backend API" /d "!BACKEND_DIR!" cmd /c ".venv\Scripts\python.exe -m uvicorn app.main:app --reload --reload-dir app --port 8000"

echo [2/2] Starting Vision Monitor Frontend UI (Port 5173)...
start "Vision Monitor - Frontend UI" /d "!FRONTEND_DIR!" cmd /c "npm run dev"

echo.
echo =======================================================
echo [SUCCESS] Vision Monitor development servers are running!
echo.
echo   • Frontend Application : http://localhost:5173
echo   • Backend Health Check : http://localhost:8000/health
echo   • API Documentation    : http://localhost:8000/docs
echo.
echo Keep both console windows open while working.
echo To stop all servers, simply close the two console windows.
echo =======================================================
echo.
pause
