@echo off
setlocal enabledelayedexpansion
title Vision Monitor

echo =======================================================
echo          Starting Vision Monitor Development Mode
echo =======================================================
echo.

cd /d "%~dp0"

if not exist "%~dp0vision-ai-backend\.venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup.bat first to configure the environment.
    echo.
    pause
    exit /b 1
)

echo [1/2] Starting Vision Monitor Backend API (Port 8000)...
start "Vision Monitor - Backend API" /d "%~dp0vision-ai-backend" cmd /c ".venv\Scripts\python.exe -m uvicorn app.main:app --reload --reload-dir app --port 8000"

echo [2/2] Starting Vision Monitor Frontend UI (Port 5173)...
start "Vision Monitor - Frontend UI" /d "%~dp0vision-ai-frontend" cmd /c "npm run dev"

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
