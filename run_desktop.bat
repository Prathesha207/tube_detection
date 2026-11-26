@echo off
setlocal
title Vision Monitor Desktop App

echo =======================================================
echo          Starting Vision Monitor Desktop App
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
    echo Run setup.bat once, then double-click run_desktop.bat again.
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
    echo Install Node.js or run setup.bat.
    pause
    exit /b 1
)

REM Free ports 8000 & 5173 if lingering from a previous session
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING 2^>nul') do (
    taskkill /f /pid %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5173 ^| findstr LISTENING 2^>nul') do (
    taskkill /f /pid %%a >nul 2>&1
)

echo [1/2] Starting Vision Monitor Backend API (Port 8000)...
start "Vision Monitor - Backend API" /d "%BACKEND_DIR%" cmd /k ".venv\Scripts\python.exe -m uvicorn app.main:app --reload --reload-dir app --host 127.0.0.1 --port 8000 --no-use-colors"

echo [2/2] Launching Vision Monitor Desktop Window...
cd /d "%FRONTEND_DIR%"
npm.cmd run electron-dev
