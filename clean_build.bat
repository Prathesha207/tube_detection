@echo off
setlocal enabledelayedexpansion
title Vision Monitor - Clean Build
echo =======================================================
echo     Vision Monitor - CLEAN Build (Windows Desktop App)
echo =======================================================
echo.
echo This will clear ALL build caches and create a fresh build.
echo.

cd /d "%~dp0"

REM ── 1. Kill any running backend / frontend processes ──
echo [1/6] Killing stale processes...
set "BACKEND_DIR=backend"
if not exist "%~dp0backend" set "BACKEND_DIR=vision-ai-backend"

set "FRONTEND_DIR=frontend"
if not exist "%~dp0frontend" set "FRONTEND_DIR=vision-ai-frontend"

taskkill /F /IM "vision-monitor.exe" 2>nul
taskkill /F /IM "backend.exe" 2>nul
timeout /t 1 /nobreak >nul

REM ── 2. Clear Python bytecache ──
echo [2/6] Clearing Python bytecache...
pushd "!BACKEND_DIR!"
for /d /r %%d in (__pycache__) do (
    if exist "%%d" rd /s /q "%%d" 2>nul
)
del /s /q *.pyc 2>nul
popd

REM ── 3. Clear PyInstaller build artifacts ──
echo [3/6] Clearing PyInstaller build artifacts...
if exist "!BACKEND_DIR!\build" rd /s /q "!BACKEND_DIR!\build"
if exist "!BACKEND_DIR!\dist" rd /s /q "!BACKEND_DIR!\dist"
if exist "!BACKEND_DIR!\backend.spec" (
    echo     Keeping backend.spec (needed by build)
)

REM ── 4. Clear Electron / Vite build artifacts ──
echo [4/6] Clearing Electron and Vite build artifacts...
if exist "!FRONTEND_DIR!\dist" rd /s /q "!FRONTEND_DIR!\dist"
if exist "!FRONTEND_DIR!\dist_app" rd /s /q "!FRONTEND_DIR!\dist_app"

REM ── 5. Clear npm cache for frontend ──
echo [5/6] Clearing npm cache...
pushd "!FRONTEND_DIR!"
call npm cache clean --force 2>nul
popd

REM ── 6. Run the normal build ──
echo [6/6] Starting fresh build...
echo.
call "%~dp0build.bat"
