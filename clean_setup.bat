@echo off
setlocal enabledelayedexpansion
title Vision Monitor - Clean Setup

echo =======================================================
echo        Vision Monitor - Clean Setup (Windows)
echo =======================================================
echo.
echo This will remove the existing Python virtual environment (.venv)
echo and perform a completely clean, fresh installation from scratch.
echo.
echo Press any key to start clean setup, or close this window to cancel.
pause >nul
echo.

cd /d %~dp0

where powershell >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PowerShell is required to run the automated setup but was not found.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0vision-ai-backend\setup_windows.ps1" -Clean
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo =======================================================
    echo [ERROR] Clean setup encountered an issue. Review the logs above.
    echo =======================================================
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo =======================================================
echo [SUCCESS] Clean setup completed successfully!
echo To launch Vision Monitor, simply double-click run.bat
echo =======================================================
pause
