@echo off
setlocal enabledelayedexpansion
title Vision Monitor Setup

echo =======================================================
echo          Vision Monitor Setup (Windows)
echo =======================================================
echo.

cd /d "%~dp0"

where powershell >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PowerShell is required to run the automated setup but was not found.
    pause
    exit /b 1
)

set "SETUP_SCRIPT="
if exist "%~dp0backend\setup_windows.ps1" (
    set "SETUP_SCRIPT=%~dp0backend\setup_windows.ps1"
) else if exist "%~dp0vision-ai-backend\setup_windows.ps1" (
    set "SETUP_SCRIPT=%~dp0vision-ai-backend\setup_windows.ps1"
)

if "%SETUP_SCRIPT%"=="" (
    echo [ERROR] setup_windows.ps1 could not be found in backend\ or vision-ai-backend\.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "!SETUP_SCRIPT!" %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo =======================================================
    echo [ERROR] Setup encountered an issue. Review the logs above.
    echo =======================================================
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo =======================================================
echo [SUCCESS] Setup completed successfully!
echo To launch Vision Monitor, simply double-click run.bat
echo =======================================================
pause
