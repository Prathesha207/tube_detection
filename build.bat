@echo off
echo =======================================================
echo     Vision Monitor Build (Windows Desktop App)
echo =======================================================
if exist "%~dp0backend\build_windows_desktop.ps1" (
    cd /d "%~dp0backend"
) else (
    cd /d "%~dp0vision-ai-backend"
)
powershell -ExecutionPolicy Bypass -File build_windows_desktop.ps1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Build failed! Please check the logs above.
    pause
    exit /b %ERRORLEVEL%
)
echo.
echo =======================================================
echo Build completed successfully!
echo You can find the installer (.exe) in: vision-ai-frontend\dist_app
echo =======================================================
pause
