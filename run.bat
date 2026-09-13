@echo off
echo Starting Vision Monitor Development Servers...

cd /d "%~dp0vision-ai-backend"
echo Starting Backend...
if exist ".venv\Scripts\activate.bat" (
    start "Vision Monitor Backend" /d "%~dp0vision-ai-backend" cmd /c "call .venv\Scripts\activate.bat && uvicorn app.main:app --reload --port 8000"
) else (
    start "Vision Monitor Backend" /d "%~dp0vision-ai-backend" cmd /c "uvicorn app.main:app --reload --port 8000"
)

cd /d "%~dp0vision-ai-frontend"
echo Starting Frontend...
start "Vision Monitor Frontend" /d "%~dp0vision-ai-frontend" cmd /c "npm run dev"

echo.
echo ==================================================
echo Vision Monitor is running!
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:5173
echo.
echo Close the two command prompt windows to stop the servers.
echo ==================================================
pause
