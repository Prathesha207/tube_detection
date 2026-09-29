#!/usr/bin/env bash
# Start backend and frontend together for Linux development.
set -euo pipefail

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$(cd "$BACKEND_DIR/../vision-ai-frontend" && pwd)"

if [[ ! -x "$BACKEND_DIR/.venv/bin/python" ]]; then
  echo "Backend environment is missing. Creating environment and running setup..."
  bash "$BACKEND_DIR/setup_linux.sh"
fi

# Fast pre-flight check (<0.05s): installs backend requirements if anything is missing
if ! "$BACKEND_DIR/.venv/bin/python" -c "import fastapi, uvicorn, cv2, torch, yaml, lap" 2>/dev/null; then
  echo "Configuring backend environment (one-time silent setup)..."
  "$BACKEND_DIR/.venv/bin/python" -m pip install --quiet -r "$BACKEND_DIR/requirements.txt"
fi

if ! "$BACKEND_DIR/.venv/bin/python" -c "import ultralytics, cv2, numpy, torch, torchvision" 2>/dev/null; then
  "$BACKEND_DIR/.venv/bin/python" -m pip install --quiet -r "$BACKEND_DIR/app/ml/tube/requirements.txt"
fi

if [[ ! -x "$FRONTEND_DIR/node_modules/.bin/vite" ]]; then
  echo "Frontend dependencies are missing. Installing node_modules..."
  cd "$FRONTEND_DIR" && npm install --include=optional
  cd "$BACKEND_DIR"
fi

# Configurable base port (default 8000, or override with BACKEND_PORT=8080)
TARGET_PORT="${BACKEND_PORT:-${PORT:-8000}}"

# Clean up any lingering dev server process on TARGET_PORT from previous sessions
if command -v fuser >/dev/null 2>&1; then
  fuser -k "${TARGET_PORT}/tcp" 2>/dev/null || true
  sleep 0.3
fi

# Function to check if a port is available without killing existing processes
is_port_free() {
  "$BACKEND_DIR/.venv/bin/python" -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(('0.0.0.0', int(sys.argv[1])))
    s.close()
    sys.exit(0)
except OSError:
    sys.exit(1)
" "$1"
}

# Auto-select an available port if the target port is already in use by another service
PORT="$TARGET_PORT"
while ! is_port_free "$PORT"; do
  echo "Port $PORT is already in use by another application. Trying port $((PORT + 1))..."
  PORT=$((PORT + 1))
  if [[ "$PORT" -gt "$((TARGET_PORT + 50))" ]]; then
    echo "ERROR: Could not find an open port in range $TARGET_PORT-$PORT."
    exit 1
  fi
done

if [[ "$PORT" != "$TARGET_PORT" ]]; then
  echo "Note: Port $TARGET_PORT was busy. Using available port $PORT for backend."
fi

cleanup() {
  echo ""
  echo "Stopping development servers..."
  if [[ -n "${BACKEND_PID:-}" ]]; then
    pkill -P "$BACKEND_PID" 2>/dev/null || true
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]]; then
    pkill -P "$FRONTEND_PID" 2>/dev/null || true
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$BACKEND_DIR"
.venv/bin/python -m uvicorn app.main:app --reload --reload-dir app --reload-exclude ".venv*" --reload-exclude "data*" --reload-exclude "recordings*" --port "$PORT" &
BACKEND_PID=$!

cd "$FRONTEND_DIR"
VITE_API_BASE_URL="http://localhost:$PORT" npm run dev -- --host 0.0.0.0 &
FRONTEND_PID=$!

echo ""
echo "=================================================="
echo " Vision Monitor Dev Servers Started"
echo " Backend:  http://localhost:$PORT"
echo " Frontend: http://localhost:5173"
echo "=================================================="
echo ""

wait
