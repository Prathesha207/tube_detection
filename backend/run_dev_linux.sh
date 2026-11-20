#!/usr/bin/env bash
# Start backend and frontend together for Linux development.
set -euo pipefail

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$BACKEND_DIR/../frontend" ]]; then
  FRONTEND_DIR="$(cd "$BACKEND_DIR/../frontend" && pwd)"
else
  FRONTEND_DIR="$(cd "$BACKEND_DIR/../vision-ai-frontend" && pwd)"
fi

if [[ ! -x "$BACKEND_DIR/.venv/bin/python" ]]; then
  echo "Backend environment is missing. Creating environment and running setup..."
  bash "$BACKEND_DIR/setup_linux.sh"
fi

# Fast pre-flight check (<0.05s): installs backend requirements if anything is missing
if ! "$BACKEND_DIR/.venv/bin/python" -c "import fastapi, uvicorn, cv2, torch, yaml" 2>/dev/null; then
  echo "Configuring backend environment (one-time silent setup)..."
  grep -v -i -E "^(torch|torchvision|ultralytics)" "$BACKEND_DIR/requirements.txt" | "$BACKEND_DIR/.venv/bin/python" -m pip install --quiet -r /dev/stdin
  "$BACKEND_DIR/.venv/bin/python" -m pip install --quiet --no-deps "ultralytics>=8.3.0"
fi

# Clean up any local CPU torch inside the venv that may shadow the host CUDA torch
if "$BACKEND_DIR/.venv/bin/python" -c "import torch; assert not torch.cuda.is_available()" 2>/dev/null; then
  echo "Detected CPU torch in .venv. Removing to uncover host CUDA PyTorch..."
  VENV_PYSITE=$("$BACKEND_DIR/.venv/bin/python" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
  if [[ -n "$VENV_PYSITE" && -d "$VENV_PYSITE" ]]; then
    rm -rf "$VENV_PYSITE"/torch* "$VENV_PYSITE"/torchvision* 2>/dev/null || true
  fi
fi

WHL_DIR="$BACKEND_DIR/app/ml/whl"
if [[ -d "$WHL_DIR" ]]; then
  LATEST_WHL=$(find "$WHL_DIR" -maxdepth 1 -name "head_tail_analyzer*.whl" 2>/dev/null | sort -V | tail -n 1)
  if [[ -n "$LATEST_WHL" ]]; then
    INSTALLED_VER=$("$BACKEND_DIR/.venv/bin/python" -c "import importlib.metadata as md; print(md.version('head_tail_analyzer'))" 2>/dev/null || true)
    WHL_NAME="$(basename "$LATEST_WHL")"
    if [[ -z "$INSTALLED_VER" || "$WHL_NAME" != *"-$INSTALLED_VER-"* ]]; then
      echo "Installing / updating ML wheel: $WHL_NAME..."
      "$BACKEND_DIR/.venv/bin/python" -m pip install --force-reinstall --no-deps "$LATEST_WHL"
    fi
  fi
fi

# Repair an incompatible CUDA wheel automatically. Device visibility alone is
# insufficient: old PyTorch builds can see a GPU but lack kernels for its arch.
if command -v nvidia-smi >/dev/null 2>&1 && \
   ! "$BACKEND_DIR/.venv/bin/python" -c "import torch; x=torch.ones((1,), device='cuda:0'); assert (x+1).item()==2; torch.cuda.synchronize(0)" 2>/dev/null; then
  echo "NVIDIA GPU detected, but installed PyTorch cannot execute its CUDA kernels. Running setup..."
  bash "$BACKEND_DIR/setup_linux.sh"
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

# Save active port marker for frontend/electron auto-detection
PORT_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/vision-monitor"
mkdir -p "$PORT_DIR" 2>/dev/null || true
echo -n "$PORT" > "$PORT_DIR/backend.port" 2>/dev/null || true
echo -n "$PORT" > "$BACKEND_DIR/backend.port" 2>/dev/null || true

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
echo "=================================================="
.venv/bin/python -c "import torch; print(f'Active ML Hardware: {\"GPU (cuda:0 - \" + torch.cuda.get_device_name(0) + \")\" if torch.cuda.is_available() else \"CPU (WARNING: 1 FPS!)\"} | PyTorch: {torch.__version__}')" 2>/dev/null || true
echo "=================================================="
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
