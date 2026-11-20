#!/usr/bin/env bash
# Build Linux desktop artifacts from a native Ubuntu/Debian machine.
# Run this script from vision-ai-backend or repo root:
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -f "$SCRIPT_DIR/run.py" ]]; then
  BACKEND_DIR="$SCRIPT_DIR"
elif [[ -f "$PWD/vision-ai-backend/run.py" ]]; then
  BACKEND_DIR="$(cd "$PWD/vision-ai-backend" && pwd -P)"
elif [[ -f "$PWD/run.py" ]]; then
  BACKEND_DIR="$PWD"
else
  echo "Error: Cannot locate vision-ai-backend directory containing run.py."
  exit 1
fi

if [[ -d "$BACKEND_DIR/../frontend" ]]; then
  FRONTEND_DIR="$(cd "$BACKEND_DIR/../frontend" && pwd -P)"
elif [[ -d "$BACKEND_DIR/../vision-ai-frontend" ]]; then
  FRONTEND_DIR="$(cd "$BACKEND_DIR/../vision-ai-frontend" && pwd -P)"
elif [[ -d "$PWD/frontend" ]]; then
  FRONTEND_DIR="$(cd "$PWD/frontend" && pwd -P)"
elif [[ -d "$PWD/vision-ai-frontend" ]]; then
  FRONTEND_DIR="$(cd "$PWD/vision-ai-frontend" && pwd -P)"
else
  echo "Error: Cannot locate frontend directory."
  exit 1
fi

VENV_DIR="$BACKEND_DIR/.venv-linux-build"
MACHINE_ARCH="$(uname -m)"

case "$MACHINE_ARCH" in
  x86_64) ELECTRON_ARCH="x64" ;;
  aarch64|arm64) ELECTRON_ARCH="arm64" ;;
  *) echo "Unsupported Linux architecture: $MACHINE_ARCH"; exit 1 ;;
esac

if [[ ! -f "$FRONTEND_DIR/package.json" ]]; then
  echo "vision-ai-frontend must be beside vision-ai-backend."
  exit 1
fi

# Detect whether this machine has an NVIDIA GPU (Desktop PCIe or Jetson Tegra)
HAS_NVIDIA_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  HAS_NVIDIA_GPU=1
elif [[ -f /etc/nv_tegra_release ]] || command -v tegrastats >/dev/null 2>&1 || [[ -d /usr/local/cuda ]]; then
  HAS_NVIDIA_GPU=1
elif python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  HAS_NVIDIA_GPU=1
fi

SKIP_PYINSTALLER="${SKIP_PYINSTALLER:-0}"
CLEAN_BUILD="${CLEAN_BUILD:-0}"
for arg in "$@"; do
  if [[ "$arg" == "--skip-pyinstaller" ]]; then
    SKIP_PYINSTALLER=1
  elif [[ "$arg" == "--clean" ]]; then
    CLEAN_BUILD=1
  fi
done

if [[ "$CLEAN_BUILD" == "1" ]]; then
  echo "========================================================"
  echo "Clean build requested. Purging ALL caches & old builds:"
  echo " - Backend build & dist directories"
  echo " - Frontend dist & dist_app (.deb, AppImage, unpacked)"
  echo " - Vite build cache (frontend/node_modules/.vite)"
  echo " - PyInstaller cache (~/.cache/pyinstaller)"
  echo " - Build virtual environment ($VENV_DIR)"
  echo "========================================================"
  rm -rf "$BACKEND_DIR/build" "$BACKEND_DIR/dist" "$FRONTEND_DIR/dist" "$FRONTEND_DIR/dist_app" "$FRONTEND_DIR/node_modules/.vite" "$VENV_DIR" ~/.cache/pyinstaller
fi

# Detect the best Python interpreter with active CUDA PyTorch
PYTHON_EXEC=""
if [[ -x "$BACKEND_DIR/.venv/bin/python" ]] && "$BACKEND_DIR/.venv/bin/python" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "✓ Detected operational CUDA PyTorch in $BACKEND_DIR/.venv"
  PYTHON_EXEC="$BACKEND_DIR/.venv/bin/python"
  VENV_DIR="$BACKEND_DIR/.venv"
elif python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  PYTHON_EXEC="python"
elif python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  PYTHON_EXEC="python3"
else
  # Check if a running GPU process (such as VLLM) has a CUDA-enabled Python
  VLLM_PID=$(pgrep -f "VLLM::EngineCore" 2>/dev/null | head -n 1 || true)
  if [[ -n "$VLLM_PID" && -e "/proc/$VLLM_PID/exe" ]]; then
    CANDIDATE_PY="$(readlink -f "/proc/$VLLM_PID/exe" 2>/dev/null || true)"
    if [[ -x "$CANDIDATE_PY" ]] && "$CANDIDATE_PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
      echo "✓ Detected CUDA-capable Python from running VLLM service: $CANDIDATE_PY"
      PYTHON_EXEC="$CANDIDATE_PY"
    fi
  fi
fi

if [[ -z "$PYTHON_EXEC" ]]; then
  PYTHON_EXEC="python3"
fi

# Always create venv with --system-site-packages so host CUDA PyTorch is directly accessible
if [[ "$VENV_DIR" != "$BACKEND_DIR/.venv" && ! -d "$VENV_DIR" ]]; then
  echo "Creating build virtual environment with --system-site-packages using $PYTHON_EXEC..."
  "$PYTHON_EXEC" -m venv --system-site-packages "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

# Ensure pyinstaller is available in the environment
if ! command -v pyinstaller >/dev/null 2>&1; then
  python -m pip install "pyinstaller>=6.5.0"
fi

# If the venv currently has a CPU-only PyTorch that shadows the host's CUDA PyTorch, remove it immediately
if python -c "import torch; assert not torch.cuda.is_available()" 2>/dev/null; then
  if "$PYTHON_EXEC" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "✓ Removing local venv CPU PyTorch to uncover host CUDA PyTorch ($PYTHON_EXEC)..."
    python -m pip uninstall -y torch torchvision 2>/dev/null || true
  fi
fi

USE_CUDA="${USE_CUDA:-1}"

# Check if CUDA PyTorch is active inside the environment
if [[ "$USE_CUDA" == "1" && "$HAS_NVIDIA_GPU" == "1" ]]; then
  if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "=========================================================================="
    echo "✓ CUDA-enabled PyTorch is already active:"
    python -c "import torch; print(f'  PyTorch: {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}')"
    echo "  Installing dependencies without overwriting CUDA PyTorch..."
    echo "=========================================================================="
    # Exclude torch, torchvision, and ultralytics so pip does not fetch CPU torch from PyPI
    grep -v -E "^(torch|torchvision|ultralytics)([><=~!]|$)" "$BACKEND_DIR/requirements.txt" | python -m pip install -r /dev/stdin
    python -m pip install --no-deps "ultralytics>=8.3.0"
  else
    # Check if host system python had CUDA PyTorch but the venv site-packages has a conflicting CPU torch
    if "$PYTHON_EXEC" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
      echo "✓ Host Python ($PYTHON_EXEC) has CUDA PyTorch. Removing shadowed venv torch..."
      python -m pip uninstall -y torch torchvision 2>/dev/null || true
      grep -v -E "^(torch|torchvision|ultralytics)([><=~!]|$)" "$BACKEND_DIR/requirements.txt" | python -m pip install -r /dev/stdin
      python -m pip install --no-deps "ultralytics>=8.3.0"
    elif [[ "$MACHINE_ARCH" == "x86_64" ]]; then
      echo "NVIDIA GPU detected on x86_64; installing official CUDA PyTorch (cu126/cu124 for Blackwell/Ada/Ampere)..."
      PYTORCH_CUDA_INDEX="${PYTORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu126}"
      python -m pip install --force-reinstall --index-url "$PYTORCH_CUDA_INDEX" torch torchvision || {
        echo "cu126 wheel fallback; trying cu124..."
        python -m pip install --force-reinstall --index-url "https://download.pytorch.org/whl/cu124" torch torchvision
      }
      grep -v -E "^(torch|torchvision|ultralytics)([><=~!]|$)" "$BACKEND_DIR/requirements.txt" | python -m pip install -r /dev/stdin
      python -m pip install --no-deps "ultralytics>=8.3.0"
    elif [[ "$MACHINE_ARCH" == "aarch64" || "$MACHINE_ARCH" == "arm64" ]]; then
      if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "✓ Found CUDA PyTorch in system Python ($(python3 -c 'import torch; print(torch.__version__)'))"
        grep -v -E "^(torch|torchvision|ultralytics)([><=~!]|$)" "$BACKEND_DIR/requirements.txt" | python -m pip install -r /dev/stdin
        python -m pip install --no-deps "ultralytics>=8.3.0"
      else
        echo "=========================================================================="
        echo "⚠️  ARM64/Blackwell platform detected, but CUDA PyTorch was not found!"
        echo "    Attempting to install JetPack/NVIDIA compatible PyTorch..."
        echo "=========================================================================="
        PY_VER="$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")' 2>/dev/null || echo "cp310")"
        if [[ "$PY_VER" == "cp310" ]]; then
          JETSON_WHEEL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+07cecf41be.nv24.08.16594639-cp310-cp310-linux_aarch64.whl"
          echo "Installing NVIDIA PyTorch: $JETSON_WHEEL ..."
          python -m pip install --no-cache "$JETSON_WHEEL" 2>/dev/null || true
        fi
        if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
          echo "✓ Successfully configured CUDA PyTorch!"
          grep -v -E "^(torch|torchvision|ultralytics)([><=~!]|$)" "$BACKEND_DIR/requirements.txt" | python -m pip install -r /dev/stdin
          python -m pip install --no-deps "ultralytics>=8.3.0"
        else
          echo "⚠️ Note: Could not auto-install NVIDIA wheel. Installing standard requirements without overwriting."
          grep -v -E "^(torch|torchvision|ultralytics)([><=~!]|$)" "$BACKEND_DIR/requirements.txt" | python -m pip install -r /dev/stdin
          python -m pip install --no-deps "ultralytics>=8.3.0"
        fi
      fi
    else
      grep -v -E "^(torch|torchvision|ultralytics)([><=~!]|$)" "$BACKEND_DIR/requirements.txt" | python -m pip install -r /dev/stdin
      python -m pip install --no-deps "ultralytics>=8.3.0"
    fi
  fi

  # Final safeguard: remove any shadowed torch introduced during pip install
  if python -c "import torch; assert not torch.cuda.is_available()" 2>/dev/null; then
    if "$PYTHON_EXEC" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
      echo "✓ Cleaning shadowed CPU PyTorch in venv..."
      python -m pip uninstall -y torch torchvision 2>/dev/null || true
    fi
  fi
else
  echo "Building with CPU-compatible PyTorch (USE_CUDA=$USE_CUDA)."
  python -m pip install -r "$BACKEND_DIR/requirements.txt"
fi

WHL_DIR="$BACKEND_DIR/app/ml/whl"
if [[ -d "$WHL_DIR" ]]; then
  LATEST_WHL=$(find "$WHL_DIR" -maxdepth 1 -name "head_tail_analyzer*.whl" 2>/dev/null | sort -V | tail -n 1)
  if [[ -n "$LATEST_WHL" ]]; then
    echo "Installing ML wheel: $(basename "$LATEST_WHL")..."
    python -m pip install --force-reinstall --no-deps "$LATEST_WHL"
  fi
fi

echo "========================================================"
echo "          Active ML Environment Verification"
echo "========================================================"
python -c "import torch; print('PyTorch Version :', torch.__version__); print('CUDA Available  :', torch.cuda.is_available()); print('Inference Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (WARNING: 1 FPS on CPU!)')"
echo "========================================================"

if [[ "$USE_CUDA" == "1" && "$HAS_NVIDIA_GPU" == "1" ]]; then
  if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "=========================================================================="
    echo "❌ FATAL: NVIDIA GPU detected, but active PyTorch cannot access CUDA!"
    echo "   Aborting build to prevent bundling a 1 FPS CPU binary into the app."
    echo "   Please run first: bash $BACKEND_DIR/setup_linux.sh"
    echo "=========================================================================="
    exit 1
  fi
fi

if [[ "$SKIP_PYINSTALLER" == "1" && -f "$BACKEND_DIR/dist/backend/backend" ]]; then
  echo "Reusing existing backend dist/backend/backend..."
else
  cd "$BACKEND_DIR"
  rm -rf "$BACKEND_DIR/build" "$BACKEND_DIR/dist"

  pyinstaller --noconfirm --clean --onedir --name backend "$BACKEND_DIR/run.py" \
    --distpath "$BACKEND_DIR/dist" \
    --workpath "$BACKEND_DIR/build" \
    --specpath "$BACKEND_DIR" \
    --add-data "$BACKEND_DIR/app/ml/model:app/ml/model" \
    --add-data "$BACKEND_DIR/app/ml/config:app/ml/config" \
    --add-data "$BACKEND_DIR/alembic:alembic" \
    --collect-all app \
    --collect-all fastapi \
    --collect-all starlette \
    --collect-all uvicorn \
    --collect-all sqlalchemy \
    --collect-all cv2 \
    --collect-all torch \
    --collect-all torchvision \
    --collect-all ultralytics \
    --collect-all sympy \
    --collect-all depthai \
    --collect-all av \
    --collect-all scipy \
    --collect-all imageio_ffmpeg \
    --collect-all head_tail_analyzer
fi

# Strip non-runtime development files directly in dist/backend
chmod +x "$BACKEND_DIR/dist/backend/backend"
find "$BACKEND_DIR/dist/backend" -name "*.a" -delete 2>/dev/null || true
rm -rf "$BACKEND_DIR/dist/backend/_internal/torch/include" 2>/dev/null || true
rm -rf "$BACKEND_DIR/dist/backend/_internal/triton" 2>/dev/null || true

cd "$FRONTEND_DIR"
npm ci --include=optional 2>/dev/null || npm install --include=optional
npm run "package:linux:$ELECTRON_ARCH"

echo
echo "Linux $ELECTRON_ARCH artifacts are in: $FRONTEND_DIR/dist_app"
