#!/usr/bin/env bash
# Install the backend and frontend correctly on a Linux development machine.
# Run from vision-ai-backend after both sibling folders are present.
set -euo pipefail

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$BACKEND_DIR/../frontend" ]]; then
  FRONTEND_DIR="$(cd "$BACKEND_DIR/../frontend" && pwd)"
else
  FRONTEND_DIR="$(cd "$BACKEND_DIR/../vision-ai-frontend" && pwd)"
fi
VENV_DIR="$BACKEND_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "$FRONTEND_DIR/package.json" ]]; then
  echo "Frontend folder with package.json must be beside backend folder."
  exit 1
fi

# Detect the best Python interpreter with active CUDA PyTorch
PYTHON_EXEC="${PYTHON_BIN:-python3}"
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
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

if [[ "${1:-}" == "--clean" || "${1:-}" == "-c" ]]; then
  echo "Clean setup requested. Removing existing .venv..."
  rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment with --system-site-packages using $PYTHON_EXEC..."
  "$PYTHON_EXEC" -m venv --system-site-packages "$VENV_DIR" || {
    echo "ERROR: Failed to create Python virtual environment."
    echo "On Ubuntu/Debian, install the required packages:"
    echo "  sudo apt update && sudo apt install -y python3-venv python3-pip python3-dev"
    exit 1
  }
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

# The base requirements remain portable. Install dependencies WITHOUT generic CPU torch
grep -v -E "^(torch|torchvision)([><=~!]|$)" "$BACKEND_DIR/requirements.txt" | python -m pip install -r /dev/stdin

# Automatically install latest head_tail_analyzer wheel
WHL_DIR="$BACKEND_DIR/app/ml/whl"
if [[ -d "$WHL_DIR" ]]; then
  LATEST_WHL=$(find "$WHL_DIR" -maxdepth 1 -name "head_tail_analyzer*.whl" 2>/dev/null | sort -V | tail -n 1)
  if [[ -n "$LATEST_WHL" ]]; then
    echo "Installing ML wheel: $(basename "$LATEST_WHL")..."
    python -m pip install --force-reinstall --no-deps "$LATEST_WHL"
  fi
fi

# Detect whether this machine has an NVIDIA GPU (Desktop PCIe, DGX Spark, or Jetson Tegra)
HAS_NVIDIA_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  HAS_NVIDIA_GPU=1
elif [[ -f /etc/nv_tegra_release ]] || command -v tegrastats >/dev/null 2>&1 || [[ -d /usr/local/cuda ]]; then
  HAS_NVIDIA_GPU=1
elif python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  HAS_NVIDIA_GPU=1
fi

if [[ "$HAS_NVIDIA_GPU" == "1" ]]; then
  echo "NVIDIA GPU detected; configuring CUDA-enabled PyTorch..."
  if python -c "import torch; x=torch.ones((1,), device='cuda:0'); assert (x+1).item()==2; torch.cuda.synchronize(0)" 2>/dev/null; then
    echo "✓ CUDA-enabled PyTorch is already operational:"
    python -c "import torch; print(f'  PyTorch: {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}')"
  else
    # Check if host system python had CUDA PyTorch but the venv has a shadowed CPU torch
    if "$PYTHON_EXEC" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
      echo "✓ Host Python ($PYTHON_EXEC) has CUDA PyTorch. Removing shadowed venv torch..."
      python -m pip uninstall -y torch torchvision 2>/dev/null || true
    elif [[ "$(uname -m)" == "x86_64" ]]; then
      echo "Installing official CUDA PyTorch for NVIDIA GPU (cu126/cu124)..."
      PYTORCH_CUDA_INDEX="${PYTORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu126}"
      python -m pip install --force-reinstall --index-url "$PYTORCH_CUDA_INDEX" torch torchvision || {
        echo "cu126 wheel fallback; trying cu124..."
        python -m pip install --force-reinstall --index-url "https://download.pytorch.org/whl/cu124" torch torchvision
      }
    else
      echo "⚠️ Note: On ARM64/Jetson/Blackwell, ensure system PyTorch with CUDA is installed."
    fi
  fi
else
  echo "No NVIDIA GPU detected; using standard PyTorch."
  python -m pip install torch torchvision
fi

# Check Luxonis OAK camera udev rules on Linux
if [[ ! -f /etc/udev/rules.d/80-movidius.rules ]]; then
  echo "--------------------------------------------------------"
  echo "NOTE: DepthAI OAK camera USB udev rules not detected."
  echo "If using USB OAK cameras without root permissions, run:"
  echo "  echo 'SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"03e7\", MODE=\"0666\"' | sudo tee /etc/udev/rules.d/80-movidius.rules"
  echo "  sudo udevadm control --reload-rules && sudo udevadm trigger"
  echo "--------------------------------------------------------"
fi

# Native Rollup/Vite modules must be installed on this exact Linux architecture.
cd "$FRONTEND_DIR"
npm install --include=optional

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo "Setup complete. Start development with: $BACKEND_DIR/run_dev_linux.sh"
