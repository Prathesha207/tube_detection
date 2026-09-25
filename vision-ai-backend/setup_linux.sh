#!/usr/bin/env bash
# Install the backend and frontend correctly on a Linux development machine.
# Run from vision-ai-backend after both sibling folders are present.
set -euo pipefail

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$(cd "$BACKEND_DIR/../vision-ai-frontend" && pwd)"
VENV_DIR="$BACKEND_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "$FRONTEND_DIR/package.json" ]]; then
  echo "vision-ai-frontend must be beside vision-ai-backend."
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR" || {
    echo "ERROR: Failed to create Python virtual environment."
    echo "On Ubuntu/Debian, install the required packages:"
    echo "  sudo apt update && sudo apt install -y python3-venv python3-pip python3-dev"
    exit 1
  }
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$BACKEND_DIR/requirements.txt"
python -m pip install -r "$BACKEND_DIR/app/ml/tube/requirements.txt"

# The base requirements remain portable. Replace generic torch with CUDA torch
# only when this Linux machine has an NVIDIA driver and GPU.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU detected; verifying CUDA-enabled PyTorch..."
  if [[ "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ]]; then
    if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
      echo "ARM64 vendor PyTorch with CUDA is already installed; keeping it."
    else
      echo "ARM64 detected, but vendor CUDA PyTorch is not installed. Continuing with CPU PyTorch."
      echo "Install the NVIDIA/platform ARM64 PyTorch package later for GPU inference."
    fi
  else
    if ! python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
      echo "Installing CUDA-enabled PyTorch..."
      PYTORCH_CUDA_INDEX="${PYTORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"
      if ! python -m pip install --force-reinstall \
        --index-url "$PYTORCH_CUDA_INDEX" \
        torch torchvision; then
        echo "CUDA wheel index $PYTORCH_CUDA_INDEX is unavailable for this Python/platform."
        echo "Retry with another supported index using PYTORCH_CUDA_INDEX=... ./setup_linux.sh"
        exit 1
      fi
    else
      echo "CUDA PyTorch is already installed and operational."
    fi
  fi
else
  echo "No NVIDIA GPU detected; keeping CPU-compatible PyTorch."
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
