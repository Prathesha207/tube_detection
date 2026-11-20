#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR"

echo "======================================================="
echo "  🚀 Vision Monitor - Clean Rebuild & Reinstall"
echo "======================================================="

# Purge any stale/contaminated CPU venv and PyInstaller caches
rm -rf "$SCRIPT_DIR/backend/.venv-linux-build" "$SCRIPT_DIR/backend/build" "$SCRIPT_DIR/backend/dist" ~/.cache/pyinstaller 2>/dev/null || true

# Remove any CPU torch installed directly in backend/.venv so host CUDA PyTorch is used
if [[ -d "$SCRIPT_DIR/backend/.venv" ]]; then
  VENV_SITE=$("$SCRIPT_DIR/backend/.venv/bin/python" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
  if [[ -n "$VENV_SITE" && -d "$VENV_SITE" ]]; then
    rm -rf "$VENV_SITE"/torch* "$VENV_SITE"/torchvision* 2>/dev/null || true
  fi
fi

chmod +x build.sh backend/build_linux_desktop.sh backend/setup_linux.sh backend/run_dev_linux.sh 2>/dev/null || true
./build.sh --clean --install "$@"
