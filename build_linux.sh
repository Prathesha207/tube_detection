#!/usr/bin/env bash
# One-click installer and builder for Vision Monitor on Linux
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BACKEND_DIR="$ROOT_DIR/backend"
if [[ ! -d "$BACKEND_DIR" ]]; then
  BACKEND_DIR="$ROOT_DIR/vision-ai-backend"
fi
FRONTEND_DIR="$ROOT_DIR/frontend"
if [[ ! -d "$FRONTEND_DIR" ]]; then
  FRONTEND_DIR="$ROOT_DIR/vision-ai-frontend"
fi

echo "========================================================"
echo "  🚀 Vision Monitor - One-Step Linux Setup & App Builder "
echo "========================================================"

# Make inner scripts executable
chmod +x "$BACKEND_DIR/build_linux_desktop.sh" 2>/dev/null || true
chmod +x "$BACKEND_DIR/setup_linux.sh" 2>/dev/null || true
chmod +x "$BACKEND_DIR/run_dev_linux.sh" 2>/dev/null || true

MODE="${1:-app}"

if [[ "$MODE" == "--dev" || "$MODE" == "dev" ]]; then
  echo "Starting Vision Monitor development environment..."
  exec "$BACKEND_DIR/run_dev_linux.sh"
fi

if [[ "$MODE" == "--setup" || "$MODE" == "setup" ]]; then
  echo "Running Vision Monitor setup..."
  exec "$BACKEND_DIR/setup_linux.sh"
fi

# Run build script inside backend
cd "$BACKEND_DIR"
echo "Starting desktop application build..."
USE_CUDA="${USE_CUDA:-auto}" "$BACKEND_DIR/build_linux_desktop.sh"

echo
echo "========================================================"
echo "  🎉 SUCCESS! Your Linux App is ready to use:           "
echo "========================================================"
find "$FRONTEND_DIR/dist_app" -name "*.AppImage" -exec ls -lh {} + 2>/dev/null || true
find "$FRONTEND_DIR/dist_app" -name "*.deb" -exec ls -lh {} + 2>/dev/null || true
echo
echo "To run the AppImage, simply execute:"
echo "  chmod +x $FRONTEND_DIR/dist_app/Vision-Monitor-*.AppImage"
echo "  $FRONTEND_DIR/dist_app/Vision-Monitor-*.AppImage"
echo
echo "Or install the Debian package:"
echo "  sudo dpkg -i $FRONTEND_DIR/dist_app/vision-monitor_*.deb"
echo "========================================================"
