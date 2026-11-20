#!/usr/bin/env bash
set -e

echo "======================================================="
echo "      Vision Monitor Build (Linux Desktop App)"
echo "======================================================="

AUTO_INSTALL=0
FILTERED_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--install" ]]; then
    AUTO_INSTALL=1
  else
    FILTERED_ARGS+=("$arg")
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -d "$SCRIPT_DIR/backend" ]]; then
  cd "$SCRIPT_DIR/backend"
else
  cd "$SCRIPT_DIR/vision-ai-backend"
fi
chmod +x build_linux_desktop.sh
USE_CUDA="${USE_CUDA:-1}" ./build_linux_desktop.sh "${FILTERED_ARGS[@]}"

echo ""
echo "======================================================="
echo "Build completed successfully!"
echo "Generated AppImage and .deb in: frontend/dist_app"
echo "======================================================="

if [[ "$AUTO_INSTALL" == "1" ]]; then
  echo ""
  echo "--> Stopping running instances and cleanly reinstalling application..."
  pkill -9 -f vision-monitor 2>/dev/null || true
  pkill -9 -f backend 2>/dev/null || true
  sleep 1

  echo "--> Removing previous /opt/vision-monitor package..."
  sudo dpkg -r vision-monitor 2>/dev/null || true
  sudo rm -rf /opt/vision-monitor /opt/Vision-Monitor 2>/dev/null || true

  DEB_FILE=$(find "$SCRIPT_DIR/frontend/dist_app" -name "*.deb" 2>/dev/null | sort -V | tail -n 1)
  if [[ -n "$DEB_FILE" && -f "$DEB_FILE" ]]; then
    echo "--> Installing: $(basename "$DEB_FILE")..."
    sudo dpkg -i "$DEB_FILE"
    echo ""
    echo "======================================================="
    echo "✓ Clean installation complete!"
    echo "Launching new Vision Monitor desktop application..."
    echo "======================================================="
    vision-monitor >/dev/null 2>&1 &
  else
    echo "Error: Could not locate generated .deb in $SCRIPT_DIR/frontend/dist_app"
    exit 1
  fi
else
  echo ""
  echo "To install the newly built clean package, run:"
  echo "  sudo dpkg -r vision-monitor 2>/dev/null || true"
  echo "  sudo rm -rf /opt/vision-monitor"
  echo "  sudo dpkg -i frontend/dist_app/*.deb"
  echo ""
  echo "Or run with auto-install:"
  echo "  ./build.sh --clean --install"
fi
