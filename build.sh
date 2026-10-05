#!/usr/bin/env bash
set -e

echo "======================================================="
echo "      Vision Monitor Build (Linux Desktop App)"
echo "======================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -d "$SCRIPT_DIR/backend" ]]; then
  cd "$SCRIPT_DIR/backend"
else
  cd "$SCRIPT_DIR/vision-ai-backend"
fi
chmod +x build_linux_desktop.sh
./build_linux_desktop.sh

echo ""
echo "======================================================="
echo "Build completed successfully!"
echo "You can find the AppImage and .deb files in:"
echo "vision-ai-frontend/dist_app"
echo "======================================================="
