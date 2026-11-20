#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR"

echo "======================================================="
echo "  🚀 Vision Monitor - Clean Rebuild & Reinstall"
echo "======================================================="

chmod +x build.sh backend/build_linux_desktop.sh 2>/dev/null || true
./build.sh --clean --install "$@"
