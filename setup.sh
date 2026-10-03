#!/usr/bin/env bash
set -e

echo "======================================================="
echo "           Vision Monitor Setup (Linux)"
echo "======================================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -d "$SCRIPT_DIR/backend" ]]; then
  cd "$SCRIPT_DIR/backend"
else
  cd "$SCRIPT_DIR/vision-ai-backend"
fi
chmod +x setup_linux.sh
./setup_linux.sh

echo ""
echo "======================================================="
echo "Setup completed successfully!"
echo "To start the application, simply execute ./run.sh"
echo "======================================================="
