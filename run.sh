#!/usr/bin/env bash
set -e

echo "Starting Vision Monitor Development Servers..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -d "$SCRIPT_DIR/backend" ]]; then
  cd "$SCRIPT_DIR/backend"
else
  cd "$SCRIPT_DIR/vision-ai-backend"
fi
chmod +x run_dev_linux.sh
./run_dev_linux.sh
