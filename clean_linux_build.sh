#!/usr/bin/env bash
set -e

echo "======================================================="
echo "       Vision Monitor Clean Linux Desktop Build"
echo "======================================================="

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$REPO_ROOT"

echo "1. Syncing latest code from GitHub..."
git fetch origin feature/ml-tube-analyzer
git reset --hard origin/feature/ml-tube-analyzer

echo "2. Cleaning old build output..."
rm -rf frontend/dist frontend/dist_app/linux* frontend/dist_app/*.AppImage frontend/dist_app/*.deb

echo "3. Installing frontend packages & applying compatibility patch..."
cd "$REPO_ROOT/frontend"
npm install --include=optional
node patch_electron_builder.js
cd "$REPO_ROOT"

echo "4. Running build..."
chmod +x build.sh backend/build_linux_desktop.sh
./build.sh "$@"

echo ""
echo "======================================================="
echo "Build complete! Your Linux packages are in:"
echo "frontend/dist_app"
echo "======================================================="
