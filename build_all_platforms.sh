#!/bin/bash
set -e

VERSION="${1:-0.1.0}"
echo "Building GoGrid Worker v${VERSION} for all platforms..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${BLUE}Building for macOS...${NC}"

    # Build scheduler (ARM64)
    echo "Building scheduler (ARM64)..."
    cargo build --release --bin corpgrid-scheduler --target aarch64-apple-darwin
    cp target/aarch64-apple-darwin/release/corpgrid-scheduler target/release/corpgrid-scheduler-aarch64-apple-darwin

    # Build scheduler (x86_64)
    echo "Building scheduler (x86_64)..."
    cargo build --release --bin corpgrid-scheduler --target x86_64-apple-darwin
    cp target/x86_64-apple-darwin/release/corpgrid-scheduler target/release/corpgrid-scheduler-x86_64-apple-darwin

    # Check if tauri-cli is installed
    if ! command -v cargo-tauri &> /dev/null; then
        echo "Installing Tauri CLI..."
        cargo install tauri-cli --version '^2.0.0'
    fi

    # Build Tauri app (ARM64)
    echo "Building Tauri app (ARM64)..."
    cd crates/tray-app/src-tauri
    cargo tauri build --target aarch64-apple-darwin
    cd ../../..

    # Build Tauri app (x86_64)
    echo "Building Tauri app (x86_64)..."
    cd crates/tray-app/src-tauri
    cargo tauri build --target x86_64-apple-darwin
    cd ../../..

    # Package for updates (ARM64)
    echo "Packaging ARM64..."
    cd target/aarch64-apple-darwin/release/bundle/macos
    tar -czf "GoGrid_Worker_${VERSION}_aarch64.app.tar.gz" "GoGrid Worker.app"

    # Sign update package
    if [ -f ~/.tauri/gogrid.key ]; then
        echo "Signing ARM64 update..."
        cargo tauri signer sign "GoGrid_Worker_${VERSION}_aarch64.app.tar.gz" --password "" --private-key ~/.tauri/gogrid.key
    fi
    cd ../../../../..

    # Package for updates (x86_64)
    echo "Packaging x86_64..."
    cd target/x86_64-apple-darwin/release/bundle/macos
    tar -czf "GoGrid_Worker_${VERSION}_x64.app.tar.gz" "GoGrid Worker.app"

    # Sign update package
    if [ -f ~/.tauri/gogrid.key ]; then
        echo "Signing x86_64 update..."
        cargo tauri signer sign "GoGrid_Worker_${VERSION}_x64.app.tar.gz" --password "" --private-key ~/.tauri/gogrid.key
    fi
    cd ../../../../..

    echo -e "${GREEN}✓ macOS builds complete!${NC}"
    echo "  ARM64 DMG: target/aarch64-apple-darwin/release/bundle/dmg/GoGrid Worker_${VERSION}_aarch64.dmg"
    echo "  ARM64 update: target/aarch64-apple-darwin/release/bundle/macos/GoGrid_Worker_${VERSION}_aarch64.app.tar.gz"
    echo "  x86_64 DMG: target/x86_64-apple-darwin/release/bundle/dmg/GoGrid Worker_${VERSION}_x64.dmg"
    echo "  x86_64 update: target/x86_64-apple-darwin/release/bundle/macos/GoGrid_Worker_${VERSION}_x64.app.tar.gz"

elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo -e "${BLUE}Building for Linux...${NC}"

    # Check dependencies
    echo "Checking dependencies..."
    DEPS=(libwebkit2gtk-4.1-dev build-essential libssl-dev libayatana-appindicator3-dev librsvg2-dev)
    MISSING=()
    for dep in "${DEPS[@]}"; do
        if ! dpkg -l | grep -q "$dep"; then
            MISSING+=("$dep")
        fi
    done

    if [ ${#MISSING[@]} -gt 0 ]; then
        echo -e "${RED}Missing dependencies: ${MISSING[*]}${NC}"
        echo "Install with: sudo apt-get install ${MISSING[*]}"
        exit 1
    fi

    # Build scheduler
    echo "Building scheduler..."
    cargo build --release --bin corpgrid-scheduler
    cp target/release/corpgrid-scheduler target/release/corpgrid-scheduler-x86_64-unknown-linux-gnu

    # Check if tauri-cli is installed
    if ! command -v cargo-tauri &> /dev/null; then
        echo "Installing Tauri CLI..."
        cargo install tauri-cli --version '^2.0.0'
    fi

    # Build Tauri app
    echo "Building Tauri app..."
    cd crates/tray-app/src-tauri
    cargo tauri build
    cd ../../..

    # Package for updates
    echo "Packaging..."
    cd target/release/bundle/appimage
    tar -czf "GoGrid_Worker_${VERSION}_amd64.AppImage.tar.gz" *.AppImage

    # Sign update package
    if [ -f ~/.tauri/gogrid.key ]; then
        echo "Signing update..."
        cargo tauri signer sign "GoGrid_Worker_${VERSION}_amd64.AppImage.tar.gz" --password "" --private-key ~/.tauri/gogrid.key
    fi
    cd ../../../..

    echo -e "${GREEN}✓ Linux build complete!${NC}"
    echo "  AppImage: target/release/bundle/appimage/gogrid-worker_${VERSION}_amd64.AppImage"
    echo "  Update package: target/release/bundle/appimage/GoGrid_Worker_${VERSION}_amd64.AppImage.tar.gz"
    echo "  DEB package: target/release/bundle/deb/gogrid-worker_${VERSION}_amd64.deb"

elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo -e "${BLUE}Building for Windows...${NC}"

    # Build scheduler
    echo "Building scheduler..."
    cargo build --release --bin corpgrid-scheduler
    copy target\release\corpgrid-scheduler.exe target\release\corpgrid-scheduler-x86_64-pc-windows-msvc.exe

    # Check if tauri-cli is installed
    if ! command -v cargo-tauri &> /dev/null; then
        echo "Installing Tauri CLI..."
        cargo install tauri-cli --version '^2.0.0'
    fi

    # Build Tauri app
    echo "Building Tauri app..."
    cd crates/tray-app/src-tauri
    cargo tauri build
    cd ../../..

    # Package for updates
    echo "Packaging..."
    cd target\release\bundle\nsis
    tar -czf "GoGrid_Worker_${VERSION}_x64-setup.nsis.zip" *.exe

    # Sign update package
    if [ -f ~/.tauri/gogrid.key ]; then
        echo "Signing update..."
        cargo tauri signer sign "GoGrid_Worker_${VERSION}_x64-setup.nsis.zip" --password "" --private-key ~/.tauri/gogrid.key
    fi
    cd ..\..\..\..

    echo -e "${GREEN}✓ Windows build complete!${NC}"
    echo "  Installer: target/release/bundle/nsis/GoGrid Worker_${VERSION}_x64-setup.exe"
    echo "  Update package: target/release/bundle/nsis/GoGrid_Worker_${VERSION}_x64-setup.nsis.zip"
    echo "  MSI installer: target/release/bundle/msi/GoGrid Worker_${VERSION}_x64_en-US.msi"
else
    echo -e "${RED}Unknown operating system: $OSTYPE${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Build complete!${NC}"
echo ""
echo "To deploy to update server, run:"
echo "  ./deploy_updates.sh"
