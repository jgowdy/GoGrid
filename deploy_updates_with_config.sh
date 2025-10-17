#!/bin/bash
set -e

# Configuration
COORDINATOR_HOST="${COORDINATOR_HOST:-coordinator.example.com}"
COORDINATOR_PORT="${COORDINATOR_PORT:-8443}"
VERSION="${1:-0.1.0}"

echo "============================================"
echo "Deploying GoGrid Worker v${VERSION}"
echo "Coordinator: $COORDINATOR_HOST:$COORDINATOR_PORT"
echo "============================================"
echo ""

# Ensure patch script exists
if [ ! -f "./patch_binary_config.sh" ]; then
    echo "Error: patch_binary_config.sh not found"
    exit 1
fi

# Create temporary working directory
WORK_DIR=$(mktemp -d)
trap "rm -rf $WORK_DIR" EXIT

echo "Working directory: $WORK_DIR"
echo ""

# Function to patch and package macOS app
patch_and_package_macos() {
    local arch=$1
    local target_dir=$2
    local app_name="GoGrid Worker.app"

    echo "→ Processing macOS $arch build..."

    # Copy app to working directory
    cp -R "$target_dir/$app_name" "$WORK_DIR/"

    # Find the binary inside the app bundle
    local binary_path="$WORK_DIR/$app_name/Contents/MacOS/gogrid-tray"

    if [ ! -f "$binary_path" ]; then
        echo "  Warning: Binary not found at $binary_path"
        return 1
    fi

    # Patch the binary
    echo "  Patching binary with $COORDINATOR_HOST:$COORDINATOR_PORT..."
    ./patch_binary_config.sh "$binary_path" "$COORDINATOR_HOST" "$COORDINATOR_PORT"

    # Package the patched app
    local output_name="GoGrid_Worker_${VERSION}_${arch}.app.tar.gz"
    echo "  Creating package: $output_name..."
    cd "$WORK_DIR"
    tar -czf "$output_name" "$app_name"

    # Sign the update package
    if [ -f ~/.tauri/gogrid.key ]; then
        echo "  Signing update package..."
        cargo tauri signer sign "$output_name" --password "" --private-key "$(cat ~/.tauri/gogrid.key)"
    else
        echo "  Warning: Signing key not found, skipping signature"
    fi

    # Move back to target directory
    mkdir -p "$target_dir"
    mv "$output_name" "$target_dir/"
    if [ -f "$output_name.sig" ]; then
        mv "$output_name.sig" "$target_dir/"
    fi

    # Clean up working directory
    rm -rf "$WORK_DIR/$app_name"
    cd - > /dev/null

    echo "  ✓ Done: $target_dir/$output_name"
    echo ""
}

# Function to patch and package Linux AppImage
patch_and_package_linux() {
    local target_dir=$1

    echo "→ Processing Linux build..."

    # Find the AppImage
    local appimage=$(find "$target_dir" -name "*.AppImage" -type f | head -1)

    if [ -z "$appimage" ]; then
        echo "  Warning: AppImage not found in $target_dir"
        return 1
    fi

    local appimage_name=$(basename "$appimage")
    echo "  Found: $appimage_name"

    # Extract AppImage
    echo "  Extracting AppImage..."
    cd "$WORK_DIR"
    "$appimage" --appimage-extract > /dev/null

    # Find the binary inside
    local binary_path="$WORK_DIR/squashfs-root/usr/bin/gogrid-tray"

    if [ ! -f "$binary_path" ]; then
        echo "  Warning: Binary not found at $binary_path"
        return 1
    fi

    # Patch the binary
    echo "  Patching binary with $COORDINATOR_HOST:$COORDINATOR_PORT..."
    ./patch_binary_config.sh "$binary_path" "$COORDINATOR_HOST" "$COORDINATOR_PORT"

    # Repackage AppImage (would require appimagetool, skip for now)
    # For updates, we can package the extracted directory
    local output_name="GoGrid_Worker_${VERSION}_amd64.AppImage.tar.gz"
    echo "  Creating package: $output_name..."
    tar -czf "$output_name" -C squashfs-root .

    # Sign the update package
    if [ -f ~/.tauri/gogrid.key ]; then
        echo "  Signing update package..."
        cargo tauri signer sign "$output_name" --password "" --private-key "$(cat ~/.tauri/gogrid.key)"
    fi

    # Move back to target directory
    mv "$output_name" "$target_dir/"
    if [ -f "$output_name.sig" ]; then
        mv "$output_name.sig" "$target_dir/"
    fi

    # Clean up
    rm -rf squashfs-root
    cd - > /dev/null

    echo "  ✓ Done: $target_dir/$output_name"
    echo ""
}

# Get absolute path to project root
PROJECT_ROOT=$(pwd)

# Process macOS ARM64 build
if [ -d "target/aarch64-apple-darwin/release/bundle/macos" ]; then
    patch_and_package_macos "aarch64" "$PROJECT_ROOT/target/aarch64-apple-darwin/release/bundle/macos"
else
    echo "→ Skipping macOS ARM64 (not built)"
fi

# Process macOS x86_64 build
if [ -d "target/x86_64-apple-darwin/release/bundle/macos" ]; then
    patch_and_package_macos "x64" "$PROJECT_ROOT/target/x86_64-apple-darwin/release/bundle/macos"
else
    echo "→ Skipping macOS x86_64 (not built)"
fi

# Process Linux build
if [ -d "target/release/bundle/appimage" ]; then
    patch_and_package_linux "$PROJECT_ROOT/target/release/bundle/appimage"
else
    echo "→ Skipping Linux (not built)"
fi

echo "============================================"
echo "Upload to server"
echo "============================================"
echo ""

# Create updates directory on server
echo "→ Creating remote directory..."
ssh "$COORDINATOR_HOST" "doas mkdir -p /opt/gogrid/updates && doas chown $USER:$USER /opt/gogrid/updates"

# Upload macOS packages
echo "→ Uploading macOS packages..."
if [ -f "target/aarch64-apple-darwin/release/bundle/macos/GoGrid_Worker_${VERSION}_aarch64.app.tar.gz" ]; then
    scp "target/aarch64-apple-darwin/release/bundle/macos/GoGrid_Worker_${VERSION}_aarch64.app.tar.gz"* "$COORDINATOR_HOST:/opt/gogrid/updates/"
    echo "  ✓ ARM64 update uploaded"
fi

if [ -f "target/x86_64-apple-darwin/release/bundle/macos/GoGrid_Worker_${VERSION}_x64.app.tar.gz" ]; then
    scp "target/x86_64-apple-darwin/release/bundle/macos/GoGrid_Worker_${VERSION}_x64.app.tar.gz"* "$COORDINATOR_HOST:/opt/gogrid/updates/"
    echo "  ✓ x86_64 update uploaded"
fi

# Upload DMGs for manual download
echo "→ Uploading DMG installers..."
if [ -f "target/aarch64-apple-darwin/release/bundle/dmg/GoGrid Worker_${VERSION}_aarch64.dmg" ]; then
    scp "target/aarch64-apple-darwin/release/bundle/dmg/GoGrid Worker_${VERSION}_aarch64.dmg" "$COORDINATOR_HOST:/opt/gogrid/updates/"
    echo "  ✓ ARM64 DMG uploaded"
fi

if [ -f "target/x86_64-apple-darwin/release/bundle/dmg/GoGrid Worker_${VERSION}_x64.dmg" ]; then
    scp "target/x86_64-apple-darwin/release/bundle/dmg/GoGrid Worker_${VERSION}_x64.dmg" "$COORDINATOR_HOST:/opt/gogrid/updates/"
    echo "  ✓ x86_64 DMG uploaded"
fi

# Upload Linux packages
if [ -f "target/release/bundle/appimage/GoGrid_Worker_${VERSION}_amd64.AppImage.tar.gz" ]; then
    echo "→ Uploading Linux packages..."
    scp "target/release/bundle/appimage/GoGrid_Worker_${VERSION}_amd64.AppImage.tar.gz"* "$COORDINATOR_HOST:/opt/gogrid/updates/"
    echo "  ✓ Linux update uploaded"
fi

echo ""
echo "============================================"
echo "✓ Deployment complete!"
echo "============================================"
echo ""
echo "Update server: https://$COORDINATOR_HOST:$COORDINATOR_PORT/updates/"
echo "Downloads page: https://$COORDINATOR_HOST:$COORDINATOR_PORT/downloads"
echo ""
echo "Binaries were patched with default coordinator: $COORDINATOR_HOST:$COORDINATOR_PORT"
echo "Users can override this via:"
echo "  - Environment variables (GOGRID_COORDINATOR_HOST, GOGRID_COORDINATOR_PORT)"
echo "  - Configuration file edit"
echo ""
