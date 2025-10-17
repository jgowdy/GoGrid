#!/bin/bash
set -e

COORDINATOR_HOST="${COORDINATOR_HOST:-coordinator.example.com}"

echo "Deploying GoGrid Worker updates to $COORDINATOR_HOST..."

# Create updates directory
ssh "$COORDINATOR_HOST" "sudo mkdir -p /opt/gogrid/updates && sudo chown $USER:$USER /opt/gogrid/updates"

# Package the macOS app for updates (Tauri expects .app.tar.gz format)
echo "Packaging macOS app..."
cd /Users/jgowdy/GoGrid/target/release/bundle/macos
tar -czf "GoGrid_Worker_0.1.0_aarch64.app.tar.gz" "GoGrid Worker.app"

# Copy to server
echo "Uploading to $COORDINATOR_HOST..."
scp "GoGrid_Worker_0.1.0_aarch64.app.tar.gz" "$COORDINATOR_HOST:/opt/gogrid/updates/"

# Also copy the DMG for manual downloads
echo "Uploading DMG..."
scp "../dmg/GoGrid Worker_0.1.0_aarch64.dmg" "$COORDINATOR_HOST:/opt/gogrid/updates/"

# Rebuild and deploy coordinator
echo "Building coordinator..."
cd /Users/jgowdy/GoGrid
cargo build --release --bin gogrid-coordinator

echo "Deploying coordinator..."
scp target/release/gogrid-coordinator "$COORDINATOR_HOST:/opt/gogrid/bin/"

echo "Restarting coordinator..."
ssh "$COORDINATOR_HOST" "sudo systemctl restart gogrid-coordinator"

echo "Deployment complete!"
echo "Update server available at: https://$COORDINATOR_HOST:8443/updates/"
