#!/bin/bash
set -e

echo "Deploying GoGrid Worker updates to bx.ee..."

# Create updates directory
ssh bx.ee "doas mkdir -p /opt/gogrid/updates && doas chown jgowdy:jgowdy /opt/gogrid/updates"

# Package the macOS app for updates (Tauri expects .app.tar.gz format)
echo "Packaging macOS app..."
cd /Users/jgowdy/GoGrid/target/release/bundle/macos
tar -czf "GoGrid_Worker_0.1.0_aarch64.app.tar.gz" "GoGrid Worker.app"

# Copy to server
echo "Uploading to bx.ee..."
scp "GoGrid_Worker_0.1.0_aarch64.app.tar.gz" bx.ee:/opt/gogrid/updates/

# Also copy the DMG for manual downloads
echo "Uploading DMG..."
scp "../dmg/GoGrid Worker_0.1.0_aarch64.dmg" bx.ee:/opt/gogrid/updates/

# Rebuild and deploy coordinator
echo "Building coordinator..."
cd /Users/jgowdy/GoGrid
cargo build --release --bin gogrid-coordinator

echo "Deploying coordinator..."
scp target/release/gogrid-coordinator bx.ee:/opt/gogrid/bin/

echo "Restarting coordinator..."
ssh bx.ee "doas systemctl restart gogrid-coordinator"

echo "Deployment complete!"
echo "Update server available at: https://bx.ee:8443/updates/"
