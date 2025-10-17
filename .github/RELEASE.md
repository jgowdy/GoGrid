# Release Process

This document describes how to build and release GoGrid Worker for all platforms.

## Automated Builds (GitHub Actions)

### Testing Builds

Every push to `main` or `develop` branches, and every pull request, will trigger automated test builds for all platforms (macOS, Linux, Windows).

### Creating a Release

#### Option 1: Tag-based Release (Recommended)

1. Update version in all relevant files:
   ```bash
   # Update version in:
   # - crates/tray-app/src-tauri/tauri.conf.json
   # - Cargo.toml (workspace)
   # - crates/*/Cargo.toml
   ```

2. Commit and create a tag:
   ```bash
   git add .
   git commit -m "Release v0.1.1"
   git tag v0.1.1
   git push origin main --tags
   ```

3. GitHub Actions will automatically:
   - Build for macOS (ARM64 & x86_64)
   - Build for Linux (x86_64)
   - Build for Windows (x86_64)
   - Create a GitHub Release with all artifacts
   - Deploy to coordinator update server

#### Option 2: Manual Workflow Dispatch

1. Go to GitHub Actions tab
2. Select "Build and Release" workflow
3. Click "Run workflow"
4. Enter the version number (e.g., 0.1.1)
5. Click "Run workflow"

This will build and deploy but won't create a GitHub Release.

## Manual Builds

### macOS

```bash
# Build scheduler (ARM64)
cargo build --release --bin corpgrid-scheduler --target aarch64-apple-darwin
cp target/aarch64-apple-darwin/release/corpgrid-scheduler target/release/corpgrid-scheduler-aarch64-apple-darwin

# Build Tauri app (ARM64)
cd crates/tray-app/src-tauri
cargo tauri build --target aarch64-apple-darwin

# Package for updates
cd ../../../target/aarch64-apple-darwin/release/bundle/macos
tar -czf "GoGrid_Worker_0.1.0_aarch64.app.tar.gz" "GoGrid Worker.app"
```

### Linux (on Ubuntu or Debian)

```bash
# Install dependencies
sudo apt-get install -y \
  libwebkit2gtk-4.1-dev \
  build-essential \
  curl \
  wget \
  file \
  libxdo-dev \
  libssl-dev \
  libayatana-appindicator3-dev \
  librsvg2-dev

# Build scheduler
cargo build --release --bin corpgrid-scheduler
cp target/release/corpgrid-scheduler target/release/corpgrid-scheduler-x86_64-unknown-linux-gnu

# Build Tauri app
cd crates/tray-app/src-tauri
cargo tauri build

# Package for updates
cd ../../../target/release/bundle/appimage
tar -czf "GoGrid_Worker_0.1.0_amd64.AppImage.tar.gz" *.AppImage
```

### Windows (on Windows machine)

```bash
# Build scheduler
cargo build --release --bin corpgrid-scheduler
copy target\release\corpgrid-scheduler.exe target\release\corpgrid-scheduler-x86_64-pc-windows-msvc.exe

# Build Tauri app
cd crates/tray-app/src-tauri
cargo tauri build

# Package for updates
cd ..\..\..\target\release\bundle\nsis
tar -czf "GoGrid_Worker_0.1.0_x64-setup.nsis.zip" *.exe
```

## Deployment to Update Server

### Required Secrets

To enable automatic deployment, add the following GitHub secrets:

1. `SSH_PRIVATE_KEY` - Private SSH key for deploying to your coordinator server
   ```bash
   # Generate a deploy key
   ssh-keygen -t ed25519 -C "github-actions@gogrid" -f ~/.ssh/gogrid_deploy

   # Add public key to your coordinator server
   ssh-copy-id -i ~/.ssh/gogrid_deploy.pub your-server.com

   # Add private key to GitHub Secrets
   cat ~/.ssh/gogrid_deploy
   ```

2. `COORDINATOR_HOST` - Hostname of your coordinator server (e.g., `coordinator.example.com`)

### Manual Deployment

```bash
# Upload macOS packages
scp target/aarch64-apple-darwin/release/bundle/macos/GoGrid_Worker_*.app.tar.gz your-server.com:/opt/gogrid/updates/
scp target/aarch64-apple-darwin/release/bundle/dmg/*.dmg your-server.com:/opt/gogrid/updates/

# Upload Linux packages
scp target/release/bundle/appimage/GoGrid_Worker_*.AppImage.tar.gz your-server.com:/opt/gogrid/updates/
scp target/release/bundle/appimage/*.AppImage your-server.com:/opt/gogrid/updates/

# Upload Windows packages
scp target/release/bundle/nsis/GoGrid_Worker_*.nsis.zip your-server.com:/opt/gogrid/updates/
scp target/release/bundle/nsis/*.exe your-server.com:/opt/gogrid/updates/

# Build and deploy coordinator
cargo build --release --bin gogrid-coordinator
scp target/release/gogrid-coordinator your-server.com:/opt/gogrid/bin/

# Restart coordinator
ssh your-server.com "sudo systemctl restart gogrid-coordinator"
```

## Artifacts Generated

### macOS
- `GoGrid_Worker_{version}_aarch64.app.tar.gz` - Auto-update package (ARM64)
- `GoGrid Worker_{version}_aarch64.dmg` - Manual installer (ARM64)
- `GoGrid_Worker_{version}_x64.app.tar.gz` - Auto-update package (Intel)
- `GoGrid Worker_{version}_x64.dmg` - Manual installer (Intel)

### Linux
- `GoGrid_Worker_{version}_amd64.AppImage.tar.gz` - Auto-update package
- `gogrid-worker_{version}_amd64.AppImage` - Manual installer
- `gogrid-worker_{version}_amd64.deb` - Debian package

### Windows
- `GoGrid_Worker_{version}_x64-setup.nsis.zip` - Auto-update package
- `GoGrid Worker_{version}_x64-setup.exe` - Manual installer
- `GoGrid Worker_{version}_x64_en-US.msi` - MSI installer

## Version Bumping Checklist

- [ ] Update `crates/tray-app/src-tauri/tauri.conf.json` - version field
- [ ] Update `Cargo.toml` (workspace root) - version field
- [ ] Update coordinator `src/main.rs` - latest_version constant
- [ ] Update `AUTO_UPDATE.md` - version references
- [ ] Update `deploy_updates.sh` - version in filenames
- [ ] Create git tag: `git tag v{version}`
- [ ] Push tag: `git push --tags`

## Testing Updates

1. Install an older version on your machine
2. Deploy a newer version to the update server
3. Wait 5 seconds (or restart the app)
4. Check logs for update detection:
   ```bash
   # macOS
   tail -f ~/Library/Logs/GoGrid\ Worker/gogrid-tray.log | grep -i update
   ```

## Monitoring

### Server Logs
```bash
# Coordinator logs
ssh your-server.com "sudo journalctl -u gogrid-coordinator -f"

# Update requests (if using nginx)
ssh your-server.com "sudo tail -f /var/log/nginx/access.log | grep /updates"
```

### Update Statistics
Check GitHub Actions logs to see build times and artifact sizes.

## Troubleshooting

### Build fails on macOS
- Ensure Xcode Command Line Tools are installed
- Check icon files exist in `crates/tray-app/src-tauri/icons/`

### Build fails on Linux
- Install all webkit2gtk dependencies
- Use Ubuntu 20.04+ or equivalent

### Build fails on Windows
- Ensure Visual Studio Build Tools are installed
- Check WebView2 is available

### Deployment fails
- Verify SSH key has correct permissions (600)
- Check `/opt/gogrid/updates` directory exists and is writable
- Verify coordinator service is running
