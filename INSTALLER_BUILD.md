# GoGrid Worker Installer Build Guide

This document describes how to build platform-specific installers for the GoGrid Worker tray application.

## Overview

The GoGrid Worker application consists of:
1. **Tray Application** (`gogrid-tray`): System tray UI built with Tauri
2. **Worker Process** (`corpgrid-scheduler`): GPU inference worker binary

Both are bundled together in platform-specific installers.

## Completed Builds

### macOS ✅

**Location**: `/Users/jgowdy/GoGrid/target/release/bundle/dmg/`

**Files Created**:
- `GoGrid Worker_0.1.0_aarch64.dmg` (27 MB) - macOS disk image installer
- `GoGrid Worker.app` - Application bundle containing:
  - `gogrid-tray` (Tauri tray app)
  - `corpgrid-scheduler` (Worker process)

**Installation**:
1. Open `GoGrid Worker_0.1.0_aarch64.dmg`
2. Drag "GoGrid Worker" to Applications folder
3. Launch from Applications or Spotlight
4. App appears in menu bar (top right)

**System Requirements**:
- macOS 10.15 (Catalina) or later
- Apple Silicon (ARM64) or Intel (x86_64)

## Building for Other Platforms

### Windows

**Prerequisites**:
- Visual Studio Build Tools
- WiX Toolset (for .msi) or NSIS (for .exe)

**Build Command**:
```powershell
# On Windows machine
cargo build --release --bin corpgrid-scheduler
cp target/release/corpgrid-scheduler.exe target/release/corpgrid-scheduler-x86_64-pc-windows-msvc.exe
cd crates/tray-app/src-tauri
cargo tauri build
```

**Output**:
- `target/release/bundle/msi/GoGrid Worker_0.1.0_x64_en-US.msi` - MSI installer
- `target/release/bundle/nsis/GoGrid Worker_0.1.0_x64-setup.exe` - NSIS installer

### Linux

**Prerequisites**:
- `build-essential`
- `libwebkit2gtk-4.0-dev`
- `libssl-dev`
- `libgtk-3-dev`
- `libayatana-appindicator3-dev`

**Build Command**:
```bash
# On Linux machine
cargo build --release --bin corpgrid-scheduler
cp target/release/corpgrid-scheduler target/release/corpgrid-scheduler-x86_64-unknown-linux-gnu
cd crates/tray-app/src-tauri
cargo tauri build
```

**Output**:
- `target/release/bundle/deb/gogrid-worker_0.1.0_amd64.deb` - Debian package
- `target/release/bundle/appimage/GoGrid_Worker_0.1.0_amd64.AppImage` - AppImage

## Build Process Details

### 1. Build Worker Binary

```bash
cd /path/to/GoGrid
cargo build --release --bin corpgrid-scheduler
```

This creates the GPU inference worker at:
- macOS: `target/release/corpgrid-scheduler`
- Windows: `target/release/corpgrid-scheduler.exe`
- Linux: `target/release/corpgrid-scheduler`

### 2. Copy Binary with Target Triple

Tauri expects the binary to have the target triple in its name:

```bash
# macOS ARM64
cp target/release/corpgrid-scheduler target/release/corpgrid-scheduler-aarch64-apple-darwin

# macOS Intel
cp target/release/corpgrid-scheduler target/release/corpgrid-scheduler-x86_64-apple-darwin

# Windows
cp target/release/corpgrid-scheduler.exe target/release/corpgrid-scheduler-x86_64-pc-windows-msvc.exe

# Linux
cp target/release/corpgrid-scheduler target/release/corpgrid-scheduler-x86_64-unknown-linux-gnu
```

### 3. Build Tauri Bundle

```bash
cd crates/tray-app/src-tauri
cargo tauri build
```

This will:
1. Compile the tray application
2. Bundle it with the worker binary
3. Create platform-specific installers
4. Sign the application (if configured)

## Configuration

The installer configuration is in `crates/tray-app/src-tauri/tauri.conf.json`:

```json
{
  "bundle": {
    "active": true,
    "targets": "all",
    "externalBin": [
      "../../../target/release/corpgrid-scheduler"
    ],
    "macOS": {
      "minimumSystemVersion": "10.15"
    },
    "windows": {
      "nsis": {
        "installMode": "perMachine"
      }
    }
  }
}
```

## Testing Installers

### macOS

```bash
# Mount DMG
open "target/release/bundle/dmg/GoGrid Worker_0.1.0_aarch64.dmg"

# Or run app bundle directly
open "target/release/bundle/macos/GoGrid Worker.app"
```

### Windows

```powershell
# Run installer
.\target\release\bundle\nsis\GoGrid Worker_0.1.0_x64-setup.exe

# Or MSI
msiexec /i "target\release\bundle\msi\GoGrid Worker_0.1.0_x64_en-US.msi"
```

### Linux

```bash
# Debian/Ubuntu
sudo dpkg -i target/release/bundle/deb/gogrid-worker_0.1.0_amd64.deb

# AppImage
chmod +x target/release/bundle/appimage/GoGrid_Worker_0.1.0_amd64.AppImage
./target/release/bundle/appimage/GoGrid_Worker_0.1.0_amd64.AppImage
```

## Bundle Contents

Each installer contains:

1. **Tray Application** (`gogrid-tray`):
   - System tray icon and menu
   - Connection to coordinator at bx.ee:8443
   - Worker process management

2. **Worker Binary** (`corpgrid-scheduler`):
   - GPU inference worker
   - Adaptive throttling
   - Resource management

3. **Icons**:
   - System tray icon
   - Application icon
   - File type icons

4. **Metadata**:
   - Version information
   - Copyright notice
   - License (MIT)

## Code Signing

### macOS

To sign the macOS application:

```bash
# Create signing certificate in Keychain Access
# Update tauri.conf.json:
{
  "macOS": {
    "signingIdentity": "Developer ID Application: Your Name (TEAM_ID)"
  }
}

# Build will automatically sign
cargo tauri build
```

### Windows

To sign the Windows installer:

```powershell
# Obtain code signing certificate
# Update tauri.conf.json:
{
  "windows": {
    "certificateThumbprint": "YOUR_CERT_THUMBPRINT"
  }
}

# Build will automatically sign
cargo tauri build
```

## Distribution

### macOS

1. Upload DMG to website/CDN
2. Optionally notarize with Apple:
   ```bash
   xcrun notarytool submit "GoGrid Worker_0.1.0_aarch64.dmg" \
     --apple-id "your@email.com" \
     --team-id "TEAM_ID" \
     --password "app-specific-password"
   ```

### Windows

1. Upload installer to website/CDN
2. Consider using Chocolatey for distribution:
   ```powershell
   choco pack
   choco push
   ```

### Linux

1. Upload packages to repository or website
2. Consider adding to package repositories:
   - Debian: Upload to PPA
   - Fedora: Submit to COPR
   - Arch: Submit to AUR

## Troubleshooting

### "resource path doesn't exist"

Ensure the worker binary is built before running `cargo tauri build`:

```bash
cargo build --release --bin corpgrid-scheduler
```

### "failed to bundle project"

Check that all icon files exist:

```bash
ls -l crates/tray-app/src-tauri/icons/
# Should show: icon.icns, icon.ico, 32x32.png, 128x128.png, etc.
```

### Binary architecture mismatch

Make sure you're building for the correct architecture:

```bash
# Check current target
rustc --version --verbose | grep host

# Build for specific target
cargo build --release --target aarch64-apple-darwin
cargo build --release --target x86_64-apple-darwin
```

### Signing issues on macOS

If unsigned, users will see "App is from an unidentified developer":

1. Right-click app → Open
2. Click "Open" in security dialog
3. Or: System Settings → Privacy & Security → "Open Anyway"

## Automated Builds

### GitHub Actions

Create `.github/workflows/build.yml`:

```yaml
name: Build Installers

on:
  push:
    tags:
      - 'v*'

jobs:
  build-macos:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
      - name: Build worker
        run: cargo build --release --bin corpgrid-scheduler
      - name: Build installer
        run: |
          cp target/release/corpgrid-scheduler target/release/corpgrid-scheduler-aarch64-apple-darwin
          cd crates/tray-app/src-tauri
          cargo tauri build
      - uses: actions/upload-artifact@v3
        with:
          name: macos-dmg
          path: target/release/bundle/dmg/*.dmg

  build-windows:
    runs-on: windows-latest
    # Similar steps for Windows

  build-linux:
    runs-on: ubuntu-latest
    # Similar steps for Linux
```

## Version Updates

To update the version:

1. Update `Cargo.toml` workspace version
2. Update `crates/tray-app/src-tauri/tauri.conf.json` version
3. Rebuild installers
4. Tag release: `git tag v0.1.1 && git push --tags`

## File Sizes

Typical installer sizes:

- **macOS DMG**: ~25-30 MB
- **Windows NSIS**: ~20-25 MB
- **Linux DEB**: ~20-25 MB
- **Linux AppImage**: ~25-30 MB

The worker binary (`corpgrid-scheduler`) is the largest component at ~70 MB due to ML model dependencies.

## Next Steps

1. **Code Signing**: Set up code signing for production releases
2. **Auto-Update**: Implement Tauri's auto-update feature
3. **Crash Reporting**: Add crash reporting service
4. **Analytics**: Add usage analytics (with user consent)
5. **Translations**: Add i18n support for multiple languages
