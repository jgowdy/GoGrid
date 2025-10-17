# Building GoGrid Worker

This guide explains how to build GoGrid Worker for all supported platforms.

## Supported Platforms

- **macOS** (ARM64 & Intel x86_64)
- **Linux** (x86_64)
- **Windows** (x86_64)

## Quick Start

### Automated Build (Recommended)

Use GitHub Actions for automated multi-platform builds:

```bash
# Push a tag to trigger release build
git tag v0.1.1
git push --tags

# Or manually trigger from GitHub Actions UI
```

See [`.github/RELEASE.md`](.github/RELEASE.md) for details.

### Local Build (Current Platform)

```bash
# Build for your current platform
./build_all_platforms.sh 0.1.0
```

## Prerequisites

### All Platforms

- Rust 1.70+ (`rustup` recommended)
- Node.js 20+
- Tauri CLI: `cargo install tauri-cli --version '^2.0.0'`

### macOS

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Add targets for cross-compilation
rustup target add aarch64-apple-darwin
rustup target add x86_64-apple-darwin
```

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get update
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
```

### Windows

```bash
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/

# Or using chocolatey
choco install visualstudio2022buildtools
choco install visualstudio2022-workload-vctools
```

## Build Process

### 1. Build Scheduler (Rust Binary)

The scheduler is a Rust binary that runs the actual GPU workload.

```bash
# macOS ARM64
cargo build --release --bin corpgrid-scheduler --target aarch64-apple-darwin

# macOS Intel
cargo build --release --bin corpgrid-scheduler --target x86_64-apple-darwin

# Linux
cargo build --release --bin corpgrid-scheduler

# Windows
cargo build --release --bin corpgrid-scheduler
```

### 2. Build Coordinator (Server Binary)

The coordinator runs on bx.ee and manages worker connections.

```bash
cargo build --release --bin gogrid-coordinator
```

### 3. Build Tray App (Tauri GUI)

The tray app is the system tray application that wraps the scheduler.

```bash
cd crates/tray-app/src-tauri

# macOS ARM64
cargo tauri build --target aarch64-apple-darwin

# macOS Intel
cargo tauri build --target x86_64-apple-darwin

# Linux
cargo tauri build

# Windows
cargo tauri build
```

### 4. Package for Distribution

Each platform requires different packaging:

#### macOS
```bash
# DMG (manual installer) - automatically created by Tauri
# Location: target/{target}/release/bundle/dmg/GoGrid Worker_0.1.0_{arch}.dmg

# .app.tar.gz (auto-updater) - create manually
cd target/aarch64-apple-darwin/release/bundle/macos
tar -czf "GoGrid_Worker_0.1.0_aarch64.app.tar.gz" "GoGrid Worker.app"
```

#### Linux
```bash
# AppImage (manual installer) - automatically created by Tauri
# Location: target/release/bundle/appimage/gogrid-worker_0.1.0_amd64.AppImage

# .AppImage.tar.gz (auto-updater) - create manually
cd target/release/bundle/appimage
tar -czf "GoGrid_Worker_0.1.0_amd64.AppImage.tar.gz" *.AppImage

# Also creates .deb package
# Location: target/release/bundle/deb/gogrid-worker_0.1.0_amd64.deb
```

#### Windows
```bash
# .exe installer (manual) - automatically created by Tauri
# Location: target/release/bundle/nsis/GoGrid Worker_0.1.0_x64-setup.exe

# .nsis.zip (auto-updater) - create manually
cd target/release/bundle/nsis
tar -czf "GoGrid_Worker_0.1.0_x64-setup.nsis.zip" *.exe

# Also creates MSI installer
# Location: target/release/bundle/msi/GoGrid Worker_0.1.0_x64_en-US.msi
```

## Build Outputs

### Directory Structure

```
target/
├── aarch64-apple-darwin/release/
│   ├── corpgrid-scheduler              # Scheduler binary (ARM64)
│   └── bundle/
│       ├── macos/
│       │   ├── GoGrid Worker.app       # macOS app bundle
│       │   └── GoGrid_Worker_0.1.0_aarch64.app.tar.gz  # Auto-update package
│       └── dmg/
│           └── GoGrid Worker_0.1.0_aarch64.dmg         # Manual installer
│
├── x86_64-apple-darwin/release/
│   └── ... (similar structure for Intel Macs)
│
├── release/
│   ├── corpgrid-scheduler              # Scheduler binary (Linux/Windows)
│   ├── gogrid-coordinator              # Coordinator server binary
│   └── bundle/
│       ├── appimage/                   # Linux packages
│       │   ├── gogrid-worker_0.1.0_amd64.AppImage
│       │   └── GoGrid_Worker_0.1.0_amd64.AppImage.tar.gz
│       ├── deb/                        # Debian packages
│       │   └── gogrid-worker_0.1.0_amd64.deb
│       ├── nsis/                       # Windows installers
│       │   ├── GoGrid Worker_0.1.0_x64-setup.exe
│       │   └── GoGrid_Worker_0.1.0_x64-setup.nsis.zip
│       └── msi/                        # Windows MSI
│           └── GoGrid Worker_0.1.0_x64_en-US.msi
```

## Cross-Compilation

### macOS: Build Both Architectures

On an ARM64 Mac, you can build for both ARM64 and Intel:

```bash
# Install both targets
rustup target add aarch64-apple-darwin x86_64-apple-darwin

# Build for both
cargo build --release --target aarch64-apple-darwin
cargo build --release --target x86_64-apple-darwin
```

### Linux to Windows (Not Recommended)

Cross-compiling from Linux to Windows is complex due to WebView2 dependencies. Use GitHub Actions or a Windows VM instead.

## Testing Builds

### Local Testing

```bash
# Run the scheduler directly
./target/release/corpgrid-scheduler

# Run the tray app
open "target/release/bundle/macos/GoGrid Worker.app"  # macOS
./target/release/bundle/appimage/*.AppImage           # Linux
target\release\bundle\nsis\*.exe                      # Windows
```

### Integration Testing

```bash
# Run all tests
cargo test --workspace

# Run scheduler tests only
cargo test -p corpgrid-scheduler

# Run coordinator tests only
cargo test -p corpgrid-coordinator
```

## Troubleshooting

### macOS: "App is damaged and can't be opened"

This happens when the app is not code-signed. On development machines:

```bash
xattr -cr "target/release/bundle/macos/GoGrid Worker.app"
```

For distribution, you need an Apple Developer account to sign the app.

### Linux: Missing WebKit2GTK

```bash
# Error: webkit2gtk-4.1 not found
sudo apt-get install libwebkit2gtk-4.1-dev

# If still not found, try webkit2gtk-4.0
sudo apt-get install libwebkit2gtk-4.0-dev
```

### Windows: MSVC Build Tools Not Found

Download and install Visual Studio Build Tools:
https://visualstudio.microsoft.com/downloads/

Or use the Visual Studio Installer to add "Desktop development with C++".

### Tauri Build Fails: Icon Not Found

Ensure icon files exist:

```bash
ls -la crates/tray-app/src-tauri/icons/
# Should show: 32x32.png, 128x128.png, 128x128@2x.png, icon.icns, icon.ico
```

Regenerate icons if needed:

```bash
cd crates/tray-app/src-tauri/icons
# See AUTO_UPDATE.md for icon generation commands
```

### Out of Disk Space

Rust builds can be large. Clean old builds:

```bash
cargo clean
rm -rf target/
```

Or use `cargo-cache`:

```bash
cargo install cargo-cache
cargo cache --autoclean
```

## Optimization

### Release Builds

Release builds are optimized by default with:
- LTO (Link-Time Optimization)
- Code stripping
- Size optimization

Configure in `Cargo.toml`:

```toml
[profile.release]
opt-level = "z"     # Optimize for size
lto = true          # Enable Link-Time Optimization
codegen-units = 1   # Better optimization
strip = true        # Strip debug symbols
```

### Build Cache

Speed up builds with `sccache`:

```bash
cargo install sccache
export RUSTC_WRAPPER=sccache
```

## CI/CD Integration

See [`.github/workflows/build-release.yml`](.github/workflows/build-release.yml) for the complete GitHub Actions configuration.

Key features:
- ✅ Parallel builds for all platforms
- ✅ Automatic artifact upload
- ✅ GitHub Release creation
- ✅ Deployment to update server
- ✅ Build caching for faster builds

## Questions?

- Build issues: Check [GitHub Issues](https://github.com/gogrid/gogrid/issues)
- Release process: See [`.github/RELEASE.md`](.github/RELEASE.md)
- Auto-updates: See [`AUTO_UPDATE.md`](AUTO_UPDATE.md)
