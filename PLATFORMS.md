# Platform Support

GoGrid Worker supports three major platforms with automated builds via GitHub Actions.

## Supported Platforms

### macOS

**Architectures:**
- ARM64 (Apple Silicon: M1, M2, M3, M4)
- x86_64 (Intel Macs)

**Requirements:**
- macOS 10.15 (Catalina) or later
- 50 MB disk space
- GPU: Metal-compatible (Apple Silicon recommended)

**Installers:**
- `.dmg` - Standard macOS disk image for manual installation
- `.app.tar.gz` - Auto-update package

**Build Platform:** macOS runners (GitHub Actions) or local macOS machine

### Linux

**Architectures:**
- x86_64 (64-bit Intel/AMD)

**Requirements:**
- Ubuntu 20.04+ / Debian 11+ / Fedora 35+ / Arch Linux
- GLIBC 2.31+
- 50 MB disk space
- GPU: NVIDIA CUDA 11.8+ (for GPU acceleration)

**Installers:**
- `.AppImage` - Universal Linux application format
- `.deb` - Debian/Ubuntu package
- `.AppImage.tar.gz` - Auto-update package

**Build Platform:** Ubuntu runners (GitHub Actions) or Linux machine with build dependencies

### Windows

**Architectures:**
- x86_64 (64-bit Intel/AMD)

**Requirements:**
- Windows 10 (1809+) or Windows 11
- WebView2 Runtime (included in Win11, auto-installed on Win10)
- 50 MB disk space
- GPU: NVIDIA CUDA 11.8+ (for GPU acceleration)

**Installers:**
- `.exe` (NSIS) - Standard Windows installer
- `.msi` - Windows Installer package
- `.nsis.zip` - Auto-update package

**Build Platform:** Windows runners (GitHub Actions) or Windows machine with Visual Studio Build Tools

## Platform-Specific Features

### macOS
- ✅ Native system tray icon (menu bar)
- ✅ Metal GPU acceleration for Apple Silicon
- ✅ Code signing ready (requires Apple Developer account)
- ✅ Universal binary support (single app for both architectures)
- ⚠️ Gatekeeper bypass required for unsigned builds (`xattr -cr`)

### Linux
- ✅ System tray via libappindicator
- ✅ CUDA GPU acceleration
- ✅ No installation required (AppImage is portable)
- ✅ DEB package for apt-based distributions
- ⚠️ Requires AppImage FUSE or kernel support

### Windows
- ✅ Native system tray icon
- ✅ CUDA GPU acceleration
- ✅ Automatic WebView2 installation
- ✅ Both NSIS and MSI installers
- ✅ Per-machine or per-user installation options

## Auto-Update Support

All platforms support automatic updates via the Tauri updater plugin:

| Platform | Update Format | Size | Frequency |
|----------|---------------|------|-----------|
| macOS ARM64 | `.app.tar.gz` | ~28 MB | Every 24h |
| macOS Intel | `.app.tar.gz` | ~28 MB | Every 24h |
| Linux | `.AppImage.tar.gz` | ~25 MB | Every 24h |
| Windows | `.nsis.zip` | ~25 MB | Every 24h |

Updates are checked:
- 5 seconds after app startup
- Every 24 hours thereafter
- Downloads and installs in background
- Prompts user to restart

## Build Matrix

### GitHub Actions (Automated)

```yaml
Platform | Runner        | Rust Target                 | Build Time
---------|---------------|----------------------------|------------
macOS ARM | macos-latest | aarch64-apple-darwin       | ~12 min
macOS x64 | macos-latest | x86_64-apple-darwin        | ~12 min
Linux     | ubuntu-latest | x86_64-unknown-linux-gnu | ~10 min
Windows   | windows-latest | x86_64-pc-windows-msvc   | ~15 min
```

### Cross-Compilation Support

- **macOS → macOS**: ✅ Can build both ARM64 and Intel from either architecture
- **macOS → Linux**: ❌ Not recommended (use GitHub Actions or Linux VM)
- **macOS → Windows**: ❌ Not possible (use GitHub Actions or Windows VM)
- **Linux → Linux**: ✅ Native build only
- **Linux → Windows**: ⚠️ Possible but complex (MinGW), not recommended
- **Windows → Windows**: ✅ Native build only

## Download Statistics

Estimated download sizes and requirements:

| Platform | Installer | Installed Size | RAM Usage | GPU VRAM |
|----------|-----------|----------------|-----------|----------|
| macOS (ARM) | 28 MB | 60 MB | 200 MB | 4 GB+ |
| macOS (Intel) | 28 MB | 60 MB | 200 MB | 4 GB+ |
| Linux | 25 MB | 55 MB | 180 MB | 4 GB+ |
| Windows | 25 MB | 55 MB | 180 MB | 4 GB+ |

## Deployment

### Update Server

All installers and update packages are hosted on your coordinator server:
```
https://your-server.com:8443/downloads/
https://your-server.com:8443/files/{package}
```

### Update Manifest Endpoints

```
GET https://your-server.com:8443/updates/{target}/{current_version}
```

Supported targets:
- `darwin-aarch64` - macOS ARM64
- `darwin-x86_64` - macOS Intel
- `linux-x86_64` - Linux x86_64
- `windows-x86_64` - Windows x86_64

## Future Platform Support

### Potential Additions

- **ARM64 Linux**: For Raspberry Pi 4/5, ARM servers
  - Status: Not yet implemented
  - Blocker: CUDA support limited on ARM Linux

- **Windows ARM64**: For Windows on ARM (Surface Pro X, etc.)
  - Status: Not yet implemented
  - Blocker: Tauri 2.0 Windows ARM support experimental

- **macOS Universal Binary**: Single .app for both architectures
  - Status: Possible with lipo tool
  - Benefit: Simpler distribution

## Testing

### Platform-Specific Tests

```bash
# macOS
cargo test --target aarch64-apple-darwin
cargo test --target x86_64-apple-darwin

# Linux
cargo test --target x86_64-unknown-linux-gnu

# Windows
cargo test --target x86_64-pc-windows-msvc
```

### Integration Tests

See `.github/workflows/build-test.yml` for CI test configuration.

## Troubleshooting

### macOS Issues

**Problem**: "App is damaged and can't be opened"
```bash
# Remove quarantine flag
xattr -cr "/Applications/GoGrid Worker.app"
```

**Problem**: Gatekeeper blocking unsigned app
- Solution: Right-click app → Open → Confirm

### Linux Issues

**Problem**: AppImage won't run
```bash
# Check FUSE support
modprobe fuse
# Or extract and run
./gogrid-worker.AppImage --appimage-extract
./squashfs-root/AppRun
```

**Problem**: System tray icon not showing
- Install: `sudo apt-get install libayatana-appindicator3-1`

### Windows Issues

**Problem**: WebView2 not found
- Download: https://go.microsoft.com/fwlink/p/?LinkId=2124703
- Or: Installer auto-installs it

**Problem**: CUDA not detected
- Install NVIDIA drivers 522.06+
- Verify: `nvidia-smi`

## Contributing

See [BUILDING.md](BUILDING.md) for detailed build instructions and [.github/RELEASE.md](.github/RELEASE.md) for the release process.
