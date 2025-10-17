# GoGrid Worker

**Distributed GPU Inference Network**

GoGrid Worker is a system tray application for running your own private distributed GPU compute network. Use idle GPU resources across your personal machines or corporate infrastructure to process AI inference workloads.

## Features

- 🖥️ **System Tray Application** - Runs quietly in the background
- 🔄 **Auto-Updates** - Seamlessly updates to the latest version
- 🎮 **GPU Accelerated** - Supports NVIDIA CUDA and Apple Metal
- 🌐 **Cross-Platform** - macOS (ARM64 & Intel), Linux, and Windows
- 📊 **Real-time Stats** - Monitor job completion and GPU usage
- ⚡ **Efficient** - Minimal resource usage when idle
- 🔒 **Private** - Run your own coordinator server

## Download

Get the latest version from our downloads page:

**[Download GoGrid Worker](https://bx.ee:8443/downloads)**

### Platform Support

| Platform | Architecture | Status |
|----------|-------------|---------|
| macOS | ARM64 (M1/M2/M3) | ✅ Available |
| macOS | Intel x86_64 | ✅ Available |
| Linux | x86_64 | ✅ Available |
| Windows | x86_64 | ✅ Available |

## Quick Start

### macOS

1. Download `GoGrid Worker.dmg`
2. Open the DMG and drag to Applications
3. Launch from Applications folder
4. Grant necessary permissions when prompted

### Linux

```bash
# Download and run AppImage
chmod +x gogrid-worker_*.AppImage
./gogrid-worker_*.AppImage

# Or install DEB package
sudo dpkg -i gogrid-worker_*.deb
```

### Windows

1. Download `GoGrid Worker Setup.exe`
2. Run the installer
3. Launch from Start Menu

## Requirements

### Hardware

- **CPU**: Any modern 64-bit processor
- **RAM**: 2 GB minimum, 4 GB recommended
- **GPU**: 
  - NVIDIA GPU with CUDA 11.8+ (recommended)
  - Apple Silicon Macs (M1/M2/M3) with Metal
  - AMD GPUs (coming soon)
- **Storage**: 100 MB free space

### Software

- **macOS**: 10.15 (Catalina) or later
- **Linux**: Ubuntu 20.04+, Debian 11+, Fedora 35+, Arch Linux
- **Windows**: Windows 10 (1809+) or Windows 11

## Building from Source

See [BUILDING.md](BUILDING.md) for detailed build instructions.

Quick build:

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone repository
git clone https://github.com/jgowdy/GoGrid.git
cd GoGrid

# Build all platforms (current platform only)
./build_all_platforms.sh 0.1.0

# Or build specific components
cargo build --release --bin corpgrid-scheduler
cd crates/tray-app/src-tauri
cargo tauri build
```

## Architecture

GoGrid consists of three main components:

### 1. Worker (This Repository)

The client application that runs on user machines:
- **Scheduler** (`corpgrid-scheduler`) - Rust binary that executes GPU workloads
- **Tray App** (`gogrid-tray`) - Tauri-based GUI for user interaction
- Connects to coordinator via QUIC protocol
- Manages GPU resources and job execution

### 2. Coordinator (Included)

Central server that manages the network:
- **Coordinator** (`gogrid-coordinator`) - Connection management and job distribution
- Serves auto-update manifests and installers
- Handles worker registration and heartbeats
- Runs on `bx.ee:8443`

### 3. Job Queue (Separate)

Backend services for job management:
- Job submission and queueing
- Job prioritization
- Management dashboard
- (Not included in this repository)

## Auto-Updates

GoGrid Worker automatically checks for updates:
- **On startup** - After 5 second delay
- **Every 24 hours** - Background check
- **Silent download** - No interruption to current work
- **Smart restart** - Prompts only when convenient

Update server: `https://bx.ee:8443/updates/`

See [AUTO_UPDATE.md](AUTO_UPDATE.md) for technical details.

## Configuration

### Default Settings

The worker connects to `bx.ee:8443` by default. Settings can be configured through the tray menu:

- **Pause/Resume** - Stop accepting new jobs
- **GPU Settings** - Limit VRAM usage
- **Network Settings** - Custom coordinator URL
- **Dashboard** - View job statistics and GPU usage

Configuration is stored at:
- **macOS**: `~/Library/Application Support/GoGrid Worker/`
- **Linux**: `~/.config/gogrid-worker/`
- **Windows**: `%APPDATA%\GoGrid Worker\`

## Contributing

Contributions are welcome! Please read our contributing guidelines first.

### Development Setup

```bash
# Clone and enter directory
git clone https://github.com/jgowdy/GoGrid.git
cd GoGrid

# Run tests
cargo test --workspace

# Run scheduler locally
cargo run --bin corpgrid-scheduler

# Run tray app in dev mode
cd crates/tray-app
npm install
npm run tauri dev
```

### CI/CD

This project uses GitHub Actions for automated builds:

- **Pull Requests** - Build and test all platforms
- **Main Branch** - Test builds on every commit
- **Tagged Releases** - Build all platforms and create GitHub release

See [.github/RELEASE.md](.github/RELEASE.md) for release process.

## Documentation

- [BUILDING.md](BUILDING.md) - Build instructions for all platforms
- [AUTO_UPDATE.md](AUTO_UPDATE.md) - Auto-update system details
- [PLATFORMS.md](PLATFORMS.md) - Platform-specific information
- [.github/RELEASE.md](.github/RELEASE.md) - Release process

## Security & Privacy

GoGrid is designed with security as a core principle for private infrastructure:

### Network Security

- **QUIC Protocol** - Modern encrypted transport using Quinn (Rust implementation)
  - TLS 1.3 encryption for all coordinator-worker communication
  - Built-in connection security and authentication
  - Resistant to connection hijacking and replay attacks

- **Configurable Endpoints** - Run your own coordinator server
  - No external dependencies on third-party services
  - Full control over network topology
  - Isolated from public networks

### Sandboxing & Isolation

- **Process Isolation** - Worker and scheduler run as separate processes
  - Tauri's security model isolates GUI from compute workloads
  - Scheduler process can be run with restricted privileges
  - Limited filesystem access via Tauri security policies

- **Resource Limits** - Configurable GPU VRAM usage caps
  - Prevents runaway jobs from consuming all resources
  - User-controlled execution boundaries

### User Experience & Non-Intrusive Operation

- **Minimal Disruption**
  - System tray application runs in background
  - No pop-ups or notifications during normal operation
  - Silent auto-updates (download in background, prompt only when ready)
  - Jobs automatically pause when user activity detected (planned)

- **Respectful Resource Usage**
  - Only uses idle GPU resources
  - Configurable GPU usage limits
  - Automatic throttling when system is under load
  - Low memory footprint when idle

- **Smart Scheduling**
  - Updates check on startup (after 5s delay) and every 24 hours
  - Update installation only prompts when convenient
  - No forced restarts
  - Jobs complete before shutdown/update

### Data Privacy

- **No Telemetry** - Zero analytics or tracking
- **Local Configuration** - All settings stored locally
- **Private Network** - Jobs and data never leave your infrastructure
- **No External Dependencies** - Can run completely air-gapped

### Reporting Vulnerabilities

Please report security vulnerabilities to: security@bx.ee

Do not open public issues for security concerns.

### Planned Security Enhancements

- **Update Signing** - Cryptographic verification of updates
  ```bash
  # Will use Tauri's built-in signing
  tauri signer generate -w ~/.tauri/gogrid.key
  tauri signer sign update.tar.gz
  ```

- **Certificate Pinning** - Pin coordinator TLS certificates
- **Worker Authentication** - Optional API keys for worker registration
- **Job Sandboxing** - Additional process isolation for inference workloads

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: https://bx.ee/docs
- **Dashboard**: https://bx.ee/dashboard
- **Issues**: https://github.com/jgowdy/GoGrid/issues
- **Email**: support@bx.ee

## Acknowledgments

Built with:
- [Rust](https://www.rust-lang.org/) - Systems programming language
- [Tauri](https://tauri.app/) - Desktop application framework
- [Quinn](https://github.com/quinn-rs/quinn) - QUIC protocol implementation
- [Candle](https://github.com/huggingface/candle) - ML framework for Rust

## Roadmap

- [x] macOS support (ARM64 & Intel)
- [x] Linux support (x86_64)
- [x] Windows support (x86_64)
- [x] Auto-update system
- [x] System tray interface
- [ ] Code signing for updates
- [ ] GPU temperature monitoring
- [ ] Advanced scheduling options
- [ ] Multi-GPU support
- [ ] AMD GPU support
- [ ] ARM Linux support

## Status

**Current Version**: 0.1.0

- ✅ Core functionality working
- ✅ Auto-updates implemented
- ✅ Multi-platform builds
- ⚠️ Beta software - use at your own risk

---

Made with ❤️ by the GoGrid team

## Configuration

### Coordinator Endpoint

By default, GoGrid Worker connects to `bx.ee:8443`. To use your own coordinator:

**Environment Variables**:
```bash
export GOGRID_COORDINATOR_HOST=your-server.com
export GOGRID_COORDINATOR_PORT=8443
```

**macOS** (launchd):
```bash
# Edit ~/Library/LaunchAgents/com.gogrid.worker.plist
<key>EnvironmentVariables</key>
<dict>
    <key>GOGRID_COORDINATOR_HOST</key>
    <string>your-server.com</string>
    <key>GOGRID_COORDINATOR_PORT</key>
    <string>8443</string>
</dict>
```

**Linux** (systemd):
```bash
# Create ~/.config/systemd/user/gogrid-worker.service.d/override.conf
[Service]
Environment="GOGRID_COORDINATOR_HOST=your-server.com"
Environment="GOGRID_COORDINATOR_PORT=8443"
```

**Windows**:
```cmd
setx GOGRID_COORDINATOR_HOST your-server.com
setx GOGRID_COORDINATOR_PORT 8443
```

### Update Server

The update endpoints are configured in `tauri.conf.json`. By default:
- Primary: `https://bx.ee:8443/updates/`
- Fallback: `https://gogrid-updates.example.com/updates/`

To host your own update server, see [AUTO_UPDATE.md](AUTO_UPDATE.md).
