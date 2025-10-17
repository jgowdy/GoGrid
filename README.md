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

Get the latest version from your coordinator server's downloads page:

**[Download GoGrid Worker](https://your-server.com:8443/downloads)**

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
- Example deployment: `coordinator.example.com:8443`

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

Example update server: `https://your-server.com:8443/updates/`

See [AUTO_UPDATE.md](AUTO_UPDATE.md) for technical details.

## Configuration

### Configuration Files

Settings can be configured through the tray menu or via configuration files:

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

Please report security vulnerabilities via GitHub Security Advisories.

Do not open public issues for security concerns.

### Implemented Security Features

- **Update Signing** ✅ - Cryptographic verification of updates using Minisign
  - Self-signed keys stored in `~/.tauri/gogrid.key`
  - Public key embedded in application for verification
  - All updates verified before installation

- **Job Sandboxing** ✅ - Process isolation for inference workloads
  - macOS: `sandbox-exec` with custom security profiles
  - Linux: Bubblewrap (bwrap) for container-like isolation
  - Windows: Job Objects for resource limits
  - Configurable resource limits (memory, CPU time, network access)

### Planned Security Enhancements

- **Certificate Pinning** - Pin coordinator TLS certificates
- **Worker Authentication** - Optional API keys for worker registration
- **Enhanced Sandboxing** - Additional filesystem restrictions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: https://github.com/jgowdy/GoGrid/issues
- **Discussions**: https://github.com/jgowdy/GoGrid/discussions

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
- [x] Code signing for updates
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

GoGrid Worker requires coordinator configuration on first run. Configure via environment variables or configuration file.

**Required Configuration**:

The worker must be configured with a coordinator server address. On first run, if environment variables are not set, you'll be prompted to configure.

**Environment Variables** (recommended):
```bash
export GOGRID_COORDINATOR_HOST=your-server.com
export GOGRID_COORDINATOR_PORT=8443
export GOGRID_UPDATE_ENDPOINTS=https://your-server.com:8443/updates/{{target}}/{{current_version}}
```

**Example**:
```bash
export GOGRID_COORDINATOR_HOST=coordinator.example.com
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

**Configuration File** (alternative):

If environment variables are not set, the worker will create a configuration file at:
- **macOS**: `~/Library/Application Support/GoGrid Worker/config.toml`
- **Linux**: `~/.config/gogrid-worker/config.toml`
- **Windows**: `%APPDATA%\GoGrid Worker\config.toml`

Example `config.toml`:
```toml
[coordinator]
host = "your-server.com"
port = 8443

[updates]
enabled = true
endpoints = ["https://your-server.com:8443/updates/{{target}}/{{current_version}}"]

[worker]
max_vram_gb = 8.0
pause_on_activity = true
```

**Note**: Environment variables take precedence over configuration file settings.

### Update Server

The update endpoints are configured in `tauri.conf.json` and via environment variables. Example configurations:
- Example: `https://coordinator.example.com:8443/updates/`
- Example: `https://your-server.com:8443/updates/`

To host your own update server, see [AUTO_UPDATE.md](AUTO_UPDATE.md).
