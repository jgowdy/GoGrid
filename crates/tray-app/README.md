# GoGrid Worker Tray Application

A cross-platform system tray application for managing the GoGrid distributed GPU inference worker.

## Features

- **System Tray Integration**: Runs in the background with a system tray icon
- **Worker Process Management**: Start, stop, and monitor the worker process
- **QUIC Connection**: Connects to the GoGrid coordinator at bx.ee:8443
- **Automatic Registration**: Registers worker with coordinator on startup
- **Heartbeat Monitoring**: Maintains connection with 30-second heartbeats
- **Resource Management**: Integrates with the scheduler's adaptive throttling system

## Architecture

### Components

1. **Tray Application** (`gogrid-tray`):
   - Built with Tauri for native performance
   - Provides system tray UI and menu
   - Manages worker process lifecycle
   - Handles coordinator communication

2. **Worker Process** (`corpgrid-scheduler`):
   - Standalone GPU inference worker
   - Runs as child process of tray app
   - Managed via signals (SIGTERM/SIGKILL on Unix)

3. **Coordinator Client**:
   - QUIC-based communication protocol
   - Worker registration and heartbeats
   - Job assignment handling

### File Structure

```
crates/tray-app/
├── src-tauri/
│   ├── src/
│   │   ├── main.rs                    # Tray app entry point
│   │   ├── lib.rs                     # Library exports
│   │   ├── coordinator_client.rs      # QUIC client implementation
│   │   └── worker_process.rs          # Process management
│   ├── icons/                         # Application icons
│   ├── Cargo.toml                     # Dependencies
│   └── tauri.conf.json               # Tauri configuration
└── README.md                          # This file
```

## Building

### Prerequisites

- Rust 1.70+
- Tauri CLI: `cargo install tauri-cli@^2.0.0`
- Platform-specific dependencies:
  - **macOS**: Xcode Command Line Tools
  - **Windows**: Visual Studio Build Tools
  - **Linux**: `build-essential`, `libwebkit2gtk-4.0-dev`, `libssl-dev`

### Development Build

```bash
cd crates/tray-app/src-tauri
cargo build
```

### Release Build

```bash
cd crates/tray-app/src-tauri
cargo build --release
```

### Create Installers

```bash
# Build for current platform
cargo tauri build

# Installers will be created in:
# - macOS: target/release/bundle/dmg/
# - Windows: target/release/bundle/nsis/ or target/release/bundle/msi/
# - Linux: target/release/bundle/deb/ or target/release/bundle/appimage/
```

## Installation

### macOS

1. Download `GoGrid Worker.dmg`
2. Open the DMG and drag "GoGrid Worker" to Applications
3. Launch from Applications or Spotlight
4. The app will appear in the menu bar (top right)

### Windows

1. Download `GoGrid Worker Setup.exe`
2. Run the installer
3. The app will start automatically and appear in the system tray

### Linux

**Debian/Ubuntu (.deb)**:
```bash
sudo dpkg -i gogrid-worker_0.1.0_amd64.deb
gogrid-worker
```

**AppImage**:
```bash
chmod +x GoGrid_Worker-0.1.0-x86_64.AppImage
./GoGrid_Worker-0.1.0-x86_64.AppImage
```

## Usage

### Tray Menu Options

- **Status**: Shows current connection and worker status
- **Pause Worker**: Stops the worker process
- **Resume Worker**: Starts the worker process
- **Settings**: Configure worker settings (coming soon)
- **Statistics**: View worker stats and earnings (coming soon)
- **Open Dashboard**: Opens bx.ee dashboard in browser
- **Quit**: Stops worker and exits the application

### Configuration

The tray app automatically:
- Locates the worker binary (corpgrid-scheduler)
- Connects to coordinator at bx.ee:8443
- Registers the worker with a unique ID
- Starts sending heartbeats every 30 seconds

### Worker Binary Path

The app looks for the worker binary in:
- **Development**: `target/debug/corpgrid-scheduler`
- **macOS Bundle**: `../MacOS/corpgrid-scheduler`
- **Windows**: `corpgrid-scheduler.exe` (same directory)
- **Linux**: `corpgrid-scheduler` (same directory)

## Development

### Running in Development

```bash
cargo tauri dev
```

This starts the app in development mode with hot-reloading.

### Debugging

Logs are written to:
- **macOS**: `~/Library/Logs/GoGrid Worker/`
- **Windows**: `%APPDATA%\GoGrid\logs\`
- **Linux**: `~/.local/share/gogrid/logs/`

View logs in real-time:
```bash
# macOS
tail -f ~/Library/Logs/GoGrid\ Worker/gogrid-tray.log

# Linux
tail -f ~/.local/share/gogrid/logs/gogrid-tray.log
```

### Testing QUIC Connection

```bash
# Test connection to coordinator
nc -zv bx.ee 8443
```

## Protocol

### QUIC Messages

**Client → Server**:
- `Register(WorkerInfo)`: Initial registration
- `Heartbeat { worker_id, status }`: Keep-alive every 30s
- `JobComplete { job_id, result }`: Job result submission
- `JobFailed { job_id, error }`: Job failure report

**Server → Client**:
- `Registered { worker_id }`: Registration confirmation
- `JobAssignment { job_id, job_data }`: New job assignment
- `Pause`: Pause worker
- `Resume`: Resume worker
- `Shutdown`: Graceful shutdown request

## Troubleshooting

### App doesn't start

1. Check if port 8443 is accessible:
   ```bash
   telnet bx.ee 8443
   ```

2. Verify worker binary exists:
   ```bash
   # macOS
   ls -la /Applications/GoGrid\ Worker.app/Contents/MacOS/corpgrid-scheduler
   ```

3. Check permissions:
   ```bash
   # Linux
   chmod +x /usr/bin/corpgrid-scheduler
   ```

### Worker process fails to start

- Check logs for error messages
- Ensure GPU drivers are installed
- Verify CUDA/Metal availability

### Connection issues

- Firewall may be blocking UDP port 8443
- Check network connectivity
- Verify coordinator is running: `curl -I https://bx.ee`

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test on all platforms
5. Submit a pull request
