# GoGrid Desktop Client Deployment Plan

**Date**: 2025-10-17
**Server**: bx.ee
**Status**: Design Phase

---

## Overview

Deploy GoGrid worker clients on Windows, macOS, and Linux with:
1. **Easy Installation**: One-click installers for each platform
2. **System Tray Application**: Minimal UI showing status
3. **Phone Home**: Secure communication with bx.ee control server
4. **Auto-Updates**: Seamless updates without user intervention
5. **Resource Management**: Automatic adaptive throttling

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Desktop Client (User Machine)             │
│                                                               │
│  ┌──────────────┐      ┌─────────────────┐                  │
│  │  Tray App    │◄────►│  Worker Process │                  │
│  │  (GUI)       │      │  (Inference)    │                  │
│  └──────────────┘      └─────────────────┘                  │
│         │                       │                            │
│         │                       │                            │
│         └───────────┬───────────┘                            │
│                     │                                        │
│                     │ Secure WebSocket/QUIC                  │
│                     │ (TLS + mTLS)                           │
└─────────────────────┼────────────────────────────────────────┘
                      │
                      │ Internet
                      │
┌─────────────────────▼────────────────────────────────────────┐
│                    bx.ee (Control Server)                     │
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐         │
│  │ Coordinator │  │  Job Queue  │  │  Metrics DB  │         │
│  └─────────────┘  └─────────────┘  └──────────────┘         │
│         │                                                     │
│         │ Port: TBD (e.g., 8443)                             │
│         │ Protocol: WSS or QUIC                              │
│         │                                                     │
│         └─────────────────────────────────────────────────   │
│                                                               │
│  nftables: Allow incoming on port 8443                       │
└───────────────────────────────────────────────────────────────┘
```

---

## Installation Strategy

### 1. Windows Installer (.msi / .exe)

**Technology**: WiX Toolset or Inno Setup

**Installation Flow**:
```
1. User downloads GoGridSetup.exe from website
2. Run installer (requires admin for service install)
3. Installer:
   - Extracts binary to C:\Program Files\GoGrid\
   - Installs as Windows Service (auto-start)
   - Creates system tray app in Startup folder
   - Generates client certificate
   - Opens firewall ports if needed
   - Starts service
4. Tray icon appears, shows "Connecting..."
5. First-time setup dialog (optional):
   - Accept terms
   - Choose resource mode (Conservative/Default/Aggressive)
   - Test GPU detection
6. Status changes to "Connected - Idle"
```

**Package Structure**:
```
GoGridSetup.exe
├── gogrid-worker.exe        # Main worker process
├── gogrid-tray.exe          # System tray UI
├── config.toml              # Default configuration
├── install-service.ps1      # Service installation script
└── README.txt
```

**Service Configuration**:
- **Name**: GoGridWorker
- **Display Name**: GoGrid Inference Worker
- **Start Type**: Automatic (Delayed Start)
- **Recovery**: Restart on failure
- **User**: Local System (for GPU access)

### 2. macOS Installer (.pkg / .dmg)

**Technology**: pkgbuild + productbuild or create-dmg

**Installation Flow**:
```
1. User downloads GoGrid.dmg from website
2. Open DMG, drag GoGrid.app to Applications
3. First launch:
   - macOS asks for accessibility permissions
   - Grant permissions for system tray access
   - Optional: Enable "Open at Login"
4. Background worker starts as LaunchAgent
5. Menu bar icon appears
6. First-time setup:
   - Accept terms
   - Choose resource mode
   - Test Metal GPU detection
7. Status: "Connected - Idle"
```

**Package Structure**:
```
GoGrid.dmg
└── GoGrid.app/
    └── Contents/
        ├── MacOS/
        │   ├── gogrid-worker       # Main worker
        │   └── gogrid-tray         # Menu bar app
        ├── Resources/
        │   ├── icon.icns
        │   └── config.toml
        └── LaunchAgents/
            └── com.gogrid.worker.plist
```

**LaunchAgent Configuration** (`~/Library/LaunchAgents/com.gogrid.worker.plist`):
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.gogrid.worker</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Applications/GoGrid.app/Contents/MacOS/gogrid-worker</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/gogrid-worker.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/gogrid-worker.err</string>
</dict>
</plist>
```

### 3. Linux Installer (.deb / .rpm / AppImage)

**Technology**:
- Debian/Ubuntu: `.deb` package
- Fedora/RHEL: `.rpm` package
- Universal: AppImage or Flatpak

**Installation Flow (systemd)**:
```
1. User downloads:
   - Ubuntu/Debian: sudo apt install ./gogrid-worker.deb
   - Fedora: sudo dnf install gogrid-worker.rpm
   - Universal: chmod +x GoGrid.AppImage && ./GoGrid.AppImage

2. Package installs:
   - Binary to /usr/bin/gogrid-worker
   - Tray app to /usr/bin/gogrid-tray
   - Systemd user service to ~/.config/systemd/user/
   - Desktop entry to ~/.local/share/applications/
   - Auto-start entry to ~/.config/autostart/

3. Enable and start service:
   systemctl --user enable gogrid-worker
   systemctl --user start gogrid-worker

4. Tray icon appears in system tray
5. First-time setup dialog
6. Status: "Connected - Idle"
```

**Systemd User Service** (`~/.config/systemd/user/gogrid-worker.service`):
```ini
[Unit]
Description=GoGrid Inference Worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/gogrid-worker
Restart=always
RestartSec=10
Environment=RUST_LOG=info

[Install]
WantedBy=default.target
```

---

## System Tray Application Design

### Minimal UI - Status Display

**Tray Icon States**:
- 🟢 **Green**: Connected and processing
- 🟡 **Yellow**: Connected but idle / paused
- 🔴 **Red**: Disconnected / error
- ⚪ **Gray**: Starting up
- 💤 **Sleep**: Paused due to high system load

**Right-Click Menu**:
```
┌─────────────────────────────────────┐
│ GoGrid Worker                       │
├─────────────────────────────────────┤
│ ✓ Status: Connected - Idle          │
│ ⚡ Mode: Conservative               │
│ 📊 Processed: 42 jobs today         │
│ 💰 Earned: $3.20 today              │
├─────────────────────────────────────┤
│ ⚙️  Settings...                     │
│ 📊 Statistics...                    │
│ ⏸️  Pause Worker                    │
│ 🔄 Restart Worker                   │
├─────────────────────────────────────┤
│ 🌐 Open Dashboard (web)             │
│ 📝 View Logs                        │
│ ❓ Help & Support                   │
├─────────────────────────────────────┤
│ ❌ Quit GoGrid                      │
└─────────────────────────────────────┘
```

**Settings Window** (simple dialog):
```
┌──────────────────────────────────────────────────────┐
│  GoGrid Worker Settings                              │
├──────────────────────────────────────────────────────┤
│                                                       │
│  Resource Mode:                                      │
│    ( ) Conservative - Desktop friendly (50% GPU)     │
│    (•) Default - Balanced (70% GPU)                  │
│    ( ) Aggressive - Maximum performance (95% GPU)   │
│                                                       │
│  [✓] Start automatically at login                   │
│  [✓] Enable adaptive throttling (pause when busy)   │
│  [✓] Show notifications                             │
│                                                       │
│  GPU Detected: NVIDIA GeForce RTX 3080 (10GB)       │
│  Backend: CUDA 12.1                                  │
│                                                       │
│  [  Cancel  ]                   [   Save   ]         │
└──────────────────────────────────────────────────────┘
```

**Statistics Window**:
```
┌──────────────────────────────────────────────────────┐
│  GoGrid Worker Statistics                            │
├──────────────────────────────────────────────────────┤
│                                                       │
│  Session Statistics (since start):                   │
│  • Uptime: 4h 32m                                    │
│  • Jobs completed: 127                               │
│  • Total tokens: 3.2M                                │
│  • Average speed: 42 tokens/sec                      │
│                                                       │
│  Resource Usage:                                     │
│  • GPU: 45% (target: 70%)                           │
│  • VRAM: 4.2 GB / 10 GB                             │
│  • CPU: 12%                                          │
│  • Memory: 2.1 GB                                    │
│                                                       │
│  Throttling:                                         │
│  • Requests throttled: 23 (18%)                      │
│  • System pauses: 2 (CPU was high)                  │
│  • Currently paused: No                              │
│                                                       │
│  Earnings:                                           │
│  • Today: $3.20                                      │
│  • This week: $18.45                                 │
│  • This month: $67.30                                │
│                                                       │
│                         [  Close  ]                   │
└──────────────────────────────────────────────────────┘
```

### Technology Stack for Tray App

**Cross-Platform Options**:

1. **Tauri** (Recommended)
   - Rust + Web UI (HTML/CSS/JS)
   - Native system tray support
   - Small binary size (~10MB)
   - Auto-update built-in
   - Cross-platform (Windows/macOS/Linux)

2. **egui** (Pure Rust)
   - Immediate mode GUI
   - No web dependencies
   - ~5MB binary
   - More manual work for system tray

3. **Slint** (Rust-native)
   - Declarative UI
   - Native look and feel
   - Good performance
   - System tray via separate crate

**Recommendation**: **Tauri** for:
- Easy development (web UI)
- Native system tray
- Auto-updates
- Cross-platform consistency
- Active community

---

## Phone Home Protocol

### Communication with bx.ee

**Protocol**: QUIC (Quinn) over TLS with mutual authentication (mTLS)

**Why QUIC**:
- ✅ Built-in multiplexing
- ✅ Fast connection establishment
- ✅ Better NAT traversal than TCP
- ✅ Resilient to packet loss
- ✅ Already in GoGrid codebase

**Connection Flow**:
```
1. Client starts up
2. Loads client certificate (mTLS)
3. Connects to bx.ee:8443 (QUIC)
4. Sends registration message:
   {
     "type": "register",
     "client_id": "uuid-here",
     "version": "0.1.0",
     "gpu_info": {
       "model": "NVIDIA RTX 3080",
       "vram": 10737418240,
       "backend": "CUDA"
     },
     "capabilities": ["llm", "quantized", "batch"],
     "resource_mode": "conservative"
   }

5. Server responds:
   {
     "type": "welcome",
     "status": "registered",
     "node_id": "node_abc123",
     "heartbeat_interval": 30,
     "config": { ... }
   }

6. Client enters ready state, awaits jobs
```

**Message Types**:

1. **Heartbeat** (every 30 seconds):
   ```json
   {
     "type": "heartbeat",
     "node_id": "node_abc123",
     "status": "idle",
     "stats": {
       "uptime_secs": 16234,
       "jobs_completed": 127,
       "gpu_utilization": 0.45,
       "vram_used": 4200000000,
       "currently_paused": false
     }
   }
   ```

2. **Job Assignment** (server → client):
   ```json
   {
     "type": "job_assignment",
     "job_id": "job_xyz789",
     "model": "mistral-7b-instruct",
     "input": {
       "prompt": "Write a poem about...",
       "max_tokens": 256,
       "temperature": 0.7
     },
     "priority": "normal",
     "timeout_secs": 300
   }
   ```

3. **Job Result** (client → server):
   ```json
   {
     "type": "job_result",
     "job_id": "job_xyz789",
     "status": "completed",
     "output": {
       "text": "Generated text here...",
       "tokens": 187,
       "completion_time_ms": 4523
     },
     "stats": {
       "tokens_per_sec": 41.3,
       "vram_peak": 5200000000
     }
   }
   ```

4. **Configuration Update** (server → client):
   ```json
   {
     "type": "config_update",
     "resource_mode": "conservative",
     "enable_adaptive_throttling": true,
     "model_update": {
       "action": "download",
       "model_id": "mistral-7b-v2",
       "url": "https://bx.ee/models/..."
     }
   }
   ```

5. **Pause/Resume** (server → client):
   ```json
   {
     "type": "command",
     "command": "pause",
     "reason": "maintenance",
     "duration_secs": 300
   }
   ```

**Security**:
- ✅ **TLS 1.3** for encryption
- ✅ **mTLS** (mutual authentication with client certificates)
- ✅ **Certificate pinning** (pin bx.ee server cert)
- ✅ **Token-based auth** (JWT for API calls)
- ✅ **No sensitive data** stored on client

### Client Certificate Generation

**First Launch**:
```rust
// Generate unique client cert on first run
let client_cert = generate_client_certificate()?;
save_to_config_dir("client.crt", &client_cert)?;

// Send CSR to server for signing
let csr = create_csr(&client_cert)?;
let response = post_to_server("/api/v1/register", csr).await?;
let signed_cert = response.signed_certificate;
save_to_config_dir("client_signed.crt", &signed_cert)?;
```

**Authentication Flow**:
1. Client generates self-signed cert
2. Sends CSR to server `/api/v1/register`
3. Server signs cert (valid for 1 year)
4. Client uses signed cert for mTLS
5. Server validates cert on each connection

---

## Server Infrastructure on bx.ee

### Components to Deploy

1. **GoGrid Coordinator** (Rust service)
   - Accepts client connections (QUIC on port 8443)
   - Job queue management
   - Load balancing across workers
   - Health monitoring
   - Metrics collection

2. **PostgreSQL Database**
   - Worker registry
   - Job queue
   - Metrics history
   - Billing records

3. **Redis Cache**
   - Active connections
   - Job status
   - Real-time stats

4. **Web Dashboard** (optional)
   - Admin interface
   - Worker monitoring
   - Job history
   - Billing/earnings

### Port Configuration

**Recommended Port**: `8443` (HTTPS alternative)

**nftables Rule**:
```bash
# Allow incoming QUIC on port 8443
sudo nft add rule inet filter input udp dport 8443 accept

# Or if using TCP fallback
sudo nft add rule inet filter input tcp dport 8443 accept
```

**Firewall Configuration**:
```bash
# Complete rule set
sudo nft add rule inet filter input \
  udp dport 8443 \
  ct state new,established \
  accept comment "GoGrid QUIC"

# Optional: Rate limiting
sudo nft add rule inet filter input \
  udp dport 8443 \
  limit rate 100/second burst 200 packets \
  accept
```

### Directory Structure on bx.ee

```
/opt/gogrid/
├── bin/
│   ├── gogrid-coordinator      # Main coordinator service
│   └── gogrid-admin            # Admin CLI tool
├── config/
│   ├── coordinator.toml        # Service configuration
│   ├── server.crt              # TLS certificate
│   └── server.key              # TLS private key
├── data/
│   └── client_certs/           # Signed client certificates
├── logs/
│   ├── coordinator.log
│   └── access.log
└── models/
    └── cache/                  # Model files for distribution
```

### systemd Service

`/etc/systemd/system/gogrid-coordinator.service`:
```ini
[Unit]
Description=GoGrid Coordinator Service
After=network-online.target postgresql.service redis.service
Wants=network-online.target

[Service]
Type=simple
User=gogrid
Group=gogrid
WorkingDirectory=/opt/gogrid
ExecStart=/opt/gogrid/bin/gogrid-coordinator --config /opt/gogrid/config/coordinator.toml
Restart=always
RestartSec=10
LimitNOFILE=65536

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/gogrid/data /opt/gogrid/logs

Environment=RUST_LOG=info

[Install]
WantedBy=multi-user.target
```

---

## Auto-Update Strategy

### Update Mechanism

**Approach**: Delta updates with signature verification

**Flow**:
```
1. Coordinator detects new version available
2. Sends update notification to connected clients:
   {
     "type": "update_available",
     "version": "0.1.1",
     "download_url": "https://bx.ee/updates/gogrid-0.1.1.delta",
     "signature": "...",
     "size_bytes": 2048576,
     "changelog": "Bug fixes and performance improvements"
   }

3. Client downloads delta patch
4. Verifies signature (Ed25519)
5. Applies patch to current binary
6. Restarts service gracefully:
   - Finish current job
   - Save state
   - Restart process
   - Reconnect to server

7. Send update confirmation:
   {
     "type": "update_complete",
     "version": "0.1.1",
     "previous_version": "0.1.0"
   }
```

**Security**:
- ✅ Ed25519 signatures on all updates
- ✅ HTTPS-only downloads
- ✅ Checksum verification (SHA-256)
- ✅ Rollback support (keep previous version)

---

## Installation Scripts

### Windows (PowerShell)

`install.ps1`:
```powershell
# GoGrid Worker Installation Script for Windows
$ErrorActionPreference = "Stop"

Write-Host "Installing GoGrid Worker..." -ForegroundColor Green

# Check admin privileges
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Error "This script must be run as Administrator"
    exit 1
}

# Create installation directory
$installPath = "$env:ProgramFiles\GoGrid"
New-Item -ItemType Directory -Force -Path $installPath | Out-Null

# Copy binaries
Copy-Item "gogrid-worker.exe" -Destination "$installPath\"
Copy-Item "gogrid-tray.exe" -Destination "$installPath\"
Copy-Item "config.toml" -Destination "$installPath\"

# Install as Windows Service
Write-Host "Installing Windows Service..."
New-Service -Name "GoGridWorker" `
    -DisplayName "GoGrid Inference Worker" `
    -Description "Distributed GPU inference worker for GoGrid" `
    -BinaryPathName "$installPath\gogrid-worker.exe" `
    -StartupType Automatic `
    -DependsOn "Tcpip"

# Set recovery options (restart on failure)
sc.exe failure GoGridWorker reset= 86400 actions= restart/60000/restart/60000/restart/60000

# Add firewall rule (if needed)
New-NetFirewallRule -DisplayName "GoGrid Worker" `
    -Direction Outbound `
    -Action Allow `
    -Protocol UDP `
    -RemotePort 8443 | Out-Null

# Create startup shortcut for tray app
$startupPath = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup"
$shortcut = (New-Object -COM WScript.Shell).CreateShortcut("$startupPath\GoGrid.lnk")
$shortcut.TargetPath = "$installPath\gogrid-tray.exe"
$shortcut.WorkingDirectory = $installPath
$shortcut.Save()

# Start service
Start-Service GoGridWorker

# Launch tray app
Start-Process "$installPath\gogrid-tray.exe"

Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "GoGrid Worker is now running. Check the system tray for status."
```

### macOS (Bash)

`install.sh`:
```bash
#!/bin/bash
set -e

echo "Installing GoGrid Worker for macOS..."

# Check if running with sudo
if [ "$EUID" -eq 0 ]; then
    echo "Error: Do not run this script with sudo"
    exit 1
fi

# Copy app to Applications
echo "Installing to /Applications..."
cp -R "GoGrid.app" /Applications/

# Install LaunchAgent
echo "Installing LaunchAgent..."
mkdir -p ~/Library/LaunchAgents
cp com.gogrid.worker.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.gogrid.worker.plist

# Add to Login Items (optional)
osascript -e 'tell application "System Events" to make login item at end with properties {path:"/Applications/GoGrid.app", hidden:false}'

echo "Installation complete!"
echo "GoGrid Worker will start automatically."
echo "Look for the menu bar icon."

# Launch tray app
open /Applications/GoGrid.app
```

### Linux (Bash)

`install.sh`:
```bash
#!/bin/bash
set -e

echo "Installing GoGrid Worker for Linux..."

# Detect package manager
if command -v apt &> /dev/null; then
    PKG_MANAGER="apt"
    INSTALL_CMD="sudo apt install -y"
elif command -v dnf &> /dev/null; then
    PKG_MANAGER="dnf"
    INSTALL_CMD="sudo dnf install -y"
else
    echo "Unsupported package manager. Please install manually."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
$INSTALL_CMD libgtk-3-0 libnotify4

# Install binary
echo "Installing binaries..."
sudo cp gogrid-worker /usr/bin/
sudo cp gogrid-tray /usr/bin/
sudo chmod +x /usr/bin/gogrid-worker /usr/bin/gogrid-tray

# Create config directory
mkdir -p ~/.config/gogrid
cp config.toml ~/.config/gogrid/

# Install systemd user service
mkdir -p ~/.config/systemd/user
cp gogrid-worker.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable gogrid-worker
systemctl --user start gogrid-worker

# Install desktop entry
mkdir -p ~/.local/share/applications
cat > ~/.local/share/applications/gogrid.desktop <<EOF
[Desktop Entry]
Type=Application
Name=GoGrid Worker
Comment=Distributed GPU Inference Worker
Exec=/usr/bin/gogrid-tray
Icon=gogrid
Terminal=false
Categories=System;
EOF

# Add to autostart
mkdir -p ~/.config/autostart
cp ~/.local/share/applications/gogrid.desktop ~/.config/autostart/

echo "Installation complete!"
echo "GoGrid Worker is now running."
echo "Check your system tray for the GoGrid icon."

# Launch tray app
/usr/bin/gogrid-tray &
```

---

## Configuration File

`config.toml`:
```toml
[server]
# Control server address
address = "bx.ee"
port = 8443
protocol = "quic"  # or "tcp" for fallback

# TLS configuration
tls_enabled = true
verify_server_cert = true
server_cert_fingerprint = "SHA256:..." # Pinned cert

[client]
# Unique client ID (generated on first run)
client_id = "uuid-will-be-generated"
node_name = "My Desktop"

# Certificate paths
client_cert = "~/.config/gogrid/client.crt"
client_key = "~/.config/gogrid/client.key"

[resources]
# Resource mode: "conservative", "default", or "aggressive"
mode = "conservative"

# Enable adaptive throttling (pause when system is busy)
enable_adaptive_throttling = true

# Custom thresholds (optional, overrides mode)
# cpu_threshold = 0.5
# gpu_threshold = 0.6
# memory_threshold = 0.8

[worker]
# Worker configuration
max_concurrent_jobs = 1
heartbeat_interval_secs = 30
reconnect_delay_secs = 10

# Auto-update
enable_auto_update = true
update_channel = "stable"  # or "beta"

[logging]
level = "info"  # debug, info, warn, error
log_file = "~/.config/gogrid/worker.log"
max_log_size_mb = 100
```

---

## Deployment Checklist

### Phase 1: Server Setup on bx.ee

- [ ] SSH to bx.ee
- [ ] Install PostgreSQL and Redis
- [ ] Create `gogrid` user
- [ ] Clone GoGrid repository
- [ ] Build coordinator service
- [ ] Create directory structure (`/opt/gogrid/`)
- [ ] Generate TLS certificates (Let's Encrypt or self-signed)
- [ ] Configure nftables (open port 8443)
- [ ] Install systemd service
- [ ] Start coordinator service
- [ ] Test connection from local machine

### Phase 2: Client Development

- [ ] Create Tauri tray application
- [ ] Implement phone-home protocol (QUIC)
- [ ] Add certificate generation
- [ ] Build Windows installer (.msi)
- [ ] Build macOS package (.dmg)
- [ ] Build Linux packages (.deb, .rpm, AppImage)
- [ ] Test on each platform
- [ ] Implement auto-update mechanism
- [ ] Add telemetry and error reporting

### Phase 3: Distribution

- [ ] Create download page
- [ ] Set up CDN for installers
- [ ] Create installation guides
- [ ] Beta testing program
- [ ] Public release
- [ ] Monitor metrics and errors

---

## Next Steps

1. **Set up bx.ee server infrastructure** (I can do this via SSH)
2. **Open port 8443 on nftables**
3. **Build coordinator service**
4. **Prototype Tauri tray app**
5. **Test phone-home connection**
6. **Build installers for each platform**

---

## Questions for You

1. **Port Number**: Is 8443 (QUIC/UDP) acceptable, or do you prefer a different port?
2. **Domain**: Use `bx.ee` directly or a subdomain like `coord.bx.ee` or `workers.bx.ee`?
3. **TLS Certificate**: Let's Encrypt (auto-renew) or self-signed for now?
4. **Billing Model**: How should we calculate earnings? (per-token, per-job, per-hour?)
5. **Beta Testing**: Ready for beta testers, or internal testing first?

---

**Status**: Design Complete - Ready to Implement
**Next Action**: Set up bx.ee server and open firewall port
