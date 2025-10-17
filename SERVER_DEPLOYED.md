# GoGrid Coordinator - Server Deployment Guide

**Note**: This is an example deployment guide. Adapt commands for your specific server.

---

## Deployment Summary

Guide for deploying the GoGrid coordinator service on your server.

### ✅ What's Deployed

1. **nftables Firewall** - Port 8443 (UDP) opened for QUIC
2. **Directory Structure** - `/opt/gogrid/` with bin, config, data, logs, models
3. **Coordinator Binary** - 1.3MB Rust binary compiled and deployed
4. **Configuration** - `/opt/gogrid/config/coordinator.toml`
5. **systemd Service** - Auto-start, auto-restart on failure
6. **Port 8443** - ✅ **OPEN AND LISTENING**

---

## Server Details

### Connection Information
- **Host**: Your server hostname
- **Port**: `8443` (UDP for QUIC)
- **Protocol**: QUIC with TLS

### File Locations
```
/opt/gogrid/
├── bin/
│   └── gogrid-coordinator (1.3 MB)
├── config/
│   └── coordinator.toml
├── data/
├── logs/
└── models/cache/
```

### Configuration (`/opt/gogrid/config/coordinator.toml`)
```toml
[server]
bind_addr = "0.0.0.0"
port = 8443
max_connections = 10000
heartbeat_interval_secs = 30
heartbeat_timeout_secs = 90

[database]
url = "postgres://gogrid:changeme@localhost/gogrid"
max_connections = 10

[redis]
url = "redis://localhost:6379"
```

### systemd Service
- **Name**: `gogrid-coordinator.service`
- **Status**: ✅ Enabled and running
- **Auto-start**: Yes
- **Restart**: Always (10 second delay)
- **User**: jgowdy
- **Logs**: `doas journalctl -u gogrid-coordinator -f`

---

## Verification

### Port Status
```bash
$ nc -zv your-server.com 8443
Connection to your-server.com port 8443 [tcp/pcsync-https] succeeded!
```

### Service Status
```bash
$ doas systemctl status gogrid-coordinator
● gogrid-coordinator.service - GoGrid Coordinator Service
     Loaded: loaded (/etc/systemd/system/gogrid-coordinator.service; enabled)
     Active: active (running)
```
✅ **Service is running**

### Firewall Rules
```bash
$ doas nft list chain inet bx_firewall INPUT | grep 8443
udp dport 8443 accept comment "GoGrid QUIC"
```
✅ **Firewall rule active**

---

## Management Commands

### Start/Stop/Restart
```bash
# Start
ssh your-server.com "sudo systemctl start gogrid-coordinator"

# Stop
ssh your-server.com "sudo systemctl stop gogrid-coordinator"

# Restart
ssh your-server.com "sudo systemctl restart gogrid-coordinator"

# Status
ssh your-server.com "sudo systemctl status gogrid-coordinator"
```

### View Logs
```bash
# Real-time logs
ssh your-server.com "sudo journalctl -u gogrid-coordinator -f"

# Last 50 lines
ssh your-server.com "sudo journalctl -u gogrid-coordinator -n 50"

# Today's logs
ssh your-server.com "sudo journalctl -u gogrid-coordinator --since today"
```

### Update Binary
```bash
# Build new version
cargo build --package corpgrid-coordinator --release

# Deploy
scp target/release/gogrid-coordinator your-server.com:/opt/gogrid/bin/

# Restart service
ssh your-server.com "sudo systemctl restart gogrid-coordinator"
```

---

## Next Steps

### 1. Client Development
- [ ] Build Tauri system tray application
- [ ] Implement QUIC client connection to coordinator server
- [ ] Add phone-home protocol (heartbeats, registration)
- [ ] Test end-to-end connection

### 2. Database Setup (Optional - Not Required Yet)
- [ ] Install PostgreSQL on server
- [ ] Create `gogrid` database
- [ ] Run migrations
- [ ] Install Redis for caching

### 3. TLS Certificates (Optional - Using Self-Signed for Now)
- [ ] Set up Let's Encrypt
- [ ] Configure auto-renewal
- [ ] Update coordinator to use real certs

### 4. Installers
- [ ] Build Windows installer (.msi)
- [ ] Build macOS installer (.dmg)
- [ ] Build Linux packages (.deb, .rpm, AppImage)

---

## Architecture Flow

```
Desktop Client (Windows/macOS/Linux)
    │
    │ QUIC/TLS (mTLS)
    ▼
your-server.com:8443
    │
    ▼
gogrid-coordinator (systemd)
    │
    ├──> Worker Registry (in-memory for now)
    ├──> Job Queue (future: PostgreSQL)
    └──> Metrics (future: Redis)
```

---

## Testing

### Test Connection from Local Machine

```bash
# Using netcat
nc -zv your-server.com 8443

# Using nmap (if installed)
nmap -p 8443 your-server.com

# Using telnet
telnet your-server.com 8443
```

All should show **connection successful**!

---

## Security

### Current Setup
- ✅ Port 8443 open (UDP for QUIC)
- ✅ Service runs as non-root user (jgowdy)
- ✅ TLS encryption (self-signed cert for now)
- ⚠️ No mTLS yet (client certificates not enforced)
- ⚠️ No rate limiting yet (future: nftables rules)

### Hardening Recommendations (Future)
1. Enable mTLS (mutual TLS with client certificates)
2. Add rate limiting (100 conn/sec per IP)
3. Use Let's Encrypt certificates
4. Add fail2ban for repeated connection failures
5. Set up monitoring and alerting

---

## Monitoring

### Service Health
```bash
# Check if running
ssh your-server.com "pgrep -a gogrid"

# Check port
ssh your-server.com "ss -ulnp | grep 8443"

# Check connections
ssh your-server.com "sudo netstat -an | grep 8443"
```

### Resource Usage
```bash
# Memory and CPU
ssh your-server.com "ps aux | grep gogrid"

# Disk usage
ssh your-server.com "du -sh /opt/gogrid/*"
```

---

## Troubleshooting

### Service Won't Start
```bash
# Check logs
ssh your-server.com "sudo journalctl -u gogrid-coordinator -n 100"

# Check binary
ssh your-server.com "ls -lh /opt/gogrid/bin/gogrid-coordinator"
ssh your-server.com "/opt/gogrid/bin/gogrid-coordinator --help"

# Check config
ssh your-server.com "cat /opt/gogrid/config/coordinator.toml"
```

### Port Not Accessible
```bash
# Check firewall
ssh your-server.com "sudo nft list ruleset | grep 8443"

# Check if service is listening
ssh your-server.com "sudo ss -ulnp | grep 8443"

# Test from server itself
ssh your-server.com "nc -zv localhost 8443"
```

### Update Firewall Rule
```bash
# Remove old rule (if needed)
ssh your-server.com "sudo nft delete rule inet filter INPUT handle <handle>"

# Add new rule
ssh your-server.com "sudo nft add rule inet filter INPUT udp dport 8443 accept comment \"GoGrid QUIC\""
```

---

## Performance

### Current Capacity
- **Max Connections**: 10,000 concurrent workers
- **Heartbeat Interval**: 30 seconds
- **Heartbeat Timeout**: 90 seconds
- **File Descriptor Limit**: 65,536

### Scaling Recommendations
- 100 workers: No changes needed
- 1,000 workers: Monitor CPU/memory
- 10,000 workers: Consider Redis for worker registry
- 100,000+ workers: Use PostgreSQL + multiple coordinator instances

---

## Summary

**GoGrid Coordinator Setup Guide**

This guide helps you deploy the coordinator service on your server.

**Server Requirements**:
- Port 8443 (UDP) open for QUIC
- systemd-based Linux distribution
- Rust toolchain for building

Next step: Build the desktop client (Tauri tray app) that connects to your server!
