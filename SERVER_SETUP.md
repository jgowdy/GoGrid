# GoGrid Server Setup Guide

This guide explains how to set up your own GoGrid coordinator server.

## Overview

The coordinator server consists of:
1. **gogrid-coordinator** - Central coordination service (QUIC + HTTP)
2. **Update server** - Serves auto-update packages and download page
3. **PostgreSQL** - Job queue and worker registry (optional)
4. **Redis** - Real-time worker state (optional)

## Quick Setup

### Automated Setup (Recommended)

```bash
# Download and run setup script
curl -fsSL https://raw.githubusercontent.com/jgowdy-godaddy/GoGrid/main/setup_server.sh | bash

# Or clone and run locally
git clone https://github.com/jgowdy-godaddy/GoGrid.git
cd GoGrid
./setup_server.sh
```

The script will:
- Install required dependencies
- Build the coordinator binary
- Create systemd service
- Configure firewall rules
- Set up update directory
- Start the coordinator

### Manual Setup

#### 1. Install Dependencies

**OpenBSD**:
```bash
doas pkg_add rust postgresql-server redis
```

**Ubuntu/Debian**:
```bash
sudo apt-get update
sudo apt-get install -y build-essential curl git postgresql redis-server
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**CentOS/RHEL**:
```bash
sudo yum install -y gcc git postgresql-server redis
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

#### 2. Clone Repository

```bash
git clone https://github.com/jgowdy-godaddy/GoGrid.git
cd GoGrid
```

#### 3. Build Coordinator

```bash
cargo build --release --bin gogrid-coordinator
```

#### 4. Create Directory Structure

```bash
# For OpenBSD
doas mkdir -p /opt/gogrid/{bin,updates,config,logs}
doas chown $USER:$USER /opt/gogrid/{bin,updates,config,logs}

# For Linux
sudo mkdir -p /opt/gogrid/{bin,updates,config,logs}
sudo chown $USER:$USER /opt/gogrid/{bin,updates,config,logs}
```

#### 5. Install Binary

```bash
cp target/release/gogrid-coordinator /opt/gogrid/bin/
chmod +x /opt/gogrid/bin/gogrid-coordinator
```

#### 6. Create Configuration File

```bash
cat > /opt/gogrid/config/coordinator.toml << 'TOML'
[coordinator]
bind_addr = "0.0.0.0"
port = 8443

[database]
# Optional: PostgreSQL for job queue
# url = "postgresql://gogrid:password@localhost/gogrid"

[redis]
# Optional: Redis for worker state
# url = "redis://localhost:6379"

[updates]
# Directory for update packages
directory = "/opt/gogrid/updates"

[security]
# Optional: TLS certificate paths
# cert_path = "/etc/letsencrypt/live/your-domain/fullchain.pem"
# key_path = "/etc/letsencrypt/live/your-domain/privkey.pem"
TOML
```

#### 7. Configure Firewall

**OpenBSD (pf)**:
```bash
# Edit /etc/pf.conf
doas vi /etc/pf.conf

# Add this line
pass in proto tcp from any to any port 8443

# Reload
doas pfctl -f /etc/pf.conf
```

**Linux (ufw)**:
```bash
sudo ufw allow 8443/tcp
sudo ufw reload
```

**Linux (firewalld)**:
```bash
sudo firewall-cmd --permanent --add-port=8443/tcp
sudo firewall-cmd --reload
```

#### 8. Create Systemd Service (Linux)

```bash
sudo tee /etc/systemd/system/gogrid-coordinator.service > /dev/null << 'SERVICE'
[Unit]
Description=GoGrid Coordinator
After=network.target

[Service]
Type=simple
User=gogrid
WorkingDirectory=/opt/gogrid
ExecStart=/opt/gogrid/bin/gogrid-coordinator --config /opt/gogrid/config/coordinator.toml
Restart=on-failure
RestartSec=10

# Logging
StandardOutput=journal
StandardError=journal

# Security
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
SERVICE

# Create user
sudo useradd -r -s /bin/false gogrid
sudo chown -R gogrid:gogrid /opt/gogrid

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable gogrid-coordinator
sudo systemctl start gogrid-coordinator
```

#### 9. Create rc.d Service (OpenBSD)

```bash
doas tee /etc/rc.d/gogrid_coordinator > /dev/null << 'RCDSCRIPT'
#!/bin/ksh
daemon="/opt/gogrid/bin/gogrid-coordinator"
daemon_flags="--config /opt/gogrid/config/coordinator.toml"
daemon_user="gogrid"

. /etc/rc.d/rc.subr

rc_reload=NO
rc_bg=YES

rc_cmd $1
RCDSCRIPT

doas chmod +x /etc/rc.d/gogrid_coordinator

# Enable and start
doas rcctl enable gogrid_coordinator
doas rcctl start gogrid_coordinator
```

## Configuration Options

### Basic Configuration

```toml
[coordinator]
bind_addr = "0.0.0.0"  # Listen on all interfaces
port = 8443             # Default port

[updates]
directory = "/opt/gogrid/updates"  # Update packages directory
```

### With Database (Optional)

```toml
[database]
url = "postgresql://gogrid:password@localhost/gogrid"
max_connections = 10
```

Setup PostgreSQL:
```bash
# Create database
sudo -u postgres createuser gogrid
sudo -u postgres createdb -O gogrid gogrid

# Set password
sudo -u postgres psql -c "ALTER USER gogrid PASSWORD 'your_password';"
```

### With Redis (Optional)

```toml
[redis]
url = "redis://localhost:6379"
pool_size = 10
```

### With TLS (Production)

```toml
[security]
cert_path = "/etc/letsencrypt/live/your-domain.com/fullchain.pem"
key_path = "/etc/letsencrypt/live/your-domain.com/privkey.pem"
```

Get Let's Encrypt certificate:
```bash
# Ubuntu/Debian
sudo apt-get install certbot
sudo certbot certonly --standalone -d your-domain.com

# OpenBSD
doas pkg_add certbot
doas certbot certonly --standalone -d your-domain.com
```

## Uploading Update Packages

### Manual Upload

```bash
# Build packages
./build_all_platforms.sh 0.1.0

# Upload to server
scp target/release/bundle/macos/GoGrid_Worker_*.app.tar.gz your-server:/opt/gogrid/updates/
scp target/release/bundle/dmg/*.dmg your-server:/opt/gogrid/updates/
```

### Using deploy_updates.sh

Edit `deploy_updates.sh` to point to your server:
```bash
#!/bin/bash
SERVER="your-server.com"
# ... rest of script
```

Then run:
```bash
./deploy_updates.sh
```

## Testing

### Check Service Status

**Linux**:
```bash
sudo systemctl status gogrid-coordinator
sudo journalctl -u gogrid-coordinator -f
```

**OpenBSD**:
```bash
doas rcctl status gogrid_coordinator
doas tail -f /var/log/daemon
```

### Test HTTP Endpoint

```bash
# Downloads page
curl -I http://your-server:8443/downloads

# Update manifest
curl http://your-server:8443/updates/darwin-aarch64/0.0.0
```

### Test QUIC Connection

```bash
# From a worker machine
export GOGRID_COORDINATOR_HOST=your-server.com
export GOGRID_COORDINATOR_PORT=8443
./target/release/corpgrid-scheduler
```

## Monitoring

### Logs

**Linux**:
```bash
# Follow logs
sudo journalctl -u gogrid-coordinator -f

# View recent logs
sudo journalctl -u gogrid-coordinator -n 100

# Search for errors
sudo journalctl -u gogrid-coordinator | grep -i error
```

**OpenBSD**:
```bash
# Follow daemon logs
doas tail -f /var/log/daemon

# View coordinator logs
doas tail -f /opt/gogrid/logs/coordinator.log
```

### Metrics

The coordinator exposes metrics at `/metrics` (if enabled):
```bash
curl http://localhost:8443/metrics
```

Use with Prometheus:
```yaml
scrape_configs:
  - job_name: 'gogrid'
    static_configs:
      - targets: ['localhost:8443']
```

## Security

### Firewall

Only expose necessary ports:
- **8443** - Coordinator (QUIC + HTTP)

### TLS Certificates

For production, always use TLS:
1. Get Let's Encrypt certificate
2. Configure cert/key paths in config
3. Coordinator will automatically use HTTPS

### User Isolation

Run coordinator as dedicated user:
```bash
# Linux
sudo useradd -r -s /bin/false gogrid

# OpenBSD  
doas useradd -s /sbin/nologin gogrid
```

### Update Package Verification

For production, sign update packages:
```bash
# Generate signing key
tauri signer generate -w ~/.tauri/gogrid.key

# Sign packages
tauri signer sign /opt/gogrid/updates/GoGrid_Worker_0.1.0_aarch64.app.tar.gz
```

Add public key to `tauri.conf.json`:
```json
{
  "plugins": {
    "updater": {
      "pubkey": "dW50cnVzdGVkIGNvbW1lbnQ6..."
    }
  }
}
```

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8443
sudo lsof -i :8443  # Linux
doas fstat -p :8443 # OpenBSD

# Kill the process or change port in config
```

### Permission Denied

```bash
# Fix ownership
sudo chown -R gogrid:gogrid /opt/gogrid

# Fix permissions
chmod 755 /opt/gogrid/bin/gogrid-coordinator
chmod 644 /opt/gogrid/config/coordinator.toml
```

### Cannot Connect

1. Check firewall rules
2. Verify service is running
3. Test locally: `curl http://localhost:8443/downloads`
4. Check logs for errors

### Database Connection Failed

```bash
# Test PostgreSQL connection
psql -U gogrid -d gogrid -h localhost

# Check PostgreSQL is running
sudo systemctl status postgresql  # Linux
doas rcctl status postgresql      # OpenBSD
```

## Upgrading

### Coordinator Binary

```bash
# Pull latest code
cd GoGrid
git pull

# Rebuild
cargo build --release --bin gogrid-coordinator

# Stop service
sudo systemctl stop gogrid-coordinator  # Linux
doas rcctl stop gogrid_coordinator      # OpenBSD

# Replace binary
cp target/release/gogrid-coordinator /opt/gogrid/bin/

# Start service
sudo systemctl start gogrid-coordinator  # Linux
doas rcctl start gogrid_coordinator      # OpenBSD
```

### Update Packages

```bash
# Build new version
./build_all_platforms.sh 0.1.1

# Upload to server
./deploy_updates.sh
```

## Backup

### Important Files

```bash
/opt/gogrid/config/coordinator.toml  # Configuration
/opt/gogrid/updates/                 # Update packages (can be rebuilt)
/var/lib/postgresql/                 # Database (if using PostgreSQL)
```

### Backup Script

```bash
#!/bin/bash
BACKUP_DIR="/backup/gogrid-$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Config
cp -r /opt/gogrid/config $BACKUP_DIR/

# Database
pg_dump -U gogrid gogrid > $BACKUP_DIR/database.sql

# Logs
cp -r /opt/gogrid/logs $BACKUP_DIR/
```

## Support

- **Documentation**: https://github.com/jgowdy-godaddy/GoGrid
- **Issues**: https://github.com/jgowdy-godaddy/GoGrid/issues
- **Discussions**: https://github.com/jgowdy-godaddy/GoGrid/discussions
