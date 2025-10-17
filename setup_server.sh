#!/bin/bash
set -e

# GoGrid Coordinator Server Setup Script
# Automatically installs and configures the GoGrid coordinator server

echo "========================================="
echo "GoGrid Coordinator Server Setup"
echo "========================================="
echo ""

# Detect OS
if [[ "$OSTYPE" == "openbsd"* ]]; then
    OS="openbsd"
    PKG_MGR="pkg_add"
    SUDO="doas"
    SERVICE_MGR="rcctl"
elif [[ -f /etc/debian_version ]]; then
    OS="debian"
    PKG_MGR="apt-get"
    SUDO="sudo"
    SERVICE_MGR="systemctl"
elif [[ -f /etc/redhat-release ]]; then
    OS="redhat"
    PKG_MGR="yum"
    SUDO="sudo"
    SERVICE_MGR="systemctl"
else
    echo "Error: Unsupported operating system"
    exit 1
fi

echo "Detected OS: $OS"
echo ""

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "Error: This script should not be run as root"
   echo "It will use $SUDO when needed"
   exit 1
fi

# Install dependencies
echo "Step 1: Installing dependencies..."
case $OS in
    openbsd)
        $SUDO pkg_add rust postgresql-server redis
        ;;
    debian)
        $SUDO apt-get update
        $SUDO apt-get install -y build-essential curl git postgresql redis-server

        # Install Rust if not present
        if ! command -v cargo &> /dev/null; then
            echo "Installing Rust..."
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            source "$HOME/.cargo/env"
        fi
        ;;
    redhat)
        $SUDO yum install -y gcc git postgresql-server redis

        # Install Rust if not present
        if ! command -v cargo &> /dev/null; then
            echo "Installing Rust..."
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            source "$HOME/.cargo/env"
        fi
        ;;
esac

echo "✓ Dependencies installed"
echo ""

# Clone or update repository
echo "Step 2: Setting up repository..."
if [[ ! -d "GoGrid" ]]; then
    echo "Cloning GoGrid repository..."
    git clone https://github.com/jgowdy-godaddy/GoGrid.git
    cd GoGrid
else
    echo "Repository already exists, updating..."
    cd GoGrid
    git pull
fi

echo "✓ Repository ready"
echo ""

# Build coordinator
echo "Step 3: Building coordinator binary..."
echo "This may take several minutes..."
cargo build --release --bin gogrid-coordinator

if [[ ! -f "target/release/gogrid-coordinator" ]]; then
    echo "Error: Build failed - binary not found"
    exit 1
fi

echo "✓ Coordinator built successfully"
echo ""

# Create directory structure
echo "Step 4: Creating directory structure..."
$SUDO mkdir -p /opt/gogrid/{bin,updates,config,logs}
$SUDO chown -R $USER:$USER /opt/gogrid

echo "✓ Directories created"
echo ""

# Install binary
echo "Step 5: Installing binary..."
cp target/release/gogrid-coordinator /opt/gogrid/bin/
chmod +x /opt/gogrid/bin/gogrid-coordinator

echo "✓ Binary installed"
echo ""

# Create configuration file
echo "Step 6: Creating configuration file..."
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

echo "✓ Configuration created at /opt/gogrid/config/coordinator.toml"
echo ""

# Configure firewall
echo "Step 7: Configuring firewall..."
case $OS in
    openbsd)
        if ! $SUDO pfctl -sr | grep -q "8443"; then
            echo "Adding firewall rule for port 8443..."
            echo "pass in proto tcp from any to any port 8443" | $SUDO tee -a /etc/pf.conf
            $SUDO pfctl -f /etc/pf.conf
            echo "✓ Firewall configured"
        else
            echo "✓ Firewall already configured"
        fi
        ;;
    debian)
        if command -v ufw &> /dev/null; then
            $SUDO ufw allow 8443/tcp
            echo "✓ UFW firewall configured"
        elif command -v firewall-cmd &> /dev/null; then
            $SUDO firewall-cmd --permanent --add-port=8443/tcp
            $SUDO firewall-cmd --reload
            echo "✓ Firewalld configured"
        else
            echo "⚠ No firewall detected, skipping"
        fi
        ;;
    redhat)
        if command -v firewall-cmd &> /dev/null; then
            $SUDO firewall-cmd --permanent --add-port=8443/tcp
            $SUDO firewall-cmd --reload
            echo "✓ Firewalld configured"
        else
            echo "⚠ No firewall detected, skipping"
        fi
        ;;
esac
echo ""

# Create service user
echo "Step 8: Creating service user..."
case $OS in
    openbsd)
        if ! id gogrid &> /dev/null; then
            $SUDO useradd -s /sbin/nologin gogrid
            echo "✓ User created"
        else
            echo "✓ User already exists"
        fi
        ;;
    *)
        if ! id gogrid &> /dev/null; then
            $SUDO useradd -r -s /bin/false gogrid
            echo "✓ User created"
        else
            echo "✓ User already exists"
        fi
        ;;
esac

$SUDO chown -R gogrid:gogrid /opt/gogrid
echo ""

# Create and start service
echo "Step 9: Creating service..."
case $OS in
    openbsd)
        $SUDO tee /etc/rc.d/gogrid_coordinator > /dev/null << 'RCDSCRIPT'
#!/bin/ksh
daemon="/opt/gogrid/bin/gogrid-coordinator"
daemon_flags="--config /opt/gogrid/config/coordinator.toml"
daemon_user="gogrid"

. /etc/rc.d/rc.subr

rc_reload=NO
rc_bg=YES

rc_cmd $1
RCDSCRIPT

        $SUDO chmod +x /etc/rc.d/gogrid_coordinator

        echo "Enabling and starting service..."
        $SUDO rcctl enable gogrid_coordinator
        $SUDO rcctl start gogrid_coordinator

        echo "✓ Service created and started"
        ;;
    *)
        $SUDO tee /etc/systemd/system/gogrid-coordinator.service > /dev/null << 'SERVICE'
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

        echo "Enabling and starting service..."
        $SUDO systemctl daemon-reload
        $SUDO systemctl enable gogrid-coordinator
        $SUDO systemctl start gogrid-coordinator

        echo "✓ Service created and started"
        ;;
esac
echo ""

# Check service status
echo "Step 10: Verifying installation..."
sleep 2

case $OS in
    openbsd)
        if $SUDO rcctl check gogrid_coordinator; then
            echo "✓ Service is running"
        else
            echo "⚠ Service may not be running, check logs"
        fi
        ;;
    *)
        if $SUDO systemctl is-active --quiet gogrid-coordinator; then
            echo "✓ Service is running"
        else
            echo "⚠ Service may not be running, check logs"
        fi
        ;;
esac
echo ""

# Display summary
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "GoGrid coordinator is installed and running on port 8443"
echo ""
echo "Configuration file: /opt/gogrid/config/coordinator.toml"
echo "Binary location: /opt/gogrid/bin/gogrid-coordinator"
echo "Updates directory: /opt/gogrid/updates/"
echo "Logs directory: /opt/gogrid/logs/"
echo ""
echo "Next steps:"
echo "1. Upload update packages to /opt/gogrid/updates/"
echo "2. Configure TLS certificates (recommended for production)"
echo "3. Set up PostgreSQL/Redis (optional)"
echo ""
echo "View service status:"
case $OS in
    openbsd)
        echo "  $SUDO rcctl status gogrid_coordinator"
        echo ""
        echo "View logs:"
        echo "  $SUDO tail -f /var/log/daemon"
        ;;
    *)
        echo "  $SUDO systemctl status gogrid-coordinator"
        echo ""
        echo "View logs:"
        echo "  $SUDO journalctl -u gogrid-coordinator -f"
        ;;
esac
echo ""
echo "Test downloads page:"
echo "  curl http://localhost:8443/downloads"
echo ""
echo "For more information, see SERVER_SETUP.md"
echo ""
