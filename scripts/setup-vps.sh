#!/bin/bash
# ─────────────────────────────────────────────────────────────
# WaveTrader VPS Setup Script
# Run this on a fresh Ubuntu 22.04/24.04 Vultr VPS
# Usage: bash setup-vps.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

echo "═══════════════════════════════════════════════════════"
echo "  WaveTrader VPS Setup"
echo "═══════════════════════════════════════════════════════"

# ── 1. System updates ────────────────────────────────────────
echo ""
echo "[1/6] Updating system packages..."
apt-get update && apt-get upgrade -y

# ── 2. Install Docker ────────────────────────────────────────
echo ""
echo "[2/6] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    echo "✓ Docker installed"
else
    echo "✓ Docker already installed"
fi

# ── 3. Install Docker Compose plugin ─────────────────────────
echo ""
echo "[3/6] Installing Docker Compose..."
if ! docker compose version &> /dev/null; then
    apt-get install -y docker-compose-plugin
    echo "✓ Docker Compose installed"
else
    echo "✓ Docker Compose already installed"
fi

# ── 4. Firewall setup ────────────────────────────────────────
echo ""
echo "[4/6] Configuring firewall..."
if command -v ufw &> /dev/null; then
    ufw allow 22/tcp    # SSH
    ufw allow 5000/tcp  # Dashboard
    ufw --force enable
    echo "✓ Firewall configured (SSH + Dashboard port 5000)"
else
    echo "⚠ ufw not found, skip firewall config"
fi

# ── 5. Create app directory ──────────────────────────────────
echo ""
echo "[5/6] Creating app directory..."
APP_DIR=/opt/wavetrader
mkdir -p "$APP_DIR"
echo "✓ App directory: $APP_DIR"

# ── 6. Create swap (prevents OOM on 4GB VPS) ─────────────────
echo ""
echo "[6/6] Setting up swap..."
if [ ! -f /swapfile ]; then
    fallocate -l 4G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo "✓ 4 GB swap created"
else
    echo "✓ Swap already exists"
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✓ VPS setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Clone your repo:"
echo "     cd /opt/wavetrader"
echo "     git clone https://github.com/Unseengap/wavetrader.git ."
echo ""
echo "  2. Copy your .env file and model checkpoint"
echo "  3. Run: docker compose up -d --build"
echo "═══════════════════════════════════════════════════════"
