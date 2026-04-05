#!/bin/bash
# ─────────────────────────────────────────────────────────────
# WaveTrader — Deploy Updates to Vultr VPS
# Run from the project root: bash scripts/deploy-update.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

VPS_IP="104.207.143.54"
VPS_USER="root"
VPS_APP_DIR="/opt/wavetrader"

echo "═══════════════════════════════════════════════════════"
echo "  WaveTrader — Deploy Update"
echo "═══════════════════════════════════════════════════════"

# ── 1. Commit & Push ─────────────────────────────────────────
echo ""
echo "[1/3] Pushing to GitHub..."
git add -A
read -p "Commit message: " MSG
git commit -m "$MSG"
git push origin main

# ── 2. Pull on VPS ───────────────────────────────────────────
echo ""
echo "[2/3] Pulling on VPS..."
ssh ${VPS_USER}@${VPS_IP} "cd ${VPS_APP_DIR} && git pull origin main"

# ── 3. Rebuild & Restart ─────────────────────────────────────
echo ""
echo "[3/3] Rebuilding containers..."
ssh ${VPS_USER}@${VPS_IP} "cd ${VPS_APP_DIR} && docker compose up -d --build"

# ── Verify ────────────────────────────────────────────────────
echo ""
echo "Checking status..."
ssh ${VPS_USER}@${VPS_IP} "cd ${VPS_APP_DIR} && docker compose ps"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✓ Update deployed!"
echo "  Dashboard: http://${VPS_IP}:5000"
echo "═══════════════════════════════════════════════════════"
