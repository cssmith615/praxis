#!/usr/bin/env bash
# setup-vps.sh — One-shot Praxis Agent setup for a fresh Ubuntu VPS.
#
# Run as root on your VPS:
#   curl -fsSL https://raw.githubusercontent.com/cssmith615/praxis/main/scripts/setup-vps.sh | bash
#
# After this script completes:
#   1. Edit /opt/praxis/.env with your API keys
#   2. docker compose -f /opt/praxis/praxis/agent/docker-compose.yml up -d --build

set -euo pipefail

REPO="https://github.com/cssmith615/praxis"
INSTALL_DIR="/opt/praxis"

echo "==> Updating system packages"
apt-get update -qq && apt-get upgrade -y -qq

echo "==> Installing Docker"
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    echo "    Docker installed: $(docker --version)"
else
    echo "    Docker already installed: $(docker --version)"
fi

echo "==> Cloning Praxis"
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "    Repository already exists — pulling latest"
    git -C "$INSTALL_DIR" pull origin main
else
    git clone "$REPO" "$INSTALL_DIR"
fi

echo "==> Setting up environment file"
if [ ! -f "$INSTALL_DIR/.env" ]; then
    cp "$INSTALL_DIR/praxis/agent/.env.example" "$INSTALL_DIR/.env"
    echo ""
    echo "  ┌─────────────────────────────────────────────────────────────────┐"
    echo "  │  NEXT STEP: fill in your API keys                               │"
    echo "  │                                                                 │"
    echo "  │  nano $INSTALL_DIR/.env                                         │"
    echo "  │                                                                 │"
    echo "  │  Required:                                                      │"
    echo "  │    ANTHROPIC_API_KEY=sk-ant-...                                 │"
    echo "  │    TELEGRAM_BOT_TOKEN=your-bot-token                            │"
    echo "  │    TELEGRAM_CHAT_IDS=your-chat-id                               │"
    echo "  └─────────────────────────────────────────────────────────────────┘"
    echo ""
else
    echo "    .env already exists — skipping"
fi

echo "==> Setting up UFW firewall"
if command -v ufw &>/dev/null; then
    ufw allow 22/tcp --quiet || true
    ufw --force enable || true
    echo "    Firewall enabled (SSH port 22 open)"
fi

echo ""
echo "==> Setup complete."
echo ""
echo "  To start the agent:"
echo "    1. nano $INSTALL_DIR/.env               # fill in your keys"
echo "    2. cd $INSTALL_DIR"
echo "    3. docker compose -f praxis/agent/docker-compose.yml up -d --build"
echo "    4. docker compose -f praxis/agent/docker-compose.yml logs -f"
echo ""
echo "  To update later:"
echo "    cd $INSTALL_DIR && git pull && docker compose -f praxis/agent/docker-compose.yml up -d --build"
