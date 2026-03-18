# Deploying Praxis Agent to a VPS

Run the Praxis Agent 24/7 on a cheap Linux server so it keeps working whether your laptop is on or not.

---

## What you need

| Item | Cost | Notes |
|------|------|-------|
| VPS | ~$4–6/month | Hetzner CX22 or DigitalOcean Basic Droplet |
| Domain (optional) | — | Not required; SSH by IP works fine |
| Your `.env` file | — | `ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_IDS` |

**Recommended VPS:** [Hetzner CX22](https://www.hetzner.com/cloud) — 2 vCPU, 4 GB RAM, €3.29/month. Pick the Ubuntu 22.04 image. DigitalOcean's $6 Basic Droplet is identical.

---

## One-time server setup

SSH into your new server, then run the setup script (or follow the steps manually below):

```bash
ssh root@your-server-ip
curl -fsSL https://raw.githubusercontent.com/cssmith615/praxis/main/scripts/setup-vps.sh | bash
```

Or manually:

```bash
# 1. Update and install Docker
apt-get update && apt-get upgrade -y
curl -fsSL https://get.docker.com | sh
systemctl enable docker

# 2. Clone the repo
git clone https://github.com/cssmith615/praxis /opt/praxis
cd /opt/praxis

# 3. Configure environment
cp praxis/agent/.env.example .env
nano .env   # fill in ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_IDS

# 4. Start the agent
docker compose -f praxis/agent/docker-compose.yml up -d --build

# 5. Verify
docker compose -f praxis/agent/docker-compose.yml logs -f
```

You should see:
```
praxis-agent  | Praxis Agent  model=claude-sonnet-4-6  mode=prod
praxis-agent  |   chat whitelist: ['your-chat-id']
praxis-agent  |   Press Ctrl+C to stop.
praxis-agent  | [SCHEDULER] started (checking every 60s)
```

Send a message to your Telegram bot. It should respond.

---

## Day-to-day operations

```bash
# View live logs
docker compose -f /opt/praxis/praxis/agent/docker-compose.yml logs -f

# Stop the agent
docker compose -f /opt/praxis/praxis/agent/docker-compose.yml stop

# Restart
docker compose -f /opt/praxis/praxis/agent/docker-compose.yml restart

# Check status
docker compose -f /opt/praxis/praxis/agent/docker-compose.yml ps
```

---

## Updating to a new version

```bash
ssh root@your-server-ip
cd /opt/praxis
git pull origin main
docker compose -f praxis/agent/docker-compose.yml up -d --build
```

The named volume `praxis_data` persists across rebuilds — your memory.db, schedule.db, and execution.log are safe.

---

## Viewing your data

```bash
# Peek inside the volume
docker run --rm -v praxis_data:/data alpine ls /data

# Copy execution log to your machine (run locally)
scp root@your-server-ip:/var/lib/docker/volumes/praxis_data/_data/execution.log ./
```

---

## Security checklist

- [ ] Set `TELEGRAM_CHAT_IDS` in `.env` — restricts who can use the agent
- [ ] Keep `.env` out of git (it is in `.gitignore` by default)
- [ ] Set up a firewall — only SSH (22) needs to be open:
  ```bash
  ufw allow 22 && ufw enable
  ```
- [ ] Optional: lock SSH to key-only auth (disable password login in `/etc/ssh/sshd_config`)
- [ ] Optional: expose the Praxis bridge (port 7821) only if you need external API access

---

## Troubleshooting

**Agent not responding in Telegram**
```bash
docker compose -f /opt/praxis/praxis/agent/docker-compose.yml logs --tail=50
```
Look for `network error` (transient, auto-retries) or `Telegram API error` (bad token).

**Container keeps restarting**
```bash
docker compose -f /opt/praxis/praxis/agent/docker-compose.yml logs --tail=20
# Usually a missing env var or bad API key
```

**Out of disk space**
```bash
docker system prune -f   # removes unused images and build cache
```

**Check how much memory the agent is using**
```bash
docker stats praxis-agent --no-stream
```
Typical usage: ~150–250 MB RAM at rest.

---

## Publishing the VS Code Extension

The packaged `.vsix` is at `praxis-vscode/praxis-lang-1.2.0.vsix`. To publish to the Marketplace:

### One-time setup

1. Go to [marketplace.visualstudio.com](https://marketplace.visualstudio.com/manage) and sign in with your Microsoft account
2. Create a publisher named `cssmith615` (matches `package.json`)
3. Go to [dev.azure.com](https://dev.azure.com) → your org → User Settings → Personal Access Tokens
4. Create a token with scope **Marketplace → Manage** and save it

### Publish

```bash
cd praxis-vscode
npm install                          # installs vsce
npx vsce publish --no-dependencies  # prompts for your PAT
```

Or package and upload manually via the Marketplace web UI:

```bash
npx vsce package --no-dependencies  # creates praxis-lang-1.2.0.vsix
# then drag-drop the .vsix at marketplace.visualstudio.com/manage
```

### Update an existing listing

Bump the version in `praxis-vscode/package.json`, then:

```bash
cd praxis-vscode
npx vsce publish --no-dependencies
```
