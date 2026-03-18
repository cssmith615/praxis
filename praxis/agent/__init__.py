"""
Praxis Agent — Sprint 21

A native Praxis agent that replaces NanoClaw.

Key properties:
- Directly executes .px programs with no translation layer
- Uses the Anthropic SDK tool-use loop (not subprocess Claude Code)
- Receives messages from a pluggable Channel (Telegram, stdin, …)
- Forwards relevant programs to the Scheduler for cron-style execution
- No new hard dependencies (anthropic + httpx already in pyproject.toml)

Quick start::

    praxis agent --token $TELEGRAM_BOT_TOKEN --chat-id $TELEGRAM_CHAT_ID

Docker production::

    docker compose -f praxis/agent/docker-compose.yml up -d
"""

from praxis.agent.core import PraxisAgent
from praxis.agent.runner import AgentRunner

__all__ = ["PraxisAgent", "AgentRunner"]
