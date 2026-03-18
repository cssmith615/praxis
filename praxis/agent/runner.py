"""
AgentRunner — wires PraxisAgent + Channel + Scheduler together.

Responsibilities
----------------
- Maintains a registry of AgentContext objects keyed by chat_id
  (one context = one conversation thread = isolated message history)
- Calls channel.send_typing() before each agent response
- Forwards the agent's reply back through the channel
- Optionally starts the Scheduler background thread
- Handles graceful shutdown on KeyboardInterrupt / SIGTERM

The runner is intentionally single-threaded per turn but can handle
multiple concurrent chat_ids because each context is independent.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
from typing import Any

from praxis.agent.channels.base import Channel, InboundMessage
from praxis.agent.context import AgentContext
from praxis.agent.core import PraxisAgent

log = logging.getLogger(__name__)


class AgentRunner:
    """
    Main entry point for running the Praxis Agent.

    Parameters
    ----------
    agent:
        A configured PraxisAgent instance.
    channel:
        A Channel implementation (TelegramChannel, StdinChannel, …).
    mode:
        Praxis execution mode ("dev" or "prod"). Default: "dev".
    provider:
        LLM provider name for the Planner ("anthropic", "openai", …).
        If None, planning is disabled but run/validate still work.
    model:
        Model id for the Planner (separate from the agent's own model).
    enable_scheduler:
        Start the Scheduler background thread. Default: False.
    enable_memory:
        Load ProgramMemory (requires sentence-transformers). Default: False.
    db_path:
        Custom path for memory + schedule databases. Default: ~/.praxis/
    """

    def __init__(
        self,
        agent: PraxisAgent,
        channel: Channel,
        mode: str = "dev",
        provider: str | None = None,
        model: str | None = None,
        enable_scheduler: bool = False,
        enable_memory: bool = False,
        db_path: str | None = None,
    ) -> None:
        self.agent = agent
        self.channel = channel
        self.mode = mode
        self.enable_scheduler = enable_scheduler
        self.enable_memory = enable_memory
        self.db_path = db_path
        self._contexts: dict[str, AgentContext] = {}
        self._lock = threading.Lock()

        # Build shared Praxis singletons
        self._planner = _build_planner(provider, model, db_path) if provider else None
        self._memory = _build_memory(db_path) if enable_memory else None
        self._scheduler = _build_scheduler(db_path) if enable_scheduler else None

        self._running = False

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Start the runner — polls the channel indefinitely.

        Blocks until a SIGINT/SIGTERM or KeyboardInterrupt.
        """
        self._running = True

        # Start scheduler if enabled
        if self._scheduler:
            self._scheduler.start()
            log.info("Scheduler started")

        # Graceful shutdown on SIGTERM
        signal.signal(signal.SIGTERM, self._handle_sigterm)

        log.info("Praxis Agent running — waiting for messages")
        try:
            for message in self.channel.poll():
                if not self._running:
                    break
                self._handle_message(message)
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def stop(self) -> None:
        """Signal the runner to stop after the current message is processed."""
        self._running = False
        if hasattr(self.channel, "stop"):
            self.channel.stop()

    # ── Internal ──────────────────────────────────────────────────────────

    def _handle_message(self, msg: InboundMessage) -> None:
        ctx = self._get_or_create_context(msg.chat_id)
        try:
            self.channel.send_typing(msg.chat_id)
            reply = self.agent.chat(msg.text, ctx)
            self.channel.send(msg.chat_id, reply)
        except Exception as exc:
            log.exception("Error handling message from %s", msg.chat_id)
            try:
                self.channel.send(msg.chat_id, f"⚠ Internal error: {exc}")
            except Exception:
                pass

    def _get_or_create_context(self, chat_id: str) -> AgentContext:
        with self._lock:
            if chat_id not in self._contexts:
                ctx = AgentContext(
                    chat_id=chat_id,
                    mode=self.mode,
                    _planner=self._planner,
                    _memory=self._memory,
                    _scheduler=self._scheduler,
                )
                self._contexts[chat_id] = ctx
            return self._contexts[chat_id]

    def _handle_sigterm(self, signum: int, frame: Any) -> None:
        log.info("SIGTERM received — shutting down")
        self.stop()

    def _shutdown(self) -> None:
        if self._scheduler:
            self._scheduler.stop()
            log.info("Scheduler stopped")
        log.info("Praxis Agent stopped")


# ──────────────────────────────────────────────────────────────────────────────
# Singleton builders
# ──────────────────────────────────────────────────────────────────────────────

def _build_planner(provider: str, model: str | None, db_path: str | None) -> Any:
    try:
        from praxis.planner import Planner
        from praxis.providers import get_provider
        p = get_provider(provider, model=model)
        mem = _build_memory(db_path)
        return Planner(provider=p, memory=mem)
    except Exception as exc:
        log.warning("Could not initialise Planner (%s) — planning disabled", exc)
        return None


def _build_memory(db_path: str | None) -> Any:
    try:
        from praxis.memory import ProgramMemory
        kwargs = {"db_path": db_path} if db_path else {}
        return ProgramMemory(**kwargs)
    except Exception as exc:
        log.warning("Could not initialise ProgramMemory (%s) — memory disabled", exc)
        return None


def _build_scheduler(db_path: str | None) -> Any:
    try:
        from praxis.scheduler import Scheduler
        from praxis.executor import Executor
        from praxis.handlers import HANDLERS
        exe = Executor(HANDLERS)
        kwargs = {"db_path": db_path} if db_path else {}
        return Scheduler(executor=exe, **kwargs)
    except Exception as exc:
        log.warning("Could not initialise Scheduler (%s) — scheduling disabled", exc)
        return None
