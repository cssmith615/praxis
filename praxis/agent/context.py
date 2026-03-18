"""
AgentContext — per-conversation state + shared Praxis singletons.

One AgentContext is created per chat session (e.g. per Telegram chat_id).
It holds:
  - The Anthropic messages list (running tool-use conversation)
  - Shared Praxis singletons: Executor, Validator, Planner, Memory, Scheduler
  - A simple key-value store for handler state (e.g. last program run)

Singletons are lazy-initialised on first access so the agent starts quickly
and we don't pay for sentence-transformer load time if memory is never used.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from praxis.executor import Executor
from praxis.handlers import HANDLERS


@dataclass
class AgentContext:
    """Per-chat conversation state."""

    chat_id: str
    mode: str = "dev"

    # Running Anthropic messages array — extended each turn
    messages: list[dict] = field(default_factory=list)

    # Arbitrary per-session state handlers can read/write
    state: dict[str, Any] = field(default_factory=dict)

    # ── Praxis singletons (injected by AgentRunner) ────────────────────────
    _executor: Any = field(default=None, repr=False)
    _validator: Any = field(default=None, repr=False)
    _planner: Any = field(default=None, repr=False)
    _memory: Any = field(default=None, repr=False)
    _scheduler: Any = field(default=None, repr=False)

    # Thread lock — tool calls can fire concurrently in PAR blocks
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ── Accessors ──────────────────────────────────────────────────────────

    @property
    def executor(self) -> Any:
        if self._executor is None:
            self._executor = Executor(HANDLERS, mode=self.mode)
        return self._executor

    @property
    def validator(self) -> Any:
        if self._validator is None:
            from praxis.validator import Validator
            self._validator = Validator()
        return self._validator

    @property
    def memory(self) -> Any:
        return self._memory  # may be None if not configured

    @property
    def planner(self) -> Any:
        return self._planner  # may be None if no provider configured

    @property
    def scheduler(self) -> Any:
        return self._scheduler  # may be None if not enabled

    # ── Message management ─────────────────────────────────────────────────

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant_message(self, content: Any) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_result(self, tool_use_id: str, result: str) -> None:
        self.messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result,
            }],
        })

    def clear(self) -> None:
        """Reset conversation (keep singletons)."""
        self.messages.clear()
        self.state.clear()
