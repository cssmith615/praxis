"""
AgentContext — per-conversation state + shared Praxis singletons.

One AgentContext is created per chat session (e.g. per Telegram chat_id).
It holds:
  - The Anthropic messages list (running tool-use conversation)
  - Shared Praxis singletons: Executor, Validator, Planner, Memory, Scheduler
  - A simple key-value store for handler state (e.g. last program run)

Singletons are lazy-initialised on first access so the agent starts quickly
and we don't pay for sentence-transformer load time if memory is never used.

Context compaction (Sprint 24B)
--------------------------------
When the conversation grows beyond COMPACT_THRESHOLD messages, maybe_compact()
summarises all but the most recent KEEP_RECENT messages using a cheap model
(claude-haiku-4-5) and replaces them with a single summary message.  This caps
context growth while preserving continuity.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

COMPACT_THRESHOLD = 20   # compact when conversation exceeds this many messages
KEEP_RECENT       = 10   # keep this many recent messages verbatim after compaction

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

    def add(self, role: str, content: str) -> None:
        """Convenience wrapper — add a plain text message with the given role."""
        self.messages.append({"role": role, "content": content})

    def clear(self) -> None:
        """Reset conversation (keep singletons)."""
        self.messages.clear()
        self.state.clear()

    # ── Context compaction ─────────────────────────────────────────────────

    def maybe_compact(self, client: Any, model: str) -> None:
        """
        If the conversation exceeds COMPACT_THRESHOLD messages, summarise the
        older portion with *model* (use a cheap model like claude-haiku-4-5)
        and replace it with a single summary message, keeping the last
        KEEP_RECENT messages verbatim.

        Safe to call with client=None — will skip compaction silently.
        """
        if client is None or len(self.messages) <= COMPACT_THRESHOLD:
            return

        to_summarise = self.messages[:-KEEP_RECENT]
        recent       = self.messages[-KEEP_RECENT:]

        # Build a plain-text transcript for the summary request
        transcript_parts = []
        for m in to_summarise:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            if isinstance(content, list):
                # tool-result or multi-block messages — extract text
                parts = [
                    b.get("content", "") if isinstance(b, dict) else str(b)
                    for b in content
                ]
                content = " ".join(str(p) for p in parts if p)
            transcript_parts.append(f"{role.upper()}: {content}")

        transcript = "\n".join(transcript_parts)

        prompt = (
            "You are summarising a conversation between a user and an AI assistant. "
            "Produce a concise summary (3-8 sentences) that captures the key topics, "
            "decisions, and context needed to continue the conversation coherently. "
            "Do not include greetings or filler.\n\n"
            f"CONVERSATION:\n{transcript}"
        )

        try:
            response = client.messages.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            summary_text = response.content[0].text
        except Exception as exc:
            log.warning("Context compaction failed (%s) — keeping full history", exc)
            return

        summary_message = {
            "role": "user",
            "content": f"[Conversation summary — earlier context]\n{summary_text}",
        }
        self.messages = [summary_message] + list(recent)
        log.info(
            "Context compacted: %d messages → 1 summary + %d recent",
            len(to_summarise),
            len(recent),
        )
