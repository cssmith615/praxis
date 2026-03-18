"""
ModelRouter — Sprint 25 multi-tier model routing.

Routes each agent turn to either a fast (cheap) model or the full model based
on the complexity of the user message.  The goal: simple, well-defined requests
(run a known program, list schedules, validate syntax) use a cheap model and
respond quickly; open-ended goals, multi-step planning, and long messages use
the full model.

Classification is heuristic — no LLM meta-call is made to classify (that would
defeat the cost savings).  The rules are deliberately conservative: when in
doubt the router falls back to the full model.

Default models
--------------
  fast : claude-haiku-4-5-20251001   (~20× cheaper than Sonnet per token)
  full : claude-sonnet-4-6           (default agent model)

Override via constructor or environment:
  PRAXIS_FAST_MODEL   fast model id
  PRAXIS_FULL_MODEL   full model id

Routing rules (evaluated in order, first match wins)
-----------------------------------------------------
  FAST if the message matches any simple-command prefix:
    run, validate, list, recall, remove schedule, status

  FAST if:
    - no planning/scheduling keywords present, AND
    - message is short (≤ FAST_MAX_CHARS characters)

  FULL otherwise (includes: goal, plan, schedule, long messages, multi-sentence
    descriptions, anything that could require multi-step reasoning)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# ── Default model IDs ──────────────────────────────────────────────────────────

_DEFAULT_FAST_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_FULL_MODEL = "claude-sonnet-4-6"

# ── Routing thresholds ────────────────────────────────────────────────────────

FAST_MAX_CHARS = 120   # messages ≤ this length can use the fast model
                       # (if no planning keywords are present)

# Simple-command prefixes — these map directly to a single known tool
_SIMPLE_PREFIXES: tuple[str, ...] = (
    "run ",
    "validate ",
    "list ",
    "list my ",
    "recall ",
    "remove schedule",
    "cancel schedule",
    "delete schedule",
    "status",
    "help",
    "hello",
    "hi ",
    "hey ",
)

# Keywords that always escalate to the full model
_COMPLEX_KEYWORDS: tuple[str, ...] = (
    "goal",
    "plan",
    "schedule this",    # creation — "schedule this every morning"
    "schedule a ",      # creation — "schedule a daily brief"
    "schedule every",   # creation — "schedule every hour"
    "add a schedule",
    "new schedule",
    "every ",
    "daily",
    "weekly",
    "hourly",
    "cron",
    "create",
    "build",
    "design",
    "generate",
    "write",
    "improve",
    "analyse",
    "analyze",
    "summarize",
    "fetch and",
    "then send",
    "and send",
)


@dataclass
class RouteDecision:
    model: str
    tier: str          # "fast" | "full"
    reason: str        # human-readable explanation (for logging)


class ModelRouter:
    """
    Heuristic model router for the Praxis Agent.

    Parameters
    ----------
    fast_model:
        Model id for simple, well-defined requests.
        Falls back to ``PRAXIS_FAST_MODEL`` env var, then the default.
    full_model:
        Model id for complex, open-ended requests.
        Falls back to ``PRAXIS_FULL_MODEL`` env var, then the default.
    enabled:
        Set to False to disable routing and always use full_model.
        Useful for debugging or high-stakes production sessions.
    """

    def __init__(
        self,
        fast_model: str | None = None,
        full_model: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.fast_model = (
            fast_model
            or os.environ.get("PRAXIS_FAST_MODEL", "")
            or _DEFAULT_FAST_MODEL
        )
        self.full_model = (
            full_model
            or os.environ.get("PRAXIS_FULL_MODEL", "")
            or _DEFAULT_FULL_MODEL
        )
        self.enabled = enabled

    def route(self, message: str) -> RouteDecision:
        """
        Classify *message* and return the model to use for this turn.
        """
        if not self.enabled:
            return RouteDecision(
                model=self.full_model,
                tier="full",
                reason="routing disabled",
            )

        normalized = message.strip().lower()

        # Rule 1: complex keyword present → full (checked first, takes priority)
        for kw in _COMPLEX_KEYWORDS:
            if kw in normalized:
                return RouteDecision(
                    model=self.full_model,
                    tier="full",
                    reason=f"complex keyword: '{kw.strip()}'",
                )

        # Rule 2: simple command prefix with no complex keywords → fast
        for prefix in _SIMPLE_PREFIXES:
            if normalized.startswith(prefix) or normalized == prefix.strip():
                return RouteDecision(
                    model=self.fast_model,
                    tier="fast",
                    reason=f"simple-command prefix: '{prefix.strip()}'",
                )

        # Rule 3: short message with no complex keywords → fast
        if len(message) <= FAST_MAX_CHARS:
            return RouteDecision(
                model=self.fast_model,
                tier="fast",
                reason=f"short message ({len(message)} chars, no complex keywords)",
            )

        # Default: full model
        return RouteDecision(
            model=self.full_model,
            tier="full",
            reason=f"long message ({len(message)} chars)",
        )
