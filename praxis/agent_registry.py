"""
Praxis Agent Registry — worker registration and capability routing.

Workers announce their supported verb sets when spawned. The registry
answers the question "which worker can handle verb X?" and stores each
worker's result queue so MSG can dispatch and JOIN can collect.

In-process use (default):
    registry = AgentRegistry()
    worker   = Worker(agent_id="data", role="data", verbs=["ING","CLN","XFRM"],
                      executor=Executor(HANDLERS))
    registry.register(worker)
    w = registry.route("ING")       # → Worker or None
    w = registry.get("data")        # → Worker or None

Redis-backed distributed use:
    Workers announce via HTTP POST /register on startup.
    The registry becomes a thin wrapper around the Redis key-space.
    See docs/multi-agent.md for the distributed deployment guide.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Worker:
    """
    A named agent that can execute Praxis programs for a specific verb set.

    Each worker has its own isolated Executor and ExecutionContext — workers
    never share mutable state with the coordinator or each other.
    """
    agent_id:  str
    role:      str
    verbs:     list[str]
    executor:  Any           # praxis.executor.Executor
    metadata:  dict = field(default_factory=dict)
    _lock:     threading.Lock = field(default_factory=threading.Lock, repr=False)

    def execute(self, program_text: str, memory: Any = None) -> dict:
        """
        Execute a Praxis program string and return a structured result dict.
        Called in a worker thread — safe to block.
        """
        from praxis.grammar import parse

        start = time.monotonic()
        try:
            program  = parse(program_text)
            results  = self.executor.execute(program, memory=memory)
            duration = int((time.monotonic() - start) * 1000)
            output   = results[-1]["output"] if results else None
            return {
                "agent_id":   self.agent_id,
                "role":       self.role,
                "status":     "ok",
                "output":     output,
                "steps":      len(results),
                "duration_ms": duration,
                "results":    results,
            }
        except Exception as exc:
            duration = int((time.monotonic() - start) * 1000)
            return {
                "agent_id":    self.agent_id,
                "role":        self.role,
                "status":      "error",
                "error":       str(exc),
                "output":      None,
                "steps":       0,
                "duration_ms": duration,
            }


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

class AgentRegistry:
    """
    In-process agent registry.

    Thread-safe: register/get/route are protected by a lock so PAR branches
    can safely spawn workers concurrently.
    """

    def __init__(self) -> None:
        self._workers: dict[str, Worker] = {}
        self._lock = threading.Lock()

    def register(self, worker: Worker) -> None:
        with self._lock:
            self._workers[worker.agent_id] = worker

    def get(self, agent_id: str) -> Optional[Worker]:
        with self._lock:
            return self._workers.get(agent_id)

    def route(self, verb: str) -> Optional[Worker]:
        """Return the first registered worker that supports this verb."""
        with self._lock:
            for w in self._workers.values():
                if verb.upper() in [v.upper() for v in w.verbs]:
                    return w
        return None

    def all_workers(self) -> list[Worker]:
        with self._lock:
            return list(self._workers.values())

    def remove(self, agent_id: str) -> bool:
        with self._lock:
            if agent_id in self._workers:
                del self._workers[agent_id]
                return True
        return False

    def capability_map(self) -> dict[str, list[str]]:
        """Returns {verb: [agent_id, ...]} for all registered workers."""
        result: dict[str, list[str]] = {}
        with self._lock:
            for w in self._workers.values():
                for v in w.verbs:
                    result.setdefault(v.upper(), []).append(w.agent_id)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# HMAC signing
# ─────────────────────────────────────────────────────────────────────────────

# Session signing key: reads from env or generates a random per-session key.
# For production, set PRAXIS_SIGN_KEY to a stable 32-byte hex secret.
_SESSION_KEY: bytes = (
    bytes.fromhex(os.environ["PRAXIS_SIGN_KEY"])
    if "PRAXIS_SIGN_KEY" in os.environ
    else os.urandom(32)
)


def sign_message(payload: str, key: bytes = _SESSION_KEY) -> str:
    """Return HMAC-SHA256 hex digest of payload using key."""
    return hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()


def verify_message(payload: str, signature: str, key: bytes = _SESSION_KEY) -> bool:
    """Verify an HMAC-SHA256 signature. Constant-time comparison."""
    expected = sign_message(payload, key)
    return hmac.compare_digest(expected, signature)
