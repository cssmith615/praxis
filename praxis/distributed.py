"""
Praxis Distributed Workers

Extends the local in-process Worker model to span separate processes and
machines.  Each remote worker runs its own Praxis bridge (FastAPI /execute)
and registers itself with a hub so coordinators can discover it.

Components
----------
RemoteWorker
    Drop-in replacement for agent_registry.Worker.  Implements the same
    .execute(program_text) interface but routes over HTTP to a remote bridge
    at `url`.  Uses only stdlib (urllib) — no new runtime dependencies.

WorkerRegistration
    Hub-side record of a registered remote worker.

RemoteWorkerHub
    In-memory registry of remote workers.  Added to bridge.py as
    /workers/* endpoints.

WorkerClient
    Thin helper for coordinator programs that want to discover and call
    workers by querying the hub rather than hardcoding URLs.

HTTP contract
-------------
Register    POST {hub}/workers/register
            body: {"agent_id": str, "role": str, "verbs": [str], "url": str}

Heartbeat   POST {hub}/workers/{agent_id}/heartbeat

Deregister  DELETE {hub}/workers/{agent_id}

List        GET  {hub}/workers  → [WorkerRegistration, ...]

Dispatch    POST {hub}/workers/dispatch/{agent_id}
            body: {"program": str, "mode": str}
            Proxies to the worker's /execute endpoint.

Worker /execute contract (already in bridge.py):
    POST {worker_url}/execute
    body: {"program": str, "mode": str}
    response: {"ok": bool, "results": [...], "errors": [...]}
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_TIMEOUT  = 30.0    # seconds for remote /execute call
HEARTBEAT_TTL    = 120.0   # seconds before a worker is marked stale


# ──────────────────────────────────────────────────────────────────────────────
# WorkerRegistration (hub-side record)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WorkerRegistration:
    agent_id:     str
    role:         str
    verbs:        list[str]
    url:          str                    # http://host:port  (no trailing slash)
    registered_at: str = field(default_factory=lambda: _now_iso())
    last_seen:    str = field(default_factory=lambda: _now_iso())

    def is_stale(self) -> bool:
        """True when the worker has not heartbeated within HEARTBEAT_TTL."""
        last = datetime.fromisoformat(self.last_seen)
        age  = (datetime.now(timezone.utc) - last).total_seconds()
        return age > HEARTBEAT_TTL

    def touch(self) -> None:
        self.last_seen = _now_iso()

    def to_dict(self) -> dict:
        return {
            "agent_id":      self.agent_id,
            "role":          self.role,
            "verbs":         self.verbs,
            "url":           self.url,
            "registered_at": self.registered_at,
            "last_seen":     self.last_seen,
            "stale":         self.is_stale(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# RemoteWorker  (coordinator-side proxy)
# ──────────────────────────────────────────────────────────────────────────────

class RemoteWorker:
    """
    Proxy for a remote Praxis worker.  Matches the Worker.execute() interface
    so msg_handler, join_handler, etc. work without modification.

    Parameters
    ----------
    agent_id : str
    role : str
    verbs : list[str]
    url : str
        Base URL of the remote bridge, e.g. "http://192.168.1.5:7821".
        Must serve POST /execute.
    timeout : float
        Per-request timeout in seconds.
    mode : str
        Default execution mode ("dev" | "prod") passed to remote /execute.
    """

    def __init__(
        self,
        agent_id: str,
        role: str,
        verbs: list[str],
        url: str,
        timeout: float = DEFAULT_TIMEOUT,
        mode: str = "dev",
    ) -> None:
        self.agent_id = agent_id
        self.role     = role
        self.verbs    = verbs
        self.url      = url.rstrip("/")
        self.timeout  = timeout
        self.mode     = mode

    # ── Public ─────────────────────────────────────────────────────────────────

    def execute(self, program_text: str, memory: Any = None) -> dict:
        """
        POST program_text to {url}/execute.  Returns a result dict matching
        the shape of in-process Worker.execute().
        """
        start = time.monotonic()
        try:
            resp = _http_post(
                f"{self.url}/execute",
                {"program": program_text, "mode": self.mode},
                timeout=self.timeout,
            )
            duration = int((time.monotonic() - start) * 1000)

            if resp.get("ok"):
                results = resp.get("results", [])
                output  = results[-1]["output"] if results else None
                return {
                    "agent_id":    self.agent_id,
                    "role":        self.role,
                    "status":      "ok",
                    "output":      output,
                    "steps":       len(results),
                    "duration_ms": duration,
                    "results":     results,
                    "remote":      True,
                    "url":         self.url,
                }
            else:
                errors = resp.get("errors", [])
                return {
                    "agent_id":    self.agent_id,
                    "role":        self.role,
                    "status":      "error",
                    "error":       "; ".join(errors) if errors else "remote error",
                    "output":      None,
                    "steps":       0,
                    "duration_ms": duration,
                    "remote":      True,
                    "url":         self.url,
                }
        except Exception as exc:
            duration = int((time.monotonic() - start) * 1000)
            return {
                "agent_id":    self.agent_id,
                "role":        self.role,
                "status":      "error",
                "error":       f"HTTP error: {exc}",
                "output":      None,
                "steps":       0,
                "duration_ms": duration,
                "remote":      True,
                "url":         self.url,
            }

    def health_check(self) -> bool:
        """Return True if the remote bridge responds to GET /health."""
        try:
            resp = _http_get(f"{self.url}/health", timeout=5.0)
            return resp.get("status") == "ok"
        except Exception:
            return False

    def __repr__(self) -> str:
        return (
            f"RemoteWorker(agent_id={self.agent_id!r}, role={self.role!r}, "
            f"verbs={self.verbs}, url={self.url!r})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# RemoteWorkerHub  (hub-side, embedded in bridge.py)
# ──────────────────────────────────────────────────────────────────────────────

class RemoteWorkerHub:
    """
    In-memory registry for remote workers.  Thread-safe for concurrent
    register/heartbeat/dispatch calls from multiple coordinator processes.

    Embedded into bridge.py — the FastAPI app calls these methods from its
    /workers/* route handlers.
    """

    def __init__(self) -> None:
        import threading
        self._workers: dict[str, WorkerRegistration] = {}
        self._lock = threading.Lock()

    def register(self, agent_id: str, role: str, verbs: list[str], url: str) -> WorkerRegistration:
        reg = WorkerRegistration(
            agent_id=agent_id,
            role=role,
            verbs=[v.upper() for v in verbs],
            url=url.rstrip("/"),
        )
        with self._lock:
            self._workers[agent_id] = reg
        return reg

    def heartbeat(self, agent_id: str) -> bool:
        with self._lock:
            reg = self._workers.get(agent_id)
            if reg is None:
                return False
            reg.touch()
            return True

    def deregister(self, agent_id: str) -> bool:
        with self._lock:
            return self._workers.pop(agent_id, None) is not None

    def get(self, agent_id: str) -> WorkerRegistration | None:
        with self._lock:
            return self._workers.get(agent_id)

    def list_all(self) -> list[WorkerRegistration]:
        with self._lock:
            return list(self._workers.values())

    def route(self, verb: str) -> WorkerRegistration | None:
        """Return first non-stale worker supporting verb."""
        verb = verb.upper()
        with self._lock:
            for reg in self._workers.values():
                if not reg.is_stale() and verb in reg.verbs:
                    return reg
        return None

    def dispatch(self, agent_id: str, program: str, mode: str = "dev") -> dict:
        """
        Hub proxies a program to a registered worker's /execute endpoint.
        Returns the worker's response dict.
        """
        reg = self.get(agent_id)
        if reg is None:
            return {"ok": False, "errors": [f"Worker '{agent_id}' not registered"]}
        if reg.is_stale():
            return {"ok": False, "errors": [f"Worker '{agent_id}' is stale (no heartbeat)"]}

        try:
            return _http_post(
                f"{reg.url}/execute",
                {"program": program, "mode": mode},
                timeout=DEFAULT_TIMEOUT,
            )
        except Exception as exc:
            return {"ok": False, "errors": [f"Dispatch failed: {exc}"]}


# ──────────────────────────────────────────────────────────────────────────────
# WorkerClient  (coordinator-side discovery helper)
# ──────────────────────────────────────────────────────────────────────────────

class WorkerClient:
    """
    Query a hub's /workers endpoints to discover and connect remote workers.

    Usage (in a coordinator program):
        client  = WorkerClient("http://hub:7821")
        workers = client.discover()          # → list[RemoteWorker]
        worker  = client.get("data_worker")  # → RemoteWorker | None
    """

    def __init__(self, hub_url: str, timeout: float = 10.0) -> None:
        self.hub_url = hub_url.rstrip("/")
        self.timeout = timeout

    def discover(self, role: str | None = None) -> list[RemoteWorker]:
        """Return all non-stale workers registered with the hub."""
        try:
            entries = _http_get(f"{self.hub_url}/workers", timeout=self.timeout)
        except Exception:
            return []

        workers = []
        for e in entries:
            if e.get("stale"):
                continue
            if role and e.get("role") != role:
                continue
            workers.append(RemoteWorker(
                agent_id=e["agent_id"],
                role=e["role"],
                verbs=e["verbs"],
                url=e["url"],
            ))
        return workers

    def get(self, agent_id: str) -> RemoteWorker | None:
        """Retrieve a specific worker by id."""
        try:
            entry = _http_get(f"{self.hub_url}/workers/{agent_id}", timeout=self.timeout)
            if entry.get("stale"):
                return None
            return RemoteWorker(
                agent_id=entry["agent_id"],
                role=entry["role"],
                verbs=entry["verbs"],
                url=entry["url"],
            )
        except Exception:
            return None

    def register(
        self, agent_id: str, role: str, verbs: list[str], worker_url: str
    ) -> bool:
        """Register a worker with the hub. Returns True on success."""
        try:
            resp = _http_post(
                f"{self.hub_url}/workers/register",
                {"agent_id": agent_id, "role": role, "verbs": verbs, "url": worker_url},
                timeout=self.timeout,
            )
            return resp.get("ok", False)
        except Exception:
            return False

    def heartbeat(self, agent_id: str) -> bool:
        try:
            resp = _http_post(
                f"{self.hub_url}/workers/{agent_id}/heartbeat",
                {},
                timeout=self.timeout,
            )
            return resp.get("ok", False)
        except Exception:
            return False

    def deregister(self, agent_id: str) -> bool:
        try:
            resp = _http_delete(f"{self.hub_url}/workers/{agent_id}", timeout=self.timeout)
            return resp.get("ok", False)
        except Exception:
            return False


# ──────────────────────────────────────────────────────────────────────────────
# HTTP helpers (stdlib only — no httpx/requests dependency)
# ──────────────────────────────────────────────────────────────────────────────

def _http_post(url: str, payload: dict, timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _http_get(url: str, timeout: float = 10.0) -> Any:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _http_delete(url: str, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(url, method="DELETE")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
