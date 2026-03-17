"""
Agent handlers — Sprint 6 full implementations.

SPAWN  create a named worker agent with a specific verb capability set
MSG    dispatch a Praxis program to a registered worker (async, via thread pool)
CAST   broadcast a program to ALL registered workers simultaneously
JOIN   collect all pending MSG futures, return unified results array
SIGN   HMAC-SHA256 sign the current payload
CAP    declare capability scope for the current agent (metadata annotation)

Multi-agent architecture
------------------------
Coordinator:
    SPAWN.data_worker(role=data, verbs=[ING,CLN,XFRM]) ->
    SPAWN.analysis_worker(role=analysis, verbs=[SUMM,EVAL,GEN]) ->
    MSG.data_worker(program="ING.sales -> CLN.normalize") ->
    MSG.analysis_worker(program="SUMM.text -> EVAL.sentiment") ->
    JOIN(timeout=30) ->
    MERGE -> OUT.telegram

Worker agents run on a shared ThreadPoolExecutor in the coordinator's process.
Each worker has its own isolated Executor and ExecutionContext.

For distributed multi-process workers, wrap the executor in a thin HTTP server
(the bridge.py already exposes /execute) and override msg_handler to use httpx
instead of the thread pool.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

# Shared thread pool for worker execution — 8 workers max by default
# Override via PRAXIS_MAX_WORKERS env var
import os
import uuid as _uuid
_MAX_WORKERS = int(os.environ.get("PRAXIS_MAX_WORKERS", 8))
_POOL = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="praxis-worker")


# ─────────────────────────────────────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────────────────────────────────────

def spawn_handler(target: list[str], params: dict, ctx) -> Any:
    """
    SPAWN — Create a named worker agent and register it in ctx.agent_registry.

    Usage:
        SPAWN.data_worker(role=data, verbs=[ING,CLN,XFRM])
        SPAWN.analysis_worker(role=analysis, verbs=[SUMM,EVAL,GEN])

    The spawned worker uses the same handler registry as the coordinator
    unless a custom `handlers` key is provided in params.
    """
    from praxis.agent_registry import AgentRegistry, Worker
    from praxis.executor import Executor

    agent_id = target[0] if target else params.get("id", "worker")
    role     = params.get("role", "worker")
    verbs    = params.get("verbs", [])

    if isinstance(verbs, str):
        verbs = [v.strip() for v in verbs.split(",")]

    # Ensure ctx has an agent_registry
    if not hasattr(ctx, "agent_registry") or ctx.agent_registry is None:
        ctx.agent_registry = AgentRegistry()
    if not hasattr(ctx, "pending_futures"):
        ctx.pending_futures = {}

    # Build worker executor — inherits the same handler set as the coordinator
    if hasattr(ctx, "_handlers") and ctx._handlers:
        worker_exe = Executor(ctx._handlers, mode=ctx.mode)
    else:
        from praxis.handlers import HANDLERS
        worker_exe = Executor(HANDLERS, mode=ctx.mode)

    worker = Worker(
        agent_id=agent_id,
        role=role,
        verbs=verbs,
        executor=worker_exe,
        metadata={"spawned_by": "coordinator"},
    )
    ctx.agent_registry.register(worker)

    return {
        "agent_id":  agent_id,
        "role":      role,
        "verbs":     verbs,
        "status":    "spawned",
        "registered": True,
    }


def msg_handler(target: list[str], params: dict, ctx) -> Any:
    """
    MSG — Dispatch a Praxis program to a registered worker (non-blocking).

    The future is stored in ctx.pending_futures. Call JOIN to collect results.

    Usage:
        MSG.data_worker(program="ING.sales -> CLN.normalize -> SUMM.text")

    If `program` param is not given, ctx.last_output is used (must be a string).
    """
    agent_id     = target[0] if target else None
    program_text = params.get("program")
    if program_text is None:
        program_text = str(ctx.last_output) if ctx.last_output is not None else ""
    timeout = float(params.get("timeout", 30.0))

    if not hasattr(ctx, "agent_registry") or ctx.agent_registry is None:
        raise RuntimeError("MSG: no AgentRegistry on context — run SPAWN first")
    if not hasattr(ctx, "pending_futures"):
        ctx.pending_futures = {}

    # Resolve worker: by agent_id first, then by verb routing
    worker = ctx.agent_registry.get(agent_id) if agent_id else None
    if worker is None and agent_id:
        worker = ctx.agent_registry.route(agent_id)   # fall back to verb routing
    if worker is None:
        raise RuntimeError(
            f"MSG: no worker registered with id or verb '{agent_id}'. "
            f"Run SPAWN first. Registered workers: "
            f"{[w.agent_id for w in ctx.agent_registry.all_workers()]}"
        )

    memory   = getattr(ctx, "memory", None)
    msg_id   = f"msg_{_uuid.uuid4().hex[:8]}"   # UUID avoids PAR race condition
    future   = _POOL.submit(worker.execute, program_text, memory)
    ctx.pending_futures[msg_id] = (worker.agent_id, future, timeout)

    return {
        "msg_id":   msg_id,
        "to":       worker.agent_id,
        "role":     worker.role,
        "program":  program_text,
        "dispatched": True,
    }


def cast_handler(target: list[str], params: dict, ctx) -> Any:
    """
    CAST — Broadcast a Praxis program to ALL registered workers simultaneously.

    Returns immediately; use JOIN to collect all results.
    """
    program_text = params.get("program")
    if program_text is None:
        program_text = str(ctx.last_output) if ctx.last_output is not None else ""

    if not hasattr(ctx, "agent_registry") or ctx.agent_registry is None:
        raise RuntimeError("CAST: no AgentRegistry on context — run SPAWN first")
    if not hasattr(ctx, "pending_futures"):
        ctx.pending_futures = {}

    workers    = ctx.agent_registry.all_workers()
    memory     = getattr(ctx, "memory", None)
    dispatched = []

    for worker in workers:
        msg_id = f"msg_{_uuid.uuid4().hex[:8]}"   # UUID avoids PAR race condition
        future = _POOL.submit(worker.execute, program_text, memory)
        ctx.pending_futures[msg_id] = (worker.agent_id, future, 30.0)
        dispatched.append(worker.agent_id)

    return {
        "broadcast_to": dispatched,
        "program":      program_text,
        "pending":      len(dispatched),
    }


def join_handler(target: list[str], params: dict, ctx) -> Any:
    """
    JOIN — Collect all pending MSG/CAST futures and unify into a results array.

    Waits up to `timeout` seconds per future (default 30s).
    Clears ctx.pending_futures after collection.

    Usage:
        JOIN
        JOIN(timeout=60)
    """
    default_timeout = float(params.get("timeout", 30.0))

    if not hasattr(ctx, "pending_futures") or not ctx.pending_futures:
        return {"joined": [], "count": 0}

    results  = []
    errors   = []

    for msg_id, (agent_id, future, timeout) in ctx.pending_futures.items():
        effective_timeout = params.get("timeout", timeout)
        try:
            result = future.result(timeout=float(effective_timeout))
            results.append(result)
            if result.get("status") == "error":
                errors.append({"msg_id": msg_id, "agent_id": agent_id, "error": result.get("error")})
        except Exception as exc:
            err = {"msg_id": msg_id, "agent_id": agent_id, "status": "error", "error": str(exc)}
            results.append(err)
            errors.append(err)

    ctx.pending_futures.clear()

    return {
        "joined":  results,
        "count":   len(results),
        "errors":  errors,
        "success": len(errors) == 0,
    }


def sign_handler(target: list[str], params: dict, ctx) -> Any:
    """
    SIGN — HMAC-SHA256 sign the current payload (ctx.last_output or params.payload).

    Uses PRAXIS_SIGN_KEY env var or a per-session random key.
    Stores signature in ctx.variables['_signature'] for downstream ASSERT/VALIDATE.

    Usage:
        MSG.worker(program="...") -> SIGN.outgoing
    """
    from praxis.agent_registry import sign_message

    payload = params.get("payload")
    if payload is None:
        payload = json.dumps(ctx.last_output, default=str) if ctx.last_output else ""

    key_hex = params.get("key")
    if key_hex:
        key = bytes.fromhex(key_hex)
        signature = sign_message(str(payload), key)
    else:
        signature = sign_message(str(payload))

    ctx.variables["_signature"] = signature

    return {
        "payload":   payload,
        "signature": signature,
        "algorithm": "hmac-sha256",
    }


def cap_handler(target: list[str], params: dict, ctx) -> Any:
    """
    CAP — Declare capability scope for the current agent.

    Metadata annotation: records what this agent is allowed to do.
    Used by the validator in production mode to enforce role-based limits.

    Usage:
        CAP.agent_name(role=data, allow=[ING,CLN,XFRM])
    """
    agent = ".".join(target) if target else "self"
    role  = params.get("role", "worker")
    allow = params.get("allow", [])

    if isinstance(allow, str):
        allow = [v.strip() for v in allow.split(",")]

    cap_entry = {"agent": agent, "role": role, "capabilities": allow}
    ctx.variables.setdefault("_capabilities", {})[agent] = cap_entry

    return cap_entry
