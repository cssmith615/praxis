"""Agent handlers: SPAWN, MSG, CAST, JOIN, SIGN, CAP — Sprint 1 stubs."""

from typing import Any


def spawn_handler(target: list[str], params: dict, ctx) -> Any:
    role = params.get("role", "worker")
    return {"agent_id": f"agent-{role}-stub", "role": role, "status": "spawned"}


def msg_handler(target: list[str], params: dict, ctx) -> Any:
    """MSG — Send Shaun message to agent. Sprint 6 wires Redis pub/sub."""
    recipient = ".".join(target)
    payload = params.get("program", ctx.last_output)
    return {"to": recipient, "payload": payload, "delivered": False, "stub": True}


def cast_handler(target: list[str], params: dict, ctx) -> Any:
    """CAST — Broadcast to all registered workers."""
    return {"broadcast": True, "payload": ctx.last_output, "stub": True}


def join_handler(target: list[str], params: dict, ctx) -> Any:
    """JOIN — Wait for all outstanding WAIT results."""
    return {"joined": True, "results": ctx.last_output, "stub": True}


def sign_handler(target: list[str], params: dict, ctx) -> Any:
    """SIGN — HMAC-SHA256 sign message. Sprint 6 uses real vault secrets."""
    import hashlib
    payload = str(ctx.last_output)
    stub_sig = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return {"payload": payload, "signature": stub_sig, "stub": True}


def cap_handler(target: list[str], params: dict, ctx) -> Any:
    """CAP — Declare capability scope for an agent."""
    role = params.get("role", "unknown")
    allow = params.get("allow", [])
    return {"agent": ".".join(target), "role": role, "capabilities": allow}
