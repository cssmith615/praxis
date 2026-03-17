"""Error handlers: ERR, RETRY, ROLLBACK — Sprint 1 stubs. Sprint 5 full implementation."""

from typing import Any


def err_handler(target: list[str], params: dict, ctx) -> Any:
    """ERR — Declare failure point and write to error log."""
    msg = params.get("msg", ".".join(target))
    print(f"[ERR] {msg}")
    return {"error": msg, "handled": True}


def retry_handler(target: list[str], params: dict, ctx) -> Any:
    """RETRY — Re-execute prior step with backoff. Sprint 5 wires real retry logic."""
    backoff = params.get("backoff", "exp")
    max_attempts = params.get("max", 3)
    return {"retried": True, "backoff": backoff, "max_attempts": max_attempts, "stub": True}


def rollback_handler(target: list[str], params: dict, ctx) -> Any:
    """ROLLBACK — Restore execution state to named SNAP. Sprint 5 full implementation."""
    checkpoint = ".".join(target) if target else "last"
    return {"rolled_back_to": checkpoint, "stub": True}
