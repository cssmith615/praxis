"""
Audit handlers: VALIDATE, ASSERT, LOG, GATE, SNAP, ANNOTATE, ROUTE

Sprint 1 stubs — Sprint 4 full implementations.
LOG and ANNOTATE are functional now (write to ctx.log / print).
"""

from __future__ import annotations
import json
import time
from typing import Any


def validate_handler(target: list[str], params: dict, ctx) -> Any:
    """VALIDATE — Schema-check prior step output. Sprint 4: jsonschema."""
    return {"valid": True, "stub": True}


def assert_handler(target: list[str], params: dict, ctx) -> Any:
    """ASSERT — Halt chain if condition is false. Sprint 4: real evaluation."""
    condition = ".".join(target)
    return {"asserted": condition, "passed": True, "stub": True}


def log_handler(target: list[str], params: dict, ctx) -> Any:
    """LOG — Structured JSON entry to execution.log. Functional in Sprint 1."""
    entry = {
        "timestamp": time.time(),
        "label": ".".join(target),
        "data": params.get("data", ctx.last_output),
        "params": params,
    }
    print(f"[LOG] {json.dumps(entry, default=str)}")
    return entry


def gate_handler(target: list[str], params: dict, ctx) -> Any:
    """
    GATE — Pause for human confirmation.

    Sprint 1: auto-approves in dev mode. Sprint 4: blocks and waits for
    Y/N response via channel (Telegram / terminal).
    """
    action = ".".join(target) if target else "continue"
    msg = params.get("msg", f"Approve {action}?")
    print(f"[GATE] {msg} (auto-approved in dev mode)")
    return {"gate": action, "approved": True, "mode": "dev-auto"}


def snap_handler(target: list[str], params: dict, ctx) -> Any:
    """SNAP — Serialize execution state to named checkpoint. Sprint 5: SQLite."""
    checkpoint = ".".join(target) if target else "snap_default"
    snapshot = {
        "checkpoint": checkpoint,
        "variables": dict(ctx.variables),
        "last_output": ctx.last_output,
    }
    return {"snapped": checkpoint, "snapshot": snapshot}


def annotate_handler(target: list[str], params: dict, ctx) -> Any:
    """ANNOTATE — Write human-readable label to log. Zero execution effect."""
    label = ".".join(target)
    msg = params.get("msg", label)
    print(f"[ANNOTATE] {msg}")
    return {"annotated": label}


def route_handler(target: list[str], params: dict, ctx) -> Any:
    """ROUTE — Evaluate output, dispatch to named handler. Sprint 4 implementation."""
    destination = ".".join(target)
    return {"routed_to": destination, "input": ctx.last_output, "stub": True}
