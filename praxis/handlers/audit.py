"""
Audit handlers — Sprint 4 full implementations.

VALIDATE  jsonschema check on prior step output
ASSERT    halt chain if condition is false  (raises AssertionFailure)
LOG       structured JSON entry to ~/.praxis/execution.log + console
GATE      terminal Y/N confirmation (dev: prompt; prod: same but GateRejected halts chain)
SNAP      serialize execution state to named SQLite checkpoint (~/.praxis/snaps.db)
ANNOTATE  write human-readable label to log (zero execution effect)
ROUTE     evaluate output value, set a routing variable in ctx
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Exception types — re-exported so callers can catch them
# ─────────────────────────────────────────────────────────────────────────────

class AssertionFailure(RuntimeError):
    """Raised by ASSERT when the condition evaluates to false. Halts the chain."""
    def __init__(self, condition: str, message: str = ""):
        detail = f" — {message}" if message else ""
        super().__init__(f"ASSERT failed: {condition}{detail}")
        self.condition = condition


class GateRejected(RuntimeError):
    """Raised by GATE when the operator responds N."""
    def __init__(self, action: str):
        super().__init__(f"GATE rejected by operator: {action}")
        self.action = action


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_PRAXIS_DIR = Path.home() / ".praxis"
_LOG_PATH   = _PRAXIS_DIR / "execution.log"
_SNAP_DB    = _PRAXIS_DIR / "snaps.db"


def _get_snap_conn() -> sqlite3.Connection:
    _PRAXIS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_SNAP_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            name        TEXT PRIMARY KEY,
            variables   TEXT NOT NULL,
            last_output TEXT,
            created_at  REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# jsonschema — optional, only needed for VALIDATE
# ─────────────────────────────────────────────────────────────────────────────

try:
    import jsonschema as _jsonschema
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False


# ─────────────────────────────────────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────────────────────────────────────

def validate_handler(target: list[str], params: dict, ctx) -> Any:
    """VALIDATE — Schema-check prior step output against a JSON Schema dict."""
    if not _HAS_JSONSCHEMA:
        raise ImportError("VALIDATE requires jsonschema: pip install jsonschema")
    schema = params.get("schema")
    if schema is None:
        is_valid = ctx.last_output is not None
        return {"valid": is_valid, "output": ctx.last_output}
    instance = ctx.last_output
    try:
        _jsonschema.validate(instance=instance, schema=schema)
        return {"valid": True, "output": instance}
    except _jsonschema.ValidationError as e:
        return {"valid": False, "error": e.message, "output": instance}


def assert_handler(target: list[str], params: dict, ctx) -> Any:
    """ASSERT — Halt chain if condition is false. Raises AssertionFailure."""
    condition = ".".join(target) if target else params.get("condition", "")
    msg = params.get("msg", params.get("message", ""))

    # Evaluate: named variable → bool, else truthiness of last_output
    var_name = condition.lstrip("$")
    if var_name and var_name in ctx.variables:
        result = bool(ctx.variables[var_name])
    elif condition:
        # Literal true/false string
        result = condition.lower() not in ("false", "0", "no", "none", "")
    else:
        result = bool(ctx.last_output)

    if not result:
        raise AssertionFailure(condition or "last_output", msg)
    return {"asserted": condition or "last_output", "passed": True}


def log_handler(target: list[str], params: dict, ctx) -> Any:
    """LOG — Structured JSON entry to ~/.praxis/execution.log + console."""
    entry = {
        "timestamp": time.time(),
        "label": ".".join(target),
        "data": params.get("data", ctx.last_output),
        "params": {k: v for k, v in params.items() if k != "data"},
    }
    line = json.dumps(entry, default=str)
    print(f"[LOG] {line}")
    _PRAXIS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return entry


def gate_handler(target: list[str], params: dict, ctx) -> Any:
    """
    GATE — Pause for human confirmation.

    Dev mode:  prompts at the terminal (Y/n). Empty input = Y.
    Prod mode: same prompt, but GateRejected raises and halts the entire chain.
    """
    action = ".".join(target) if target else "continue"
    msg = params.get("msg", f"GATE: approve '{action}'? [Y/n] ")
    try:
        response = input(msg).strip().lower()
    except (EOFError, KeyboardInterrupt):
        response = "n"

    approved = response in ("", "y", "yes")
    if not approved:
        raise GateRejected(action)
    return {"gate": action, "approved": True}


def snap_handler(target: list[str], params: dict, ctx) -> Any:
    """SNAP — Serialize current execution state to a named SQLite checkpoint."""
    name = ".".join(target) if target else params.get("name", "default")
    snapshot_vars = dict(ctx.variables)
    conn = _get_snap_conn()
    conn.execute(
        "INSERT OR REPLACE INTO snapshots (name, variables, last_output, created_at) VALUES (?, ?, ?, ?)",
        (
            name,
            json.dumps(snapshot_vars, default=str),
            json.dumps(ctx.last_output, default=str),
            time.time(),
        ),
    )
    conn.commit()
    conn.close()
    return {"snapped": name, "variables": list(snapshot_vars.keys()), "db": str(_SNAP_DB)}


def annotate_handler(target: list[str], params: dict, ctx) -> Any:
    """ANNOTATE — Write human-readable label to log. Zero execution effect."""
    label = ".".join(target)
    msg = params.get("msg", label)
    print(f"[ANNOTATE] {msg}")
    return {"annotated": label}


def route_handler(target: list[str], params: dict, ctx) -> Any:
    """
    ROUTE — Evaluate last_output and set a routing variable in ctx.

    Usage:  ROUTE.dest(match=success, else=fallback)
    Sets ctx.variables['dest'] = 'success' or 'fallback'.
    Downstream IF.dest or CALL.dest can then branch on this value.
    """
    var_name = target[0] if target else "route"
    current = ctx.last_output
    match_value = params.get("match")
    else_value = params.get("else", "unmatched")

    if match_value is not None and str(current) == str(match_value):
        routed_to = str(match_value)
    else:
        routed_to = str(else_value)

    ctx.set_var(var_name, routed_to)
    return {"route_var": var_name, "routed_to": routed_to, "input": current}
