"""
Error handlers — Sprint 5 full implementations.

ERR      declare a failure point; write structured entry to ~/.praxis/errors.log;
         optionally set a recovery plan name in ctx.variables['_recover_plan']

RETRY    — handled natively by Executor._exec_retry (not dispatched here)
ROLLBACK — handled natively by Executor._exec_rollback (not dispatched here)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_PRAXIS_DIR  = Path.home() / ".praxis"
_ERR_LOG     = _PRAXIS_DIR / "errors.log"


def err_handler(target: list[str], params: dict, ctx) -> Any:
    """
    ERR — Declare a failure point.

    Writes a structured JSON entry to ~/.praxis/errors.log.
    Sets ctx.variables['last_error'] with the error dict.
    If params['recover'] is set, stores the plan name in
    ctx.variables['_recover_plan'] for downstream CALL to pick up.

    Usage:
        FETCH.api -> ERR.fetch_failed(msg="API unreachable", code=503)
        FETCH.api -> ERR.fetch_failed(msg="API unreachable", recover=handle_outage)
    """
    label = ".".join(target) if target else "error"
    msg   = params.get("msg", label)
    code  = str(params.get("code", "ERR"))
    recover = params.get("recover")

    entry = {
        "timestamp": time.time(),
        "label":     label,
        "code":      code,
        "message":   msg,
        "last_output": ctx.last_output,
        "variables": {k: v for k, v in ctx.variables.items() if not k.startswith("_")},
    }

    _PRAXIS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_ERR_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")

    print(f"[ERR] {code}: {msg}")

    ctx.variables["last_error"] = entry
    if recover:
        ctx.variables["_recover_plan"] = str(recover)

    return entry
