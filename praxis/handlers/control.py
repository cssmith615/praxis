"""Control handlers: FORK — others (IF/LOOP/PAR/SET/CALL/SKIP/BREAK/WAIT) are executor-native."""

from typing import Any


def fork_handler(target: list[str], params: dict, ctx) -> Any:
    """FORK — conditional routing, evaluated at runtime. Sprint 1 stub."""
    return {"forked": True, "stub": True}
