"""Deploy handlers: BUILD, DEP, TEST — Sprint 1 stubs. Sprint 4 uses TS bridge."""

from typing import Any


def build_handler(target: list[str], params: dict, ctx) -> Any:
    artifact = ".".join(target)
    return {"artifact": artifact, "status": "built", "stub": True}


def dep_handler(target: list[str], params: dict, ctx) -> Any:
    """DEP — Deploy. Requires GATE in production mode (enforced by validator + executor)."""
    artifact = ".".join(target)
    env = params.get("env", "dev")
    return {"artifact": artifact, "env": env, "status": "deployed", "stub": True}


def test_handler(target: list[str], params: dict, ctx) -> Any:
    suite = ".".join(target) if target else "all"
    return {"suite": suite, "passed": 42, "failed": 0, "stub": True}
