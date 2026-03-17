"""
Deploy handlers — Sprint 4 subprocess implementations.

BUILD  run a build command via subprocess, return structured output
DEP    run a deploy command; GATE required in prod mode (validator enforces statically)
TEST   run pytest (or custom cmd), parse pass/fail/error counts from output
"""
from __future__ import annotations

import re
import subprocess
from typing import Any


def _run(cmd: str, cwd: str | None = None, timeout: int = 120) -> dict:
    """Run a shell command and return structured output."""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=timeout,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "success": result.returncode == 0,
    }


def build_handler(target: list[str], params: dict, ctx) -> Any:
    """BUILD — Run a build command. Provide cmd= or defaults to 'build <target>'."""
    artifact = ".".join(target) if target else "default"
    cmd = params.get("cmd", f"build {artifact}")
    cwd = params.get("cwd")
    timeout = int(params.get("timeout", 120))
    result = _run(cmd, cwd=cwd, timeout=timeout)
    return {
        "artifact": artifact,
        "status": "built" if result["success"] else "failed",
        **result,
    }


def dep_handler(target: list[str], params: dict, ctx) -> Any:
    """DEP — Deploy artifact. GATE required in prod mode (enforced by validator)."""
    artifact = ".".join(target) if target else "default"
    env = params.get("env", "dev")
    cmd = params.get("cmd", f"deploy {artifact} --env {env}")
    cwd = params.get("cwd")
    timeout = int(params.get("timeout", 180))
    result = _run(cmd, cwd=cwd, timeout=timeout)
    return {
        "artifact": artifact,
        "env": env,
        "status": "deployed" if result["success"] else "failed",
        **result,
    }


def test_handler(target: list[str], params: dict, ctx) -> Any:
    """TEST — Run test suite. Defaults to pytest. Parses pass/fail/error counts."""
    suite = ".".join(target) if target else ""
    cmd = params.get("cmd", f"pytest {suite} -q" if suite else "pytest -q")
    cwd = params.get("cwd")
    timeout = int(params.get("timeout", 300))
    result = _run(cmd, cwd=cwd, timeout=timeout)

    # Parse pytest summary line
    passed = failed = errors = 0
    if result["stdout"]:
        m = re.search(r"(\d+) passed", result["stdout"])
        if m:
            passed = int(m.group(1))
        m = re.search(r"(\d+) failed", result["stdout"])
        if m:
            failed = int(m.group(1))
        m = re.search(r"(\d+) error", result["stdout"])
        if m:
            errors = int(m.group(1))

    return {
        "suite": suite or "all",
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "success": result["success"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }
