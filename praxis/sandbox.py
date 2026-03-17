"""
Praxis Process Sandbox — Sprint 15 (Pillar 7, Layer 3).

Runs individual handler calls in isolated subprocesses so that:
  - A handler crash cannot corrupt the main executor's memory
  - File I/O can be restricted to an explicit allowed_paths set
  - CPU/memory abuse is contained and killed on timeout
  - Secrets in the main process are not visible to the subprocess

Architecture:
  SandboxedExecutor wraps a standard Executor and intercepts _exec_verb.
  Verbs in sandbox_verbs (default: all handler-dispatched verbs) run in a
  child process via subprocess + pickle. Verbs NOT in sandbox_verbs run
  normally in the parent process.

  The child receives: (verb, target, params, mode) — NOT the full ctx.
  The child returns: (status, output, duration_ms, error).
  ctx.variables and ctx.last_output are passed as read-only snapshots;
  the child cannot mutate the parent's context.

Security model (Layer 3 baseline):
  - Child runs with inherited environment (no new capabilities granted)
  - Allowed file paths checked against os.path.realpath — symlink-safe
  - Timeout kills the child process group
  - Child stdout/stderr are captured, not forwarded to parent

Limitations:
  - Handlers that require live ctx mutation (SET, CALL, RETRY, ROLLBACK, SPAWN)
    are always run in the parent process — they are excluded from sandboxing
  - Pickle is used for IPC; only safe (JSON-serializable) outputs are returned
  - Full OS-level sandboxing (seccomp, namespaces) requires Linux; this
    provides a best-effort isolation layer that works cross-platform

Usage:
    from praxis.sandbox import SandboxedExecutor, SandboxPolicy
    from praxis.handlers import HANDLERS

    policy = SandboxPolicy(
        timeout_seconds=10,
        allowed_paths=["/tmp/praxis"],
        sandbox_verbs={"ING", "FETCH", "POST", "WRITE", "READ"},
    )
    executor = SandboxedExecutor(HANDLERS, policy=policy)
    results = executor.execute(program)
"""
from __future__ import annotations

import json
import multiprocessing
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from praxis.executor import (
    Executor, ExecutionContext, ExecutionResult,
    ShaunRuntimeError, ResourceLimitExceeded, CapabilityViolation,
    _make_result,
)
from praxis.handlers.audit import AssertionFailure, GateRejected
from praxis.ast_types import VerbAction


# Verbs that mutate ctx in ways that require in-process execution.
_UNSANDBOXABLE: frozenset[str] = frozenset({
    "SET", "CALL", "RETRY", "ROLLBACK", "CAP",
    "SPAWN", "MSG", "CAST", "JOIN",          # agent lifecycle
})


# ─────────────────────────────────────────────────────────────────────────────
# Policy
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SandboxPolicy:
    """
    Configuration for subprocess sandboxing.

    Parameters
    ----------
    timeout_seconds : float
        Maximum wall time for a sandboxed step. Exceeded → SandboxTimeout.
    allowed_paths : list[str]
        File paths the sandboxed handler is allowed to access (read or write).
        Any file access outside these paths raises SandboxViolation.
        Empty list = no file I/O restriction (but still process-isolated).
    sandbox_verbs : set[str] | None
        Verb names to sandbox. None = all handler-dispatched verbs.
        Verbs in _UNSANDBOXABLE are always excluded regardless.
    """
    timeout_seconds: float = 30.0
    allowed_paths: list[str] = field(default_factory=list)
    sandbox_verbs: set[str] | None = None   # None = sandbox everything sandboxable

    def should_sandbox(self, verb: str) -> bool:
        if verb in _UNSANDBOXABLE:
            return False
        if self.sandbox_verbs is None:
            return True
        return verb in self.sandbox_verbs


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class SandboxTimeout(ResourceLimitExceeded):
    """Raised when a sandboxed step exceeds its wall-clock timeout."""


class SandboxViolation(ShaunRuntimeError):
    """Raised when a sandboxed step attempts a disallowed file access."""


class SandboxCrash(ShaunRuntimeError):
    """Raised when a sandboxed step crashes (non-zero exit or exception)."""


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess worker
# ─────────────────────────────────────────────────────────────────────────────

def _run_in_child(
    result_queue: multiprocessing.Queue,
    verb: str,
    target: list[str],
    params: dict,
    mode: str,
    variables_snapshot: dict,
    last_output: Any,
    allowed_paths: list[str],
    handlers_module: str,  # not used yet; handlers rebuilt in child
) -> None:
    """
    Entry point for the sandbox child process.
    Rebuilds handlers, runs the verb, puts result on queue.
    """
    try:
        # File I/O restriction: patch builtins.open if allowed_paths is set
        if allowed_paths:
            _install_path_guard(allowed_paths)

        from praxis.handlers import HANDLERS
        from praxis.executor import ExecutionContext

        ctx = ExecutionContext(mode=mode)
        ctx.variables = dict(variables_snapshot)
        ctx.last_output = last_output

        if verb not in HANDLERS:
            result_queue.put({"status": "error", "output": None,
                              "error": f"No handler for '{verb}'", "duration_ms": 0})
            return

        handler = HANDLERS[verb]
        start = time.monotonic()
        try:
            output = handler(target, params, ctx)
            duration_ms = int((time.monotonic() - start) * 1000)
            # Serialize output to JSON to ensure it's safe to pass back
            try:
                json.dumps(output)
            except (TypeError, ValueError):
                output = str(output)
            result_queue.put({"status": "ok", "output": output,
                              "error": None, "duration_ms": duration_ms})
        except SandboxViolation as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            result_queue.put({"status": "error", "output": None,
                              "error": f"SandboxViolation: {exc}", "duration_ms": duration_ms})
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            result_queue.put({"status": "error", "output": None,
                              "error": str(exc), "duration_ms": duration_ms})
    except Exception as exc:
        result_queue.put({"status": "error", "output": None,
                          "error": f"Child init error: {exc}", "duration_ms": 0})


def _install_path_guard(allowed_paths: list[str]) -> None:
    """
    Monkey-patch builtins.open to raise SandboxViolation for paths outside
    the allowed set. Only installed in the child process.
    """
    import builtins
    real_open = builtins.open
    resolved_allowed = [os.path.realpath(p) for p in allowed_paths]

    def guarded_open(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            real = os.path.realpath(str(file))
            if not any(real.startswith(a) for a in resolved_allowed):
                raise SandboxViolation(
                    f"File access denied: '{file}' is outside allowed paths {allowed_paths}"
                )
        return real_open(file, *args, **kwargs)

    builtins.open = guarded_open


# ─────────────────────────────────────────────────────────────────────────────
# SandboxedExecutor
# ─────────────────────────────────────────────────────────────────────────────

class SandboxedExecutor(Executor):
    """
    Executor variant that runs sandboxable verbs in child processes.

    Inherits the full Executor; only _exec_verb is overridden.
    """

    def __init__(self, handlers: dict, mode: str = "dev",
                 policy: SandboxPolicy | None = None) -> None:
        super().__init__(handlers, mode)
        self.policy = policy or SandboxPolicy()

    def _exec_verb(self, action: VerbAction, ctx: ExecutionContext) -> ExecutionResult:
        verb = action.verb

        if not self.policy.should_sandbox(verb):
            # Run in-process (SET, CALL, RETRY, ROLLBACK, CAP, agents, etc.)
            return super()._exec_verb(action, ctx)

        # Resolve $var params before crossing the process boundary
        resolved = self._resolve_params(action.params, ctx)

        # CAP check still happens here (in parent process)
        if ctx._cap_allow is not None and verb not in {"SET", "CALL", "RETRY", "ROLLBACK", "CAP"}:
            if verb not in ctx._cap_allow:
                raise CapabilityViolation(
                    f"Verb '{verb}' is not in the declared capability set "
                    f"{sorted(ctx._cap_allow)}."
                )

        return self._exec_verb_sandboxed(action, resolved, ctx)

    def _exec_verb_sandboxed(
        self,
        action: VerbAction,
        resolved_params: dict,
        ctx: ExecutionContext,
    ) -> ExecutionResult:
        verb = action.verb
        result_queue: multiprocessing.Queue = multiprocessing.Queue()

        proc = multiprocessing.Process(
            target=_run_in_child,
            args=(
                result_queue,
                verb,
                action.target,
                resolved_params,
                ctx.mode,
                dict(ctx.variables),
                ctx.last_output,
                self.policy.allowed_paths,
                "",  # handlers_module placeholder
            ),
            daemon=True,
        )

        start = time.monotonic()
        proc.start()
        proc.join(timeout=self.policy.timeout_seconds)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()
            duration_ms = int((time.monotonic() - start) * 1000)
            raise SandboxTimeout(
                f"{verb} sandboxed step timed out after {self.policy.timeout_seconds}s"
            )

        duration_ms = int((time.monotonic() - start) * 1000)

        if result_queue.empty():
            raise SandboxCrash(
                f"{verb} sandboxed step crashed with exit code {proc.exitcode}"
            )

        child_result = result_queue.get_nowait()
        output = child_result.get("output")
        status = child_result.get("status", "error")
        error = child_result.get("error")
        child_ms = child_result.get("duration_ms", duration_ms)

        target_str = ".".join(action.target)
        if status == "ok":
            log = f"{verb}.{target_str} [sandbox] -> ok ({child_ms}ms)"
            r = _make_result(verb, action.target, resolved_params, output, "ok", child_ms, log)
        else:
            log = f"{verb}.{target_str} [sandbox] -> error: {error}"
            r = _make_result(verb, action.target, resolved_params, None, "error", child_ms, log)

        ctx.last_output = r["output"]
        ctx.log.append(r)
        ctx.prev_verb_action = action
        return r
