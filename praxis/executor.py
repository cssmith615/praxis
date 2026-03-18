"""
Praxis executor.

Walks a validated Program AST and dispatches each VerbAction to a registered
handler function. Returns a list of ExecutionResult dicts — one per step.

Execution semantics:
  Chain     — steps run sequentially; each step receives the prior step's output
              as ctx.last_output; SET.varname captures last_output into ctx.variables
  PAR(...)  — all branches run in parallel via ThreadPoolExecutor; results merged
  IF/ELSE   — condition evaluated; matching body executed; unmatched = []
  LOOP      — body re-executed until `until` condition is True or MAX_LOOP_DEPTH hit
  Block     — all statements executed in order
  BREAK     — raises BreakSignal; caught by LOOP executor
  SKIP      — returns a skipped result immediately; no handler call
  WAIT      — stub for Sprint 1; returns an ok result with no output

Persistent variables (Sprint 27):
  SET.varname(persist=true)  — writes variable to ~/.praxis/kv.db for cross-run persistence
  LOAD.varname               — reads from ~/.praxis/kv.db into ctx.variables[varname]

Handler contract:
  def my_handler(target: list[str], params: dict, ctx: ExecutionContext) -> Any:
      ...
      return output  # any value; stored as ExecutionResult["output"]

  Handlers must not raise unless they want to mark the step as "error".
  Return None for no meaningful output.
"""

from __future__ import annotations

import json as _json
import sqlite3 as _sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeout, as_completed
from pathlib import Path as _Path
from typing import Any, Literal, TypedDict

from praxis.handlers.audit import AssertionFailure, GateRejected

from praxis.ast_types import (
    Program, Chain, VerbAction, ParBlock, IfStmt, LoopStmt,
    Block, GoalDecl, PlanDecl, Skip, Break, Wait,
    VarRef, NamedCond, FuncCond, Comparison, OrExpr, AndExpr, NotExpr,
)

# ──────────────────────────────────────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────────────────────────────────────

class ExecutionResult(TypedDict):
    verb: str
    target: list[str]
    params: dict
    output: Any
    status: Literal["ok", "error", "skipped"]
    duration_ms: int
    log_entry: str          # AgentRx-compatible structured log string


# ──────────────────────────────────────────────────────────────────────────────
# Execution context — shared across a single program run
# ──────────────────────────────────────────────────────────────────────────────

class ExecutionContext:
    def __init__(
        self,
        mode: str = "dev",
        memory: Any = None,
        handlers: Any = None,
        timeout_seconds: float | None = None,
        max_step_ms: int | None = None,
        max_output_bytes: int | None = None,
    ) -> None:
        self.variables: dict[str, Any] = {}
        self.last_output: Any = None
        self.log: list[ExecutionResult] = []
        self.plan_registry: dict[str, PlanDecl] = {}
        self.mode = mode
        self.memory = memory            # optional ProgramMemory for SEARCH verb
        self.prev_verb_action: Any = None  # last non-RETRY VerbAction; used by RETRY
        self.agent_registry: Any = None    # AgentRegistry, populated by SPAWN
        self.pending_futures: dict = {}    # msg_id → (agent_id, Future, timeout)
        self._handlers = handlers          # passed to spawned workers
        # Resource limits
        self.timeout_seconds = timeout_seconds
        self.max_step_ms = max_step_ms
        self.max_output_bytes = max_output_bytes
        self._start_time: float = time.monotonic()
        # CAP enforcement — None means unrestricted
        self._cap_allow: set[str] | None = None

    def set_var(self, name: str, value: Any) -> None:
        self.variables[name] = value

    def get_var(self, name: str) -> Any:
        if name not in self.variables:
            raise ShaunRuntimeError(f"Undefined variable: ${name}")
        return self.variables[name]


# ──────────────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────────────

class ShaunRuntimeError(Exception):
    pass


class ResourceLimitExceeded(ShaunRuntimeError):
    """Raised when a resource limit (time or output size) is exceeded."""


class CapabilityViolation(ShaunRuntimeError):
    """Raised when a verb is executed outside the declared CAP allow-list."""


class UnregisteredVerbError(ShaunRuntimeError):
    pass


# Verbs that are always allowed even when a CAP allow-list is active.
# These are native executor constructs, not handler-dispatched operations.
_CAP_NATIVE_VERBS: frozenset[str] = frozenset({
    "SET", "LOAD", "CALL", "RETRY", "ROLLBACK", "CAP",
})

# ── Persistent KV helpers (SET persist=true / LOAD) ──────────────────────────
_KV_DB_PATH = _Path.home() / ".praxis" / "kv.db"
_KV_VAR_NS  = "praxis_var::"   # namespace prefix to avoid collisions


def _kv_write(name: str, value: Any) -> None:
    """Write a variable to the persistent KV store."""
    _KV_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _sqlite3.connect(str(_KV_DB_PATH))
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO kv (key, value, updated_at) VALUES (?, ?, ?)",
            (_KV_VAR_NS + name, _json.dumps(value), time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )
        conn.commit()
    finally:
        conn.close()


def _kv_read(name: str) -> Any:
    """Read a variable from the persistent KV store. Returns None if not found."""
    if not _KV_DB_PATH.exists():
        return None
    conn = _sqlite3.connect(str(_KV_DB_PATH))
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        row = conn.execute(
            "SELECT value FROM kv WHERE key = ?", (_KV_VAR_NS + name,)
        ).fetchone()
    finally:
        conn.close()
    return _json.loads(row[0]) if row else None


class RetryExhausted(ShaunRuntimeError):
    """Raised when RETRY exhausts all attempts without success."""
    def __init__(self, verb: str, attempts: int):
        super().__init__(f"RETRY: '{verb}' failed after {attempts} attempt(s)")
        self.verb = verb
        self.attempts = attempts


class _BreakSignal(Exception):
    """Internal — raised by BREAK, caught by LOOP executor."""


# ──────────────────────────────────────────────────────────────────────────────
# Executor
# ──────────────────────────────────────────────────────────────────────────────

MAX_LOOP_DEPTH = 10


class Executor:
    """
    Execute a validated Shaun Program AST.

    Parameters
    ----------
    handlers : dict[str, Callable]
        Map of VERB → handler function.
    mode : "dev" | "prod"
        In prod mode, GATE is enforced at runtime (not just statically).
    """

    def __init__(self, handlers: dict, mode: str = "dev") -> None:
        self.handlers = handlers
        self.mode = mode

    def execute(
        self,
        program: Program,
        memory: Any = None,
        timeout_seconds: float | None = None,
        max_step_ms: int | None = None,
        max_output_bytes: int | None = None,
        cap_allow: set[str] | None = None,
        initial_variables: dict[str, Any] | None = None,
    ) -> list[ExecutionResult]:
        ctx = ExecutionContext(
            mode=self.mode,
            memory=memory,
            handlers=self.handlers,
            timeout_seconds=timeout_seconds,
            max_step_ms=max_step_ms,
            max_output_bytes=max_output_bytes,
        )
        if cap_allow is not None:
            ctx._cap_allow = set(cap_allow)
        if initial_variables:
            ctx.variables.update(initial_variables)

        # Register all PLAN declarations so CALL can find them
        for stmt in program.statements:
            if isinstance(stmt, PlanDecl):
                ctx.plan_registry[stmt.name] = stmt

        results: list[ExecutionResult] = []
        for stmt in program.statements:
            if isinstance(stmt, (GoalDecl, PlanDecl)):
                continue
            results.extend(self._exec(stmt, ctx))

        return results

    # ── Node dispatch ──────────────────────────────────────────────────────────

    def _exec(self, node: Any, ctx: ExecutionContext) -> list[ExecutionResult]:
        if isinstance(node, Chain):
            return self._exec_chain(node, ctx)
        if isinstance(node, VerbAction):
            return [self._exec_verb(node, ctx)]
        if isinstance(node, ParBlock):
            return self._exec_par(node, ctx)
        if isinstance(node, IfStmt):
            return self._exec_if(node, ctx)
        if isinstance(node, LoopStmt):
            return self._exec_loop(node, ctx)
        if isinstance(node, Block):
            return self._exec_block(node, ctx)
        if isinstance(node, Skip):
            r = _make_result("SKIP", [], {}, None, "skipped", 0, "SKIP → no-op")
            ctx.log.append(r)
            return [r]
        if isinstance(node, Wait):
            import time as _time
            secs = float(node.params.get("seconds", 0)) if hasattr(node, "params") else 0
            if secs > 0:
                _time.sleep(secs)
            r = _make_result("WAIT", [], {}, {"slept_seconds": secs}, "ok", int(secs * 1000), f"WAIT → {secs}s")
            ctx.log.append(r)
            return [r]
        if isinstance(node, Break):
            raise _BreakSignal()
        if isinstance(node, (GoalDecl, PlanDecl)):
            return []
        raise ShaunRuntimeError(f"Unhandled AST node type: {type(node).__name__}")

    # ── Chain ──────────────────────────────────────────────────────────────────

    def _exec_chain(self, chain: Chain, ctx: ExecutionContext) -> list[ExecutionResult]:
        results: list[ExecutionResult] = []
        for step in chain.steps:
            step_results = self._exec(step, ctx)
            if step_results:
                ctx.last_output = step_results[-1].get("output")
            results.extend(step_results)
        return results

    # ── Verb action ────────────────────────────────────────────────────────────

    def _exec_verb(self, action: VerbAction, ctx: ExecutionContext) -> ExecutionResult:
        verb = action.verb

        # SET captures last_output into a named variable — no handler needed
        if verb == "SET":
            var_name = action.target[0] if action.target else "_"
            ctx.set_var(var_name, ctx.last_output)
            persist = str(action.params.get("persist", "")).lower() in ("true", "1", "yes")
            if persist:
                _kv_write(var_name, ctx.last_output)
            log = f"SET.{var_name} ← {ctx.last_output!r}" + (" [persisted]" if persist else "")
            r = _make_result("SET", action.target, action.params, ctx.last_output, "ok", 0, log)
            ctx.log.append(r)
            return r

        # LOAD reads a persisted variable from kv.db into ctx.variables
        if verb == "LOAD":
            var_name = action.target[0] if action.target else "_"
            value = _kv_read(var_name)
            ctx.set_var(var_name, value)
            ctx.last_output = value
            log = f"LOAD.{var_name} → {value!r}"
            r = _make_result("LOAD", action.target, {}, value, "ok", 0, log)
            ctx.log.append(r)
            return r

        # CALL executes a named PLAN
        if verb == "CALL":
            plan_name = action.target[0] if action.target else ""
            if plan_name not in ctx.plan_registry:
                raise ShaunRuntimeError(f"CALL: no plan named '{plan_name}'")
            plan = ctx.plan_registry[plan_name]
            call_results = self._exec_block(plan.body, ctx)
            ctx.last_output = call_results[-1]["output"] if call_results else None
            return call_results[-1] if call_results else _make_result(
                "CALL", action.target, {}, None, "ok", 0, f"CALL.{plan_name} → empty plan"
            )

        # RETRY — re-execute prev action with backoff (native, no handler needed)
        if verb == "RETRY":
            return self._exec_retry(action, ctx)

        # ROLLBACK — restore ctx from a SNAP checkpoint (native, no handler needed)
        if verb == "ROLLBACK":
            return self._exec_rollback(action, ctx)

        # CAP enforcement — check verb against declared allow-list
        if ctx._cap_allow is not None and verb not in _CAP_NATIVE_VERBS:
            if verb not in ctx._cap_allow:
                raise CapabilityViolation(
                    f"Verb '{verb}' is not in the declared capability set "
                    f"{sorted(ctx._cap_allow)}. Add it to CAP.allow or remove the step."
                )

        # Resolve $var references in params before dispatch
        resolved = self._resolve_params(action.params, ctx)

        if verb not in self.handlers:
            raise UnregisteredVerbError(
                f"No handler registered for verb '{verb}'. "
                "Add it to handlers/__init__.py."
            )

        # Wall-clock budget check before dispatching next step
        if ctx.timeout_seconds is not None:
            elapsed = time.monotonic() - ctx._start_time
            if elapsed >= ctx.timeout_seconds:
                raise ResourceLimitExceeded(
                    f"Program timeout: exceeded {ctx.timeout_seconds}s wall-clock limit"
                )

        handler = self.handlers[verb]
        start = time.monotonic()
        try:
            if ctx.max_step_ms is not None:
                step_timeout = ctx.max_step_ms / 1000.0
                _pool = ThreadPoolExecutor(max_workers=1)
                _fut = _pool.submit(handler, action.target, resolved, ctx)
                _pool.shutdown(wait=False)  # don't block on thread cleanup
                try:
                    output = _fut.result(timeout=step_timeout)
                except _FutureTimeout:
                    duration_ms = int((time.monotonic() - start) * 1000)
                    raise ResourceLimitExceeded(
                        f"{verb} exceeded per-step limit of {ctx.max_step_ms}ms"
                    )
            else:
                output = handler(action.target, resolved, ctx)

            duration_ms = int((time.monotonic() - start) * 1000)

            # Output size enforcement
            if ctx.max_output_bytes is not None and output is not None:
                try:
                    out_bytes = len(_json.dumps(output).encode())
                except (TypeError, ValueError):
                    out_bytes = len(str(output).encode())
                if out_bytes > ctx.max_output_bytes:
                    raise ResourceLimitExceeded(
                        f"{verb} output size {out_bytes} bytes exceeds limit of {ctx.max_output_bytes} bytes"
                    )

            target_str = ".".join(action.target)
            log = f"{verb}.{target_str} -> ok ({duration_ms}ms)"
            r = _make_result(verb, action.target, resolved, output, "ok", duration_ms, log)
        except (AssertionFailure, GateRejected, ResourceLimitExceeded, CapabilityViolation):
            raise  # these halt the chain — do not swallow
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            target_str = ".".join(action.target)
            log = f"{verb}.{target_str} -> error: {exc}"
            r = _make_result(verb, action.target, resolved, None, "error", duration_ms, log)

        ctx.last_output = r["output"]
        ctx.log.append(r)
        ctx.prev_verb_action = action  # track for RETRY
        return r

    # ── RETRY ──────────────────────────────────────────────────────────────────

    def _exec_retry(self, action: VerbAction, ctx: ExecutionContext) -> ExecutionResult:
        """Re-execute the previous VerbAction with backoff until success or exhausted."""
        # Nothing to retry if last step succeeded
        last = ctx.log[-1] if ctx.log else None
        if last is None or last["status"] != "error":
            r = _make_result("RETRY", [], action.params, ctx.last_output, "ok", 0,
                             "RETRY -> no prior failure, skipped")
            ctx.log.append(r)
            return r

        prev = ctx.prev_verb_action
        if prev is None:
            raise ShaunRuntimeError("RETRY: no prior verb action recorded in context")

        attempts    = int(action.params.get("attempts", 3))
        backoff_mode = action.params.get("backoff", "exp")

        for attempt in range(attempts):
            wait_s = _backoff_seconds(attempt, backoff_mode)
            if wait_s > 0:
                time.sleep(wait_s)
            result = self._exec_verb(prev, ctx)
            if result["status"] == "ok":
                r = _make_result("RETRY", [], action.params, result["output"], "ok",
                                 0, f"RETRY -> '{prev.verb}' succeeded on attempt {attempt + 1}/{attempts}")
                ctx.log.append(r)
                return r
            ctx.log.append(result)

        raise RetryExhausted(prev.verb, attempts)

    # ── ROLLBACK ───────────────────────────────────────────────────────────────

    def _exec_rollback(self, action: VerbAction, ctx: ExecutionContext) -> ExecutionResult:
        """Restore ctx.variables and ctx.last_output from a named SNAP checkpoint."""
        checkpoint = action.target[0] if action.target else action.params.get("to", "default")
        snap_db = _Path.home() / ".praxis" / "snaps.db"

        if not snap_db.exists():
            raise ShaunRuntimeError("ROLLBACK: no snap database found — run SNAP first")

        conn = _sqlite3.connect(str(snap_db))
        row = conn.execute(
            "SELECT variables, last_output FROM snapshots WHERE name = ?", (checkpoint,)
        ).fetchone()
        conn.close()

        if row is None:
            raise ShaunRuntimeError(f"ROLLBACK: checkpoint '{checkpoint}' not found")

        ctx.variables   = _json.loads(row[0])
        ctx.last_output = _json.loads(row[1])
        r = _make_result("ROLLBACK", [checkpoint], {}, {"restored": checkpoint},
                         "ok", 0, f"ROLLBACK -> restored checkpoint '{checkpoint}'")
        ctx.log.append(r)
        return r

    # ── PAR ────────────────────────────────────────────────────────────────────

    def _exec_par(self, par: ParBlock, ctx: ExecutionContext) -> list[ExecutionResult]:
        results: list[ExecutionResult] = []
        with ThreadPoolExecutor(max_workers=len(par.branches)) as pool:
            futures = {
                pool.submit(self._exec, branch, ctx): i
                for i, branch in enumerate(par.branches)
            }
            # Collect in submission order for deterministic output
            ordered = sorted(futures.items(), key=lambda kv: kv[1])
            for future, _ in ordered:
                results.extend(future.result())
        if results:
            ctx.last_output = results[-1]["output"]
        return results

    # ── IF ─────────────────────────────────────────────────────────────────────

    def _exec_if(self, stmt: IfStmt, ctx: ExecutionContext) -> list[ExecutionResult]:
        if self._eval_expr(stmt.condition, ctx):
            return self._exec(stmt.then_body, ctx)
        elif stmt.else_body is not None:
            return self._exec(stmt.else_body, ctx)
        return []

    # ── LOOP ───────────────────────────────────────────────────────────────────

    def _exec_loop(self, stmt: LoopStmt, ctx: ExecutionContext) -> list[ExecutionResult]:
        results: list[ExecutionResult] = []
        for _ in range(MAX_LOOP_DEPTH):
            if self._eval_expr(stmt.until, ctx):
                break
            try:
                results.extend(self._exec(stmt.body, ctx))
            except _BreakSignal:
                break
        return results

    # ── Block ──────────────────────────────────────────────────────────────────

    def _exec_block(self, block: Block, ctx: ExecutionContext) -> list[ExecutionResult]:
        results: list[ExecutionResult] = []
        for stmt in block.statements:
            results.extend(self._exec(stmt, ctx))
        return results

    # ── Expression evaluation ──────────────────────────────────────────────────

    def _eval_expr(self, expr: Any, ctx: ExecutionContext) -> bool:
        if isinstance(expr, NamedCond):
            return bool(ctx.variables.get(expr.name, False))
        if isinstance(expr, FuncCond):
            # Stub: function conditions default to False; Sprint 2 will wire these
            # to real evaluators via the handler registry.
            return bool(ctx.variables.get(expr.name, False))
        if isinstance(expr, VarRef):
            return bool(ctx.get_var(expr.name))
        if isinstance(expr, Comparison):
            left = self._resolve_operand(expr.left, ctx)
            right = self._resolve_operand(expr.right, ctx)
            return _compare(left, expr.op, right)
        if isinstance(expr, OrExpr):
            return any(self._eval_expr(e, ctx) for e in expr.operands)
        if isinstance(expr, AndExpr):
            return all(self._eval_expr(e, ctx) for e in expr.operands)
        if isinstance(expr, NotExpr):
            return not self._eval_expr(expr.operand, ctx)
        return False

    def _resolve_operand(self, operand: Any, ctx: ExecutionContext) -> Any:
        if isinstance(operand, VarRef):
            return ctx.get_var(operand.name)
        return operand

    def _resolve_params(self, params: dict, ctx: ExecutionContext) -> dict:
        resolved = {}
        for k, v in params.items():
            if isinstance(v, VarRef):
                resolved[k] = ctx.get_var(v.name)
            elif isinstance(v, list):
                resolved[k] = [
                    ctx.get_var(i.name) if isinstance(i, VarRef) else i
                    for i in v
                ]
            else:
                resolved[k] = v
        return resolved


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_result(
    verb: str,
    target: list[str],
    params: dict,
    output: Any,
    status: str,
    duration_ms: int,
    log_entry: str,
) -> ExecutionResult:
    return {
        "verb": verb,
        "target": target,
        "params": params,
        "output": output,
        "status": status,
        "duration_ms": duration_ms,
        "log_entry": log_entry,
    }


def _backoff_seconds(attempt: int, mode: str) -> float:
    """Return seconds to wait before attempt N (attempt 0 = first retry = no wait)."""
    if attempt == 0:
        return 0.0
    if mode == "exp":
        return min(2.0 ** attempt, 30.0)   # 2, 4, 8, 16, 30 …
    if mode == "linear":
        return float(attempt) * 2.0         # 2, 4, 6, 8 …
    if mode == "fixed":
        return 5.0
    return 0.0


def _compare(left: Any, op: str, right: Any) -> bool:
    _ops = {
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }
    try:
        return _ops[op](left, right)
    except TypeError:
        return False
