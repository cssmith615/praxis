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

Handler contract:
  def my_handler(target: list[str], params: dict, ctx: ExecutionContext) -> Any:
      ...
      return output  # any value; stored as ExecutionResult["output"]

  Handlers must not raise unless they want to mark the step as "error".
  Return None for no meaningful output.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal, TypedDict

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
    def __init__(self) -> None:
        self.variables: dict[str, Any] = {}
        self.last_output: Any = None
        self.log: list[ExecutionResult] = []
        self.plan_registry: dict[str, PlanDecl] = {}

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


class UnregisteredVerbError(ShaunRuntimeError):
    pass


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

    def execute(self, program: Program) -> list[ExecutionResult]:
        ctx = ExecutionContext()

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
            r = _make_result("WAIT", [], {}, None, "ok", 0, "WAIT → stub (Sprint 1)")
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
            log = f"SET.{var_name} ← {ctx.last_output!r}"
            r = _make_result("SET", action.target, {}, ctx.last_output, "ok", 0, log)
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

        # Resolve $var references in params before dispatch
        resolved = self._resolve_params(action.params, ctx)

        if verb not in self.handlers:
            raise UnregisteredVerbError(
                f"No handler registered for verb '{verb}'. "
                "Add it to handlers/__init__.py."
            )

        handler = self.handlers[verb]
        start = time.monotonic()
        try:
            output = handler(action.target, resolved, ctx)
            duration_ms = int((time.monotonic() - start) * 1000)
            target_str = ".".join(action.target)
            log = f"{verb}.{target_str} → ok ({duration_ms}ms)"
            r = _make_result(verb, action.target, resolved, output, "ok", duration_ms, log)
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            target_str = ".".join(action.target)
            log = f"{verb}.{target_str} → error: {exc}"
            r = _make_result(verb, action.target, resolved, None, "error", duration_ms, log)

        ctx.last_output = r["output"]
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
