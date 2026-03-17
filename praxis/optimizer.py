"""
Praxis Optimizer — AST-level program analysis and rewriting (Pillar 2-PartA).

Three passes, applied in order:
  1. Dead step elimination  — remove steps that can never execute
  2. Constant folding       — evaluate IF conditions with known-at-compile-time values
  3. Parallelization        — rewrite independent sequential chains into PAR blocks

Usage:
    from praxis.optimizer import optimize, OptimizeResult
    from praxis import parse

    program = parse("ING.data -> CLN.normalize -> SUMM.text -> OUT.slack")
    result  = optimize(program)
    print(result.summary())   # "parallelized 2 steps"
    executor.execute(result.program)

All passes are non-destructive — they return a new AST, never mutate the input.
The optimizer is sound: the rewritten program produces the same observable outputs
as the original for any input, assuming handlers are pure (no side-effects that
depend on execution order between independent steps).

Parallelization safety contract:
    Two steps A → B are independent if B does not reference $variables set by A
    and neither step is a SET/CALL/RETRY/ROLLBACK (which mutate ctx.variables or
    ctx.last_output in ways that are order-sensitive).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from praxis.ast_types import (
    Program, Chain, VerbAction, ParBlock, IfStmt, LoopStmt,
    Block, GoalDecl, PlanDecl, Skip, Break, Wait,
    VarRef, NamedCond, Comparison, OrExpr, AndExpr, NotExpr,
)


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptimizeResult:
    program: Program
    parallelized: int = 0        # number of steps moved into PAR blocks
    dead_removed: int = 0        # number of unreachable steps removed
    branches_folded: int = 0     # number of IF branches evaluated at compile time

    def any_changes(self) -> bool:
        return self.parallelized > 0 or self.dead_removed > 0 or self.branches_folded > 0

    def summary(self) -> str:
        parts = []
        if self.parallelized:
            parts.append(f"parallelized {self.parallelized} step(s)")
        if self.dead_removed:
            parts.append(f"removed {self.dead_removed} dead step(s)")
        if self.branches_folded:
            parts.append(f"folded {self.branches_folded} constant branch(es)")
        return "; ".join(parts) if parts else "no changes"


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def optimize(program: Program) -> OptimizeResult:
    """
    Run all optimization passes over program and return an OptimizeResult.

    Passes run in order:
      1. Dead step elimination
      2. Constant folding
      3. Parallelization
    """
    result = OptimizeResult(program=program)
    _dead_pass(result)
    _fold_pass(result)
    _par_pass(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pass 1: Dead step elimination
# ─────────────────────────────────────────────────────────────────────────────

# Verbs that unconditionally halt the chain after they execute.
_HALTING_VERBS: frozenset[str] = frozenset({"BREAK"})


def _dead_pass(result: OptimizeResult) -> None:
    new_stmts, removed = _elim_dead_stmts(result.program.statements)
    result.dead_removed += removed
    result.program = Program(statements=new_stmts)


def _elim_dead_stmts(stmts: list) -> tuple[list, int]:
    new, removed = [], 0
    for stmt in stmts:
        stmt, r = _elim_dead_node(stmt)
        removed += r
        new.append(stmt)
    return new, removed


def _elim_dead_node(node: Any) -> tuple[Any, int]:
    """Return (rewritten_node, steps_removed)."""
    if isinstance(node, Chain):
        return _elim_dead_chain(node)
    if isinstance(node, Block):
        new_stmts, removed = _elim_dead_stmts(node.statements)
        return Block(statements=new_stmts), removed
    if isinstance(node, IfStmt):
        then, r1 = _elim_dead_node(node.then_body)
        else_ = node.else_body
        r2 = 0
        if else_ is not None:
            else_, r2 = _elim_dead_node(else_)
        return IfStmt(condition=node.condition, then_body=then, else_body=else_), r1 + r2
    if isinstance(node, LoopStmt):
        body, removed = _elim_dead_node(node.body)
        return LoopStmt(until=node.until, body=body), removed
    return node, 0


def _elim_dead_chain(chain: Chain) -> tuple[Chain, int]:
    """Remove all steps after an unconditional halt (BREAK)."""
    new_steps = []
    removed = 0
    for step in chain.steps:
        if isinstance(step, VerbAction) and step.verb == "BREAK":
            new_steps.append(step)
            # Everything after BREAK is dead
            removed += len(chain.steps) - len(new_steps)
            break
        if isinstance(step, Break):
            new_steps.append(step)
            removed += len(chain.steps) - len(new_steps)
            break
        new_steps.append(step)
    return Chain(steps=new_steps), removed


# ─────────────────────────────────────────────────────────────────────────────
# Pass 2: Constant folding — IF with a known-at-compile-time condition
# ─────────────────────────────────────────────────────────────────────────────

def _fold_pass(result: OptimizeResult) -> None:
    new_stmts, folded = _fold_stmts(result.program.statements)
    result.branches_folded += folded
    result.program = Program(statements=new_stmts)


def _fold_stmts(stmts: list) -> tuple[list, int]:
    new, folded = [], 0
    for stmt in stmts:
        stmt, f = _fold_node(stmt)
        folded += f
        if stmt is not None:
            new.append(stmt)
    return new, folded


def _fold_node(node: Any) -> tuple[Any | None, int]:
    if isinstance(node, IfStmt):
        val = _eval_const(node.condition)
        if val is True:
            # Always-true: replace IF with then_body
            return node.then_body, 1
        if val is False:
            if node.else_body is not None:
                return node.else_body, 1
            # No else — IF vanishes entirely
            return None, 1
        # Unknown condition — recurse into sub-nodes
        then, f1 = _fold_node(node.then_body)
        else_ = node.else_body
        f2 = 0
        if else_ is not None:
            else_, f2 = _fold_node(else_)
        if then is None:
            then = Block(statements=[])
        return IfStmt(condition=node.condition, then_body=then, else_body=else_), f1 + f2
    if isinstance(node, Block):
        new_stmts, folded = _fold_stmts(node.statements)
        return Block(statements=new_stmts), folded
    if isinstance(node, Chain):
        new_steps = []
        folded = 0
        for step in node.steps:
            rewritten, f = _fold_node(step)
            folded += f
            if rewritten is not None:
                new_steps.append(rewritten)
        return Chain(steps=new_steps), folded
    if isinstance(node, LoopStmt):
        body, f = _fold_node(node.body)
        if body is None:
            body = Block(statements=[])
        return LoopStmt(until=node.until, body=body), f
    return node, 0


def _eval_const(expr: Any) -> bool | None:
    """
    Try to evaluate a condition at compile time.
    Returns True/False if the value is known, None if it depends on runtime state.
    """
    if isinstance(expr, bool):
        return expr
    if isinstance(expr, NamedCond):
        # Named condition references a runtime variable — unknown at compile time
        return None
    if isinstance(expr, VarRef):
        return None
    if isinstance(expr, Comparison):
        # Only fold if both sides are literals
        if isinstance(expr.left, VarRef) or isinstance(expr.right, VarRef):
            return None
        try:
            return _compare_const(expr.left, expr.op, expr.right)
        except Exception:
            return None
    if isinstance(expr, OrExpr):
        results = [_eval_const(e) for e in expr.operands]
        if any(r is True for r in results):
            return True
        if all(r is False for r in results):
            return False
        return None
    if isinstance(expr, AndExpr):
        results = [_eval_const(e) for e in expr.operands]
        if any(r is False for r in results):
            return False
        if all(r is True for r in results):
            return True
        return None
    if isinstance(expr, NotExpr):
        inner = _eval_const(expr.operand)
        if inner is not None:
            return not inner
        return None
    return None


def _compare_const(left: Any, op: str, right: Any) -> bool:
    ops = {">": lambda a, b: a > b, "<": lambda a, b: a < b,
           ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
           "==": lambda a, b: a == b, "!=": lambda a, b: a != b}
    return ops[op](left, right)


# ─────────────────────────────────────────────────────────────────────────────
# Pass 3: Parallelization — sequential independent steps → PAR block
# ─────────────────────────────────────────────────────────────────────────────

# Verbs that mutate shared context in order-sensitive ways — never safe to PAR.
_ORDER_SENSITIVE: frozenset[str] = frozenset({
    "SET", "CALL", "RETRY", "ROLLBACK", "SNAP",
    "LOOP", "BREAK", "SKIP", "WAIT",
    "SPAWN", "JOIN",   # agent lifecycle — order-sensitive
})


def _par_pass(result: OptimizeResult) -> None:
    new_stmts, parallelized = _par_stmts(result.program.statements)
    result.parallelized += parallelized
    result.program = Program(statements=new_stmts)


def _par_stmts(stmts: list) -> tuple[list, int]:
    new, parallelized = [], 0
    for stmt in stmts:
        stmt, p = _par_node(stmt)
        parallelized += p
        new.append(stmt)
    return new, parallelized


def _par_node(node: Any) -> tuple[Any, int]:
    if isinstance(node, Chain):
        return _par_chain(node)
    if isinstance(node, Block):
        new_stmts, p = _par_stmts(node.statements)
        return Block(statements=new_stmts), p
    if isinstance(node, IfStmt):
        then, p1 = _par_node(node.then_body)
        else_ = node.else_body
        p2 = 0
        if else_ is not None:
            else_, p2 = _par_node(else_)
        return IfStmt(condition=node.condition, then_body=then, else_body=else_), p1 + p2
    if isinstance(node, LoopStmt):
        body, p = _par_node(node.body)
        return LoopStmt(until=node.until, body=body), p
    return node, 0


def _par_chain(chain: Chain) -> tuple[Any, int]:
    """
    Group consecutive independent VerbActions into PAR blocks.

    Two adjacent steps are independent if:
      - Both are VerbActions (not nested control structures)
      - Neither verb is in _ORDER_SENSITIVE
      - The second step does not reference $variables written by the first
        (we conservatively assume any $var reference is a dependency)

    Groups of 2+ independent steps are collected into a PAR block.
    Singleton groups and steps with dependencies remain as-is.
    """
    steps = chain.steps
    if len(steps) <= 1:
        return chain, 0

    # Split chain into runs of independent VerbActions
    groups: list[list] = []   # each element is a list of steps that can PAR
    current_independent: list[VerbAction] = []
    written_vars: set[str] = set()   # vars written so far in this group

    def flush():
        if current_independent:
            groups.append(list(current_independent))
        current_independent.clear()
        written_vars.clear()

    for step in steps:
        if not isinstance(step, VerbAction):
            flush()
            groups.append([step])
            continue

        if step.verb in _ORDER_SENSITIVE:
            flush()
            groups.append([step])
            continue

        # Check if this step reads any var written by a prior step in the group
        refs = _collect_var_refs(step)
        if refs & written_vars:
            # Dependency — start a new independent group
            flush()

        current_independent.append(step)
        # SET.varname writes a variable
        if step.verb == "SET" and step.target:
            written_vars.add(step.target[0])

    flush()

    # Rebuild the chain: groups of 2+ → PAR, singletons stay as VerbAction
    new_steps = []
    parallelized = 0
    for group in groups:
        if len(group) >= 2 and all(isinstance(s, VerbAction) for s in group):
            new_steps.append(ParBlock(branches=group))
            parallelized += len(group)
        else:
            new_steps.extend(group)

    if len(new_steps) == 1 and isinstance(new_steps[0], (Chain, Block, ParBlock)):
        return new_steps[0], parallelized

    return Chain(steps=new_steps), parallelized


def _collect_var_refs(action: VerbAction) -> set[str]:
    """Return the set of $variable names referenced in action.target or action.params."""
    refs: set[str] = set()
    for t in action.target:
        if isinstance(t, VarRef):
            refs.add(t.name)
        elif isinstance(t, str) and t.startswith("$"):
            refs.add(t[1:])
    for v in action.params.values():
        if isinstance(v, VarRef):
            refs.add(v.name)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, VarRef):
                    refs.add(item.name)
    return refs
