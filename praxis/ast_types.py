"""
Typed AST nodes for the Praxis language.

All nodes are immutable dataclasses. Every node that carries a compound value
holds it in a typed field — no raw dicts or untyped trees escape this module.

Node hierarchy:
  ASTNode = GoalDecl | PlanDecl | Chain | VerbAction | ParBlock |
            IfStmt | LoopStmt | Block | Skip | Break | Wait
  Expr    = NamedCond | FuncCond | Comparison | OrExpr | AndExpr |
            NotExpr | VarRef
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Union


# ──────────────────────────────────────────────────────────────────────────────
# Aliases (defined at bottom after all classes)
# ──────────────────────────────────────────────────────────────────────────────

# Forward-declared — see bottom of file
ASTNode = Any
Expr = Any


# ──────────────────────────────────────────────────────────────────────────────
# Declarations
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GoalDecl:
    """GOAL:forecast_sales"""
    name: str


@dataclass
class PlanDecl:
    """PLAN:check_flights { ING.flights -> EVAL.price -> OUT.telegram }"""
    name: str
    body: "Block"


# ──────────────────────────────────────────────────────────────────────────────
# Actions and sequences
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Chain:
    """Two or more actions joined by ->"""
    steps: list[ASTNode]


@dataclass
class VerbAction:
    """ING.sales.db(format=csv)  |  SET.score  |  OUT.telegram(msg="done")"""
    verb: str
    target: list[str]          # dot-separated path segments
    params: dict[str, Any]     # key=value pairs; values may be VarRef


@dataclass
class ParBlock:
    """PAR(ING.sales, ING.marketing, ING.crm)"""
    branches: list[ASTNode]


@dataclass
class Block:
    """{ statement+ } — multi-step body for IF/LOOP/PLAN"""
    statements: list[ASTNode]


@dataclass
class Skip:
    """SKIP — no-op, used in ELSE branches"""


@dataclass
class Break:
    """BREAK — exit the enclosing LOOP"""


@dataclass
class Wait:
    """WAIT — suspend until a signal arrives (multi-agent)"""


# ──────────────────────────────────────────────────────────────────────────────
# Control flow
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IfStmt:
    """IF.expr -> body  |  IF.expr -> body ELSE -> body"""
    condition: Expr
    then_body: ASTNode
    else_body: ASTNode | None = None


@dataclass
class LoopStmt:
    """LOOP(body, until=expr)"""
    body: ASTNode
    until: Expr


# ──────────────────────────────────────────────────────────────────────────────
# Expressions (conditions, comparisons, boolean logic)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NamedCond:
    """A bare identifier used as a boolean condition: price_drop, ready, changed"""
    name: str


@dataclass
class FuncCond:
    """A function-style condition: price_drop(threshold=200), state_changed(key=denver)"""
    name: str
    params: dict[str, Any]


@dataclass
class Comparison:
    """left op right: $score > 0.9  |  $price < 200  |  $status == "ok" """
    left: Any           # VarRef | int | float | str | str(pct)
    op: str             # ">=" | "<=" | "!=" | "==" | ">" | "<"
    right: Any          # same


@dataclass
class OrExpr:
    """cond_a OR cond_b OR ..."""
    operands: list[Expr]


@dataclass
class AndExpr:
    """cond_a AND cond_b AND ..."""
    operands: list[Expr]


@dataclass
class NotExpr:
    """NOT cond"""
    operand: Expr


@dataclass
class VarRef:
    """$varname — reference a variable captured by SET.varname"""
    name: str


# ──────────────────────────────────────────────────────────────────────────────
# Convenience type alias (not enforced at runtime — for readability)
# ──────────────────────────────────────────────────────────────────────────────

# Kept as a string alias; actual isinstance checks use the concrete types above
Param = tuple[str, Any]   # (key, value) pair from params transformer

_AST_NODE_TYPES = (
    GoalDecl, PlanDecl, Chain, VerbAction, ParBlock, Block,
    IfStmt, LoopStmt, Skip, Break, Wait,
)

_EXPR_TYPES = (
    NamedCond, FuncCond, Comparison, OrExpr, AndExpr, NotExpr, VarRef,
)


@dataclass
class Program:
    """Root node — the full parsed Praxis program."""
    statements: list[ASTNode]

    def goals(self) -> list[GoalDecl]:
        return [s for s in self.statements if isinstance(s, GoalDecl)]

    def plans(self) -> list[PlanDecl]:
        return [s for s in self.statements if isinstance(s, PlanDecl)]

    def plan_names(self) -> set[str]:
        return {p.name for p in self.plans()}
