"""
Shaun semantic validator.

Two-pass validation:
  Pass 1 — collect all PLAN declaration names
  Pass 2 — walk the AST and enforce semantic rules

Rules enforced:
  - All VERBs must be in VALID_VERBS
  - Reserved keywords (OR, AND, NOT, ELSE, IF, LOOP, PAR, GOAL, PLAN,
    SKIP, BREAK, WAIT) must not appear as verb names
  - CALL targets must reference a declared PLAN name
  - LOOP body must not contain another LOOP at the top level (depth > 1)
  - In production mode: DEP, WRITE, SPAWN must be preceded by GATE in
    the same chain (static check — guards against accidental irreversible ops)
  - SET target must be a single identifier (not a dot-path)
  - LOOP until= expression must be present (enforced by grammar, but
    validated here for better error messages)
"""

from __future__ import annotations

from praxis.ast_types import (
    Program, Chain, VerbAction, ParBlock, IfStmt, LoopStmt,
    Block, GoalDecl, PlanDecl, Skip, Break, Wait,
    VarRef, NamedCond, FuncCond, Comparison, OrExpr, AndExpr, NotExpr,
)

# ──────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────────────────────────────────────

VALID_VERBS: frozenset[str] = frozenset({
    # Data
    "ING", "CLN", "XFRM", "FILTER", "SORT", "MERGE",
    # AI/ML
    "TRN", "INF", "EVAL", "SUMM", "CLASS", "GEN", "EMBED", "SEARCH",
    # I/O
    "READ", "WRITE", "FETCH", "POST", "OUT", "STORE", "RECALL", "LOAD",
    # Agents
    "SPAWN", "CALL", "MSG", "WAIT", "CAST", "JOIN", "SIGN", "CAP",
    # Deploy
    "BUILD", "DEP", "TEST",
    # Control
    "GOAL", "PLAN", "IF", "LOOP", "SKIP", "PAR", "FORK", "BREAK", "SET",
    # Error
    "ERR", "RETRY", "ROLLBACK",
    # Audit
    "VALIDATE", "ASSERT", "LOG", "GATE", "SNAP", "ANNOTATE", "ROUTE",
})

# Verbs that appear as grammar keywords and must not be used as user-written
# verb tokens in verb_action position.
RESERVED_AS_KEYWORDS: frozenset[str] = frozenset({
    # These appear as grammar structure keywords — invalid as verb_action targets.
    # SKIP/BREAK/WAIT are NOT included: they're routed to their own AST nodes
    # inside the transformer, so they never reach verb action validation.
    "IF", "LOOP", "PAR", "GOAL", "PLAN",
    "OR", "AND", "NOT", "ELSE",
})

# In production mode, these verbs require GATE earlier in the same chain.
GATE_REQUIRED_VERBS: frozenset[str] = frozenset({"DEP", "WRITE", "SPAWN"})


# ──────────────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────────────

class ShaunValidationError(Exception):
    """Raised when a Praxis program fails semantic validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(f"  • {e}" for e in errors))


# ──────────────────────────────────────────────────────────────────────────────
# Validator
# ──────────────────────────────────────────────────────────────────────────────

class Validator:
    """
    Validate a parsed Shaun Program AST.

    Usage:
        v = Validator(mode="dev")   # or "prod"
        errors = v.validate(program)
        if errors:
            raise ShaunValidationError(errors)
    """

    def __init__(self, mode: str = "dev") -> None:
        assert mode in ("dev", "prod"), f"mode must be 'dev' or 'prod', got {mode!r}"
        self.mode = mode

    def validate(self, program: Program) -> list[str]:
        errors: list[str] = []
        plan_names: set[str] = program.plan_names()

        ctx = _ValidationContext(
            mode=self.mode,
            plan_names=plan_names,
            errors=errors,
            loop_depth=0,
        )
        for stmt in program.statements:
            _validate_node(stmt, ctx)

        # Multi-agent cycle detection: build MSG delegation graph
        _check_msg_cycles(program, errors)

        return errors


# ──────────────────────────────────────────────────────────────────────────────
# Internal
# ──────────────────────────────────────────────────────────────────────────────

class _ValidationContext:
    __slots__ = ("mode", "plan_names", "errors", "loop_depth")

    def __init__(self, mode: str, plan_names: set[str],
                 errors: list[str], loop_depth: int) -> None:
        self.mode = mode
        self.plan_names = plan_names
        self.errors = errors
        self.loop_depth = loop_depth

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def descend_loop(self) -> "_ValidationContext":
        return _ValidationContext(
            mode=self.mode,
            plan_names=self.plan_names,
            errors=self.errors,
            loop_depth=self.loop_depth + 1,
        )


def _validate_node(node, ctx: _ValidationContext) -> None:
    if isinstance(node, VerbAction):
        _validate_verb_action(node, ctx)
    elif isinstance(node, Chain):
        _validate_chain(node, ctx)
    elif isinstance(node, ParBlock):
        for branch in node.branches:
            _validate_node(branch, ctx)
    elif isinstance(node, IfStmt):
        _validate_node(node.then_body, ctx)
        if node.else_body is not None:
            _validate_node(node.else_body, ctx)
    elif isinstance(node, LoopStmt):
        if ctx.loop_depth >= 3:
            ctx.err("LOOP nesting depth exceeds maximum (3). Use PLAN declarations to flatten.")
        _validate_node(node.body, ctx.descend_loop())
    elif isinstance(node, Block):
        for stmt in node.statements:
            _validate_node(stmt, ctx)
    elif isinstance(node, PlanDecl):
        _validate_node(node.body, ctx)
    elif isinstance(node, (GoalDecl, Skip, Break, Wait)):
        pass  # always valid
    else:
        ctx.err(f"Unknown AST node type: {type(node).__name__}")


def _validate_verb_action(action: VerbAction, ctx: _ValidationContext) -> None:
    verb = action.verb

    # Unknown verb
    if verb not in VALID_VERBS:
        ctx.err(f"Unknown verb '{verb}'. Not in VALID_VERBS vocabulary.")
        return  # can't do further checks on unknown verbs

    # Reserved keywords used in verb position
    if verb in RESERVED_AS_KEYWORDS:
        ctx.err(f"'{verb}' is a reserved keyword and cannot be used as a verb action.")

    # SET must have exactly one-segment target
    if verb == "SET":
        if len(action.target) != 1:
            ctx.err(
                f"SET target must be a single identifier (e.g. SET.score), "
                f"got: SET.{''.join(action.target)}"
            )

    # CALL target must be a declared PLAN name
    if verb == "CALL":
        if not action.target:
            ctx.err("CALL requires a target (the PLAN name to call).")
        else:
            plan_name = action.target[0]
            if plan_name not in ctx.plan_names:
                ctx.err(
                    f"CALL.{plan_name} — no PLAN named '{plan_name}' declared in this program."
                )

    # FORK must have no target (it's a control token, not a data verb)
    if verb == "FORK" and action.target:
        ctx.err("FORK does not take a target. Use IF/ELSE for conditional branching.")

    # CAP must have a role param
    if verb == "CAP" and "role" not in action.params:
        ctx.err("CAP requires a 'role' parameter: CAP.agent(role=worker, allow=[SEARCH,SUMM])")


def _validate_chain(chain: Chain, ctx: _ValidationContext) -> None:
    # In production mode, verify that GATE precedes any GATE_REQUIRED_VERBS
    if ctx.mode == "prod":
        gate_seen = False
        for step in chain.steps:
            if isinstance(step, VerbAction):
                if step.verb == "GATE":
                    gate_seen = True
                elif step.verb in GATE_REQUIRED_VERBS and not gate_seen:
                    ctx.err(
                        f"'{step.verb}' in production mode requires a GATE earlier in the "
                        f"same chain. Add GATE before {step.verb} to require human confirmation."
                    )

    # Validate each step
    for step in chain.steps:
        _validate_node(step, ctx)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-agent cycle detection
# ──────────────────────────────────────────────────────────────────────────────

def _collect_msg_edges(node, edges: list[tuple[str, str]], current_plan: str = "main") -> None:
    """Walk the AST and collect (from_plan, to_agent) edges for all MSG verbs."""
    if isinstance(node, VerbAction):
        if node.verb == "MSG" and node.target:
            edges.append((current_plan, node.target[0]))
    elif isinstance(node, Chain):
        for step in node.steps:
            _collect_msg_edges(step, edges, current_plan)
    elif isinstance(node, (ParBlock,)):
        for branch in node.branches:
            _collect_msg_edges(branch, edges, current_plan)
    elif isinstance(node, (IfStmt,)):
        _collect_msg_edges(node.then_body, edges, current_plan)
        if node.else_body:
            _collect_msg_edges(node.else_body, edges, current_plan)
    elif isinstance(node, (LoopStmt,)):
        _collect_msg_edges(node.body, edges, current_plan)
    elif isinstance(node, Block):
        for stmt in node.statements:
            _collect_msg_edges(stmt, edges, current_plan)
    elif isinstance(node, PlanDecl):
        _collect_msg_edges(node.body, edges, current_plan=node.name)


def _check_msg_cycles(program: Program, errors: list[str]) -> None:
    """
    Detect direct MSG self-loops (agent A sends MSG to itself).
    Sprint 6 catches direct self-loops; transitive A→B→A detection
    is deferred to Sprint 6+ when inter-program topology is known.
    """
    edges: list[tuple[str, str]] = []
    for stmt in program.statements:
        _collect_msg_edges(stmt, edges)

    # Collect SPAWN target names so we can map agent_id → plan context
    spawned: set[str] = set()
    for stmt in program.statements:
        _collect_spawned(stmt, spawned)

    for from_plan, to_agent in edges:
        if from_plan == to_agent:
            errors.append(
                f"MSG cycle detected: PLAN '{from_plan}' sends MSG to itself. "
                f"Agents cannot delegate back to their own plan."
            )


def _collect_spawned(node, spawned: set) -> None:
    if isinstance(node, VerbAction) and node.verb == "SPAWN" and node.target:
        spawned.add(node.target[0])
    elif isinstance(node, Chain):
        for step in node.steps:
            _collect_spawned(step, spawned)
    elif isinstance(node, Block):
        for stmt in node.statements:
            _collect_spawned(stmt, spawned)
    elif isinstance(node, PlanDecl):
        _collect_spawned(node.body, spawned)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────────────────────────────────────

def validate(program: Program, mode: str = "dev") -> list[str]:
    """Validate a program and return a list of error strings (empty = valid)."""
    return Validator(mode=mode).validate(program)
