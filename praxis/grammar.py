"""
Praxis grammar definition and parse-tree-to-AST transformer.

Design decisions:
  - VERB  = /[A-Z][A-Z0-9]{1,7}/   (all-caps, 2-8 chars)
  - IDENTIFIER = /[a-z_][a-zA-Z0-9_]*/  (lowercase start — avoids VERB/IDENTIFIER
    terminal ambiguity; verbs are always all-caps by convention)
  - $varname  — variable reference; SET.name captures prior step output
  - {...} blocks — multi-step bodies in IF/LOOP/PLAN declarations
  - PLAN:name { ... } — named reusable plan, called via CALL.planname
  - Expressions — full comparison ops, AND/OR/NOT, function conditions
  - List values — [VERB, VERB] for CAP.allow and similar params
  - // comments — ignored by parser
"""

from __future__ import annotations

from lark import Lark, Transformer, Token, Tree, v_args
from praxis.ast_types import (
    Program, Chain, VerbAction, ParBlock, IfStmt, LoopStmt,
    Block, GoalDecl, PlanDecl, Skip, Break, Wait,
    VarRef, NamedCond, FuncCond, Comparison, OrExpr, AndExpr, NotExpr,
    Param,
)

# ──────────────────────────────────────────────────────────────────────────────
# Grammar
# ──────────────────────────────────────────────────────────────────────────────

SHAUN_GRAMMAR = r"""
    // ── Top-level ──────────────────────────────────────────────────────────────
    program:    statement+

    ?statement: goal_decl
              | plan_decl
              | chain
              | if_stmt
              | loop_stmt
              | action

    // ── Declarations ───────────────────────────────────────────────────────────
    goal_decl:  "GOAL:" IDENTIFIER
    plan_decl:  "PLAN:" IDENTIFIER block

    // ── Chains and actions ─────────────────────────────────────────────────────
    // A chain is 2+ actions joined by ->
    // A single action is promoted directly (no Chain wrapper for lone actions)
    chain:      action ("->" action)+

    ?action:    verb_action
              | par_block
              | if_stmt
              | loop_stmt
              | "SKIP"    -> skip
              | "BREAK"   -> break_stmt
              | "WAIT"    -> wait_stmt

    // target is optional — allows bare verbs like MERGE, SKIP-like usage in chains
    verb_action: VERB ("." target)? ("(" params ")")?

    par_block:  "PAR(" action ("," action)+ ")"

    // ── Control flow ───────────────────────────────────────────────────────────
    // body = single action OR a multi-step {...} block
    // IF single-action:  IF.cond -> OUT.telegram ELSE -> SKIP
    // IF multi-step:     IF.cond -> { ING.data -> CLN.null -> OUT.result } ELSE -> SKIP
    //
    // Priority .10 ensures the Earley forest resolver picks if_stmt over a chain
    // when it sees "IF." — both are grammatically valid parses of "IF.x -> Y".
    if_stmt.10: "IF." expr "->" body ("ELSE" "->" body)?

    // LOOP single-action: LOOP(EVAL.metric, until=$score > 0.9)
    // LOOP multi-step:    LOOP({ ING.data -> TRN.lstm -> SET.score }, until=$score > 0.9)
    loop_stmt:  "LOOP(" body "," "until" "=" expr ")"

    ?body:      block | action

    block:      "{" statement+ "}"

    // ── Expressions ────────────────────────────────────────────────────────────
    // Precedence (low → high): OR < AND < NOT < atom
    ?expr:      or_expr

    or_expr:    and_expr ("OR" and_expr)*
    and_expr:   not_expr ("AND" not_expr)*
    ?not_expr:  "NOT" atom_expr   -> not_expr
              | atom_expr

    ?atom_expr: comparison
              | func_cond
              | var_ref           -> var_cond
              | IDENTIFIER        -> named_cond
              | "(" expr ")"

    // Comparisons: $score > 0.9  |  $price < 200  |  $status == "ok"
    comparison: operand COMP_OP operand

    // Function conditions: price_drop(threshold=200)  |  state_changed(key=denver)
    func_cond:  IDENTIFIER "(" params ")"

    ?operand:   var_ref
              | NUMBER            -> num_operand
              | ESCAPED_STRING    -> str_operand
              | PERCENTAGE        -> pct_operand
              | IDENTIFIER        -> id_operand

    // ── Values and references ──────────────────────────────────────────────────
    // $varname — reference a variable set by SET.varname
    var_ref:    "$" IDENTIFIER

    target:     IDENTIFIER ("." IDENTIFIER)*

    params:     param ("," param)*
    param:      IDENTIFIER "=" value

    ?value:     ESCAPED_STRING    -> string_val
              | PERCENTAGE        -> pct_val
              | NUMBER            -> num_val
              | var_ref           -> var_val
              | list_val
              | IDENTIFIER        -> word_val

    // List values: [SEARCH, SUMM, GEN] — used in CAP.allow, MERGE.src, etc.
    list_val:   "[" (IDENTIFIER ("," IDENTIFIER)*)? "]"

    // ── Terminals ──────────────────────────────────────────────────────────────
    // COMP_OP defined before VERB to avoid any priority clash
    COMP_OP:    ">=" | "<=" | "!=" | "==" | ">" | "<"

    // Percentages before NUMBER to avoid partial match
    PERCENTAGE: /[0-9]+(\.[0-9]+)?%/

    // VERB = all-caps 2-8 chars. String literals like "IF." "GOAL:" "PAR("
    // have higher Lark priority and are matched first in their grammar positions.
    VERB:       /[A-Z][A-Z0-9]{1,7}/

    // IDENTIFIER starts with lowercase or underscore — no overlap with VERB
    IDENTIFIER: /[a-z_][a-zA-Z0-9_]*/

    %import common.NUMBER
    %import common.ESCAPED_STRING
    %import common.WS
    %import common.NEWLINE
    %ignore WS
    %ignore NEWLINE
    %ignore /\/\/.*/
"""

# ──────────────────────────────────────────────────────────────────────────────
# Parser factory
# ──────────────────────────────────────────────────────────────────────────────

_PARSER: Lark | None = None


def make_parser(force_rebuild: bool = False) -> Lark:
    """Return a cached Lark parser instance."""
    global _PARSER
    if _PARSER is None or force_rebuild:
        _PARSER = Lark(
            SHAUN_GRAMMAR,
            start="program",
            parser="earley",
            ambiguity="resolve",
        )
    return _PARSER


# ──────────────────────────────────────────────────────────────────────────────
# AST Transformer — converts Lark parse tree → typed AST nodes
# ──────────────────────────────────────────────────────────────────────────────

@v_args(inline=True)
class ShaunTransformer(Transformer):

    # ── Top-level ───────────────────────────────────────────────────────────────

    def program(self, *statements):
        return Program(statements=list(statements))

    # ── Declarations ────────────────────────────────────────────────────────────

    def goal_decl(self, name):
        return GoalDecl(name=str(name))

    def plan_decl(self, name, body):
        return PlanDecl(name=str(name), body=body)

    # ── Chains and actions ──────────────────────────────────────────────────────

    def chain(self, *steps):
        return Chain(steps=list(steps))

    def verb_action(self, verb, target=None, params=None):
        v = str(verb)
        # SKIP/BREAK/WAIT may be parsed as verb_action when target is optional.
        # Map them back to their dedicated AST nodes.
        if v == "SKIP":
            return Skip()
        if v == "BREAK":
            return Break()
        if v == "WAIT":
            return Wait()
        # target may be absent for bare verbs like MERGE, JOIN, ROLLBACK, etc.
        if isinstance(target, dict):
            # called as verb_action(verb, params) — target was omitted
            params = target
            target = []
        return VerbAction(
            verb=v,
            target=target or [],
            params=params or {},
        )

    def par_block(self, *branches):
        return ParBlock(branches=list(branches))

    def skip(self):
        return Skip()

    def break_stmt(self):
        return Break()

    def wait_stmt(self):
        return Wait()

    # ── Control flow ────────────────────────────────────────────────────────────

    def if_stmt(self, condition, then_body, else_body=None):
        return IfStmt(condition=condition, then_body=then_body, else_body=else_body)

    def loop_stmt(self, body, until):
        return LoopStmt(body=body, until=until)

    def block(self, *statements):
        return Block(statements=list(statements))

    # ── Expressions ─────────────────────────────────────────────────────────────

    def or_expr(self, *operands):
        if len(operands) == 1:
            return operands[0]
        return OrExpr(operands=list(operands))

    def and_expr(self, *operands):
        if len(operands) == 1:
            return operands[0]
        return AndExpr(operands=list(operands))

    def not_expr(self, operand):
        return NotExpr(operand=operand)

    def named_cond(self, name):
        return NamedCond(name=str(name))

    def var_cond(self, var_ref):
        return var_ref  # VarRef node used directly as a boolean condition

    def func_cond(self, name, params=None):
        return FuncCond(name=str(name), params=params or {})

    def comparison(self, left, op, right):
        return Comparison(left=left, op=str(op), right=right)

    # Operands
    def num_operand(self, n):
        s = str(n)
        return float(s) if "." in s else int(s)

    def str_operand(self, s):
        return str(s)[1:-1]  # strip surrounding quotes

    def pct_operand(self, p):
        return str(p)  # keep as "5%" string; runtime parses to float

    def id_operand(self, name):
        return str(name)

    # ── Values ──────────────────────────────────────────────────────────────────

    def var_ref(self, name):
        return VarRef(name=str(name))

    def target(self, *parts):
        return [str(p) for p in parts]

    def params(self, *pairs):
        return dict(pairs)

    def param(self, key, value):
        return (str(key), value)

    def string_val(self, s):
        return str(s)[1:-1]  # strip quotes

    def num_val(self, n):
        s = str(n)
        return float(s) if "." in s else int(s)

    def pct_val(self, p):
        return str(p)

    def var_val(self, var_ref):
        return var_ref

    def word_val(self, w):
        return str(w)

    def list_val(self, *items):
        return list(str(i) for i in items)


# ──────────────────────────────────────────────────────────────────────────────
# Public parse function
# ──────────────────────────────────────────────────────────────────────────────

def parse(source: str) -> Program:
    """Parse a Shaun source string and return a typed AST Program."""
    parser = make_parser()
    tree = parser.parse(source)
    return ShaunTransformer().transform(tree)
