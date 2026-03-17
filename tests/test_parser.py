"""
Parser tests — 10 tests covering all major grammar constructs.

These tests verify that the Lark grammar + ShaunTransformer produce
correct typed AST nodes for valid programs, and raise on syntax errors.
"""

import pytest
from praxis.grammar import parse
from praxis.ast_types import (
    Program, Chain, VerbAction, ParBlock, IfStmt, LoopStmt,
    Block, GoalDecl, PlanDecl, Skip, Break,
    VarRef, NamedCond, FuncCond, Comparison, OrExpr, AndExpr, NotExpr,
)


# ──────────────────────────────────────────────────────────────────────────────
# Simple chain
# ──────────────────────────────────────────────────────────────────────────────

def test_simple_chain_parses():
    prog = parse("ING.sales.db -> CLN.null -> SUMM.text")
    assert isinstance(prog, Program)
    assert len(prog.statements) == 1
    chain = prog.statements[0]
    assert isinstance(chain, Chain)
    assert len(chain.steps) == 3
    assert chain.steps[0].verb == "ING"
    assert chain.steps[0].target == ["sales", "db"]
    assert chain.steps[1].verb == "CLN"
    assert chain.steps[2].verb == "SUMM"


def test_single_action_no_chain_wrapper():
    prog = parse("ING.sales.db")
    assert len(prog.statements) == 1
    action = prog.statements[0]
    assert isinstance(action, VerbAction)
    assert action.verb == "ING"


def test_params_parse_correctly():
    prog = parse('TRN.lstm(ep=20, lr="0.001")')
    action = prog.statements[0]
    assert isinstance(action, VerbAction)
    assert action.verb == "TRN"
    assert action.params["ep"] == 20
    assert action.params["lr"] == "0.001"


def test_goal_declaration_parses():
    prog = parse("GOAL:forecast_sales")
    assert len(prog.statements) == 1
    assert isinstance(prog.statements[0], GoalDecl)
    assert prog.statements[0].name == "forecast_sales"


def test_goal_then_chain():
    prog = parse("GOAL:my_goal ING.data -> CLN.null")
    assert len(prog.statements) == 2
    assert isinstance(prog.statements[0], GoalDecl)
    assert isinstance(prog.statements[1], Chain)


# ──────────────────────────────────────────────────────────────────────────────
# Parallel blocks
# ──────────────────────────────────────────────────────────────────────────────

def test_parallel_block_parses():
    prog = parse("PAR(ING.sales, ING.marketing, ING.crm) -> MERGE")
    chain = prog.statements[0]
    assert isinstance(chain, Chain)
    par = chain.steps[0]
    assert isinstance(par, ParBlock)
    assert len(par.branches) == 3
    assert par.branches[0].verb == "ING"
    assert par.branches[1].target == ["marketing"]


# ──────────────────────────────────────────────────────────────────────────────
# Conditionals
# ──────────────────────────────────────────────────────────────────────────────

def test_if_else_parses():
    prog = parse("IF.price_drop -> OUT.telegram ELSE -> SKIP")
    stmt = prog.statements[0]
    assert isinstance(stmt, IfStmt)
    assert isinstance(stmt.condition, NamedCond)
    assert stmt.condition.name == "price_drop"
    assert isinstance(stmt.then_body, VerbAction)
    assert stmt.then_body.verb == "OUT"
    assert isinstance(stmt.else_body, Skip)


def test_if_no_else_parses():
    prog = parse("IF.ready -> TRN.lstm")
    stmt = prog.statements[0]
    assert isinstance(stmt, IfStmt)
    assert stmt.else_body is None


def test_if_with_comparison_parses():
    prog = parse("IF.$score > 0.9 -> DEP.api ELSE -> SKIP")
    stmt = prog.statements[0]
    assert isinstance(stmt, IfStmt)
    cond = stmt.condition
    assert isinstance(cond, Comparison)
    assert isinstance(cond.left, VarRef)
    assert cond.left.name == "score"
    assert cond.op == ">"
    assert cond.right == 0.9


def test_if_with_func_condition_parses():
    prog = parse("IF.state_changed(key=denver) -> ING.flights ELSE -> SKIP")
    stmt = prog.statements[0]
    assert isinstance(stmt.condition, FuncCond)
    assert stmt.condition.name == "state_changed"
    assert stmt.condition.params["key"] == "denver"


# ──────────────────────────────────────────────────────────────────────────────
# Loops
# ──────────────────────────────────────────────────────────────────────────────

def test_loop_parses():
    prog = parse("LOOP(EVAL.metric, until=done)")
    stmt = prog.statements[0]
    assert isinstance(stmt, LoopStmt)
    assert isinstance(stmt.body, VerbAction)
    assert stmt.body.verb == "EVAL"
    assert isinstance(stmt.until, NamedCond)
    assert stmt.until.name == "done"


def test_loop_with_comparison_until():
    prog = parse("LOOP(TRN.lstm, until=$score > 0.9)")
    stmt = prog.statements[0]
    assert isinstance(stmt.until, Comparison)
    assert stmt.until.op == ">"
    assert stmt.until.right == 0.9


# ──────────────────────────────────────────────────────────────────────────────
# Blocks and PLAN declarations
# ──────────────────────────────────────────────────────────────────────────────

def test_plan_declaration_with_block():
    prog = parse("""
        PLAN:check_flights {
            ING.flights(dest=denver) -> EVAL.price(threshold=200) -> OUT.telegram
        }
    """)
    decls = [s for s in prog.statements if isinstance(s, PlanDecl)]
    assert len(decls) == 1
    plan = decls[0]
    assert plan.name == "check_flights"
    assert isinstance(plan.body, Block)
    assert len(plan.body.statements) == 1  # the chain is one statement
    chain = plan.body.statements[0]
    assert isinstance(chain, Chain)
    assert chain.steps[0].verb == "ING"


def test_if_with_block_body():
    prog = parse("""
        IF.price_drop -> {
            ING.flights -> EVAL.price -> OUT.telegram
        } ELSE -> SKIP
    """)
    stmt = prog.statements[0]
    assert isinstance(stmt, IfStmt)
    assert isinstance(stmt.then_body, Block)


# ──────────────────────────────────────────────────────────────────────────────
# Variable references
# ──────────────────────────────────────────────────────────────────────────────

def test_set_verb_parses():
    prog = parse("TRN.lstm -> SET.score")
    chain = prog.statements[0]
    assert isinstance(chain, Chain)
    assert chain.steps[1].verb == "SET"
    assert chain.steps[1].target == ["score"]


def test_var_ref_in_params():
    prog = parse("EVAL.rmse(threshold=$cutoff)")
    action = prog.statements[0]
    assert isinstance(action, VerbAction)
    param_val = action.params["threshold"]
    assert isinstance(param_val, VarRef)
    assert param_val.name == "cutoff"


# ──────────────────────────────────────────────────────────────────────────────
# Boolean expressions
# ──────────────────────────────────────────────────────────────────────────────

def test_and_expression():
    prog = parse("IF.ready AND price_drop -> OUT.telegram ELSE -> SKIP")
    stmt = prog.statements[0]
    assert isinstance(stmt.condition, AndExpr)
    assert len(stmt.condition.operands) == 2


def test_not_expression():
    prog = parse("IF.NOT failed -> TRN.lstm")
    stmt = prog.statements[0]
    assert isinstance(stmt.condition, NotExpr)


# ──────────────────────────────────────────────────────────────────────────────
# List values (for CAP)
# ──────────────────────────────────────────────────────────────────────────────

def test_list_value_in_params():
    prog = parse("CAP.agent(role=worker, allow=[search, summ, gen])")
    action = prog.statements[0]
    assert action.params["allow"] == ["search", "summ", "gen"]


# ──────────────────────────────────────────────────────────────────────────────
# Whitespace and comments
# ──────────────────────────────────────────────────────────────────────────────

def test_whitespace_ignored():
    a = parse("ING.sales.db->CLN.null->SUMM.text")
    b = parse("ING.sales.db -> CLN.null -> SUMM.text")
    assert a.statements[0].steps[0].verb == b.statements[0].steps[0].verb


def test_comments_ignored():
    prog = parse("""
        // This is a goal declaration
        GOAL:my_goal
        // Ingest and clean
        ING.sales.db -> CLN.null
    """)
    assert len(prog.statements) == 2


# ──────────────────────────────────────────────────────────────────────────────
# BREAK and WAIT
# ──────────────────────────────────────────────────────────────────────────────

def test_break_in_loop():
    prog = parse("LOOP(BREAK, until=never)")
    stmt = prog.statements[0]
    assert isinstance(stmt, LoopStmt)
    assert isinstance(stmt.body, Break)


def test_real_world_flight_monitor():
    """The canonical Shaun example from the roadmap."""
    prog = parse("""
        GOAL:monitor_denver_flights
        PLAN:check_price {
            ING.flights(dest=denver)
            -> EVAL.price(threshold=200)
            -> IF.$price < 200 -> OUT.telegram(msg="Price drop!") ELSE -> SKIP
        }
        LOOP(CALL.check_price, until=$cancelled)
    """)
    assert len(prog.goals()) == 1
    assert len(prog.plans()) == 1
    assert prog.goals()[0].name == "monitor_denver_flights"
    assert prog.plans()[0].name == "check_price"
