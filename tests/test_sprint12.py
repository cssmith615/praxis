"""
Sprint 12 tests — Optimizer: parallelization, dead step elimination, constant folding.
"""
from __future__ import annotations

import pytest

from praxis import parse
from praxis.optimizer import optimize, OptimizeResult
from praxis.ast_types import (
    Program, Chain, VerbAction, ParBlock, IfStmt, Block,
    NamedCond, Comparison,
)
from praxis.executor import Executor
from praxis.handlers import HANDLERS


# ─── helpers ──────────────────────────────────────────────────────────────────

def _opt(text: str) -> OptimizeResult:
    return optimize(parse(text))


def _flat_verbs(program: Program) -> list:
    """Flatten all VerbAction verbs in order for easy assertions."""
    verbs = []
    def _walk(node):
        if isinstance(node, Program):
            for s in node.statements: _walk(s)
        elif isinstance(node, Chain):
            for s in node.steps: _walk(s)
        elif isinstance(node, Block):
            for s in node.statements: _walk(s)
        elif isinstance(node, ParBlock):
            for b in node.branches: _walk(b)
        elif isinstance(node, VerbAction):
            verbs.append(node.verb)
        elif isinstance(node, IfStmt):
            _walk(node.then_body)
            if node.else_body: _walk(node.else_body)
    _walk(program)
    return verbs


def _exe():
    return Executor(dict(HANDLERS))


# ══════════════════════════════════════════════════════════════════════════════
# OptimizeResult
# ══════════════════════════════════════════════════════════════════════════════

class TestOptimizeResult:
    def test_any_changes_false_when_all_zero(self):
        r = OptimizeResult(program=parse("LOG.x"))
        assert r.any_changes() is False

    def test_any_changes_true_when_parallelized(self):
        r = OptimizeResult(program=parse("LOG.x"), parallelized=2)
        assert r.any_changes() is True

    def test_any_changes_true_when_dead_removed(self):
        r = OptimizeResult(program=parse("LOG.x"), dead_removed=1)
        assert r.any_changes() is True

    def test_any_changes_true_when_branches_folded(self):
        r = OptimizeResult(program=parse("LOG.x"), branches_folded=1)
        assert r.any_changes() is True

    def test_summary_no_changes(self):
        assert "no changes" in OptimizeResult(program=parse("LOG.x")).summary()

    def test_summary_shows_parallelized_count(self):
        r = OptimizeResult(program=parse("LOG.x"), parallelized=3)
        assert "3" in r.summary()
        assert "parallel" in r.summary()

    def test_summary_shows_dead_count(self):
        r = OptimizeResult(program=parse("LOG.x"), dead_removed=2)
        assert "2" in r.summary()
        assert "dead" in r.summary()

    def test_summary_shows_folded_count(self):
        r = OptimizeResult(program=parse("LOG.x"), branches_folded=1)
        assert "1" in r.summary()
        assert "fold" in r.summary()


# ══════════════════════════════════════════════════════════════════════════════
# optimize() identity: no-change programs
# ══════════════════════════════════════════════════════════════════════════════

class TestOptimizeIdentity:
    def test_single_step_no_change(self):
        r = _opt("LOG.x")
        assert r.any_changes() is False

    def test_returns_program(self):
        r = _opt("LOG.x -> ANNOTATE.y")
        assert isinstance(r.program, Program)

    def test_original_program_not_mutated(self):
        prog = parse("LOG.x -> ANNOTATE.y")
        original_id = id(prog)
        optimize(prog)
        assert id(prog) == original_id


# ══════════════════════════════════════════════════════════════════════════════
# Dead step elimination
# ══════════════════════════════════════════════════════════════════════════════

class TestDeadStepElimination:
    def _make_break_chain(self, *verbs_after_break):
        """Build a Program: LOG.a -> BREAK -> <verbs_after_break>"""
        steps = [
            VerbAction(verb="LOG", target=["a"], params={}),
            VerbAction(verb="BREAK", target=[], params={}),
        ] + [VerbAction(verb=v, target=["x"], params={}) for v in verbs_after_break]
        return Program(statements=[Chain(steps=steps)])

    def test_no_break_no_removal(self):
        r = _opt("LOG.a -> ANNOTATE.b -> OUT.c")
        assert r.dead_removed == 0

    def test_steps_after_break_removed(self):
        prog = self._make_break_chain("LOG", "ANNOTATE")
        r = optimize(prog)
        assert r.dead_removed == 2

    def test_break_itself_preserved(self):
        prog = self._make_break_chain("LOG")
        r = optimize(prog)
        verbs = _flat_verbs(r.program)
        assert "BREAK" in verbs

    def test_steps_before_break_preserved(self):
        prog = self._make_break_chain("LOG")
        r = optimize(prog)
        verbs = _flat_verbs(r.program)
        assert "LOG" in verbs   # the first LOG (before BREAK)

    def test_single_dead_step_counted(self):
        prog = self._make_break_chain("ANNOTATE")
        r = optimize(prog)
        assert r.dead_removed == 1

    def test_chain_length_after_elimination(self):
        prog = self._make_break_chain("A", "B", "C")
        r = optimize(prog)
        chain = r.program.statements[0]
        assert isinstance(chain, Chain)
        assert len(chain.steps) == 2   # LOG + BREAK


# ══════════════════════════════════════════════════════════════════════════════
# Constant folding
# ══════════════════════════════════════════════════════════════════════════════

class TestConstantFolding:
    def _always_true_if(self, then_verb="LOG", else_verb="ANNOTATE"):
        """IF 1 == 1 THEN LOG.x ELSE ANNOTATE.y"""
        cond = Comparison(left=1, op="==", right=1)
        then = VerbAction(verb=then_verb, target=["x"], params={})
        else_ = VerbAction(verb=else_verb, target=["y"], params={})
        return Program(statements=[IfStmt(condition=cond, then_body=then, else_body=else_)])

    def _always_false_if(self, then_verb="LOG", else_verb="ANNOTATE"):
        """IF 1 == 2 THEN LOG.x ELSE ANNOTATE.y"""
        cond = Comparison(left=1, op="==", right=2)
        then = VerbAction(verb=then_verb, target=["x"], params={})
        else_ = VerbAction(verb=else_verb, target=["y"], params={})
        return Program(statements=[IfStmt(condition=cond, then_body=then, else_body=else_)])

    def test_always_true_if_folds_to_then_body(self):
        r = optimize(self._always_true_if())
        assert r.branches_folded == 1
        # Program should no longer contain an IfStmt
        assert not any(isinstance(s, IfStmt) for s in r.program.statements)

    def test_always_true_if_keeps_then_verb(self):
        r = optimize(self._always_true_if(then_verb="LOG"))
        verbs = _flat_verbs(r.program)
        assert "LOG" in verbs

    def test_always_true_if_discards_else_verb(self):
        r = optimize(self._always_true_if(then_verb="LOG", else_verb="ANNOTATE"))
        verbs = _flat_verbs(r.program)
        assert "ANNOTATE" not in verbs

    def test_always_false_if_folds_to_else_body(self):
        r = optimize(self._always_false_if())
        assert r.branches_folded == 1
        verbs = _flat_verbs(r.program)
        assert "ANNOTATE" in verbs
        assert "LOG" not in verbs

    def test_always_false_if_no_else_removes_stmt(self):
        cond = Comparison(left=1, op="==", right=2)
        then = VerbAction(verb="LOG", target=["x"], params={})
        prog = Program(statements=[IfStmt(condition=cond, then_body=then, else_body=None)])
        r = optimize(prog)
        assert r.branches_folded == 1
        # Statement should be gone
        assert len(r.program.statements) == 0

    def test_runtime_condition_not_folded(self):
        cond = NamedCond(name="my_var")
        then = VerbAction(verb="LOG", target=["x"], params={})
        prog = Program(statements=[IfStmt(condition=cond, then_body=then, else_body=None)])
        r = optimize(prog)
        assert r.branches_folded == 0
        assert any(isinstance(s, IfStmt) for s in r.program.statements)

    def test_literal_comparison_gt(self):
        cond = Comparison(left=5, op=">", right=3)
        then = VerbAction(verb="LOG", target=["x"], params={})
        prog = Program(statements=[IfStmt(condition=cond, then_body=then, else_body=None)])
        r = optimize(prog)
        assert r.branches_folded == 1
        verbs = _flat_verbs(r.program)
        assert "LOG" in verbs

    def test_literal_comparison_false_gt(self):
        cond = Comparison(left=2, op=">", right=10)
        then = VerbAction(verb="LOG", target=["x"], params={})
        prog = Program(statements=[IfStmt(condition=cond, then_body=then, else_body=None)])
        r = optimize(prog)
        assert r.branches_folded == 1
        assert len(r.program.statements) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Parallelization
# ══════════════════════════════════════════════════════════════════════════════

class TestParallelization:
    def test_two_independent_steps_become_par(self):
        r = _opt("LOG.a -> ANNOTATE.b")
        assert r.parallelized == 2
        # Program should contain a PAR block
        stmt = r.program.statements[0]
        assert isinstance(stmt, ParBlock)

    def test_three_independent_steps_become_par(self):
        r = _opt("LOG.a -> ANNOTATE.b -> OUT.c")
        assert r.parallelized == 3

    def test_single_step_not_parallelized(self):
        r = _opt("LOG.a")
        assert r.parallelized == 0

    def test_order_sensitive_verb_breaks_par_group(self):
        """SET is order-sensitive — LOG -> SET -> ANNOTATE cannot fully PAR."""
        steps = [
            VerbAction(verb="LOG", target=["a"], params={}),
            VerbAction(verb="SET", target=["x"], params={}),
            VerbAction(verb="ANNOTATE", target=["b"], params={}),
        ]
        prog = Program(statements=[Chain(steps=steps)])
        r = optimize(prog)
        # SET is order-sensitive — LOG stays alone, ANNOTATE stays alone
        # Neither group of 1 becomes a PAR
        assert r.parallelized == 0

    def test_par_result_executes_correctly(self):
        """Optimized PAR program should produce same results as original chain."""
        r = _opt("LOG.a -> ANNOTATE.b")
        ex = _exe()
        results = ex.execute(r.program)
        assert len(results) == 2
        assert all(res["status"] == "ok" for res in results)

    def test_four_independent_steps_all_parallelized(self):
        r = _opt("LOG.a -> ANNOTATE.b -> OUT.c -> ING.d")
        assert r.parallelized == 4

    def test_spawn_verb_not_parallelized(self):
        steps = [
            VerbAction(verb="LOG", target=["a"], params={}),
            VerbAction(verb="SPAWN", target=["w"], params={"role": "data", "verbs": []}),
        ]
        prog = Program(statements=[Chain(steps=steps)])
        r = optimize(prog)
        assert r.parallelized == 0

    def test_join_verb_not_parallelized(self):
        steps = [
            VerbAction(verb="LOG", target=["a"], params={}),
            VerbAction(verb="JOIN", target=[], params={}),
        ]
        prog = Program(statements=[Chain(steps=steps)])
        r = optimize(prog)
        assert r.parallelized == 0


# ══════════════════════════════════════════════════════════════════════════════
# All passes combined
# ══════════════════════════════════════════════════════════════════════════════

class TestCombinedPasses:
    def test_dead_steps_and_par_combined(self):
        """Remove dead steps then PAR the survivors."""
        steps = [
            VerbAction(verb="LOG", target=["a"], params={}),
            VerbAction(verb="ANNOTATE", target=["b"], params={}),
            VerbAction(verb="BREAK", target=[], params={}),
            VerbAction(verb="OUT", target=["dead"], params={}),
        ]
        prog = Program(statements=[Chain(steps=steps)])
        r = optimize(prog)
        assert r.dead_removed == 1
        # LOG + ANNOTATE should have been PAR'd
        assert r.parallelized == 2

    def test_optimize_result_executable(self):
        """Fully optimized program must still execute without errors."""
        r = _opt("LOG.a -> ANNOTATE.b -> LOG.c -> OUT.d")
        ex = _exe()
        results = ex.execute(r.program)
        assert all(res["status"] == "ok" for res in results)
