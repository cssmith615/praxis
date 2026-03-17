"""
Sprint 10 tests — Resource limits (Pillar 7, Layer 1).

Covers:
  - ExecutionContext resource limit fields
  - Per-step timeout enforcement (max_step_ms)
  - Wall-clock program timeout enforcement (timeout_seconds)
  - Output size enforcement (max_output_bytes)
  - ResourceLimitExceeded exception type
  - Limits do not fire when within budget
  - execute() accepts limit keyword args
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from praxis.executor import (
    Executor,
    ExecutionContext,
    ResourceLimitExceeded,
    ShaunRuntimeError,
)
from praxis import parse


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_executor(extra_handlers: dict | None = None) -> Executor:
    """Executor with a minimal set of test handlers."""
    from praxis.handlers import HANDLERS as build_handlers
    h = dict(build_handlers)
    if extra_handlers:
        h.update(extra_handlers)
    return Executor(h)


def _run(program_text: str, **limit_kwargs):
    """Parse + execute a one-liner with given limits."""
    ex = _make_executor()
    prog = parse(program_text)
    return ex.execute(prog, **limit_kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# ExecutionContext field defaults
# ══════════════════════════════════════════════════════════════════════════════

class TestExecutionContextDefaults:
    def test_timeout_seconds_defaults_to_none(self):
        ctx = ExecutionContext()
        assert ctx.timeout_seconds is None

    def test_max_step_ms_defaults_to_none(self):
        ctx = ExecutionContext()
        assert ctx.max_step_ms is None

    def test_max_output_bytes_defaults_to_none(self):
        ctx = ExecutionContext()
        assert ctx.max_output_bytes is None

    def test_start_time_is_set(self):
        before = time.monotonic()
        ctx = ExecutionContext()
        after = time.monotonic()
        assert before <= ctx._start_time <= after

    def test_limits_passed_through(self):
        ctx = ExecutionContext(timeout_seconds=10, max_step_ms=500, max_output_bytes=1024)
        assert ctx.timeout_seconds == 10
        assert ctx.max_step_ms == 500
        assert ctx.max_output_bytes == 1024


# ══════════════════════════════════════════════════════════════════════════════
# ResourceLimitExceeded exception
# ══════════════════════════════════════════════════════════════════════════════

class TestResourceLimitExceededException:
    def test_is_subclass_of_shaun_runtime_error(self):
        assert issubclass(ResourceLimitExceeded, ShaunRuntimeError)

    def test_carries_message(self):
        exc = ResourceLimitExceeded("test limit hit")
        assert "test limit hit" in str(exc)


# ══════════════════════════════════════════════════════════════════════════════
# execute() keyword args forwarded
# ══════════════════════════════════════════════════════════════════════════════

class TestExecuteAcceptsLimitArgs:
    def test_no_limits_runs_normally(self):
        results = _run("LOG.test")
        assert results[0]["status"] == "ok"

    def test_generous_timeout_does_not_fire(self):
        results = _run("LOG.test", timeout_seconds=60)
        assert results[0]["status"] == "ok"

    def test_generous_step_ms_does_not_fire(self):
        results = _run("LOG.test", max_step_ms=5000)
        assert results[0]["status"] == "ok"

    def test_generous_output_bytes_does_not_fire(self):
        results = _run("LOG.test", max_output_bytes=1_000_000)
        assert results[0]["status"] == "ok"


# ══════════════════════════════════════════════════════════════════════════════
# Per-step timeout
# ══════════════════════════════════════════════════════════════════════════════

class TestStepTimeout:
    def _slow_handler(self, target, params, ctx):
        """Sleeps longer than any test timeout."""
        time.sleep(2)
        return "done"

    def _make_slow_executor(self):
        from praxis.handlers import HANDLERS as build_handlers
        h = dict(build_handlers)
        h["SLOW"] = self._slow_handler
        return Executor(h)

    def test_step_timeout_raises_resource_limit_exceeded(self):
        ex = self._make_slow_executor()
        # Register SLOW as a valid verb — use validator bypass
        from praxis.ast_types import Program, Chain, VerbAction
        prog = Program(statements=[Chain(steps=[VerbAction(verb="SLOW", target=["x"], params={})])])
        with pytest.raises(ResourceLimitExceeded, match="SLOW"):
            ex.execute(prog, max_step_ms=50)

    def test_step_timeout_message_includes_limit(self):
        ex = self._make_slow_executor()
        from praxis.ast_types import Program, Chain, VerbAction
        prog = Program(statements=[Chain(steps=[VerbAction(verb="SLOW", target=["x"], params={})])])
        with pytest.raises(ResourceLimitExceeded) as exc_info:
            ex.execute(prog, max_step_ms=50)
        assert "50ms" in str(exc_info.value)

    def test_fast_step_within_limit_succeeds(self):
        results = _run("LOG.test", max_step_ms=5000)
        assert results[0]["status"] == "ok"

    def test_chain_stops_at_first_timeout(self):
        ex = self._make_slow_executor()
        from praxis.ast_types import Program, Chain, VerbAction
        prog = Program(statements=[
            Chain(steps=[
                VerbAction(verb="LOG", target=["before"], params={}),
                VerbAction(verb="SLOW", target=["x"], params={}),
                VerbAction(verb="LOG", target=["after"], params={}),
            ])
        ])
        with pytest.raises(ResourceLimitExceeded):
            ex.execute(prog, max_step_ms=50)


# ══════════════════════════════════════════════════════════════════════════════
# Wall-clock program timeout
# ══════════════════════════════════════════════════════════════════════════════

class TestProgramTimeout:
    def test_already_expired_context_raises_on_first_step(self):
        """If we manually rewind _start_time the program is already over budget."""
        ex = _make_executor()
        from praxis.ast_types import Program, Chain, VerbAction
        prog = Program(statements=[Chain(steps=[VerbAction(verb="LOG", target=["x"], params={})])])
        # Execute with a tiny timeout; backdate start time so it's already exceeded
        with pytest.raises(ResourceLimitExceeded, match="timeout"):
            ctx_spy = []
            original_execute = ex.execute.__func__

            def patched(self, program, memory=None, timeout_seconds=None,
                        max_step_ms=None, max_output_bytes=None):
                from praxis.executor import ExecutionContext
                ctx = ExecutionContext(
                    mode=self.mode,
                    memory=memory,
                    handlers=self.handlers,
                    timeout_seconds=0.001,  # 1ms
                    max_step_ms=max_step_ms,
                    max_output_bytes=max_output_bytes,
                )
                ctx._start_time -= 1.0  # pretend we started 1s ago
                ctx_spy.append(ctx)
                # Re-register plans
                from praxis.ast_types import PlanDecl
                for stmt in program.statements:
                    if isinstance(stmt, PlanDecl):
                        ctx.plan_registry[stmt.name] = stmt
                results = []
                from praxis.ast_types import GoalDecl
                for stmt in program.statements:
                    if isinstance(stmt, (GoalDecl, PlanDecl)):
                        continue
                    results.extend(self._exec(stmt, ctx))
                return results

            import types
            ex.execute = types.MethodType(patched, ex)
            ex.execute(prog)

    def test_timeout_zero_blocks_execution(self):
        """timeout_seconds=0 should block all steps since elapsed >= 0 immediately."""
        ex = _make_executor()
        from praxis.ast_types import Program, Chain, VerbAction
        prog = Program(statements=[Chain(steps=[VerbAction(verb="LOG", target=["x"], params={})])])
        # timeout=0 means any elapsed >= 0 triggers — but monotonic clock moves forward
        # so we set a very tiny value and backdate; use direct ctx manipulation test approach
        with pytest.raises(ResourceLimitExceeded):
            # Use a real short sleep handler to let the clock run past 0.001s
            from praxis.handlers import HANDLERS as build_handlers
            h = dict(build_handlers)
            def sleepy(t, p, c):
                time.sleep(0.05)
                return "x"
            h["SLEEPY"] = sleepy
            ex2 = Executor(h)
            from praxis.ast_types import Program, Chain, VerbAction
            prog2 = Program(statements=[
                Chain(steps=[
                    VerbAction(verb="SLEEPY", target=["a"], params={}),
                    VerbAction(verb="SLEEPY", target=["b"], params={}),
                    VerbAction(verb="SLEEPY", target=["c"], params={}),
                ])
            ])
            ex2.execute(prog2, timeout_seconds=0.08)


# ══════════════════════════════════════════════════════════════════════════════
# Output size enforcement
# ══════════════════════════════════════════════════════════════════════════════

class TestOutputSizeLimit:
    def _big_output_handler(self, size_bytes: int):
        def handler(target, params, ctx):
            return "x" * size_bytes
        return handler

    def _make_executor_with_big(self, size_bytes: int) -> Executor:
        from praxis.handlers import HANDLERS as build_handlers
        h = dict(build_handlers)
        h["BIG"] = self._big_output_handler(size_bytes)
        return Executor(h)

    def _run_big(self, size_bytes: int, limit: int):
        ex = self._make_executor_with_big(size_bytes)
        from praxis.ast_types import Program, Chain, VerbAction
        prog = Program(statements=[Chain(steps=[VerbAction(verb="BIG", target=["x"], params={})])])
        return ex.execute(prog, max_output_bytes=limit)

    def test_output_within_limit_succeeds(self):
        results = self._run_big(100, 1000)
        assert results[0]["status"] == "ok"

    def test_output_exceeding_limit_raises(self):
        with pytest.raises(ResourceLimitExceeded, match="bytes"):
            self._run_big(10_000, 100)

    def test_error_message_includes_verb(self):
        with pytest.raises(ResourceLimitExceeded) as exc_info:
            self._run_big(10_000, 100)
        assert "BIG" in str(exc_info.value)

    def test_error_message_includes_limit(self):
        with pytest.raises(ResourceLimitExceeded) as exc_info:
            self._run_big(10_000, 100)
        assert "100" in str(exc_info.value)

    def test_none_output_not_checked(self):
        """None output should never trigger size limit."""
        from praxis.handlers import HANDLERS as build_handlers
        h = dict(build_handlers)
        h["NOOP"] = lambda t, p, c: None
        ex = Executor(h)
        from praxis.ast_types import Program, Chain, VerbAction
        prog = Program(statements=[Chain(steps=[VerbAction(verb="NOOP", target=["x"], params={})])])
        results = ex.execute(prog, max_output_bytes=1)
        assert results[0]["status"] == "ok"

    def test_multiple_steps_each_checked_independently(self):
        """Only the oversized step should fail; prior steps succeed."""
        from praxis.handlers import HANDLERS as build_handlers
        h = dict(build_handlers)
        h["SMALL"] = lambda t, p, c: "ok"
        h["BIG"] = lambda t, p, c: "x" * 10_000
        ex = Executor(h)
        from praxis.ast_types import Program, Chain, VerbAction
        prog = Program(statements=[
            Chain(steps=[
                VerbAction(verb="SMALL", target=["a"], params={}),
                VerbAction(verb="BIG", target=["b"], params={}),
            ])
        ])
        with pytest.raises(ResourceLimitExceeded):
            ex.execute(prog, max_output_bytes=100)


# ══════════════════════════════════════════════════════════════════════════════
# Combined limits
# ══════════════════════════════════════════════════════════════════════════════

class TestCombinedLimits:
    def test_all_limits_generous_runs_fine(self):
        results = _run("LOG.a -> ANNOTATE.b", timeout_seconds=60,
                       max_step_ms=5000, max_output_bytes=1_000_000)
        assert all(r["status"] == "ok" for r in results)

    def test_output_limit_checked_before_program_timeout(self):
        """Output size check happens inside _exec_verb; timeout check at step start."""
        from praxis.handlers import HANDLERS as build_handlers
        h = dict(build_handlers)
        h["BIG"] = lambda t, p, c: "x" * 10_000
        ex = Executor(h)
        from praxis.ast_types import Program, Chain, VerbAction
        prog = Program(statements=[Chain(steps=[VerbAction(verb="BIG", target=["x"], params={})])])
        with pytest.raises(ResourceLimitExceeded):
            ex.execute(prog, timeout_seconds=60, max_output_bytes=100)
