"""
Sprint 11 tests — CAP enforcement at runtime (Pillar 7, Layer 2).

Covers:
  - CapabilityViolation exception type
  - cap_handler activates _cap_allow on the live context
  - Verbs in allow-list execute normally
  - Verbs outside allow-list raise CapabilityViolation
  - Native verbs (SET, CALL, RETRY, ROLLBACK, CAP) always allowed
  - Programs without CAP run unrestricted
  - Executor.execute(cap_allow=...) pre-enforces before any step
  - SPAWN workers enforce their verb set
  - CapabilityViolation message names the offending verb
"""
from __future__ import annotations

import pytest

from praxis import parse
from praxis.executor import (
    Executor,
    ExecutionContext,
    CapabilityViolation,
    ShaunRuntimeError,
    _CAP_NATIVE_VERBS,
)
from praxis.handlers import HANDLERS
from praxis.ast_types import Program, Chain, VerbAction, Block


# ─── helpers ──────────────────────────────────────────────────────────────────

def _exe() -> Executor:
    return Executor(dict(HANDLERS))


def _run(text: str, **kwargs):
    return _exe().execute(parse(text), **kwargs)


def _run_ast(program: Program, **kwargs):
    return _exe().execute(program, **kwargs)


def _prog(*verbs) -> Program:
    """Build a Program with a chain of bare VerbActions."""
    steps = [VerbAction(verb=v, target=["x"], params={}) for v in verbs]
    return Program(statements=[Chain(steps=steps)])


# ══════════════════════════════════════════════════════════════════════════════
# CapabilityViolation exception
# ══════════════════════════════════════════════════════════════════════════════

class TestCapabilityViolationException:
    def test_is_subclass_of_shaun_runtime_error(self):
        assert issubclass(CapabilityViolation, ShaunRuntimeError)

    def test_carries_message(self):
        exc = CapabilityViolation("WRITE not in cap set")
        assert "WRITE not in cap set" in str(exc)


# ══════════════════════════════════════════════════════════════════════════════
# _CAP_NATIVE_VERBS constant
# ══════════════════════════════════════════════════════════════════════════════

class TestCapNativeVerbs:
    def test_set_is_always_allowed(self):
        assert "SET" in _CAP_NATIVE_VERBS

    def test_call_is_always_allowed(self):
        assert "CALL" in _CAP_NATIVE_VERBS

    def test_cap_itself_is_always_allowed(self):
        assert "CAP" in _CAP_NATIVE_VERBS

    def test_retry_is_always_allowed(self):
        assert "RETRY" in _CAP_NATIVE_VERBS

    def test_rollback_is_always_allowed(self):
        assert "ROLLBACK" in _CAP_NATIVE_VERBS


# ══════════════════════════════════════════════════════════════════════════════
# No CAP declared → unrestricted
# ══════════════════════════════════════════════════════════════════════════════

class TestNoCap:
    def test_program_without_cap_runs_any_verb(self):
        results = _run("LOG.test -> ANNOTATE.x")
        assert all(r["status"] == "ok" for r in results)

    def test_context_cap_allow_defaults_to_none(self):
        ctx = ExecutionContext()
        assert ctx._cap_allow is None


# ══════════════════════════════════════════════════════════════════════════════
# execute(cap_allow=...) pre-enforcement
# ══════════════════════════════════════════════════════════════════════════════

class TestExecuteCapAllow:
    def test_allowed_verb_executes(self):
        results = _run_ast(_prog("LOG"), cap_allow={"LOG"})
        assert results[0]["status"] == "ok"

    def test_disallowed_verb_raises(self):
        with pytest.raises(CapabilityViolation, match="WRITE"):
            _run_ast(_prog("WRITE"), cap_allow={"LOG"})

    def test_multiple_allowed_verbs(self):
        results = _run_ast(_prog("LOG", "ANNOTATE"), cap_allow={"LOG", "ANNOTATE"})
        assert len(results) == 2
        assert all(r["status"] == "ok" for r in results)

    def test_violation_names_verb_in_message(self):
        with pytest.raises(CapabilityViolation) as exc_info:
            _run_ast(_prog("FETCH"), cap_allow={"LOG"})
        assert "FETCH" in str(exc_info.value)

    def test_violation_lists_allowed_verbs(self):
        with pytest.raises(CapabilityViolation) as exc_info:
            _run_ast(_prog("FETCH"), cap_allow={"LOG", "ANNOTATE"})
        msg = str(exc_info.value)
        assert "LOG" in msg or "ANNOTATE" in msg

    def test_empty_allow_set_blocks_all_handler_verbs(self):
        with pytest.raises(CapabilityViolation):
            _run_ast(_prog("LOG"), cap_allow=set())

    def test_cap_allow_none_means_unrestricted(self):
        results = _run_ast(_prog("LOG"), cap_allow=None)
        assert results[0]["status"] == "ok"


# ══════════════════════════════════════════════════════════════════════════════
# CAP verb in program activates enforcement
# ══════════════════════════════════════════════════════════════════════════════

class TestCapVerbActivation:
    def test_cap_sets_cap_allow_on_context(self):
        """After CAP runs, _cap_allow is set to the declared verbs."""
        # Grammar requires lowercase in list values; handler normalizes to uppercase
        results = _run("CAP.self(role=test, allow=[log]) -> LOG.test")
        assert results[0]["verb"] == "CAP"
        assert results[0]["status"] == "ok"
        assert results[1]["verb"] == "LOG"
        assert results[1]["status"] == "ok"

    def test_verb_after_cap_not_in_allow_raises(self):
        with pytest.raises(CapabilityViolation, match="WRITE"):
            _run("CAP.self(role=test, allow=[log]) -> WRITE.file")

    def test_cap_allow_survives_multi_step_chain(self):
        """Three allowed verbs in sequence all succeed."""
        results = _run(
            "CAP.self(role=test, allow=[log, annotate, out]) -> "
            "LOG.a -> ANNOTATE.b -> OUT.c"
        )
        assert all(r["status"] == "ok" for r in results)

    def test_cap_self_is_not_blocked_by_its_own_declaration(self):
        """CAP is in _CAP_NATIVE_VERBS, so re-declaring CAP is never a violation."""
        results = _run("CAP.self(role=a, allow=[log]) -> CAP.self(role=b, allow=[log, annotate])")
        assert all(r["status"] == "ok" for r in results)

    def test_second_cap_declaration_updates_allow_list(self):
        """A second CAP call replaces the first allow-list."""
        results = _run(
            "CAP.self(role=a, allow=[log]) -> "
            "CAP.self(role=b, allow=[log, annotate]) -> "
            "ANNOTATE.x"
        )
        assert results[-1]["status"] == "ok"

    def test_empty_cap_allow_blocks_subsequent_verbs(self):
        with pytest.raises(CapabilityViolation):
            _run("CAP.self(role=locked, allow=[]) -> LOG.x")


# ══════════════════════════════════════════════════════════════════════════════
# Native verbs bypass CAP
# ══════════════════════════════════════════════════════════════════════════════

class TestNativeVerbsBypassCap:
    def test_set_always_allowed(self):
        # CAP allows nothing but SET should still work
        results = _run_ast(
            Program(statements=[Chain(steps=[
                VerbAction(verb="LOG", target=["x"], params={}),
                VerbAction(verb="SET", target=["myvar"], params={}),
            ])]),
            cap_allow={"LOG"}  # SET not in cap_allow
        )
        assert results[1]["verb"] == "SET"
        assert results[1]["status"] == "ok"

    def test_cap_verb_always_allowed(self):
        # Even with a restrictive cap_allow, CAP itself runs
        results = _run_ast(
            Program(statements=[Chain(steps=[
                VerbAction(verb="CAP", target=["self"], params={"role": "x", "allow": ["LOG"]}),
                VerbAction(verb="LOG", target=["y"], params={}),
            ])]),
            cap_allow={"LOG"}
        )
        assert all(r["status"] == "ok" for r in results)


# ══════════════════════════════════════════════════════════════════════════════
# SPAWN worker CAP enforcement
# ══════════════════════════════════════════════════════════════════════════════

class TestSpawnWorkerCapEnforcement:
    def test_worker_allows_declared_verbs(self):
        from praxis.agent_registry import Worker
        from praxis.executor import Executor
        worker = Worker(
            agent_id="w1",
            role="data",
            verbs=["LOG", "ANNOTATE"],
            cap_allow={"LOG", "ANNOTATE"},
            executor=Executor(dict(HANDLERS)),
        )
        result = worker.execute("LOG.test")
        assert result["status"] == "ok"

    def test_worker_blocks_undeclared_verbs(self):
        from praxis.agent_registry import Worker
        from praxis.executor import Executor
        worker = Worker(
            agent_id="w2",
            role="data",
            verbs=["LOG"],
            cap_allow={"LOG"},
            executor=Executor(dict(HANDLERS)),
        )
        # WRITE is not in cap_allow — should produce error status
        result = worker.execute("WRITE.file")
        assert result["status"] == "error"
        assert "WRITE" in result["error"]

    def test_worker_with_no_cap_allow_runs_unrestricted(self):
        from praxis.agent_registry import Worker
        from praxis.executor import Executor
        worker = Worker(
            agent_id="w3",
            role="any",
            verbs=[],
            cap_allow=None,
            executor=Executor(dict(HANDLERS)),
        )
        result = worker.execute("LOG.test -> ANNOTATE.y")
        assert result["status"] == "ok"

    def test_spawn_sets_cap_allow_from_verbs(self):
        """spawn_handler creates Worker with cap_allow=set(verbs); verbs normalized to uppercase."""
        results = _run(
            'SPAWN.myworker(role=data, verbs=[log, annotate]) -> '
            'MSG.myworker(program="LOG.hello")'
        )
        spawn_r = results[0]
        assert spawn_r["status"] == "ok"
        # Handler normalizes to uppercase
        assert "LOG" in spawn_r["output"]["verbs"]
