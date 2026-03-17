"""
Sprint 15 tests — Process isolation / sandbox (Pillar 7, Layer 3).
"""
from __future__ import annotations

import multiprocessing
import tempfile
import time
from pathlib import Path

import pytest

from praxis import parse
from praxis.sandbox import (
    SandboxedExecutor, SandboxPolicy,
    SandboxTimeout, SandboxViolation, SandboxCrash,
    _UNSANDBOXABLE, _install_path_guard,
)
from praxis.executor import ShaunRuntimeError, CapabilityViolation
from praxis.handlers import HANDLERS
from praxis.ast_types import Program, Chain, VerbAction


# ─── helpers ──────────────────────────────────────────────────────────────────

def _exe(policy: SandboxPolicy | None = None) -> SandboxedExecutor:
    return SandboxedExecutor(dict(HANDLERS), policy=policy)


def _run(text: str, policy: SandboxPolicy | None = None):
    return _exe(policy).execute(parse(text))


def _prog(*verbs) -> Program:
    return Program(statements=[Chain(steps=[
        VerbAction(verb=v, target=["x"], params={}) for v in verbs
    ])])


# ══════════════════════════════════════════════════════════════════════════════
# SandboxPolicy
# ══════════════════════════════════════════════════════════════════════════════

class TestSandboxPolicy:
    def test_default_timeout(self):
        assert SandboxPolicy().timeout_seconds == 30.0

    def test_default_allowed_paths_empty(self):
        assert SandboxPolicy().allowed_paths == []

    def test_default_sandbox_verbs_is_none(self):
        assert SandboxPolicy().sandbox_verbs is None

    def test_should_sandbox_handler_verb(self):
        assert SandboxPolicy().should_sandbox("LOG") is True

    def test_should_sandbox_fetch(self):
        assert SandboxPolicy().should_sandbox("FETCH") is True

    def test_should_not_sandbox_set(self):
        assert SandboxPolicy().should_sandbox("SET") is False

    def test_should_not_sandbox_call(self):
        assert SandboxPolicy().should_sandbox("CALL") is False

    def test_should_not_sandbox_spawn(self):
        assert SandboxPolicy().should_sandbox("SPAWN") is False

    def test_explicit_sandbox_verbs_respected(self):
        policy = SandboxPolicy(sandbox_verbs={"LOG"})
        assert policy.should_sandbox("LOG") is True
        assert policy.should_sandbox("ANNOTATE") is False

    def test_unsandboxable_excluded_even_if_in_set(self):
        policy = SandboxPolicy(sandbox_verbs={"SET", "LOG"})
        assert policy.should_sandbox("SET") is False
        assert policy.should_sandbox("LOG") is True


# ══════════════════════════════════════════════════════════════════════════════
# _UNSANDBOXABLE constant
# ══════════════════════════════════════════════════════════════════════════════

class TestUnsandboxableConstant:
    def test_set_excluded(self):
        assert "SET" in _UNSANDBOXABLE

    def test_call_excluded(self):
        assert "CALL" in _UNSANDBOXABLE

    def test_spawn_excluded(self):
        assert "SPAWN" in _UNSANDBOXABLE

    def test_join_excluded(self):
        assert "JOIN" in _UNSANDBOXABLE


# ══════════════════════════════════════════════════════════════════════════════
# Exception hierarchy
# ══════════════════════════════════════════════════════════════════════════════

class TestExceptions:
    def test_sandbox_timeout_is_resource_limit(self):
        from praxis.executor import ResourceLimitExceeded
        assert issubclass(SandboxTimeout, ResourceLimitExceeded)

    def test_sandbox_violation_is_runtime_error(self):
        assert issubclass(SandboxViolation, ShaunRuntimeError)

    def test_sandbox_crash_is_runtime_error(self):
        assert issubclass(SandboxCrash, ShaunRuntimeError)


# ══════════════════════════════════════════════════════════════════════════════
# Basic sandboxed execution
# ══════════════════════════════════════════════════════════════════════════════

class TestBasicSandboxedExecution:
    def test_sandboxed_log_returns_ok(self):
        results = _run("LOG.test",
                       policy=SandboxPolicy(sandbox_verbs={"LOG"}))
        assert results[0]["status"] == "ok"

    def test_sandboxed_step_has_verb(self):
        results = _run("LOG.test",
                       policy=SandboxPolicy(sandbox_verbs={"LOG"}))
        assert results[0]["verb"] == "LOG"

    def test_sandboxed_chain_returns_all_steps(self):
        results = _run("LOG.a -> ANNOTATE.b",
                       policy=SandboxPolicy(sandbox_verbs={"LOG", "ANNOTATE"}))
        assert len(results) == 2

    def test_unsandboxed_verb_runs_in_process(self):
        # SET is not sandboxed — should work normally and update ctx
        policy = SandboxPolicy(sandbox_verbs={"LOG"})
        exe = _exe(policy)
        prog = parse("LOG.x -> SET.myvar")
        results = exe.execute(prog)
        assert len(results) == 2
        assert results[1]["verb"] == "SET"
        assert results[1]["status"] == "ok"

    def test_log_entry_shows_sandbox_marker(self):
        results = _run("LOG.test",
                       policy=SandboxPolicy(sandbox_verbs={"LOG"}))
        assert "[sandbox]" in results[0]["log_entry"]


# ══════════════════════════════════════════════════════════════════════════════
# Sandbox timeout
# ══════════════════════════════════════════════════════════════════════════════

class TestSandboxTimeout:
    def test_timeout_raises_sandbox_timeout(self):
        """A verb whose handler sleeps longer than timeout_seconds → SandboxTimeout."""
        # We can't easily inject a slow handler into the child process without
        # it being importable, so we use the WAIT verb which calls time.sleep.
        # But WAIT is native in the executor... let's instead use a very short
        # timeout and a verb that does real work.
        # Strategy: use timeout=0.001 and any verb — the process startup alone
        # should exceed the timeout.
        policy = SandboxPolicy(
            timeout_seconds=0.001,  # 1ms — process startup alone exceeds this
            sandbox_verbs={"LOG"},
        )
        with pytest.raises(SandboxTimeout):
            _run("LOG.test", policy=policy)

    def test_timeout_message_contains_verb(self):
        policy = SandboxPolicy(timeout_seconds=0.001, sandbox_verbs={"LOG"})
        with pytest.raises(SandboxTimeout) as exc_info:
            _run("LOG.test", policy=policy)
        assert "LOG" in str(exc_info.value)


# ══════════════════════════════════════════════════════════════════════════════
# File path restriction
# ══════════════════════════════════════════════════════════════════════════════

class TestPathGuard:
    def test_allowed_path_permits_access(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            allowed = str(Path(tmpdir).resolve())
            import builtins
            real_open = builtins.open
            try:
                _install_path_guard([allowed])
                # Writing inside allowed dir should work
                test_file = Path(tmpdir) / "test.txt"
                test_file.write_text("ok")
                assert test_file.read_text() == "ok"
            finally:
                builtins.open = real_open

    def test_disallowed_path_raises_sandbox_violation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            allowed = str(Path(tmpdir).resolve())
            import builtins
            real_open = builtins.open
            try:
                _install_path_guard([allowed])
                with pytest.raises(SandboxViolation):
                    open("/tmp/outside_allowed_dir.txt", "w")
            finally:
                builtins.open = real_open

    def test_violation_message_contains_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import builtins
            real_open = builtins.open
            try:
                _install_path_guard([str(Path(tmpdir).resolve())])
                with pytest.raises(SandboxViolation) as exc_info:
                    open("/etc/passwd", "r")
                assert "/etc/passwd" in str(exc_info.value)
            finally:
                builtins.open = real_open


# ══════════════════════════════════════════════════════════════════════════════
# SandboxedExecutor inherits from Executor
# ══════════════════════════════════════════════════════════════════════════════

class TestInheritance:
    def test_is_executor_subclass(self):
        from praxis.executor import Executor
        assert issubclass(SandboxedExecutor, Executor)

    def test_cap_enforcement_still_works(self):
        """CAP enforcement in parent process still raises CapabilityViolation."""
        policy = SandboxPolicy(sandbox_verbs={"WRITE"})
        exe = _exe(policy)
        prog = parse("CAP.self(role=test, allow=[log]) -> WRITE.file")
        with pytest.raises(CapabilityViolation):
            exe.execute(prog)
