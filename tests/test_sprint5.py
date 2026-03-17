"""
Sprint 5 tests — ERR, RETRY, ROLLBACK, Scheduler.
All tests are offline; no real HTTP calls or Anthropic API usage.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from praxis.executor import (
    Executor, ExecutionContext,
    RetryExhausted, ShaunRuntimeError,
    AssertionFailure,
)
from praxis.handlers import HANDLERS
from praxis.handlers.error import err_handler
from praxis.grammar import parse
from praxis.scheduler import Scheduler, ScheduledProgram


# ─── helpers ─────────────────────────────────────────────────────────────────

def _ctx(**kwargs):
    ctx = ExecutionContext()
    for k, v in kwargs.items():
        setattr(ctx, k, v)
    return ctx


def _make_executor(extra_handlers: dict | None = None):
    h = dict(HANDLERS)
    if extra_handlers:
        h.update(extra_handlers)
    return Executor(h)


# ══════════════════════════════════════════════════════════════════════════════
# ERR handler
# ══════════════════════════════════════════════════════════════════════════════

class TestErrHandler:
    def test_err_writes_to_log(self, tmp_path, monkeypatch):
        monkeypatch.setattr("praxis.handlers.error._PRAXIS_DIR", tmp_path)
        monkeypatch.setattr("praxis.handlers.error._ERR_LOG", tmp_path / "errors.log")
        ctx = _ctx(last_output="bad_value")
        result = err_handler(["api_down"], {"msg": "API unreachable", "code": "503"}, ctx)
        assert result["code"] == "503"
        assert result["message"] == "API unreachable"
        log = tmp_path / "errors.log"
        assert log.exists()
        entry = json.loads(log.read_text().strip())
        assert entry["code"] == "503"

    def test_err_sets_last_error_in_ctx(self, tmp_path, monkeypatch):
        monkeypatch.setattr("praxis.handlers.error._PRAXIS_DIR", tmp_path)
        monkeypatch.setattr("praxis.handlers.error._ERR_LOG", tmp_path / "errors.log")
        ctx = _ctx(last_output=None)
        err_handler(["fail"], {"msg": "oops"}, ctx)
        assert "last_error" in ctx.variables
        assert ctx.variables["last_error"]["message"] == "oops"

    def test_err_sets_recover_plan(self, tmp_path, monkeypatch):
        monkeypatch.setattr("praxis.handlers.error._PRAXIS_DIR", tmp_path)
        monkeypatch.setattr("praxis.handlers.error._ERR_LOG", tmp_path / "errors.log")
        ctx = _ctx(last_output=None)
        err_handler(["fail"], {"msg": "need recovery", "recover": "handle_outage"}, ctx)
        assert ctx.variables.get("_recover_plan") == "handle_outage"

    def test_err_without_recover_does_not_set_recover_plan(self, tmp_path, monkeypatch):
        monkeypatch.setattr("praxis.handlers.error._PRAXIS_DIR", tmp_path)
        monkeypatch.setattr("praxis.handlers.error._ERR_LOG", tmp_path / "errors.log")
        ctx = _ctx(last_output=None)
        err_handler(["fail"], {"msg": "simple error"}, ctx)
        assert "_recover_plan" not in ctx.variables


# ══════════════════════════════════════════════════════════════════════════════
# RETRY — native executor verb
# ══════════════════════════════════════════════════════════════════════════════

class TestRetry:
    def test_retry_skipped_when_no_prior_error(self):
        program = parse("STORE.x(key=x, value=1) -> RETRY(attempts=3)")
        exe = _make_executor()
        results = exe.execute(program)
        # RETRY should be a no-op when STORE succeeds
        retry_result = next(r for r in results if r["verb"] == "RETRY")
        assert retry_result["status"] == "ok"
        assert "no prior failure" in retry_result["log_entry"]

    def test_retry_succeeds_on_second_attempt(self):
        call_count = {"n": 0}

        def flaky_handler(target, params, ctx):
            call_count["n"] += 1
            # Fails on calls 1 and 2 (initial + first retry), succeeds on call 3
            if call_count["n"] < 3:
                raise RuntimeError("flaky failure")
            return "success"

        program = parse("FLAKY.x -> RETRY(attempts=3, backoff=fixed)")
        exe = _make_executor({"FLAKY": flaky_handler})

        with patch("praxis.executor.time.sleep"):
            results = exe.execute(program)

        # Initial call (fail) + 2 retry attempts (fail, succeed) = 3 total calls
        assert call_count["n"] == 3
        retry_r = next(r for r in results if r["verb"] == "RETRY")
        assert retry_r["status"] == "ok"
        assert "attempt 2/3" in retry_r["log_entry"]

    def test_retry_raises_retry_exhausted_after_max_attempts(self):
        def always_fails(target, params, ctx):
            raise RuntimeError("always fails")

        program = parse("FAIL.x -> RETRY(attempts=2, backoff=fixed)")
        exe = _make_executor({"FAIL": always_fails})

        with patch("praxis.executor.time.sleep"):
            with pytest.raises(RetryExhausted) as exc_info:
                exe.execute(program)

        assert exc_info.value.attempts == 2
        assert "FAIL" in exc_info.value.verb

    def test_retry_exp_backoff_calls_sleep_with_increasing_durations(self):
        attempt_sleeps = []

        def failing(target, params, ctx):
            raise RuntimeError("fail")

        program = parse("FAIL.x -> RETRY(attempts=3, backoff=exp)")
        exe = _make_executor({"FAIL": failing})

        with patch("praxis.executor.time.sleep", side_effect=lambda s: attempt_sleeps.append(s)):
            with pytest.raises(RetryExhausted):
                exe.execute(program)

        # attempt=0: _backoff_seconds returns 0.0 → no sleep call
        # attempt=1: returns 2.0 → sleep(2.0)
        # attempt=2: returns 4.0 → sleep(4.0)
        assert len(attempt_sleeps) == 2
        assert attempt_sleeps[0] == 2.0
        assert attempt_sleeps[1] == 4.0

    def test_retry_not_in_handlers_dict(self):
        """RETRY must be handled natively, not via the HANDLERS registry."""
        assert "RETRY" not in HANDLERS

    def test_rollback_not_in_handlers_dict(self):
        """ROLLBACK must be handled natively, not via the HANDLERS registry."""
        assert "ROLLBACK" not in HANDLERS


# ══════════════════════════════════════════════════════════════════════════════
# ROLLBACK — native executor verb
# ══════════════════════════════════════════════════════════════════════════════

class TestRollback:
    def test_rollback_restores_ctx_from_snap(self, tmp_path, monkeypatch):
        """ROLLBACK should restore variables and last_output from a SNAP checkpoint."""
        from praxis.handlers.audit import snap_handler
        from praxis.ast_types import VerbAction
        import praxis.executor as emod

        # Build a context with known state
        ctx = ExecutionContext()
        ctx.variables["score"] = 100
        ctx.last_output = "before"

        # Write snap to tmp_path/.praxis/snaps.db (mirrors real runtime path)
        praxis_dir = tmp_path / ".praxis"
        praxis_dir.mkdir()
        snap_db = praxis_dir / "snaps.db"
        with patch("praxis.handlers.audit._SNAP_DB", snap_db), \
             patch("praxis.handlers.audit._PRAXIS_DIR", praxis_dir):
            snap_handler(["my_snap"], {}, ctx)

        # Corrupt context state
        ctx.variables["score"] = 999
        ctx.last_output = "corrupted"

        # Patch executor's _Path.home() so ROLLBACK resolves to tmp_path
        orig_path = emod._Path

        class _MockPath:
            @staticmethod
            def home():
                return tmp_path

        emod._Path = _MockPath
        try:
            rollback_action = VerbAction(verb="ROLLBACK", target=["my_snap"], params={})
            exe = Executor(HANDLERS)
            result = exe._exec_rollback(rollback_action, ctx)
        finally:
            emod._Path = orig_path

        assert result["status"] == "ok"
        assert ctx.variables["score"] == 100
        assert ctx.last_output == "before"

    def test_rollback_raises_when_no_snap_db(self, tmp_path):
        from praxis.ast_types import VerbAction
        import praxis.executor as emod

        ctx = ExecutionContext()
        rollback_action = VerbAction(verb="ROLLBACK", target=["nonexistent"], params={})
        exe = Executor(HANDLERS)

        class PatchedPath:
            @staticmethod
            def home():
                return tmp_path
            def __truediv__(self, other):
                return tmp_path / other

        orig = emod._Path
        emod._Path = PatchedPath
        try:
            with pytest.raises(ShaunRuntimeError, match="no snap database"):
                exe._exec_rollback(rollback_action, ctx)
        finally:
            emod._Path = orig


# ══════════════════════════════════════════════════════════════════════════════
# Scheduler
# ══════════════════════════════════════════════════════════════════════════════

class TestScheduler:
    def _make_sched(self, tmp_path, executor=None, triage_fn=None):
        import praxis.scheduler as smod
        smod._SCHEDULE_DB = tmp_path / "schedule.db"
        sched = Scheduler(executor=executor, triage_fn=triage_fn)
        sched._db_path = tmp_path / "schedule.db"
        sched._init_db()
        return sched

    def test_add_and_list(self, tmp_path):
        sched = self._make_sched(tmp_path)
        sid = sched.add("monitor prices", "ING.data -> LOG.out", 3600)
        programs = sched.list_programs()
        assert len(programs) == 1
        assert programs[0].id == sid
        assert programs[0].goal == "monitor prices"
        assert programs[0].interval_seconds == 3600

    def test_remove(self, tmp_path):
        sched = self._make_sched(tmp_path)
        sid = sched.add("test goal", "LOG.out", 60)
        assert sched.remove(sid) is True
        assert sched.list_programs() == []

    def test_remove_nonexistent_returns_false(self, tmp_path):
        sched = self._make_sched(tmp_path)
        assert sched.remove("nonexistent") is False

    def test_run_pending_skips_not_due(self, tmp_path):
        sched = self._make_sched(tmp_path)
        # Add with future next_run_at
        sched.add("future task", "LOG.out", 9999, run_immediately=False)
        results = sched.run_pending()
        assert results == []

    def test_run_pending_executes_due_program(self, tmp_path):
        mock_executor = MagicMock()
        mock_executor.execute.return_value = [
            {"verb": "LOG", "output": "done", "log_entry": "LOG -> ok", "status": "ok"}
        ]

        sched = self._make_sched(tmp_path, executor=mock_executor)
        sched.add("due task", "LOG.out", 60, run_immediately=True)

        import praxis.scheduler as smod
        with patch.object(smod, "parse", return_value=MagicMock()):
            results = sched.run_pending()

        assert len(results) == 1
        assert results[0].status == "ok"
        assert results[0].steps == 1
        mock_executor.execute.assert_called_once()

    def test_run_pending_handles_error_gracefully(self, tmp_path):
        mock_executor = MagicMock()
        mock_executor.execute.side_effect = RuntimeError("program failed")

        sched = self._make_sched(tmp_path, executor=mock_executor)
        sched.add("failing task", "LOG.x", 60, run_immediately=True)

        import praxis.scheduler as smod
        with patch.object(smod, "parse", return_value=MagicMock()):
            results = sched.run_pending()

        assert results[0].status == "error"
        assert "program failed" in results[0].error

    def test_triage_skip_prevents_execution(self, tmp_path):
        mock_executor = MagicMock()
        always_skip = lambda goal, last_output: False  # never run

        sched = self._make_sched(tmp_path, executor=mock_executor, triage_fn=always_skip)
        sched.add("triage task", "ING.data", 60, run_immediately=True)
        results = sched.run_pending()

        assert results[0].status == "triage_skip"
        mock_executor.execute.assert_not_called()

    def test_triage_allows_execution_when_true(self, tmp_path):
        mock_executor = MagicMock()
        mock_executor.execute.return_value = [
            {"verb": "LOG", "output": "ok", "log_entry": "ok", "status": "ok"}
        ]
        always_run = lambda goal, last_output: True

        sched = self._make_sched(tmp_path, executor=mock_executor, triage_fn=always_run)
        sched.add("allowed task", "LOG.out", 60, run_immediately=True)

        import praxis.scheduler as smod
        with patch.object(smod, "parse", return_value=MagicMock()):
            results = sched.run_pending()

        assert results[0].status == "ok"
        mock_executor.execute.assert_called_once()

    def test_next_run_updated_after_execution(self, tmp_path):
        mock_executor = MagicMock()
        mock_executor.execute.return_value = []

        sched = self._make_sched(tmp_path, executor=mock_executor)
        sid = sched.add("update test", "LOG.x", 300, run_immediately=True)
        before = time.time()

        import praxis.scheduler as smod
        with patch.object(smod, "parse", return_value=MagicMock()):
            sched.run_pending()

        programs = sched.list_programs()
        assert programs[0].next_run_at > before
        assert programs[0].last_run_at is not None

    def test_enable_disable(self, tmp_path):
        mock_executor = MagicMock()
        sched = self._make_sched(tmp_path, executor=mock_executor)
        sid = sched.add("toggle task", "LOG.x", 60, run_immediately=True)
        sched.enable(sid, False)
        results = sched.run_pending()
        assert results == []  # disabled, not run
        mock_executor.execute.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# End-to-end: ERR in a chain does not halt (it's a handler, not an exception)
# ══════════════════════════════════════════════════════════════════════════════

class TestErrInChain:
    def test_err_in_chain_continues_execution(self, tmp_path, monkeypatch):
        monkeypatch.setattr("praxis.handlers.error._PRAXIS_DIR", tmp_path)
        monkeypatch.setattr("praxis.handlers.error._ERR_LOG", tmp_path / "errors.log")

        # ERR is a handler that logs and returns — it does NOT halt the chain
        program = parse('ERR.step1(msg="something went wrong") -> LOG.after_err')
        exe = _make_executor()

        with patch("praxis.handlers.audit._PRAXIS_DIR", tmp_path), \
             patch("praxis.handlers.audit._LOG_PATH", tmp_path / "execution.log"):
            results = exe.execute(program)

        verbs = [r["verb"] for r in results]
        assert "ERR" in verbs
        assert "LOG" in verbs
        assert results[-1]["status"] == "ok"
