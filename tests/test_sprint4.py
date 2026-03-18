"""
Sprint 4 handler tests — I/O, Audit, Deploy.

All tests use tmp paths / mock HTTP so they run offline without side effects.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from praxis.executor import ExecutionContext, Executor
from praxis.handlers.audit import (
    AssertionFailure,
    GateRejected,
    annotate_handler,
    assert_handler,
    gate_handler,
    log_handler,
    route_handler,
    snap_handler,
    validate_handler,
)
from praxis.handlers.deploy import build_handler, dep_handler
from praxis.handlers.deploy import test_handler as run_suite_handler
from praxis.handlers.io import (
    out_handler,
    read_handler,
    recall_handler,
    store_handler,
    write_handler,
)
from praxis.handlers import HANDLERS


# ─── helpers ─────────────────────────────────────────────────────────────────

def _ctx(**kwargs):
    ctx = ExecutionContext()
    for k, v in kwargs.items():
        setattr(ctx, k, v)
    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# I/O handlers
# ══════════════════════════════════════════════════════════════════════════════

class TestReadWrite:
    def test_write_and_read_roundtrip(self, tmp_path):
        path = str(tmp_path / "hello.txt")
        ctx = _ctx(last_output="hello praxis")
        result = write_handler([], {"path": path, "content": "hello praxis"}, ctx)
        assert result["written"] == path
        assert result["bytes"] == 12

        ctx2 = _ctx(last_output=None)
        content = read_handler([], {"path": path}, ctx2)
        assert content == "hello praxis"

    def test_write_append_mode(self, tmp_path):
        path = str(tmp_path / "log.txt")
        ctx = _ctx(last_output=None)
        write_handler([], {"path": path, "content": "line1\n"}, ctx)
        write_handler([], {"path": path, "content": "line2\n", "mode": "a"}, ctx)
        content = read_handler([], {"path": path}, ctx)
        assert "line1" in content and "line2" in content

    def test_read_missing_file_raises(self):
        ctx = _ctx(last_output=None)
        with pytest.raises(FileNotFoundError):
            read_handler([], {"path": "/nonexistent/file.txt"}, ctx)

    def test_write_uses_last_output_when_no_content(self, tmp_path):
        path = str(tmp_path / "out.txt")
        ctx = _ctx(last_output="from_last_output")
        write_handler([], {"path": path}, ctx)
        assert read_handler([], {"path": path}, _ctx(last_output=None)) == "from_last_output"


class TestFetchPost:
    def test_fetch_returns_json(self):
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()
        with patch("praxis.handlers.io.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_response
            ctx = _ctx(last_output=None)
            result = from_praxis_fetch([], {"url": "https://example.com/api"}, ctx)
        assert result == {"status": "ok"}

    def test_post_sends_json_body(self):
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"created": True}
        mock_response.raise_for_status = MagicMock()
        with patch("praxis.handlers.io.httpx") as mock_httpx:
            mock_httpx.post.return_value = mock_response
            ctx = _ctx(last_output=None)
            from praxis.handlers.io import post_handler
            result = post_handler([], {"url": "https://example.com/api", "body": {"x": 1}}, ctx)
        assert result == {"created": True}


def from_praxis_fetch(target, params, ctx):
    from praxis.handlers.io import fetch_handler
    return fetch_handler(target, params, ctx)


class TestStoreRecall:
    def test_store_and_recall(self, tmp_path, monkeypatch):
        monkeypatch.setattr("praxis.handlers.io._KV_DB_PATH", tmp_path / "kv.db")
        ctx = _ctx(last_output="42")
        store_handler(["score"], {"key": "score", "value": 42}, ctx)
        result = recall_handler(["score"], {"key": "score"}, ctx)
        assert result["found"] is True
        assert result["value"] == 42

    def test_recall_missing_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr("praxis.handlers.io._KV_DB_PATH", tmp_path / "kv.db")
        ctx = _ctx(last_output=None)
        result = recall_handler([], {"key": "does_not_exist"}, ctx)
        assert result["found"] is False
        assert result["value"] is None

    def test_store_overwrite(self, tmp_path, monkeypatch):
        monkeypatch.setattr("praxis.handlers.io._KV_DB_PATH", tmp_path / "kv.db")
        ctx = _ctx(last_output=None)
        store_handler([], {"key": "x", "value": 1}, ctx)
        store_handler([], {"key": "x", "value": 2}, ctx)
        result = recall_handler([], {"key": "x"}, ctx)
        assert result["value"] == 2


class TestOut:
    def test_out_prints_to_console(self, capsys):
        ctx = _ctx(last_output="hello")
        result = out_handler(["console"], {"msg": "hello world"}, ctx)
        captured = capsys.readouterr()
        assert "hello world" in captured.out
        assert result["delivered"] is True

    def test_out_uses_last_output_when_no_msg(self, capsys):
        ctx = _ctx(last_output="auto message")
        out_handler(["console"], {}, ctx)
        captured = capsys.readouterr()
        assert "auto message" in captured.out

    def test_out_custom_channel(self, capsys):
        from praxis.handlers.io import register_out_channel, _OUT_CHANNELS
        received = []
        register_out_channel("test_chan", lambda msg, p: received.append(msg))
        ctx = _ctx(last_output="hi")
        result = out_handler(["test_chan"], {"msg": "hi"}, ctx)
        assert received == ["hi"]
        assert result["delivered"] is True
        del _OUT_CHANNELS["test_chan"]

    def test_out_telegram_sends_message(self):
        """OUT.telegram calls the Telegram API with correct params."""
        from unittest.mock import patch, MagicMock
        ctx = _ctx(last_output="hello from praxis")
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"ok": true, "result": {}}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("praxis.handlers.io.urllib.request.urlopen", return_value=mock_resp):
            result = out_handler(
                ["telegram"],
                {"msg": "hello", "token": "tok123", "chat_id": "999"},
                ctx,
            )
        assert result["channel"] == "telegram"
        assert result["delivered"] is True
        assert result["chat_id"] == "999"

    def test_out_telegram_reads_env_vars(self):
        """OUT.telegram falls back to env vars for credentials."""
        import os
        from unittest.mock import patch, MagicMock
        ctx = _ctx(last_output=None)
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"ok": true, "result": {}}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        env = {"TELEGRAM_BOT_TOKEN": "envtok", "TELEGRAM_CHAT_ID": "111"}
        with patch.dict(os.environ, env):
            with patch("praxis.handlers.io.urllib.request.urlopen", return_value=mock_resp):
                result = out_handler(["telegram"], {"msg": "test"}, ctx)
        assert result["delivered"] is True

    def test_out_telegram_raises_without_token(self):
        """OUT.telegram raises clearly when no token is available."""
        import os
        from unittest.mock import patch
        ctx = _ctx(last_output=None)
        env_without_token = {k: v for k, v in os.environ.items()
                             if k not in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")}
        with patch.dict(os.environ, env_without_token, clear=True):
            try:
                out_handler(["telegram"], {"msg": "hi", "chat_id": "999"}, ctx)
                assert False, "Should have raised"
            except RuntimeError as e:
                assert "TELEGRAM_BOT_TOKEN" in str(e)

    def test_out_telegram_splits_long_message(self):
        """OUT.telegram sends multiple API calls for messages over 4096 chars."""
        from unittest.mock import patch, MagicMock, call
        ctx = _ctx(last_output=None)
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"ok": true, "result": {}}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        long_msg = "x" * 5000
        with patch("praxis.handlers.io.urllib.request.urlopen", return_value=mock_resp) as mock_open:
            result = out_handler(
                ["telegram"],
                {"msg": long_msg, "token": "tok", "chat_id": "999"},
                ctx,
            )
        assert result["chunks"] == 2
        assert mock_open.call_count == 2


# ══════════════════════════════════════════════════════════════════════════════
# Audit handlers
# ══════════════════════════════════════════════════════════════════════════════

class TestAssert:
    def test_assert_passes_when_var_true(self):
        ctx = _ctx(last_output=None)
        ctx.variables["flag"] = True
        result = assert_handler(["flag"], {}, ctx)
        assert result["passed"] is True

    def test_assert_fails_when_var_false(self):
        ctx = _ctx(last_output=None)
        ctx.variables["flag"] = False
        with pytest.raises(AssertionFailure, match="flag"):
            assert_handler(["flag"], {}, ctx)

    def test_assert_uses_last_output_when_no_target(self):
        ctx = _ctx(last_output=True)
        result = assert_handler([], {}, ctx)
        assert result["passed"] is True

    def test_assert_fails_on_falsy_last_output(self):
        ctx = _ctx(last_output=None)
        with pytest.raises(AssertionFailure):
            assert_handler([], {}, ctx)

    def test_assert_includes_message(self):
        ctx = _ctx(last_output=False)
        with pytest.raises(AssertionFailure, match="custom error"):
            assert_handler([], {"msg": "custom error"}, ctx)


class TestGate:
    def test_gate_approves_on_y(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "y")
        ctx = _ctx(last_output=None)
        result = gate_handler(["deploy"], {}, ctx)
        assert result["approved"] is True

    def test_gate_approves_on_empty(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")
        ctx = _ctx(last_output=None)
        result = gate_handler(["deploy"], {}, ctx)
        assert result["approved"] is True

    def test_gate_rejects_on_n(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "n")
        ctx = _ctx(last_output=None)
        with pytest.raises(GateRejected, match="deploy"):
            gate_handler(["deploy"], {}, ctx)

    def test_gate_rejects_on_eof(self, monkeypatch):
        def raise_eof(_): raise EOFError
        monkeypatch.setattr("builtins.input", raise_eof)
        ctx = _ctx(last_output=None)
        with pytest.raises(GateRejected):
            gate_handler(["deploy"], {}, ctx)


class TestSnap:
    def test_snap_saves_state(self, tmp_path, monkeypatch):
        monkeypatch.setattr("praxis.handlers.audit._SNAP_DB", tmp_path / "snaps.db")
        monkeypatch.setattr("praxis.handlers.audit._PRAXIS_DIR", tmp_path)
        ctx = _ctx(last_output="result_value")
        ctx.variables["x"] = 42
        result = snap_handler(["checkpoint_1"], {}, ctx)
        assert result["snapped"] == "checkpoint_1"
        assert "x" in result["variables"]

        conn = sqlite3.connect(str(tmp_path / "snaps.db"))
        row = conn.execute("SELECT variables FROM snapshots WHERE name='checkpoint_1'").fetchone()
        conn.close()
        assert row is not None
        saved_vars = json.loads(row[0])
        assert saved_vars["x"] == 42


class TestLog:
    def test_log_writes_to_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("praxis.handlers.audit._PRAXIS_DIR", tmp_path)
        monkeypatch.setattr("praxis.handlers.audit._LOG_PATH", tmp_path / "execution.log")
        ctx = _ctx(last_output="logged_value")
        log_handler(["step.one"], {"data": "logged_value"}, ctx)
        log_path = tmp_path / "execution.log"
        assert log_path.exists()
        entry = json.loads(log_path.read_text().strip())
        assert entry["label"] == "step.one"


class TestRoute:
    def test_route_matches(self):
        ctx = _ctx(last_output="success")
        result = route_handler(["dest"], {"match": "success", "else": "fallback"}, ctx)
        assert result["routed_to"] == "success"
        assert ctx.variables["dest"] == "success"

    def test_route_falls_to_else(self):
        ctx = _ctx(last_output="error")
        result = route_handler(["dest"], {"match": "success", "else": "fallback"}, ctx)
        assert result["routed_to"] == "fallback"
        assert ctx.variables["dest"] == "fallback"


class TestValidate:
    def test_validate_passes_null_schema(self):
        pytest.importorskip("jsonschema")
        ctx = _ctx(last_output={"key": "value"})
        result = validate_handler([], {}, ctx)
        assert result["valid"] is True


# ══════════════════════════════════════════════════════════════════════════════
# Deploy handlers
# ══════════════════════════════════════════════════════════════════════════════

class TestDeploy:
    def test_test_handler_runs_and_parses(self):
        ctx = _ctx(last_output=None)
        with patch("praxis.handlers.deploy._run") as mock_run:
            mock_run.return_value = {
                "returncode": 0,
                "stdout": "3 passed in 0.12s",
                "stderr": "",
                "success": True,
            }
            result = run_suite_handler([], {}, ctx)
        assert result["passed"] == 3
        assert result["failed"] == 0
        assert result["success"] is True

    def test_build_handler_failure(self):
        ctx = _ctx(last_output=None)
        with patch("praxis.handlers.deploy._run") as mock_run:
            mock_run.return_value = {
                "returncode": 1,
                "stdout": "",
                "stderr": "build failed",
                "success": False,
            }
            result = build_handler(["myapp"], {}, ctx)
        assert result["status"] == "failed"
        assert result["artifact"] == "myapp"

    def test_dep_handler_success(self):
        ctx = _ctx(last_output=None)
        with patch("praxis.handlers.deploy._run") as mock_run:
            mock_run.return_value = {
                "returncode": 0,
                "stdout": "deployed",
                "stderr": "",
                "success": True,
            }
            result = dep_handler(["myapp"], {"env": "staging"}, ctx)
        assert result["status"] == "deployed"
        assert result["env"] == "staging"


# ══════════════════════════════════════════════════════════════════════════════
# Executor integration — AssertionFailure and GateRejected propagate
# ══════════════════════════════════════════════════════════════════════════════

class TestExecutorPropagation:
    def test_assert_failure_propagates_through_executor(self):
        from praxis.grammar import parse

        program = parse("FETCH.api -> ASSERT.ok")

        def always_false_assert(target, params, ctx):
            raise AssertionFailure("ok", "simulated failure")

        handlers = dict(HANDLERS)
        handlers["ASSERT"] = always_false_assert
        handlers["FETCH"] = lambda t, p, c: {"data": "result"}

        executor = Executor(handlers)
        with pytest.raises(AssertionFailure):
            executor.execute(program)

    def test_gate_rejected_propagates_through_executor(self, monkeypatch):
        from praxis.grammar import parse

        monkeypatch.setattr("builtins.input", lambda _: "n")
        program = parse("GATE.deploy")

        executor = Executor(HANDLERS)
        with pytest.raises(GateRejected):
            executor.execute(program)
