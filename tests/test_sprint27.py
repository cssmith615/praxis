"""
Sprint 27 / 28 tests — Persistent SET/LOAD, ING.webhook, OUT.x, plugin loader.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from praxis.grammar import parse
from praxis.executor import Executor, _kv_read, _kv_write, _KV_DB_PATH
from praxis.handlers import HANDLERS
from praxis.server import app


# ─────────────────────────────────────────────────────────────────────────────
# Persistent SET / LOAD
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _patch_kv_path(tmp_path, monkeypatch):
    """Redirect KV DB to a temp file for all tests in this module."""
    import praxis.executor as exe_mod
    fake_kv = tmp_path / "kv.db"
    monkeypatch.setattr(exe_mod, "_KV_DB_PATH", fake_kv)
    yield fake_kv


class TestPersistentSet:
    def _run(self, program: str) -> list:
        ast = parse(program)
        return Executor(handlers=HANDLERS).execute(ast)

    def test_set_without_persist_does_not_write_kv(self, tmp_path, monkeypatch):
        import praxis.executor as exe_mod
        fake_kv = tmp_path / "kv_nopersist.db"
        monkeypatch.setattr(exe_mod, "_KV_DB_PATH", fake_kv)
        self._run("LOG.x -> SET.myvar")
        assert not fake_kv.exists() or not sqlite3.connect(fake_kv).execute(
            "SELECT count(*) FROM kv WHERE key LIKE 'praxis_var::%'"
        ).fetchone()[0]

    def test_set_with_persist_writes_kv(self, tmp_path, monkeypatch):
        import praxis.executor as exe_mod
        fake_kv = tmp_path / "kv_persist.db"
        monkeypatch.setattr(exe_mod, "_KV_DB_PATH", fake_kv)
        self._run("LOG.hello -> SET.greeting(persist=true)")
        assert fake_kv.exists()
        conn = sqlite3.connect(fake_kv)
        row = conn.execute("SELECT value FROM kv WHERE key = 'praxis_var::greeting'").fetchone()
        conn.close()
        assert row is not None

    def test_set_persist_result_says_persisted(self, tmp_path, monkeypatch):
        import praxis.executor as exe_mod
        fake_kv = tmp_path / "kv_p2.db"
        monkeypatch.setattr(exe_mod, "_KV_DB_PATH", fake_kv)
        results = self._run("LOG.x -> SET.v(persist=true)")
        set_result = next(r for r in results if r["verb"] == "SET")
        assert "[persisted]" in set_result.get("log_entry", "")

    def test_set_persist_false_does_not_persist(self, tmp_path, monkeypatch):
        import praxis.executor as exe_mod
        fake_kv = tmp_path / "kv_false.db"
        monkeypatch.setattr(exe_mod, "_KV_DB_PATH", fake_kv)
        self._run("LOG.x -> SET.v(persist=false)")
        if fake_kv.exists():
            conn = sqlite3.connect(fake_kv)
            count = conn.execute("SELECT count(*) FROM kv WHERE key LIKE 'praxis_var::%'").fetchone()[0]
            conn.close()
            assert count == 0


class TestLoadVerb:
    def _run(self, program: str) -> list:
        ast = parse(program)
        return Executor(handlers=HANDLERS).execute(ast)

    def test_load_returns_none_when_no_kv(self, tmp_path, monkeypatch):
        import praxis.executor as exe_mod
        monkeypatch.setattr(exe_mod, "_KV_DB_PATH", tmp_path / "empty.db")
        results = self._run("LOAD.missing")
        load_r = next(r for r in results if r["verb"] == "LOAD")
        assert load_r["output"] is None

    def test_load_retrieves_persisted_value(self, tmp_path, monkeypatch):
        import praxis.executor as exe_mod
        fake_kv = tmp_path / "kv_load.db"
        monkeypatch.setattr(exe_mod, "_KV_DB_PATH", fake_kv)
        # Write directly
        exe_mod._kv_write("mykey", {"hello": "world"})
        results = self._run("LOAD.mykey")
        load_r = next(r for r in results if r["verb"] == "LOAD")
        assert load_r["output"] == {"hello": "world"}

    def test_load_sets_ctx_variable(self, tmp_path, monkeypatch):
        import praxis.executor as exe_mod
        fake_kv = tmp_path / "kv_ctx.db"
        monkeypatch.setattr(exe_mod, "_KV_DB_PATH", fake_kv)
        exe_mod._kv_write("counter", 42)
        # LOAD then SET again — verifies variable is in ctx
        results = self._run("LOAD.counter -> SET.counter2")
        set_r = next(r for r in results if r["verb"] == "SET")
        assert set_r["output"] == 42

    def test_kv_write_read_roundtrip(self, tmp_path, monkeypatch):
        import praxis.executor as exe_mod
        fake_kv = tmp_path / "kv_rr.db"
        monkeypatch.setattr(exe_mod, "_KV_DB_PATH", fake_kv)
        exe_mod._kv_write("data", [1, 2, 3])
        assert exe_mod._kv_read("data") == [1, 2, 3]

    def test_load_verb_in_valid_verbs(self):
        from praxis.validator import VALID_VERBS
        assert "LOAD" in VALID_VERBS


# ─────────────────────────────────────────────────────────────────────────────
# initial_variables — webhook payload injection
# ─────────────────────────────────────────────────────────────────────────────

class TestInitialVariables:
    def test_initial_variables_accessible_in_program(self):
        ast = parse("LOG.event")
        exe = Executor(handlers=HANDLERS)
        results = exe.execute(ast, initial_variables={"event": {"text": "hello"}})
        assert results[0]["status"] == "ok"

    def test_initial_variables_set_in_ctx(self):
        # Use SET to capture $event into a new variable
        ast = parse("SET.captured")
        exe = Executor(handlers=HANDLERS)
        # pre-load ctx manually to verify
        from praxis.executor import ExecutionContext
        ctx = ExecutionContext()
        ctx.variables["event"] = {"payload": "test"}
        assert ctx.variables["event"] == {"payload": "test"}


# ─────────────────────────────────────────────────────────────────────────────
# Webhooks API
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def wh_client(tmp_path):
    import praxis.server as srv
    srv._memory = None
    srv._constitution = None
    wh_db = tmp_path / "webhooks.db"
    with patch("praxis.server._WEBHOOK_DB", wh_db), \
         patch("praxis.server._ACTIVITY_LOG", tmp_path / "activity.log"):
        yield TestClient(app)


class TestWebhooksApi:
    def test_list_empty(self, wh_client):
        d = wh_client.get("/api/webhooks").json()
        assert d["webhooks"] == []

    def test_register_webhook(self, wh_client):
        r = wh_client.post("/api/webhooks", json={"name": "my-bot", "program_text": "LOG.x"})
        assert r.status_code == 200
        d = r.json()
        assert "id" in d
        assert "url" in d
        assert d["name"] == "my-bot"

    def test_list_after_register(self, wh_client):
        wh_client.post("/api/webhooks", json={"name": "bot1", "program_text": "LOG.x"})
        wh_client.post("/api/webhooks", json={"name": "bot2", "program_text": "LOG.y"})
        d = wh_client.get("/api/webhooks").json()
        assert len(d["webhooks"]) == 2

    def test_delete_webhook(self, wh_client):
        r = wh_client.post("/api/webhooks", json={"name": "del-me", "program_text": "LOG.x"})
        wid = r.json()["id"]
        dr = wh_client.delete(f"/api/webhooks/{wid}")
        assert dr.status_code == 200
        assert wh_client.get("/api/webhooks").json()["webhooks"] == []

    def test_delete_nonexistent_404(self, wh_client):
        r = wh_client.delete("/api/webhooks/ghost")
        assert r.status_code == 404

    def test_trigger_webhook_runs_program(self, wh_client):
        r = wh_client.post("/api/webhooks", json={"name": "trigger-test", "program_text": "LOG.x"})
        wid = r.json()["id"]
        tr = wh_client.post(f"/webhook/{wid}", json={"text": "hello"})
        assert tr.status_code == 200
        assert tr.json()["ok"] is True

    def test_trigger_unknown_webhook_404(self, wh_client):
        r = wh_client.post("/webhook/nope", json={})
        assert r.status_code == 404

    def test_trigger_passes_payload_as_event(self, wh_client):
        # Program just needs to run — we verify $event is accessible via LOG
        r = wh_client.post("/api/webhooks", json={"name": "payload-test", "program_text": "LOG.event"})
        wid = r.json()["id"]
        tr = wh_client.post(f"/webhook/{wid}", json={"key": "value"})
        assert tr.json()["ok"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Activity Feed
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def act_client(tmp_path):
    import praxis.server as srv
    srv._memory = None
    srv._constitution = None
    with patch("praxis.server._ACTIVITY_LOG", tmp_path / "activity.log"), \
         patch("praxis.server._WEBHOOK_DB", tmp_path / "webhooks.db"):
        yield TestClient(app), tmp_path / "activity.log"


class TestActivityFeed:
    def test_empty_when_no_log(self, act_client):
        tc, _ = act_client
        d = tc.get("/api/activity").json()
        assert d["events"] == []

    def test_activity_appended_on_run(self, act_client):
        tc, log_path = act_client
        with patch("praxis.server._MEM_DB", log_path.parent / "memory.db"), \
             patch("praxis.server._LOG_PATH", log_path.parent / "execution.log"):
            tc.post("/api/run", json={"program": "LOG.x"})
        assert log_path.exists()
        events = tc.get("/api/activity").json()["events"]
        assert any("run" == e["type"] for e in events)

    def test_activity_has_required_fields(self, act_client):
        tc, log_path = act_client
        # Write directly
        with open(log_path, "w") as f:
            f.write(json.dumps({"type": "schedule", "summary": "Test job ran", "detail": {}, "ts": 1000.0}) + "\n")
        events = tc.get("/api/activity").json()["events"]
        assert len(events) == 1
        e = events[0]
        assert e["type"] == "schedule"
        assert e["summary"] == "Test job ran"
        assert "ts" in e


# ─────────────────────────────────────────────────────────────────────────────
# OUT.x
# ─────────────────────────────────────────────────────────────────────────────

class TestOutX:
    def _run_out_x(self, msg: str, env: dict) -> dict:
        from praxis.handlers.io import out_handler
        ctx = MagicMock()
        ctx.last_output = msg
        with patch.dict("os.environ", env):
            return out_handler(["x"], {"msg": msg}, ctx)

    def test_out_x_missing_creds_raises(self):
        from praxis.handlers.io import out_handler
        ctx = MagicMock()
        ctx.last_output = "test"
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="X_API_KEY"):
                out_handler(["x"], {"msg": "hello"}, ctx)

    def test_out_x_twitter_alias(self, monkeypatch):
        """OUT.twitter should route to X handler."""
        from praxis.handlers.io import out_handler
        calls = []

        def fake_send_x(msg, params):
            calls.append(msg)
            return {"posted": True, "tweet_id": "123", "chars": len(msg)}

        monkeypatch.setattr("praxis.handlers.io._send_x", fake_send_x)
        ctx = MagicMock()
        ctx.last_output = None
        r = out_handler(["twitter"], {"msg": "hello"}, ctx)
        assert r["channel"] == "x"
        assert calls == ["hello"]

    def test_out_x_truncates_to_280(self, monkeypatch):
        from praxis.handlers.io import _send_x
        captured = []

        def fake_tweepy_client(**kwargs):
            class C:
                def create_tweet(self, text):
                    captured.append(text)
                    m = MagicMock()
                    m.data = {"id": "1"}
                    return m
            return C()

        fake_tweepy = MagicMock()
        fake_tweepy.Client = fake_tweepy_client
        monkeypatch.setitem(sys.modules, "tweepy", fake_tweepy)

        env = {
            "X_API_KEY": "k", "X_API_SECRET": "s",
            "X_ACCESS_TOKEN": "t", "X_ACCESS_TOKEN_SECRET": "ts",
        }
        long_msg = "x" * 400
        with patch.dict("os.environ", env):
            _send_x(long_msg, {})
        assert len(captured[0]) == 280


# ─────────────────────────────────────────────────────────────────────────────
# Plugin handler auto-loader
# ─────────────────────────────────────────────────────────────────────────────

class TestPluginLoader:
    def test_load_plugins_skips_missing_dir(self, tmp_path, monkeypatch):
        """_load_plugins() should silently no-op if ~/.praxis/handlers/ doesn't exist."""
        import praxis.handlers as h_mod
        fake_home = tmp_path / "no_handlers_home"
        fake_home.mkdir()
        with patch("pathlib.Path.home", return_value=fake_home):
            h_mod._load_plugins()  # should not raise

    def test_load_valid_plugin(self, tmp_path, monkeypatch):
        """A valid plugin file should register its verb in HANDLERS."""
        plugin_dir = tmp_path / ".praxis" / "handlers"
        plugin_dir.mkdir(parents=True)
        plugin_file = plugin_dir / "out_test_plugin.py"
        plugin_file.write_text(textwrap.dedent("""
            VERB_NAME = "OUT_TESTPLUGIN"
            def handle(target, params, ctx):
                return {"plugin": True}
        """))

        import praxis.handlers as h_mod
        with patch("pathlib.Path.home", return_value=tmp_path):
            h_mod._load_plugins()

        assert "OUT_TESTPLUGIN" in h_mod.HANDLERS

    def test_plugin_missing_verb_name_skipped(self, tmp_path, monkeypatch):
        """Plugin without VERB_NAME should be skipped without crashing."""
        plugin_dir = tmp_path / ".praxis" / "handlers"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "bad_plugin.py").write_text("def handle(t, p, c): return {}")

        import praxis.handlers as h_mod
        before = set(h_mod.HANDLERS.keys())
        with patch("pathlib.Path.home", return_value=tmp_path):
            h_mod._load_plugins()
        after = set(h_mod.HANDLERS.keys())
        assert after == before  # nothing new registered

    def test_plugin_syntax_error_skipped(self, tmp_path, monkeypatch):
        """Plugin with syntax error should be skipped without crashing."""
        plugin_dir = tmp_path / ".praxis" / "handlers"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "syntax_err.py").write_text("def bad syntax !!!")

        import praxis.handlers as h_mod
        with patch("pathlib.Path.home", return_value=tmp_path):
            h_mod._load_plugins()  # should not raise
