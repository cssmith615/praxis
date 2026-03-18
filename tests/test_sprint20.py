"""
Sprint 20 — Distributed Workers

Tests cover:
  - WorkerRegistration: stale detection, touch, to_dict
  - RemoteWorkerHub: register, heartbeat, deregister, list, route, dispatch
  - RemoteWorker: execute (mocked HTTP), health_check, error handling
  - WorkerClient: discover, get, register, heartbeat, deregister (mocked HTTP)
  - SPAWN handler: url= param creates RemoteWorker; no url= creates local Worker
  - bridge.py /workers/* endpoints (FastAPI TestClient)
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# WorkerRegistration
# ──────────────────────────────────────────────────────────────────────────────

class TestWorkerRegistration:
    def _make(self, **kwargs):
        from praxis.distributed import WorkerRegistration
        defaults = dict(agent_id="w1", role="data", verbs=["ING", "CLN"], url="http://host:7821")
        defaults.update(kwargs)
        return WorkerRegistration(**defaults)

    def test_not_stale_immediately(self):
        reg = self._make()
        assert not reg.is_stale()

    def test_stale_after_ttl(self):
        from praxis.distributed import HEARTBEAT_TTL
        reg = self._make()
        # Backdate last_seen
        old = (datetime.now(timezone.utc) - timedelta(seconds=HEARTBEAT_TTL + 1)).isoformat()
        reg.last_seen = old
        assert reg.is_stale()

    def test_touch_resets_staleness(self):
        from praxis.distributed import HEARTBEAT_TTL
        reg = self._make()
        old = (datetime.now(timezone.utc) - timedelta(seconds=HEARTBEAT_TTL + 1)).isoformat()
        reg.last_seen = old
        assert reg.is_stale()
        reg.touch()
        assert not reg.is_stale()

    def test_to_dict_keys(self):
        reg = self._make()
        d = reg.to_dict()
        for key in ("agent_id", "role", "verbs", "url", "registered_at", "last_seen", "stale"):
            assert key in d

    def test_to_dict_stale_false_when_fresh(self):
        assert not self._make().to_dict()["stale"]

    def test_to_dict_stale_true_when_old(self):
        from praxis.distributed import HEARTBEAT_TTL
        reg = self._make()
        reg.last_seen = (
            datetime.now(timezone.utc) - timedelta(seconds=HEARTBEAT_TTL + 5)
        ).isoformat()
        assert reg.to_dict()["stale"]


# ──────────────────────────────────────────────────────────────────────────────
# RemoteWorkerHub
# ──────────────────────────────────────────────────────────────────────────────

class TestRemoteWorkerHub:
    @pytest.fixture
    def hub(self):
        from praxis.distributed import RemoteWorkerHub
        return RemoteWorkerHub()

    def test_register_and_get(self, hub):
        reg = hub.register("w1", "data", ["ING", "CLN"], "http://h:7821")
        assert reg.agent_id == "w1"
        assert hub.get("w1") is not None

    def test_verbs_normalized_uppercase(self, hub):
        reg = hub.register("w1", "data", ["ing", "cln"], "http://h:7821")
        assert "ING" in reg.verbs
        assert "CLN" in reg.verbs

    def test_get_unknown_returns_none(self, hub):
        assert hub.get("unknown") is None

    def test_heartbeat_updates_last_seen(self, hub):
        from praxis.distributed import HEARTBEAT_TTL
        hub.register("w1", "data", ["ING"], "http://h:7821")
        reg = hub.get("w1")
        old = (datetime.now(timezone.utc) - timedelta(seconds=HEARTBEAT_TTL + 1)).isoformat()
        reg.last_seen = old
        assert hub.heartbeat("w1") is True
        assert not hub.get("w1").is_stale()

    def test_heartbeat_unknown_returns_false(self, hub):
        assert hub.heartbeat("nobody") is False

    def test_deregister(self, hub):
        hub.register("w1", "data", ["ING"], "http://h:7821")
        assert hub.deregister("w1") is True
        assert hub.get("w1") is None

    def test_deregister_unknown_returns_false(self, hub):
        assert hub.deregister("nobody") is False

    def test_list_all(self, hub):
        hub.register("w1", "data", ["ING"], "http://h1:7821")
        hub.register("w2", "analysis", ["SUMM"], "http://h2:7821")
        assert len(hub.list_all()) == 2

    def test_route_by_verb(self, hub):
        hub.register("w1", "data", ["ING", "CLN"], "http://h:7821")
        reg = hub.route("ING")
        assert reg is not None
        assert reg.agent_id == "w1"

    def test_route_unknown_verb_returns_none(self, hub):
        hub.register("w1", "data", ["ING"], "http://h:7821")
        assert hub.route("SUMM") is None

    def test_route_skips_stale(self, hub):
        from praxis.distributed import HEARTBEAT_TTL
        hub.register("w1", "data", ["ING"], "http://h:7821")
        reg = hub.get("w1")
        reg.last_seen = (
            datetime.now(timezone.utc) - timedelta(seconds=HEARTBEAT_TTL + 5)
        ).isoformat()
        assert hub.route("ING") is None

    def test_dispatch_unregistered(self, hub):
        result = hub.dispatch("nobody", "LOG.test")
        assert result["ok"] is False
        assert "not registered" in result["errors"][0]

    def test_dispatch_stale_worker(self, hub):
        from praxis.distributed import HEARTBEAT_TTL
        hub.register("w1", "data", ["ING"], "http://h:7821")
        reg = hub.get("w1")
        reg.last_seen = (
            datetime.now(timezone.utc) - timedelta(seconds=HEARTBEAT_TTL + 5)
        ).isoformat()
        result = hub.dispatch("w1", "LOG.test")
        assert result["ok"] is False

    def test_dispatch_calls_http(self, hub):
        hub.register("w1", "data", ["ING"], "http://h:7821")
        with patch("praxis.distributed._http_post", return_value={"ok": True, "results": []}) as mock_post:
            hub.dispatch("w1", "LOG.test", mode="dev")
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "/execute" in call_args[0][0]


# ──────────────────────────────────────────────────────────────────────────────
# RemoteWorker
# ──────────────────────────────────────────────────────────────────────────────

class TestRemoteWorker:
    @pytest.fixture
    def worker(self):
        from praxis.distributed import RemoteWorker
        return RemoteWorker(
            agent_id="rw1", role="data", verbs=["ING", "CLN"],
            url="http://remote:7821", mode="dev",
        )

    def test_execute_success(self, worker):
        mock_resp = {
            "ok": True,
            "results": [
                {"verb": "LOG", "target": ["test"], "params": {}, "output": "logged",
                 "status": "ok", "duration_ms": 5, "log_entry": ""},
            ],
            "errors": [],
        }
        with patch("praxis.distributed._http_post", return_value=mock_resp):
            result = worker.execute("LOG.test")
        assert result["status"] == "ok"
        assert result["remote"] is True
        assert result["agent_id"] == "rw1"
        assert result["steps"] == 1
        assert result["output"] == "logged"

    def test_execute_remote_error_response(self, worker):
        mock_resp = {"ok": False, "results": [], "errors": ["parse error"]}
        with patch("praxis.distributed._http_post", return_value=mock_resp):
            result = worker.execute("BAD!!!")
        assert result["status"] == "error"
        assert "parse error" in result["error"]

    def test_execute_http_exception(self, worker):
        with patch("praxis.distributed._http_post", side_effect=Exception("connection refused")):
            result = worker.execute("LOG.test")
        assert result["status"] == "error"
        assert "connection refused" in result["error"]

    def test_execute_empty_results(self, worker):
        mock_resp = {"ok": True, "results": [], "errors": []}
        with patch("praxis.distributed._http_post", return_value=mock_resp):
            result = worker.execute("LOG.test")
        assert result["output"] is None

    def test_health_check_ok(self, worker):
        with patch("praxis.distributed._http_get", return_value={"status": "ok"}):
            assert worker.health_check() is True

    def test_health_check_fail(self, worker):
        with patch("praxis.distributed._http_get", side_effect=Exception("timeout")):
            assert worker.health_check() is False

    def test_repr(self, worker):
        r = repr(worker)
        assert "rw1" in r
        assert "http://remote:7821" in r

    def test_url_trailing_slash_stripped(self):
        from praxis.distributed import RemoteWorker
        w = RemoteWorker("w", "r", [], "http://host:7821/")
        assert not w.url.endswith("/")


# ──────────────────────────────────────────────────────────────────────────────
# WorkerClient
# ──────────────────────────────────────────────────────────────────────────────

class TestWorkerClient:
    @pytest.fixture
    def client(self):
        from praxis.distributed import WorkerClient
        return WorkerClient("http://hub:7821")

    def test_discover_returns_remote_workers(self, client):
        mock_entries = [
            {"agent_id": "w1", "role": "data", "verbs": ["ING"], "url": "http://h1:7821",
             "registered_at": "2026-01-01T00:00:00+00:00",
             "last_seen": "2026-01-01T00:00:00+00:00", "stale": False},
        ]
        with patch("praxis.distributed._http_get", return_value=mock_entries):
            workers = client.discover()
        assert len(workers) == 1
        assert workers[0].agent_id == "w1"

    def test_discover_filters_stale(self, client):
        mock_entries = [
            {"agent_id": "w1", "role": "data", "verbs": ["ING"], "url": "http://h1:7821",
             "registered_at": "2026-01-01T00:00:00+00:00",
             "last_seen": "2026-01-01T00:00:00+00:00", "stale": True},
        ]
        with patch("praxis.distributed._http_get", return_value=mock_entries):
            workers = client.discover()
        assert workers == []

    def test_discover_filters_by_role(self, client):
        mock_entries = [
            {"agent_id": "w1", "role": "data", "verbs": ["ING"], "url": "http://h1:7821",
             "registered_at": "t", "last_seen": "t", "stale": False},
            {"agent_id": "w2", "role": "analysis", "verbs": ["SUMM"], "url": "http://h2:7821",
             "registered_at": "t", "last_seen": "t", "stale": False},
        ]
        with patch("praxis.distributed._http_get", return_value=mock_entries):
            workers = client.discover(role="data")
        assert len(workers) == 1
        assert workers[0].role == "data"

    def test_discover_http_error_returns_empty(self, client):
        with patch("praxis.distributed._http_get", side_effect=Exception("network error")):
            assert client.discover() == []

    def test_get_returns_remote_worker(self, client):
        mock_entry = {"agent_id": "w1", "role": "data", "verbs": ["ING"], "url": "http://h1:7821",
                      "registered_at": "t", "last_seen": "t", "stale": False}
        with patch("praxis.distributed._http_get", return_value=mock_entry):
            w = client.get("w1")
        assert w is not None
        assert w.agent_id == "w1"

    def test_get_stale_returns_none(self, client):
        mock_entry = {"agent_id": "w1", "role": "data", "verbs": ["ING"], "url": "http://h1:7821",
                      "registered_at": "t", "last_seen": "t", "stale": True}
        with patch("praxis.distributed._http_get", return_value=mock_entry):
            assert client.get("w1") is None

    def test_register_success(self, client):
        with patch("praxis.distributed._http_post", return_value={"ok": True, "agent_id": "w1"}):
            assert client.register("w1", "data", ["ING"], "http://h:7821") is True

    def test_register_failure(self, client):
        with patch("praxis.distributed._http_post", side_effect=Exception("refused")):
            assert client.register("w1", "data", ["ING"], "http://h:7821") is False

    def test_heartbeat_success(self, client):
        with patch("praxis.distributed._http_post", return_value={"ok": True}):
            assert client.heartbeat("w1") is True

    def test_deregister_success(self, client):
        with patch("praxis.distributed._http_delete", return_value={"ok": True}):
            assert client.deregister("w1") is True


# ──────────────────────────────────────────────────────────────────────────────
# SPAWN handler — remote vs local routing
# ──────────────────────────────────────────────────────────────────────────────

class TestSpawnHandlerDistributed:
    def _make_ctx(self):
        ctx = MagicMock()
        ctx.mode = "dev"
        ctx.agent_registry = None
        ctx.pending_futures = {}
        ctx._handlers = None
        return ctx

    def test_spawn_local_no_url(self):
        from praxis.handlers.agents import spawn_handler
        ctx = self._make_ctx()
        result = spawn_handler(["my_worker"], {"role": "data", "verbs": ["ING"]}, ctx)
        assert result["remote"] is False
        assert result["status"] == "spawned"
        assert ctx.agent_registry is not None

    def test_spawn_remote_with_url(self):
        from praxis.handlers.agents import spawn_handler
        from praxis.distributed import RemoteWorker
        ctx = self._make_ctx()
        result = spawn_handler(
            ["my_worker"],
            {"role": "data", "verbs": ["ING"], "url": "http://remote:7821"},
            ctx,
        )
        assert result["remote"] is True
        assert result["url"] == "http://remote:7821"
        assert result["status"] == "spawned"
        # The registered entity should be a RemoteWorker
        registered = ctx.agent_registry.get("my_worker")
        assert isinstance(registered, RemoteWorker)

    def test_spawn_remote_registers_in_context(self):
        from praxis.handlers.agents import spawn_handler
        ctx = self._make_ctx()
        spawn_handler(
            ["rw"],
            {"role": "data", "verbs": ["LOG"], "url": "http://r:7821"},
            ctx,
        )
        assert ctx.agent_registry.get("rw") is not None

    def test_spawn_remote_worker_execute_routes_http(self):
        from praxis.handlers.agents import spawn_handler
        ctx = self._make_ctx()
        spawn_handler(
            ["rw"],
            {"role": "data", "verbs": ["LOG"], "url": "http://r:7821"},
            ctx,
        )
        worker = ctx.agent_registry.get("rw")
        mock_resp = {"ok": True, "results": [], "errors": []}
        with patch("praxis.distributed._http_post", return_value=mock_resp) as mock_post:
            worker.execute("LOG.test")
            mock_post.assert_called_once()


# ──────────────────────────────────────────────────────────────────────────────
# Bridge /workers/* endpoints
# ──────────────────────────────────────────────────────────────────────────────

class TestBridgeWorkerEndpoints:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from praxis.bridge import app, _worker_hub
        # Clear hub state before each test
        for wid in list(_worker_hub._workers.keys()):
            _worker_hub.deregister(wid)
        return TestClient(app)

    def test_register(self, client):
        resp = client.post("/workers/register", json={
            "agent_id": "w1", "role": "data", "verbs": ["ING", "CLN"],
            "url": "http://remote:7821"
        })
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_list_workers(self, client):
        client.post("/workers/register", json={
            "agent_id": "w2", "role": "data", "verbs": ["ING"], "url": "http://h:7821"
        })
        resp = client.get("/workers")
        assert resp.status_code == 200
        workers = resp.json()
        assert any(w["agent_id"] == "w2" for w in workers)

    def test_get_worker(self, client):
        client.post("/workers/register", json={
            "agent_id": "w3", "role": "analysis", "verbs": ["SUMM"], "url": "http://h:7821"
        })
        resp = client.get("/workers/w3")
        assert resp.status_code == 200
        assert resp.json()["agent_id"] == "w3"

    def test_get_worker_not_found(self, client):
        resp = client.get("/workers/nobody")
        assert resp.status_code == 404

    def test_heartbeat(self, client):
        client.post("/workers/register", json={
            "agent_id": "w4", "role": "data", "verbs": ["ING"], "url": "http://h:7821"
        })
        resp = client.post("/workers/w4/heartbeat")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_heartbeat_unknown(self, client):
        resp = client.post("/workers/nobody/heartbeat")
        assert resp.json()["ok"] is False

    def test_deregister(self, client):
        client.post("/workers/register", json={
            "agent_id": "w5", "role": "data", "verbs": ["ING"], "url": "http://h:7821"
        })
        resp = client.delete("/workers/w5")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        # Should be gone
        resp2 = client.get("/workers/w5")
        assert resp2.status_code == 404

    def test_dispatch_unregistered(self, client):
        resp = client.post("/workers/dispatch/nobody", json={"program": "LOG.test"})
        assert resp.status_code == 200
        assert resp.json()["ok"] is False

    def test_dispatch_calls_worker(self, client):
        client.post("/workers/register", json={
            "agent_id": "w6", "role": "data", "verbs": ["LOG"], "url": "http://r:7821"
        })
        mock_resp = {"ok": True, "results": [], "errors": []}
        with patch("praxis.distributed._http_post", return_value=mock_resp):
            resp = client.post("/workers/dispatch/w6", json={"program": "LOG.test"})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_health_still_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
