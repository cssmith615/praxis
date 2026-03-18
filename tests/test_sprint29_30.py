"""
Sprint 29 / 30 tests — Constitutional Audit Reports and praxis install registry.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from praxis.server import app


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 29 — Constitutional Audit
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def audit_client(tmp_path):
    import praxis.server as srv
    srv._memory = None
    srv._constitution = None
    act_log = tmp_path / "activity.log"
    wh_db   = tmp_path / "webhooks.db"
    mem_db  = tmp_path / "memory.db"
    log_p   = tmp_path / "execution.log"
    const_p = tmp_path / "constitution.md"
    const_p.write_text("# test\n[verb:LOG] ALWAYS use LOG for audit trails.\n")

    from praxis.memory import ProgramMemory
    from praxis.constitution import Constitution
    srv._memory       = ProgramMemory(db_path=str(mem_db))
    srv._constitution = Constitution(const_p)

    with patch("praxis.server._ACTIVITY_LOG", act_log), \
         patch("praxis.server._WEBHOOK_DB",   wh_db), \
         patch("praxis.server._LOG_PATH",     log_p), \
         patch("praxis.server._MEM_DB",       mem_db):
        yield TestClient(app), {"activity": act_log, "const": const_p}


class TestConstitutionalAudit:
    def test_audit_endpoint_empty(self, audit_client):
        tc, _ = audit_client
        d = tc.get("/api/audit").json()
        assert d["audits"] == []

    def test_run_produces_audit_record(self, audit_client):
        tc, paths = audit_client
        tc.post("/api/run", json={"program": "LOG.x"})
        d = tc.get("/api/audit").json()
        assert len(d["audits"]) >= 1

    def test_audit_has_required_fields(self, audit_client):
        tc, _ = audit_client
        tc.post("/api/run", json={"program": "LOG.x"})
        entry = tc.get("/api/audit").json()["audits"][0]
        assert entry["type"] == "audit"
        assert "summary" in entry
        assert "ts" in entry
        det = entry.get("detail", {})
        assert "verbs" in det
        assert "rules_checked" in det
        assert "violations" in det

    def test_audit_detects_verbs_used(self, audit_client):
        tc, _ = audit_client
        tc.post("/api/run", json={"program": "LOG.x"})
        entry = tc.get("/api/audit").json()["audits"][0]
        assert "LOG" in entry["detail"]["verbs"]

    def test_audit_all_passed_when_no_errors(self, audit_client):
        tc, _ = audit_client
        tc.post("/api/run", json={"program": "LOG.x"})
        entry = tc.get("/api/audit").json()["audits"][0]
        assert entry["detail"]["violations"] == []
        assert "passed" in entry["summary"].lower()

    def test_audit_checks_constitutional_rules(self, audit_client):
        tc, _ = audit_client
        tc.post("/api/run", json={"program": "LOG.x"})
        entry = tc.get("/api/audit").json()["audits"][0]
        # LOG verb has a rule in the test constitution — should be checked
        assert entry["detail"]["rules_checked"] >= 1

    def test_audit_summary_includes_verbs(self, audit_client):
        tc, _ = audit_client
        tc.post("/api/run", json={"program": "LOG.step"})
        entry = tc.get("/api/audit").json()["audits"][0]
        assert "LOG" in entry["summary"]

    def test_audit_limit_param(self, audit_client):
        tc, _ = audit_client
        for _ in range(5):
            tc.post("/api/run", json={"program": "LOG.x"})
        d = tc.get("/api/audit?limit=3").json()
        assert len(d["audits"]) <= 3

    def test_audit_webhook_trigger_also_audited(self, audit_client, tmp_path):
        tc, _ = audit_client
        wh_db = tmp_path / "wh2.db"
        with patch("praxis.server._WEBHOOK_DB", wh_db):
            r = tc.post("/api/webhooks", json={"name": "test", "program_text": "LOG.event"})
            wid = r.json()["id"]
            tc.post(f"/webhook/{wid}", json={"text": "hello"})
        d = tc.get("/api/audit").json()
        assert any("webhook" in e["summary"].lower() for e in d["audits"])


class TestAuditRunHelper:
    def test_audit_run_returns_dict(self, audit_client, tmp_path):
        from praxis.server import _audit_run
        act_log = tmp_path / "act2.log"
        with patch("praxis.server._ACTIVITY_LOG", act_log):
            result = _audit_run("LOG.x", [{"verb": "LOG", "target": ["x"], "status": "ok", "log_entry": ""}], label="test")
        assert "verbs" in result
        assert "violations" in result
        assert "summary" in result

    def test_audit_run_detects_violations(self, audit_client, tmp_path):
        from praxis.server import _audit_run
        act_log = tmp_path / "act3.log"
        bad_results = [{"verb": "LOG", "target": ["x"], "status": "error", "log_entry": "something failed"}]
        with patch("praxis.server._ACTIVITY_LOG", act_log):
            result = _audit_run("LOG.x", bad_results, label="test")
        assert len(result["violations"]) == 1

    def test_audit_run_no_violation_on_ok_steps(self, audit_client, tmp_path):
        from praxis.server import _audit_run
        act_log = tmp_path / "act4.log"
        ok_results = [{"verb": "LOG", "target": ["x"], "status": "ok", "log_entry": ""}]
        with patch("praxis.server._ACTIVITY_LOG", act_log):
            result = _audit_run("LOG.x", ok_results, label="test")
        assert result["violations"] == []


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 30 — Program Registry
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLE_REGISTRY = {
    "version": "1",
    "programs": [
        {
            "name": "news-brief",
            "description": "Fetch top HN stories and summarize",
            "author": "cssmith615",
            "version": "1.0.0",
            "tags": ["news", "summarize"],
            "program": "FETCH.data(src=\"https://example.com\") -> SUMM.text -> OUT.telegram",
        },
        {
            "name": "price-alert",
            "description": "Alert on price drops",
            "author": "cssmith615",
            "version": "1.0.0",
            "tags": ["price", "alert"],
            "program": "FETCH.price -> EVAL.threshold -> OUT.telegram",
        },
    ],
}


class TestRegistryFetch:
    def test_fetch_registry_returns_programs(self):
        from praxis.registry import fetch_registry
        with patch("urllib.request.urlopen") as mock_url:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(_SAMPLE_REGISTRY).encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_url.return_value = mock_resp
            programs = fetch_registry("http://fake.registry/index.json")
        assert len(programs) == 2
        assert programs[0].name == "news-brief"

    def test_fetch_registry_falls_back_to_local(self, tmp_path):
        from praxis.registry import fetch_registry
        local = tmp_path / "index.json"
        local.write_text(json.dumps(_SAMPLE_REGISTRY))
        with patch("urllib.request.urlopen", side_effect=Exception("offline")), \
             patch("praxis.registry._LOCAL_REGISTRY", local):
            programs = fetch_registry()
        assert len(programs) == 2

    def test_fetch_registry_raises_when_no_source(self, tmp_path):
        from praxis.registry import fetch_registry, RegistryError
        with patch("urllib.request.urlopen", side_effect=Exception("offline")), \
             patch("praxis.registry._LOCAL_REGISTRY", tmp_path / "nonexistent.json"):
            with pytest.raises(RegistryError):
                fetch_registry()


class TestRegistrySearch:
    def _mock_fetch(self, monkeypatch):
        from praxis import registry as reg_mod
        programs = [reg_mod.RegistryProgram(p) for p in _SAMPLE_REGISTRY["programs"]]
        monkeypatch.setattr(reg_mod, "fetch_registry", lambda *a, **kw: programs)

    def test_search_empty_returns_all(self, monkeypatch):
        self._mock_fetch(monkeypatch)
        from praxis.registry import search_registry
        results = search_registry("")
        assert len(results) == 2

    def test_search_by_name(self, monkeypatch):
        self._mock_fetch(monkeypatch)
        from praxis.registry import search_registry
        results = search_registry("news")
        assert len(results) == 1
        assert results[0].name == "news-brief"

    def test_search_by_tag(self, monkeypatch):
        self._mock_fetch(monkeypatch)
        from praxis.registry import search_registry
        results = search_registry("alert")
        assert any(r.name == "price-alert" for r in results)

    def test_search_case_insensitive(self, monkeypatch):
        self._mock_fetch(monkeypatch)
        from praxis.registry import search_registry
        results = search_registry("NEWS")
        assert len(results) == 1

    def test_search_no_match_returns_empty(self, monkeypatch):
        self._mock_fetch(monkeypatch)
        from praxis.registry import search_registry
        results = search_registry("xyzzy_no_match")
        assert results == []


class TestRegistryInstall:
    def _mock_fetch(self, monkeypatch, programs=None):
        from praxis import registry as reg_mod
        data = programs or _SAMPLE_REGISTRY["programs"]
        progs = [reg_mod.RegistryProgram(p) for p in data]
        monkeypatch.setattr(reg_mod, "fetch_registry", lambda *a, **kw: progs)

    def test_install_stores_in_memory(self, monkeypatch, tmp_path):
        self._mock_fetch(monkeypatch)
        from praxis.registry import install_program
        from praxis.memory import ProgramMemory
        mem = ProgramMemory(db_path=str(tmp_path / "mem.db"))
        prog = install_program("news-brief", memory=mem)
        assert prog.name == "news-brief"
        assert mem.count() == 1

    def test_install_not_found_raises(self, monkeypatch):
        self._mock_fetch(monkeypatch)
        from praxis.registry import install_program, RegistryError
        with pytest.raises(RegistryError, match="not found"):
            install_program("nonexistent-program")

    def test_install_program_fields(self, monkeypatch, tmp_path):
        self._mock_fetch(monkeypatch)
        from praxis.registry import install_program
        from praxis.memory import ProgramMemory
        mem = ProgramMemory(db_path=str(tmp_path / "mem2.db"))
        prog = install_program("price-alert", memory=mem)
        assert prog.description == "Alert on price drops"
        assert prog.author == "cssmith615"


class TestRegistryPublish:
    def test_publish_creates_files(self, tmp_path):
        from praxis.registry import publish_program
        program_text = "LOG.x -> OUT.telegram"
        out_path = tmp_path / "my-prog.px"
        meta = publish_program(
            program_text=program_text,
            name="my-prog",
            description="Test program",
            tags=["test"],
            author="me",
            output_path=out_path,
        )
        assert out_path.exists()
        assert out_path.with_suffix(".json").exists()
        assert meta["name"] == "my-prog"
        assert meta["tags"] == ["test"]

    def test_publish_metadata_structure(self, tmp_path):
        from praxis.registry import publish_program
        out = tmp_path / "prog.px"
        meta = publish_program("LOG.x", "prog", "A program", tags=["a", "b"], output_path=out)
        assert "name" in meta
        assert "description" in meta
        assert "version" in meta
        assert "program" in meta
        assert meta["program"] == "LOG.x"

    def test_registry_program_repr(self):
        from praxis.registry import RegistryProgram
        p = RegistryProgram({"name": "test", "description": "A test program"})
        assert "test" in repr(p)


class TestLocalRegistryIndex:
    def test_local_registry_is_valid_json(self):
        local = Path(__file__).parent.parent / "registry" / "index.json"
        assert local.exists(), "registry/index.json should exist"
        data = json.loads(local.read_text())
        assert "version" in data
        assert "programs" in data
        assert len(data["programs"]) > 0

    def test_local_registry_programs_have_required_fields(self):
        local = Path(__file__).parent.parent / "registry" / "index.json"
        data = json.loads(local.read_text())
        for prog in data["programs"]:
            assert "name" in prog, f"missing name in {prog}"
            assert "description" in prog
            assert "program" in prog or "url" in prog
