"""
Sprint F tests — OUT.pagerduty, OUT.jira, CAP.remediate.

All HTTP calls are mocked — no credentials or network required.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from praxis.executor import ExecutionContext
from praxis.handlers import HANDLERS
from praxis.handlers.io import _send_pagerduty, _send_jira
from praxis.handlers.agents import _cap_remediate, _REMEDIATE_ACTIONS


# ── OUT.pagerduty ─────────────────────────────────────────────────────────────

def _pd_mock(dedup_key="abc123"):
    resp = MagicMock()
    resp.json.return_value = {"status": "success", "dedup_key": dedup_key, "message": "Event processed"}
    resp.raise_for_status = MagicMock()
    return resp


def test_out_pagerduty_sends_correct_payload(monkeypatch):
    monkeypatch.setenv("PAGERDUTY_ROUTING_KEY", "test-key-xyz")
    with patch("httpx.post", return_value=_pd_mock()) as mock_post:
        result = HANDLERS["OUT"](["pagerduty"], {"summary": "High CPU spike", "severity": "error"}, ExecutionContext())
    assert result["channel"] == "pagerduty"
    assert result["status"] == "success"
    call_json = mock_post.call_args[1]["json"]
    assert call_json["routing_key"] == "test-key-xyz"
    assert call_json["payload"]["summary"] == "High CPU spike"
    assert call_json["payload"]["severity"] == "error"
    assert call_json["event_action"] == "trigger"


def test_out_pagerduty_missing_key_raises(monkeypatch):
    monkeypatch.delenv("PAGERDUTY_ROUTING_KEY", raising=False)
    with pytest.raises(ValueError, match="PAGERDUTY_ROUTING_KEY"):
        _send_pagerduty("alert", {})


def test_out_pagerduty_severity_defaults_to_error(monkeypatch):
    monkeypatch.setenv("PAGERDUTY_ROUTING_KEY", "k")
    with patch("httpx.post", return_value=_pd_mock()) as mock_post:
        _send_pagerduty("test msg", {})
    assert mock_post.call_args[1]["json"]["payload"]["severity"] == "error"


def test_out_pagerduty_invalid_severity_coerced(monkeypatch):
    monkeypatch.setenv("PAGERDUTY_ROUTING_KEY", "k")
    with patch("httpx.post", return_value=_pd_mock()) as mock_post:
        _send_pagerduty("test", {"severity": "ultra"})
    assert mock_post.call_args[1]["json"]["payload"]["severity"] == "error"


def test_out_pagerduty_passes_dedup_key(monkeypatch):
    monkeypatch.setenv("PAGERDUTY_ROUTING_KEY", "k")
    with patch("httpx.post", return_value=_pd_mock("my-dedup")) as mock_post:
        _send_pagerduty("test", {"dedup_key": "my-dedup"})
    assert mock_post.call_args[1]["json"]["dedup_key"] == "my-dedup"


def test_out_pagerduty_no_dedup_key_omitted(monkeypatch):
    monkeypatch.setenv("PAGERDUTY_ROUTING_KEY", "k")
    with patch("httpx.post", return_value=_pd_mock()) as mock_post:
        _send_pagerduty("test", {})
    assert "dedup_key" not in mock_post.call_args[1]["json"]


def test_out_pagerduty_summary_uses_msg_fallback(monkeypatch):
    monkeypatch.setenv("PAGERDUTY_ROUTING_KEY", "k")
    with patch("httpx.post", return_value=_pd_mock()) as mock_post:
        HANDLERS["OUT"](["pagerduty"], {"msg": "fallback message"}, ExecutionContext())
    payload = mock_post.call_args[1]["json"]["payload"]
    assert payload["summary"] == "fallback message"


# ── OUT.jira ──────────────────────────────────────────────────────────────────

def _jira_mock(key="SEC-42"):
    resp = MagicMock()
    resp.json.return_value = {"id": "10042", "key": key, "self": f"https://example.atlassian.net/rest/api/3/issue/10042"}
    resp.raise_for_status = MagicMock()
    return resp


def test_out_jira_sends_correct_structure(monkeypatch):
    monkeypatch.setenv("JIRA_BASE_URL", "https://example.atlassian.net")
    monkeypatch.setenv("JIRA_EMAIL", "user@example.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "token123")
    with patch("httpx.post", return_value=_jira_mock()) as mock_post:
        result = HANDLERS["OUT"](
            ["jira"],
            {"project": "SEC", "summary": "SQL injection detected", "priority": "Critical"},
            ExecutionContext(),
        )
    assert result["channel"] == "jira"
    assert result["key"] == "SEC-42"
    assert "SEC-42" in result["url"]
    body = mock_post.call_args[1]["json"]
    assert body["fields"]["project"]["key"] == "SEC"
    assert body["fields"]["summary"] == "SQL injection detected"
    assert body["fields"]["priority"]["name"] == "Critical"


def test_out_jira_missing_env_vars_raises(monkeypatch):
    monkeypatch.delenv("JIRA_BASE_URL",    raising=False)
    monkeypatch.delenv("JIRA_EMAIL",       raising=False)
    monkeypatch.delenv("JIRA_API_TOKEN",   raising=False)
    with pytest.raises(ValueError, match="JIRA_BASE_URL"):
        _send_jira("test", {})


def test_out_jira_uses_basic_auth(monkeypatch):
    monkeypatch.setenv("JIRA_BASE_URL",  "https://test.atlassian.net")
    monkeypatch.setenv("JIRA_EMAIL",     "admin@test.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "mytoken")
    with patch("httpx.post", return_value=_jira_mock()) as mock_post:
        _send_jira("msg", {})
    headers = mock_post.call_args[1]["headers"]
    assert headers["Authorization"].startswith("Basic ")


def test_out_jira_builds_browse_url(monkeypatch):
    monkeypatch.setenv("JIRA_BASE_URL",  "https://myorg.atlassian.net")
    monkeypatch.setenv("JIRA_EMAIL",     "x@x.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "tok")
    with patch("httpx.post", return_value=_jira_mock("OPS-7")):
        result = _send_jira("msg", {})
    assert result["url"] == "https://myorg.atlassian.net/browse/OPS-7"


def test_out_jira_parses_comma_labels(monkeypatch):
    monkeypatch.setenv("JIRA_BASE_URL",  "https://x.atlassian.net")
    monkeypatch.setenv("JIRA_EMAIL",     "x@x.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "tok")
    with patch("httpx.post", return_value=_jira_mock()) as mock_post:
        _send_jira("msg", {"labels": "security, incident, p1"})
    labels = mock_post.call_args[1]["json"]["fields"]["labels"]
    assert "security" in labels
    assert "incident" in labels


def test_out_jira_defaults_project_sec(monkeypatch):
    monkeypatch.setenv("JIRA_BASE_URL",  "https://x.atlassian.net")
    monkeypatch.setenv("JIRA_EMAIL",     "x@x.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "tok")
    with patch("httpx.post", return_value=_jira_mock()) as mock_post:
        _send_jira("msg", {})
    assert mock_post.call_args[1]["json"]["fields"]["project"]["key"] == "SEC"


# ── CAP.remediate ─────────────────────────────────────────────────────────────

def test_cap_remediate_dry_run_does_not_execute():
    ctx = ExecutionContext()
    result = HANDLERS["CAP"](["remediate", "isolate"], {"target": "10.0.0.5", "reason": "IOC match"}, ctx)
    assert result["action"] == "isolate"
    assert result["dry_run"] is True
    assert result["executed"] is False
    assert "dry run" in result["note"].lower()


def test_cap_remediate_all_valid_actions():
    ctx = ExecutionContext()
    for action in _REMEDIATE_ACTIONS:
        r = HANDLERS["CAP"](["remediate", action], {"target": "host1"}, ctx)
        assert r["action"] == action


def test_cap_remediate_invalid_action_raises():
    ctx = ExecutionContext()
    with pytest.raises(ValueError, match="unknown action"):
        HANDLERS["CAP"](["remediate", "nuke"], {}, ctx)


def test_cap_remediate_returns_required_fields():
    ctx = ExecutionContext()
    result = HANDLERS["CAP"](["remediate", "block"], {"target": "1.2.3.4", "reason": "C2 traffic"}, ctx)
    for key in ("action", "target", "reason", "environment", "approver", "dry_run", "timestamp", "executed"):
        assert key in result


def test_cap_remediate_execute_marks_executed():
    ctx = ExecutionContext()
    result = HANDLERS["CAP"](["remediate", "patch"], {"target": "webapp-01", "dry_run": "false"}, ctx)
    assert result["executed"] is True
    assert result["dry_run"] is False


def test_cap_remediate_environment_defaults_prod():
    ctx = ExecutionContext()
    result = HANDLERS["CAP"](["remediate", "notify"], {"target": "soc@company.com"}, ctx)
    assert result["environment"] == "prod"


def test_cap_remediate_records_approver():
    ctx = ExecutionContext()
    result = HANDLERS["CAP"](["remediate", "rollback"], {"target": "api-service", "approver": "alice"}, ctx)
    assert result["approver"] == "alice"


def test_cap_still_handles_allow_list():
    """Original CAP.agent_name(allow=[...]) must still work."""
    ctx = ExecutionContext()
    result = HANDLERS["CAP"](["worker"], {"role": "data", "allow": ["ing", "cln"]}, ctx)
    assert result["agent"] == "worker"
    assert "ING" in result["capabilities"]
