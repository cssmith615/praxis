"""
Sprint E tests — ING.siem, ING.threat_intel, EVAL.risk.

All network and LLM calls are mocked.  No credentials or internet required.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from praxis.security import (
    normalize_siem_alert,
    fetch_threat_intel,
    score_risk,
    _detect_format,
    _parse_risk_response,
    _score_to_priority,
    _cvss_to_severity,
)
from praxis.executor import ExecutionContext
from praxis.handlers import HANDLERS


# ── SIEM format detection ─────────────────────────────────────────────────────

def test_detect_splunk():
    assert _detect_format({"_time": "2024-01-01", "sourcetype": "ids", "_raw": "..."}) == "splunk"

def test_detect_elastic():
    assert _detect_format({"@timestamp": "2024-01-01T00:00:00Z", "event": {"kind": "alert"}, "host": {"name": "web1"}}) == "elastic"

def test_detect_qradar():
    assert _detect_format({"qid": 100, "magnitude": 7, "sourceIP": "1.2.3.4"}) == "qradar"
    assert _detect_format({"offense": {}, "description": "Brute force"}) == "qradar"

def test_detect_cef_from_raw_key():
    assert _detect_format({"_raw_cef": "CEF:0|..."}) == "cef"

def test_detect_cef_from_message():
    assert _detect_format({"message": "CEF:0|Vendor|Product|1.0|100|Login|5|src=1.2.3.4"}) == "cef"

def test_detect_leef_from_raw_key():
    assert _detect_format({"_raw_leef": "LEEF:1.0|..."}) == "leef"

def test_detect_generic_fallback():
    assert _detect_format({"description": "something happened", "severity": 3}) == "generic"


# ── SIEM normalization ────────────────────────────────────────────────────────

def test_normalize_splunk_fields():
    raw = {
        "_time": "2024-01-15T10:00:00Z",
        "sourcetype": "suricata",
        "index": "security",
        "_raw": "...",
        "severity": 3,
        "signature": "ET SCAN Port Scan Detected",
        "src_ip": "10.0.0.5",
        "dest_ip": "192.168.1.1",
    }
    alert = normalize_siem_alert(raw, fmt="splunk")
    assert alert["format"] == "splunk"
    assert alert["timestamp"] == "2024-01-15T10:00:00Z"
    assert alert["source_ip"] == "10.0.0.5"
    assert alert["dest_ip"] == "192.168.1.1"
    assert alert["severity"] == 6       # 3 * 2
    assert "Port Scan" in alert["description"]


def test_normalize_elastic_fields():
    raw = {
        "@timestamp": "2024-01-15T10:00:00Z",
        "event": {"kind": "alert", "severity": 70, "action": "Brute force login"},
        "source": {"ip": "203.0.113.1"},
        "destination": {"ip": "10.0.0.10"},
        "message": "Failed login attempt",
    }
    alert = normalize_siem_alert(raw, fmt="elastic")
    assert alert["format"] == "elastic"
    assert alert["source_ip"] == "203.0.113.1"
    assert alert["dest_ip"] == "10.0.0.10"
    assert 1 <= alert["severity"] <= 10
    assert alert["description"] == "Failed login attempt"


def test_normalize_qradar_fields():
    raw = {
        "id": "OFF-12345",
        "magnitude": 8,
        "startTime": 1705305600000,   # epoch ms
        "sourceIP": "172.16.0.5",
        "destinationIP": "10.0.1.50",
        "description": "SQL Injection attempt",
        "category": "Application Attack",
    }
    alert = normalize_siem_alert(raw, fmt="qradar")
    assert alert["format"] == "qradar"
    assert alert["severity"] == 8
    assert alert["source_ip"] == "172.16.0.5"
    assert "SQL" in alert["description"]
    assert "2024" in alert["timestamp"]   # epoch ms → ISO timestamp


def test_normalize_cef_string():
    cef = "CEF:0|Cisco|ASA|9.0|106023|Deny TCP|7|src=1.2.3.4 dst=5.6.7.8 msg=Access denied"
    alert = normalize_siem_alert(cef)
    assert alert["format"] == "cef"
    assert alert["severity"] == 7
    assert alert["source_ip"] == "1.2.3.4"
    assert alert["dest_ip"] == "5.6.7.8"
    assert "Deny TCP" in alert["description"]


def test_normalize_leef_string():
    leef = "LEEF:1.0|IBM|QRadar|1.0|Login_Failed|devTime=2024-01-15T10:00:00Z\tsrc=9.9.9.9\tdst=10.0.0.1\tsev=6"
    alert = normalize_siem_alert(leef)
    assert alert["format"] == "leef"
    assert alert["severity"] == 6
    assert alert["source_ip"] == "9.9.9.9"
    assert alert["dest_ip"] == "10.0.0.1"


def test_normalize_generic_fields():
    raw = {"severity": 4, "description": "Anomaly detected", "source_ip": "1.1.1.1"}
    alert = normalize_siem_alert(raw, fmt="generic")
    assert alert["format"] == "generic"
    assert alert["severity"] == 4
    assert alert["source_ip"] == "1.1.1.1"


def test_normalize_json_string_input():
    raw_str = json.dumps({"_time": "2024-01-01", "sourcetype": "ids", "_raw": "x", "severity": 2})
    alert = normalize_siem_alert(raw_str)
    assert alert["format"] == "splunk"


def test_severity_always_clamped_1_10():
    for sev in [-5, 0, 100, 999]:
        raw = {"severity": sev, "description": "test"}
        alert = normalize_siem_alert(raw, fmt="generic")
        assert 1 <= alert["severity"] <= 10


def test_normalize_produces_required_keys():
    alert = normalize_siem_alert({"message": "test"})
    for key in ("id", "timestamp", "source_ip", "dest_ip", "severity", "description", "format", "tags", "raw"):
        assert key in alert


# ── ING.siem handler ──────────────────────────────────────────────────────────

def test_ing_siem_from_param():
    ctx = ExecutionContext()
    raw = {"_time": "2024-01-01", "sourcetype": "ids", "_raw": "x", "severity": 2, "signature": "Test"}
    result = HANDLERS["ING"](["siem"], {"alert": raw}, ctx)
    assert result["format"] == "splunk"


def test_ing_siem_from_pipe():
    ctx = ExecutionContext()
    ctx.last_output = {"@timestamp": "2024-01-01", "event": {"kind": "alert"}, "host": {"name": "web1"}, "message": "Attack"}
    result = HANDLERS["ING"](["siem"], {}, ctx)
    assert result["format"] == "elastic"


def test_ing_siem_missing_input_raises():
    ctx = ExecutionContext()
    with pytest.raises(ValueError, match="requires alert="):
        HANDLERS["ING"](["siem"], {}, ctx)


def test_ing_siem_format_override():
    ctx = ExecutionContext()
    # CEF-like data but we force format=generic
    raw = {"_raw_cef": "CEF:0|X|Y|1|1|test|5|src=1.2.3.4"}
    result = HANDLERS["ING"](["siem"], {"alert": raw, "format": "generic"}, ctx)
    assert result["format"] == "generic"


# ── ING.threat_intel handler ──────────────────────────────────────────────────

def test_ing_threat_intel_missing_src_raises():
    ctx = ExecutionContext()
    with pytest.raises(ValueError, match="requires src="):
        HANDLERS["ING"](["threat_intel"], {}, ctx)


def test_ing_threat_intel_unknown_src_raises():
    ctx = ExecutionContext()
    with pytest.raises(ValueError, match="Unknown threat intel src"):
        HANDLERS["ING"](["threat_intel"], {"src": "shodan"}, ctx)


def test_ing_threat_intel_nvd_missing_cve_raises():
    ctx = ExecutionContext()
    with pytest.raises(ValueError, match="cve_id="):
        HANDLERS["ING"](["threat_intel"], {"src": "nvd"}, ctx)


def test_ing_threat_intel_mitre_missing_technique_raises():
    ctx = ExecutionContext()
    with pytest.raises(ValueError, match="technique="):
        HANDLERS["ING"](["threat_intel"], {"src": "mitre"}, ctx)


def test_ing_threat_intel_nvd_fetches_api():
    ctx = ExecutionContext()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "vulnerabilities": [{
            "cve": {
                "id": "CVE-2024-1234",
                "descriptions": [{"lang": "en", "value": "A critical buffer overflow"}],
                "metrics": {
                    "cvssMetricV31": [{"cvssData": {"baseScore": 9.8}}]
                },
                "published": "2024-01-15",
                "references": [{"url": "https://example.com/advisory"}],
            }
        }]
    }
    mock_response.raise_for_status = MagicMock()
    with patch("httpx.get", return_value=mock_response):
        result = HANDLERS["ING"](["threat_intel"], {"src": "nvd", "cve_id": "CVE-2024-1234"}, ctx)
    assert isinstance(result, list)
    intel = result[0]
    assert intel["id"] == "CVE-2024-1234"
    assert intel["cvss_score"] == 9.8
    assert intel["severity"] == 10     # CVSS 9.8 → severity 10
    assert "buffer overflow" in intel["description"]


def test_ing_threat_intel_mitre_uses_stub_without_cache(tmp_path, monkeypatch):
    """Without local cache, MITRE returns a structured stub."""
    ctx = ExecutionContext()
    # Point home to tmp_path so cache doesn't exist
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    result = HANDLERS["ING"](["threat_intel"], {"src": "mitre", "technique": "T1190"}, ctx)
    intel = result[0]
    assert intel["id"] == "T1190"
    assert "attack.mitre.org" in intel["url"]
    assert intel.get("cached") is False


def test_ing_threat_intel_mitre_reads_local_cache(tmp_path, monkeypatch):
    """With local STIX cache, MITRE returns real technique data."""
    ctx = ExecutionContext()
    cache_dir = tmp_path / ".praxis" / "cache"
    cache_dir.mkdir(parents=True)
    stix = {
        "objects": [
            {
                "type": "attack-pattern",
                "name": "Exploit Public-Facing Application",
                "description": "Adversaries may exploit internet-facing software.",
                "kill_chain_phases": [{"phase_name": "initial-access"}],
                "x_mitre_platforms": ["Windows", "Linux"],
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1190", "url": "https://attack.mitre.org/techniques/T1190/"}
                ],
            }
        ]
    }
    (cache_dir / "mitre-attack.json").write_text(json.dumps(stix))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    result = HANDLERS["ING"](["threat_intel"], {"src": "mitre", "technique": "T1190"}, ctx)
    intel = result[0]
    assert intel["name"] == "Exploit Public-Facing Application"
    assert "initial-access" in intel["tactics"]


def test_ing_threat_intel_generic_fetches_url():
    ctx = ExecutionContext()
    mock_response = MagicMock()
    mock_response.text = json.dumps({"indicators": [{"type": "ip", "value": "1.2.3.4"}]})
    mock_response.raise_for_status = MagicMock()
    with patch("httpx.get", return_value=mock_response):
        result = HANDLERS["ING"](["threat_intel"], {"src": "generic", "url": "https://example.com/feed"}, ctx)
    assert result[0]["source"] == "generic"
    assert result[0]["type"] == "json"


# ── EVAL.risk handler ─────────────────────────────────────────────────────────

def test_eval_risk_returns_score_dict():
    ctx = ExecutionContext()
    ctx.last_output = {"severity": 8, "description": "Brute force detected", "format": "splunk"}
    mock_response = json.dumps({
        "score": 7,
        "rationale": "Brute force on prod login matches prior incident pattern.",
        "priority": "high",
        "mitre_techniques": ["T1110"],
        "recommended_actions": ["block source IP", "notify SOC"],
    })
    with patch("praxis.handlers.ai_ml._llm_call", return_value=mock_response):
        result = HANDLERS["EVAL"](["risk"], {"context": "Runbook: block on 5 fails."}, ctx)
    assert result["score"] == 7
    assert result["priority"] == "high"
    assert "T1110" in result["mitre_techniques"]
    assert len(result["recommended_actions"]) >= 1


def test_eval_risk_score_clamped_1_10():
    ctx = ExecutionContext()
    ctx.last_output = {"description": "test"}
    for raw_score in [0, -5, 15, 100]:
        mock_json = json.dumps({"score": raw_score, "rationale": "test", "priority": "low",
                                "mitre_techniques": [], "recommended_actions": []})
        with patch("praxis.handlers.ai_ml._llm_call", return_value=mock_json):
            result = HANDLERS["EVAL"](["risk"], {}, ctx)
        assert 1 <= result["score"] <= 10


def test_eval_risk_fallback_on_malformed_json():
    ctx = ExecutionContext()
    ctx.last_output = {"description": "test"}
    with patch("praxis.handlers.ai_ml._llm_call", return_value='The risk score is "score": 8 here.'):
        result = HANDLERS["EVAL"](["risk"], {}, ctx)
    assert result["score"] == 8


# ── _parse_risk_response unit tests ──────────────────────────────────────────

def test_parse_risk_response_clean_json():
    raw = '{"score": 9, "rationale": "Critical CVE", "priority": "critical", "mitre_techniques": ["T1190"], "recommended_actions": ["patch now"]}'
    r = _parse_risk_response(raw)
    assert r["score"] == 9
    assert r["priority"] == "critical"
    assert r["mitre_techniques"] == ["T1190"]


def test_parse_risk_response_json_with_preamble():
    raw = 'Here is my assessment:\n{"score": 5, "rationale": "Medium risk", "priority": "medium", "mitre_techniques": [], "recommended_actions": []}'
    r = _parse_risk_response(raw)
    assert r["score"] == 5


def test_parse_risk_response_no_json_fallback():
    raw = "I cannot determine the risk score without more context."
    r = _parse_risk_response(raw)
    assert r["score"] == 5   # default
    assert "priority" in r


# ── _score_to_priority and _cvss_to_severity ──────────────────────────────────

@pytest.mark.parametrize("score,expected", [
    (1, "low"), (3, "low"), (4, "medium"), (6, "medium"),
    (7, "high"), (8, "high"), (9, "critical"), (10, "critical"),
])
def test_score_to_priority(score, expected):
    assert _score_to_priority(score) == expected


@pytest.mark.parametrize("cvss,expected", [
    (None, 5), (0.0, 2), (3.9, 2), (4.0, 5), (6.9, 5),
    (7.0, 8), (8.9, 8), (9.0, 10), (10.0, 10),
])
def test_cvss_to_severity(cvss, expected):
    assert _cvss_to_severity(cvss) == expected
