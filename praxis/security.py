"""
Security intelligence module — Sprint E.

Provides core logic for three new verb targets:
  ING.siem        — normalize SIEM alerts (Splunk/Elastic/QRadar/CEF/LEEF/generic)
  ING.threat_intel — fetch CVE/ATT&CK/generic threat intel
  EVAL.risk        — LLM-graded risk score (1–10) grounded in retrieved context

Security invariants:
  - No API keys or credentials accepted through params — env vars only
  - All inputs treated as untrusted until normalized
  - Risk scores without context (RECALL.docs output) are flagged as ungrounded
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── SIEM normalization ────────────────────────────────────────────────────────

def normalize_siem_alert(raw: Any, fmt: str = "auto") -> dict:
    """
    Normalize a SIEM alert to a standard SecurityAlert dict.

    raw: dict, JSON string, or raw text (CEF/LEEF)
    fmt: "auto" | "splunk" | "elastic" | "qradar" | "cef" | "leef" | "generic"

    Returns:
      {id, timestamp, source_ip, dest_ip, severity (1-10),
       description, format, tags, raw}
    """
    if isinstance(raw, str):
        raw = _parse_raw_string(raw)
    if not isinstance(raw, dict):
        raw = {"message": str(raw)}

    if fmt == "auto":
        fmt = _detect_format(raw)

    normalizers = {
        "splunk":  _normalize_splunk,
        "elastic": _normalize_elastic,
        "qradar":  _normalize_qradar,
        "cef":     _normalize_cef,
        "leef":    _normalize_leef,
    }
    return normalizers.get(fmt, _normalize_generic)(raw)


def _parse_raw_string(s: str) -> Any:
    s = s.strip()
    if s.upper().startswith("CEF:"):
        return {"_raw_cef": s}
    if s.upper().startswith("LEEF:"):
        return {"_raw_leef": s}
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return {"message": s}


def _detect_format(data: dict) -> str:
    if "_raw_cef" in data or str(data.get("message", "")).upper().startswith("CEF:"):
        return "cef"
    if "_raw_leef" in data or str(data.get("message", "")).upper().startswith("LEEF:"):
        return "leef"
    if "_time" in data and ("sourcetype" in data or "_raw" in data):
        return "splunk"
    if "@timestamp" in data and ("event" in data or "host" in data):
        return "elastic"
    if "qid" in data or "magnitude" in data or "offense" in data:
        return "qradar"
    return "generic"


def _alert_id(raw: dict) -> str:
    return hashlib.sha256(
        json.dumps(raw, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _clamp(value: Any, default: int = 5) -> int:
    try:
        return min(10, max(1, int(float(str(value)))))
    except (ValueError, TypeError):
        return default


def _normalize_splunk(raw: dict) -> dict:
    sev_raw = raw.get("severity", raw.get("alert_severity", 3))
    # Splunk 1-5 → 1-10
    severity = min(10, max(1, int(float(str(sev_raw))) * 2))
    return {
        "id":          raw.get("event_id", _alert_id(raw)),
        "timestamp":   str(raw.get("_time", "")),
        "source_ip":   raw.get("src_ip", raw.get("src")),
        "dest_ip":     raw.get("dest_ip", raw.get("dest")),
        "severity":    severity,
        "description": str(raw.get("signature", raw.get("search_name", raw.get("_raw", "Splunk alert")))),
        "format":      "splunk",
        "tags":        [str(raw.get("sourcetype", "")), str(raw.get("index", ""))],
        "raw":         raw,
    }


def _normalize_elastic(raw: dict) -> dict:
    event = raw.get("event", {}) if isinstance(raw.get("event"), dict) else {}
    src   = raw.get("source", {}) if isinstance(raw.get("source"), dict) else {}
    dst   = raw.get("destination", {}) if isinstance(raw.get("destination"), dict) else {}
    # ECS event.severity 0-100 → 1-10
    sev_raw  = event.get("severity", event.get("risk_score", 50))
    severity = max(1, min(10, round(float(sev_raw) / 10) or 5))
    return {
        "id":          str(raw.get("event.id", _alert_id(raw))),
        "timestamp":   str(raw.get("@timestamp", "")),
        "source_ip":   src.get("ip"),
        "dest_ip":     dst.get("ip"),
        "severity":    severity,
        "description": str(raw.get("message", event.get("action", "Elastic alert"))),
        "format":      "elastic",
        "tags":        [str(event.get("kind", "")), str(event.get("category", ""))],
        "raw":         raw,
    }


def _normalize_qradar(raw: dict) -> dict:
    magnitude = _clamp(raw.get("magnitude", 5))
    return {
        "id":          str(raw.get("id", _alert_id(raw))),
        "timestamp":   _qradar_time(raw.get("startTime", "")),
        "source_ip":   raw.get("sourceIP", raw.get("sourceAddress")),
        "dest_ip":     raw.get("destinationIP", raw.get("destinationAddress")),
        "severity":    magnitude,
        "description": str(raw.get("description", raw.get("offenseType", "QRadar offense"))),
        "format":      "qradar",
        "tags":        [str(raw.get("category", "")), str(raw.get("status", ""))],
        "raw":         raw,
    }


def _qradar_time(ts: Any) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc).isoformat()
    except (ValueError, TypeError):
        return str(ts)


def _parse_cef_extensions(ext_str: str) -> dict:
    result = {}
    for m in re.finditer(r'(\w+)=((?:[^\\=\s]|\\.)*)', ext_str):
        result[m.group(1)] = m.group(2).replace("\\=", "=").replace("\\|", "|")
    return result


def _normalize_cef(raw: dict) -> dict:
    cef_str = raw.get("_raw_cef", raw.get("message", ""))
    # CEF:0|vendor|product|version|id|name|severity|extensions
    parts = str(cef_str).split("|", 7)
    ext   = _parse_cef_extensions(parts[7]) if len(parts) >= 8 else {}
    sev   = _clamp(parts[6].strip() if len(parts) > 6 else ext.get("sev", "5"))
    return {
        "id":          _alert_id(raw),
        "timestamp":   str(ext.get("rt", ext.get("end", ""))),
        "source_ip":   ext.get("src"),
        "dest_ip":     ext.get("dst"),
        "severity":    sev,
        "description": str(parts[5].strip() if len(parts) > 5 else ext.get("msg", "CEF alert")),
        "format":      "cef",
        "tags":        [parts[1].strip() if len(parts) > 1 else "",
                        parts[2].strip() if len(parts) > 2 else ""],
        "raw":         raw,
    }


def _parse_leef_extensions(ext_str: str) -> dict:
    result = {}
    for pair in ext_str.split("\t"):
        if "=" in pair:
            k, _, v = pair.partition("=")
            result[k.strip()] = v.strip()
    return result


def _normalize_leef(raw: dict) -> dict:
    leef_str = raw.get("_raw_leef", raw.get("message", ""))
    parts = str(leef_str).split("|", 5)
    ext   = _parse_leef_extensions(parts[5]) if len(parts) >= 6 else {}
    sev   = _clamp(ext.get("sev", ext.get("severity", "5")))
    return {
        "id":          _alert_id(raw),
        "timestamp":   str(ext.get("devTime", "")),
        "source_ip":   ext.get("src"),
        "dest_ip":     ext.get("dst"),
        "severity":    sev,
        "description": str(parts[4].strip() if len(parts) > 4 else "LEEF alert"),
        "format":      "leef",
        "tags":        [parts[1].strip() if len(parts) > 1 else "",
                        parts[2].strip() if len(parts) > 2 else ""],
        "raw":         raw,
    }


def _normalize_generic(raw: dict) -> dict:
    sev = _clamp(raw.get("severity", raw.get("priority", raw.get("level", 5))))
    return {
        "id":          _alert_id(raw),
        "timestamp":   str(raw.get("timestamp", raw.get("time", raw.get("date", "")))),
        "source_ip":   raw.get("source_ip", raw.get("src_ip", raw.get("src"))),
        "dest_ip":     raw.get("dest_ip", raw.get("dst_ip", raw.get("dst"))),
        "severity":    sev,
        "description": str(raw.get("description", raw.get("message", raw.get("msg", "Security alert")))),
        "format":      "generic",
        "tags":        [],
        "raw":         raw,
    }


# ── Threat intel fetching ─────────────────────────────────────────────────────

def fetch_threat_intel(src: str, **params) -> list[dict]:
    """
    Fetch threat intelligence.

    src=nvd     params: cve_id="CVE-2024-1234"
    src=mitre   params: technique="T1190"
    src=generic params: url="https://..."
    """
    if src == "nvd":
        cve_id = params.get("cve_id", "")
        if not cve_id:
            raise ValueError("ING.threat_intel src=nvd requires cve_id= parameter")
        return [_fetch_nvd_cve(cve_id)]
    if src == "mitre":
        technique = params.get("technique", "")
        if not technique:
            raise ValueError("ING.threat_intel src=mitre requires technique= parameter")
        return [_fetch_mitre_technique(technique)]
    if src == "generic":
        url = params.get("url", "")
        if not url:
            raise ValueError("ING.threat_intel src=generic requires url= parameter")
        return [_fetch_generic_feed(url)]
    raise ValueError(
        f"Unknown threat intel src: {src!r}. Use nvd, mitre, or generic."
    )


def _fetch_nvd_cve(cve_id: str) -> dict:
    """Fetch CVE from NVD API 2.0. No key required."""
    import httpx
    url  = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    data  = resp.json()
    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return {"id": cve_id, "source": "nvd", "found": False}
    cve = vulns[0]["cve"]
    description = next(
        (d["value"] for d in cve.get("descriptions", []) if d.get("lang") == "en"),
        "No description available",
    )
    cvss_score = _extract_cvss(cve.get("metrics", {}))
    return {
        "id":          cve_id,
        "source":      "nvd",
        "title":       cve_id,
        "description": description,
        "cvss_score":  cvss_score,
        "severity":    _cvss_to_severity(cvss_score),
        "published":   cve.get("published", ""),
        "references":  [r["url"] for r in cve.get("references", [])[:5]],
    }


def _extract_cvss(metrics: dict) -> float | None:
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if key in metrics and metrics[key]:
            return metrics[key][0].get("cvssData", {}).get("baseScore")
    return None


def _cvss_to_severity(score) -> int:
    if score is None:
        return 5
    s = float(score)
    if s >= 9.0: return 10
    if s >= 7.0: return 8
    if s >= 4.0: return 5
    return 2


def _fetch_mitre_technique(technique_id: str) -> dict:
    """
    Look up ATT&CK technique.

    Uses local cache at ~/.praxis/cache/mitre-attack.json if present
    (download from: https://raw.githubusercontent.com/mitre/cti/master/
     enterprise-attack/enterprise-attack.json).

    Returns a structured stub with reference URL if cache is absent.
    """
    tid        = technique_id.upper()
    cache_path = Path.home() / ".praxis" / "cache" / "mitre-attack.json"

    if cache_path.exists():
        try:
            stix = json.loads(cache_path.read_text(encoding="utf-8"))
            for obj in stix.get("objects", []):
                if obj.get("type") != "attack-pattern":
                    continue
                for ref in obj.get("external_references", []):
                    if ref.get("source_name") == "mitre-attack" and ref.get("external_id") == tid:
                        tactics = [p["phase_name"] for p in obj.get("kill_chain_phases", [])]
                        return {
                            "id":          tid,
                            "source":      "mitre",
                            "name":        obj.get("name", tid),
                            "description": obj.get("description", "")[:1000],
                            "tactics":     tactics,
                            "platforms":   obj.get("x_mitre_platforms", []),
                            "url":         f"https://attack.mitre.org/techniques/{tid}/",
                        }
        except Exception:
            pass

    # Stub — cache not available
    return {
        "id":          tid,
        "source":      "mitre",
        "name":        f"ATT&CK {tid}",
        "description": f"MITRE ATT&CK technique {tid}.",
        "tactics":     [],
        "platforms":   [],
        "url":         f"https://attack.mitre.org/techniques/{tid}/",
        "cached":      False,
        "note":        (
            "Cache MITRE ATT&CK locally for full details: "
            "curl -o ~/.praxis/cache/mitre-attack.json "
            "https://raw.githubusercontent.com/mitre/cti/master/"
            "enterprise-attack/enterprise-attack.json"
        ),
    }


def _fetch_generic_feed(url: str) -> dict:
    """Fetch a generic threat feed URL."""
    import httpx
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    content = resp.text
    try:
        return {
            "source":     "generic",
            "url":        url,
            "type":       "json",
            "data":       json.loads(content),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    except (json.JSONDecodeError, ValueError):
        return {
            "source":     "generic",
            "url":        url,
            "type":       "text",
            "content":    content[:2000],
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }


# ── Risk scoring ──────────────────────────────────────────────────────────────

_RISK_PROMPT = """\
You are a senior security analyst. Score the risk of this alert for our specific environment.

Environment:
{environment}

Knowledge base context (from security runbooks and prior incidents):
{context}

Alert:
{alert}

Score 1–10:
  1–3: Low — monitor, no immediate action
  4–6: Medium — investigate, create Jira ticket
  7–8: High — escalate to on-call
  9–10: Critical — page immediately, initiate IR

Return ONLY valid JSON:
{{
  "score": <int 1-10>,
  "rationale": "<1-2 sentence justification referencing the context>",
  "priority": "<critical|high|medium|low>",
  "mitre_techniques": ["<T-number>", ...],
  "recommended_actions": ["<action>", ...]
}}"""


def score_risk(
    alert_data: Any,
    context: str = "",
    environment: str = "production environment",
    provider: str = "claude",
    model: str | None = None,
) -> dict:
    """
    LLM-graded risk score for a security alert.

    alert_data: SecurityAlert dict (from normalize_siem_alert) or any dict/str
    context:    RAG-retrieved context block from RECALL.docs
                (empty context is flagged — constitutional rule requires grounding)
    environment: description of the target environment

    Returns: {score, rationale, priority, mitre_techniques, recommended_actions}
    """
    from praxis.handlers.ai_ml import _llm_call

    alert_str = (
        json.dumps(alert_data, indent=2, default=str)
        if isinstance(alert_data, dict)
        else str(alert_data)
    )
    ctx_str = context if context else "(no context retrieved — score may be ungrounded)"
    prompt  = _RISK_PROMPT.format(
        environment=environment,
        context=ctx_str,
        alert=alert_str,
    )
    raw = _llm_call(prompt, provider, model, max_tokens=512)
    return _parse_risk_response(raw)


def _parse_risk_response(raw: str) -> dict:
    """Extract JSON risk dict from LLM response with safe fallback."""
    # Find the outermost JSON object
    match = re.search(r'\{[^{}]*"score"[^{}]*\}', raw, re.DOTALL)
    if match:
        try:
            data  = json.loads(match.group())
            score = max(1, min(10, int(data.get("score", 5))))
            return {
                "score":               score,
                "rationale":           str(data.get("rationale", "")),
                "priority":            data.get("priority", _score_to_priority(score)),
                "mitre_techniques":    list(data.get("mitre_techniques", [])),
                "recommended_actions": list(data.get("recommended_actions", [])),
            }
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
    # Fallback: pull score via regex
    m = re.search(r'"score"\s*:\s*(\d+)', raw)
    score = max(1, min(10, int(m.group(1)))) if m else 5
    return {
        "score":               score,
        "rationale":           raw[:200].strip(),
        "priority":            _score_to_priority(score),
        "mitre_techniques":    [],
        "recommended_actions": [],
    }


def _score_to_priority(score: int) -> str:
    if score >= 9: return "critical"
    if score >= 7: return "high"
    if score >= 4: return "medium"
    return "low"
