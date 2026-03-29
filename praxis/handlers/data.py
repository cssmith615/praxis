"""
Data handlers: ING, CLN, XFRM, FILTER, SORT, MERGE

Sprint 1: ING reads from a file path or returns mock data.
          All others are pass-through stubs that log and return input.
Sprint 4: Real implementations replace these.
Sprint A: ING.docs — ingest text/markdown/PDF files and URLs into chunks.
"""

from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

try:
    import httpx as _httpx
    _HTTPX = True
except ImportError:
    _HTTPX = False


def ing_handler(target: list[str], params: dict, ctx) -> Any:
    """
    ING — Ingest / load data.

    target:  dot-path describing the data source
             e.g. ['sales', 'db'] | ['flights'] | ['docs']

    params (ING.docs):
      src=        file path, directory glob, or https:// URL
      chunk_size= max chars per chunk (default 400)
      overlap=    overlap chars between chunks (default 50)
      format=     'csv' | 'json' | 'auto' (non-docs targets)

    params (ING.siem):
      alert=      raw alert dict, JSON string, or CEF/LEEF text (or pipe from ctx)
      format=     'auto' | 'splunk' | 'elastic' | 'qradar' | 'cef' | 'leef' | 'generic'

    params (ING.threat_intel):
      src=        'nvd' | 'mitre' | 'generic'
      cve_id=     CVE ID for src=nvd  (e.g. CVE-2024-1234)
      technique=  ATT&CK ID for src=mitre  (e.g. T1190)
      url=        feed URL for src=generic

    Returns:
      ING.docs         → list[{id, text, source, chunk_index, char_count}]
      ING.siem         → SecurityAlert dict
      ING.threat_intel → list[ThreatIntel dict]
      others           → list[dict] (CSV/JSON) or mock data
    """
    if target and target[0] == "docs":
        return _ing_docs(params)
    if target and target[0] == "siem":
        return _ing_siem(params, ctx)
    if target and target[0] == "threat_intel":
        return _ing_threat_intel(params)

    source = ".".join(target)
    path: str | None = params.get("path")
    fmt: str = params.get("format", "auto")

    if path and os.path.exists(path):
        return _read_file(path, fmt)

    # Mock data keyed by source name so tests get predictable results
    mock_data = {
        "sales.db":   [{"id": 1, "amount": 100.0}, {"id": 2, "amount": 250.0}],
        "flights":    [{"dest": "denver", "price": 189.0}, {"dest": "austin", "price": 145.0}],
        "marketing":  [{"campaign": "q1", "clicks": 5000}],
        "crm":        [{"customer": "acme", "revenue": 12000}],
    }
    data = mock_data.get(source, [{"source": source, "stub": True}])
    ctx.last_output = data
    return data


def cln_handler(target: list[str], params: dict, ctx) -> Any:
    """
    CLN — Clean / normalize data.

    target: cleaning strategy  e.g. ['null'] | ['dedupe'] | ['normalize']
    Input:  ctx.last_output (expected: list[dict])
    Returns: cleaned list[dict]
    """
    data = ctx.last_output
    if data is None:
        return []
    strategy = target[0] if target else "null"

    if strategy == "null" and isinstance(data, list):
        # Remove rows where any value is None
        return [row for row in data if all(v is not None for v in row.values())]

    if strategy == "dedupe" and isinstance(data, list):
        seen = set()
        result = []
        for row in data:
            key = str(sorted(row.items()))
            if key not in seen:
                seen.add(key)
                result.append(row)
        return result

    # Passthrough for unknown strategies
    return data


def xfrm_handler(target: list[str], params: dict, ctx) -> Any:
    """
    XFRM — Transform data.

    Strategies (first element of target):
      slice(limit=N, offset=M)   — take N items starting at M (default 0)
      pluck(field=name)          — extract a single field from each dict in a list
      join(sep=", ")             — join a list of strings into one string
      flatten                    — flatten a list of lists one level deep
      keys                       — return the keys of the first dict in a list
      values(field=name)         — alias for pluck
      (anything else)            — passthrough
    """
    data = ctx.last_output
    strategy = target[0] if target else "passthrough"

    if strategy == "slice":
        if not isinstance(data, list):
            return data
        limit = int(params.get("limit", len(data)))
        offset = int(params.get("offset", 0))
        return data[offset:offset + limit]

    if strategy in ("pluck", "values"):
        field = params.get("field", target[1] if len(target) > 1 else None)
        if not field or not isinstance(data, list):
            return data
        return [item[field] for item in data if isinstance(item, dict) and field in item]

    if strategy == "join":
        sep = params.get("sep", "\n")
        if isinstance(data, list):
            return sep.join(str(item) for item in data)
        return str(data) if data is not None else ""

    if strategy == "flatten":
        if isinstance(data, list):
            result = []
            for item in data:
                if isinstance(item, list):
                    result.extend(item)
                else:
                    result.append(item)
            return result
        return data

    if strategy == "keys":
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return list(data[0].keys())
        if isinstance(data, dict):
            return list(data.keys())
        return data

    # Passthrough for unknown strategies
    return data


def filter_handler(target: list[str], params: dict, ctx) -> Any:
    """
    FILTER — Filter rows.

    Strategies:
      field(name=x, value=y)     — keep rows where row[name] == value
      field(name=x, gt=y)        — keep rows where row[name] > y
      field(name=x, lt=y)        — keep rows where row[name] < y
      (anything else)            — passthrough
    """
    data = ctx.last_output
    if not isinstance(data, list):
        return data

    strategy = target[0] if target else "passthrough"

    if strategy == "field":
        name = params.get("name", target[1] if len(target) > 1 else None)
        if not name:
            return data
        if "value" in params:
            val = params["value"]
            return [row for row in data if isinstance(row, dict) and str(row.get(name)) == str(val)]
        if "gt" in params:
            threshold = float(params["gt"])
            return [row for row in data if isinstance(row, dict) and float(row.get(name, 0)) > threshold]
        if "lt" in params:
            threshold = float(params["lt"])
            return [row for row in data if isinstance(row, dict) and float(row.get(name, 0)) < threshold]

    return data


def sort_handler(target: list[str], params: dict, ctx) -> Any:
    """SORT — Sort a list. Params: field=name, order=asc|desc."""
    data = ctx.last_output
    if not isinstance(data, list):
        return data

    field = params.get("field", target[0] if target else None)
    reverse = params.get("order", "asc").lower() == "desc"

    if field:
        return sorted(data, key=lambda x: x.get(field, 0) if isinstance(x, dict) else x,
                      reverse=reverse)
    return sorted(data, reverse=reverse)


def merge_handler(target: list[str], params: dict, ctx) -> Any:
    """MERGE — Merge datasets. Stub: returns combined list."""
    data = ctx.last_output
    if isinstance(data, list):
        return data
    return [data] if data is not None else []


# ── ING.docs helpers ─────────────────────────────────────────────────────────

def _ing_docs(params: dict) -> list[dict]:
    src = params.get("src", "").strip()
    if not src:
        raise ValueError("ING.docs requires src= parameter")

    chunk_size = max(50, int(params.get("chunk_size", 400)))
    overlap = max(0, int(params.get("overlap", 50)))
    if overlap >= chunk_size:
        overlap = chunk_size // 4

    p = Path(src)
    if p.is_dir():
        return _ing_docs_dir(p, chunk_size, overlap)

    if src.startswith(("http://", "https://")):
        text = _fetch_url(src)
        source = src
    else:
        if not p.exists():
            raise FileNotFoundError(f"ING.docs: not found: {src}")
        text = _read_doc(p)
        source = str(p.resolve())

    return _chunk_text(text, source, chunk_size, overlap)


_DOC_EXTENSIONS = {".txt", ".md", ".pdf", ".json"}


def _ing_docs_dir(directory: Path, chunk_size: int, overlap: int) -> list[dict]:
    """Ingest all supported files in a directory (recursive)."""
    chunks: list[dict] = []
    for file in sorted(directory.rglob("*")):
        if file.is_file() and file.suffix.lower() in _DOC_EXTENSIONS:
            try:
                text = _read_doc(file)
                if text.strip():
                    chunks.extend(_chunk_text(text, str(file.resolve()), chunk_size, overlap))
            except Exception:
                pass  # Skip unreadable files; don't abort the whole directory
    return chunks


def _fetch_url(url: str) -> str:
    if not _HTTPX:
        raise ImportError("ING.docs URL fetching requires httpx (pip install praxis-lang[bridge])")
    resp = _httpx.get(url, timeout=15, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def _read_doc(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("PDF ingestion requires pdfplumber: pip install praxis-lang[rag]")
        with pdfplumber.open(str(path)) as pdf:
            return "\n\n".join(page.extract_text() or "" for page in pdf.pages)
    if ext == ".json":
        return _json_to_text(path)
    return path.read_text(encoding="utf-8")


def _json_to_text(path: Path) -> str:
    """Convert JSON to readable text for embedding.

    Chuck decision files (contain 'decision' + 'rejected' keys) are formatted
    as structured prose so semantic search works naturally.
    All other JSON falls back to an indented dump.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return path.read_text(encoding="utf-8", errors="replace")

    if isinstance(data, dict) and "decision" in data and "rejected" in data:
        parts = [f"Decision: {data.get('decision', '')}"]
        if data.get("rejected"):
            parts.append(f"Rejected: {', '.join(data['rejected'])}")
        if data.get("reason"):
            parts.append(f"Reason: {data['reason']}")
        if data.get("constraints"):
            parts.append(f"Constraints: {', '.join(data['constraints'])}")
        if data.get("tags"):
            parts.append(f"Tags: {', '.join(data['tags'])}")
        if data.get("id"):
            parts.append(f"ID: {data['id']}")
        if data.get("date"):
            parts.append(f"Date: {data['date']}")
        return "\n".join(parts)

    return json.dumps(data, indent=2)


def _chunk_text(text: str, source: str, chunk_size: int, overlap: int) -> list[dict]:
    # Split on paragraph breaks; rejoin small paragraphs up to chunk_size
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    raw_chunks: list[str] = []
    buf = ""

    for para in paragraphs:
        candidate = (buf + "\n\n" + para).strip() if buf else para
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf:
                raw_chunks.append(buf)
            if len(para) > chunk_size:
                # Sliding window over oversized paragraphs
                for i in range(0, len(para), chunk_size - overlap):
                    seg = para[i : i + chunk_size]
                    if seg.strip():
                        raw_chunks.append(seg.strip())
                buf = para[-(overlap):].strip() if overlap else ""
            else:
                buf = para

    if buf:
        raw_chunks.append(buf)

    return [
        {
            "id": hashlib.sha256(f"{source}|{i}".encode()).hexdigest(),
            "text": chunk,
            "source": source,
            "chunk_index": i,
            "char_count": len(chunk),
        }
        for i, chunk in enumerate(raw_chunks)
    ]


# ── ING.siem / ING.threat_intel helpers ──────────────────────────────────────

def _ing_siem(params: dict, ctx) -> dict:
    from praxis.security import normalize_siem_alert
    raw = params.get("alert") or ctx.last_output
    if raw is None:
        raise ValueError("ING.siem requires alert= parameter or piped input from a prior step")
    return normalize_siem_alert(raw, fmt=params.get("format", "auto"))


def _ing_threat_intel(params: dict) -> list[dict]:
    from praxis.security import fetch_threat_intel
    src = params.get("src", "")
    if not src:
        raise ValueError("ING.threat_intel requires src= parameter (nvd, mitre, or generic)")
    extra = {k: v for k, v in params.items() if k != "src"}
    return fetch_threat_intel(src, **extra)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _read_file(path: str, fmt: str) -> list[dict]:
    ext = os.path.splitext(path)[1].lower()
    if fmt == "auto":
        fmt = "csv" if ext == ".csv" else "json"

    if fmt == "csv":
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    if fmt == "json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]

    return [{"path": path, "raw": open(path).read()}]
