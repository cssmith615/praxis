"""
I/O handlers — Sprint 4 full implementations.

READ    file read, returns content string
WRITE   file write, respects GATE in production mode
FETCH   httpx GET, returns parsed JSON or raw text
POST    httpx POST with JSON body
OUT     dispatches to named channel (console default; extensible via register_out_channel)
STORE   SQLite key/value write  (~/.praxis/kv.db)
RECALL  SQLite key/value read
SEARCH  vector search over program memory (delegates to ctx.memory)
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

# httpx is a core dependency for FETCH/POST
try:
    import httpx
    _HTTPX = True
except ImportError:
    _HTTPX = False

# ─────────────────────────────────────────────────────────────────────────────
# KV store (STORE / RECALL)  —  ~/.praxis/kv.db
# ─────────────────────────────────────────────────────────────────────────────

_KV_DB_PATH = Path.home() / ".praxis" / "kv.db"


def _get_kv_conn() -> sqlite3.Connection:
    _KV_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_KV_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kv (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# OUT channel registry — extend at runtime via register_out_channel()
# ─────────────────────────────────────────────────────────────────────────────

_OUT_CHANNELS: dict[str, Any] = {}


def register_out_channel(name: str, fn) -> None:
    """Register a custom OUT channel.  fn(msg: str, params: dict) -> Any"""
    _OUT_CHANNELS[name] = fn


# ─────────────────────────────────────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────────────────────────────────────

def read_handler(target: list[str], params: dict, ctx) -> Any:
    """READ — Read a file. Path from params['path'] or dot-joined target."""
    path = params.get("path", ".".join(target))
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"READ: file not found: {path}")
    except PermissionError:
        raise PermissionError(f"READ: permission denied: {path}")


def write_handler(target: list[str], params: dict, ctx) -> Any:
    """WRITE — Write content to a file. mode='w' (default) or 'a' to append."""
    path = params.get("path", ".".join(target))
    content = params.get("content", str(ctx.last_output) if ctx.last_output is not None else "")
    mode = params.get("mode", "w")
    try:
        with open(path, mode, encoding="utf-8") as f:
            f.write(str(content))
        return {"written": path, "bytes": len(str(content)), "mode": mode}
    except PermissionError:
        raise PermissionError(f"WRITE: permission denied: {path}")


def fetch_handler(target: list[str], params: dict, ctx) -> Any:
    """FETCH — HTTP GET. Returns parsed JSON dict or raw text string.

    Fan-out mode: if the URL template contains '$item' and ctx.last_output is a
    list, FETCH substitutes each item into the URL and returns a list of responses.
    """
    if not _HTTPX:
        raise ImportError("FETCH requires httpx: pip install praxis-lang[bridge]")
    url_template = params.get("url") or params.get("src") or ".".join(target)
    headers = params.get("headers", {})
    timeout = float(params.get("timeout", 10))

    # Fan-out: iterate when template has $item and last output is a list
    if "$item" in url_template and isinstance(ctx.last_output, list):
        results = []
        for item in ctx.last_output:
            url = url_template.replace("$item", str(item))
            resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            results.append(resp.json() if "application/json" in ct else resp.text)
        return results

    url = url_template
    response = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        return response.json()
    return response.text


def post_handler(target: list[str], params: dict, ctx) -> Any:
    """POST — HTTP POST with JSON or text body."""
    if not _HTTPX:
        raise ImportError("POST requires httpx: pip install praxis-lang[bridge]")
    url = params.get("url") or params.get("src") or ".".join(target)
    body = params.get("body", ctx.last_output)
    headers = params.get("headers", {})
    timeout = float(params.get("timeout", 10))
    if isinstance(body, (dict, list)):
        response = httpx.post(url, json=body, headers=headers, timeout=timeout)
    else:
        response = httpx.post(url, content=str(body), headers=headers, timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        return response.json()
    return response.text


def out_handler(target: list[str], params: dict, ctx) -> Any:
    """OUT — Send to a named channel. Defaults to console. Extend via register_out_channel()."""
    channel = target[0] if target else "console"
    msg = params.get("msg", str(ctx.last_output) if ctx.last_output is not None else "")
    if channel in _OUT_CHANNELS:
        result = _OUT_CHANNELS[channel](msg, params)
        return {"channel": channel, "msg": msg, "delivered": True, "result": result}
    print(f"[OUT.{channel}] {msg}")
    return {"channel": channel, "msg": msg, "delivered": True}


def store_handler(target: list[str], params: dict, ctx) -> Any:
    """STORE — Persist key/value to SQLite at ~/.praxis/kv.db."""
    key = params.get("key", ".".join(target))
    value = params.get("value", ctx.last_output)
    conn = _get_kv_conn()
    conn.execute(
        "INSERT OR REPLACE INTO kv (key, value, updated_at) VALUES (?, ?, ?)",
        (key, json.dumps(value, default=str), time.time()),
    )
    conn.commit()
    conn.close()
    return {"stored": key, "db": str(_KV_DB_PATH)}


def recall_handler(target: list[str], params: dict, ctx) -> Any:
    """RECALL — Retrieve from SQLite key/value store."""
    key = params.get("name", params.get("key", ".".join(target)))
    conn = _get_kv_conn()
    row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
    conn.close()
    if row is None:
        return {"key": key, "found": False, "value": None}
    return {"key": key, "found": True, "value": json.loads(row[0])}


def search_handler(target: list[str], params: dict, ctx) -> Any:
    """SEARCH — Vector search over ProgramMemory. Requires ctx.memory to be set."""
    query = params.get("query", str(ctx.last_output) if ctx.last_output is not None else ".".join(target))
    k = int(params.get("k", 3))
    if not hasattr(ctx, "memory") or ctx.memory is None:
        raise RuntimeError(
            "SEARCH requires a ProgramMemory instance. "
            "Pass memory=your_memory to Executor.execute()."
        )
    results = ctx.memory.retrieve_similar(query, k=k)
    return [
        {"id": r.id, "goal": r.goal_text, "similarity": r.similarity, "program": r.shaun_program}
        for r in results
    ]
