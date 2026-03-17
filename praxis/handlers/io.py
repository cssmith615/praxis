"""I/O handlers: READ, WRITE, FETCH, POST, OUT, STORE, RECALL — Sprint 1 stubs."""

from typing import Any

# In-memory store for Sprint 1 (Sprint 4 moves to SQLite)
_STORE: dict[str, Any] = {}


def read_handler(target: list[str], params: dict, ctx) -> Any:
    path = params.get("path", ".".join(target))
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"[READ stub — file not found: {path}]"


def write_handler(target: list[str], params: dict, ctx) -> Any:
    path = params.get("path", ".".join(target))
    content = params.get("content", str(ctx.last_output))
    return {"written": path, "bytes": len(str(content)), "stub": True}


def fetch_handler(target: list[str], params: dict, ctx) -> Any:
    """FETCH — HTTP GET stub. Sprint 4 uses httpx."""
    url = params.get("url", ".".join(target))
    return {"url": url, "status": 200, "body": f"[FETCH stub — {url}]"}


def post_handler(target: list[str], params: dict, ctx) -> Any:
    """POST — HTTP POST stub."""
    url = params.get("url", ".".join(target))
    return {"url": url, "status": 200, "stub": True}


def out_handler(target: list[str], params: dict, ctx) -> Any:
    """OUT — Send to channel (Telegram, Slack, Notion, etc.)."""
    channel = target[0] if target else "console"
    msg = params.get("msg", str(ctx.last_output))
    print(f"[OUT.{channel}] {msg}")
    return {"channel": channel, "msg": msg, "delivered": True}


def store_handler(target: list[str], params: dict, ctx) -> Any:
    """STORE — Persist to key/value memory."""
    key = params.get("key", ".".join(target))
    value = params.get("value", ctx.last_output)
    _STORE[key] = value
    return {"stored": key}


def recall_handler(target: list[str], params: dict, ctx) -> Any:
    """RECALL — Retrieve from key/value memory."""
    key = params.get("name", params.get("key", ".".join(target)))
    value = _STORE.get(key)
    if value is None:
        return {"key": key, "found": False, "value": None}
    return {"key": key, "found": True, "value": value}
