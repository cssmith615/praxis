"""
I/O handlers — Sprint 4 full implementations.

READ    file read, returns content string
WRITE   file write, respects GATE in production mode
FETCH   httpx GET, returns parsed JSON or raw text
POST    httpx POST with JSON body
OUT     dispatches to named channel (console default; telegram/slack/discord/pagerduty/jira built-in)
STORE   SQLite key/value write  (~/.praxis/kv.db)
RECALL  SQLite key/value read   (RECALL.docs → one-step RAG context block, Sprint A)
SEARCH  vector search over program memory (delegates to ctx.memory)
"""
from __future__ import annotations

import base64
import json
import os
import sqlite3
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from praxis.embeddings import EmbeddingsDB

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


def _send_telegram(msg: str, params: dict) -> dict:
    """Send a message via Telegram Bot API using env vars for credentials."""
    token = params.get("token") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = params.get("chat_id") or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token:
        raise RuntimeError("OUT.telegram requires TELEGRAM_BOT_TOKEN env var or token= param")
    if not chat_id:
        raise RuntimeError("OUT.telegram requires TELEGRAM_CHAT_ID env var or chat_id= param")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    # Split long messages at Telegram's 4096-char limit
    chunks = [msg[i:i + 4096] for i in range(0, max(len(msg), 1), 4096)]
    for chunk in chunks:
        data = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "Markdown",
        }).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
        if not result.get("ok"):
            raise RuntimeError(f"Telegram API error: {result.get('description', result)}")
    return {"delivered": True, "chunks": len(chunks), "chat_id": chat_id}


def _send_slack(msg: str, params: dict) -> dict:
    """Send a message to a Slack incoming webhook (no bot setup required)."""
    webhook = params.get("webhook") or os.environ.get("SLACK_WEBHOOK_URL", "")
    if not webhook:
        raise ValueError(
            "OUT.slack requires a webhook URL — pass webhook= param or set SLACK_WEBHOOK_URL"
        )
    payload = json.dumps({"text": msg}).encode()
    req = urllib.request.Request(webhook, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=10) as resp:
        pass  # Slack returns "ok" as plain text with HTTP 200
    return {"ok": True, "channel": "slack"}


def _send_discord(msg: str, params: dict) -> dict:
    """Send a message to a Discord incoming webhook."""
    webhook = params.get("webhook") or os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not webhook:
        raise ValueError(
            "OUT.discord requires a webhook URL — pass webhook= param or set DISCORD_WEBHOOK_URL"
        )
    # Discord has a 2000-char message limit
    chunks = [msg[i:i + 2000] for i in range(0, max(len(msg), 1), 2000)]
    for chunk in chunks:
        payload = json.dumps({"content": chunk}).encode()
        req = urllib.request.Request(webhook, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=10) as resp:
            pass  # Discord returns 204 No Content on success
    return {"ok": True, "channel": "discord", "chunks": len(chunks)}


def _send_x(msg: str, params: dict) -> dict:
    """OUT.x — Post a tweet via X API v2 (OAuth 1.0a).

    Required env vars (or params):
      X_API_KEY / x_api_key
      X_API_SECRET / x_api_secret
      X_ACCESS_TOKEN / x_access_token
      X_ACCESS_TOKEN_SECRET / x_access_token_secret

    Uses tweepy if available; falls back to manual OAuth 1.0a signing.
    Message is truncated to 280 characters automatically.
    """
    text = msg[:280]

    api_key    = params.get("x_api_key")    or os.environ.get("X_API_KEY", "")
    api_secret = params.get("x_api_secret") or os.environ.get("X_API_SECRET", "")
    access_tok = params.get("x_access_token") or os.environ.get("X_ACCESS_TOKEN", "")
    access_sec = params.get("x_access_token_secret") or os.environ.get("X_ACCESS_TOKEN_SECRET", "")

    if not all([api_key, api_secret, access_tok, access_sec]):
        raise ValueError(
            "OUT.x requires X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET"
        )

    try:
        import tweepy  # type: ignore[import]
        client = tweepy.Client(
            consumer_key=api_key, consumer_secret=api_secret,
            access_token=access_tok, access_token_secret=access_sec,
        )
        resp = client.create_tweet(text=text)
        tweet_id = resp.data["id"] if resp.data else None
        return {"posted": True, "tweet_id": tweet_id, "chars": len(text)}
    except ImportError:
        pass  # fall through to manual OAuth

    # Manual OAuth 1.0a signing (stdlib only — no tweepy required)
    import base64
    import hashlib
    import hmac
    import uuid

    url    = "https://api.twitter.com/2/tweets"
    nonce  = uuid.uuid4().hex
    ts     = str(int(time.time()))

    params_oauth = {
        "oauth_consumer_key":     api_key,
        "oauth_nonce":            nonce,
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp":        ts,
        "oauth_token":            access_tok,
        "oauth_version":          "1.0",
    }

    # Signature base string
    def _pct(s: str) -> str:
        return urllib.parse.quote(str(s), safe="")

    param_str = "&".join(f"{_pct(k)}={_pct(v)}" for k, v in sorted(params_oauth.items()))
    base_str  = f"POST&{_pct(url)}&{_pct(param_str)}"
    sign_key  = f"{_pct(api_secret)}&{_pct(access_sec)}"
    sig = base64.b64encode(
        hmac.new(sign_key.encode(), base_str.encode(), hashlib.sha1).digest()
    ).decode()

    params_oauth["oauth_signature"] = sig
    auth_header = "OAuth " + ", ".join(
        f'{_pct(k)}="{_pct(v)}"' for k, v in sorted(params_oauth.items())
    )

    body    = json.dumps({"text": text}).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Authorization": auth_header, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as r:
        data = json.loads(r.read().decode())
    tweet_id = data.get("data", {}).get("id")
    return {"posted": True, "tweet_id": tweet_id, "chars": len(text)}


def _send_pagerduty(msg: str, params: dict) -> dict:
    """OUT.pagerduty — Create a PagerDuty incident via Events API v2.

    Required env var: PAGERDUTY_ROUTING_KEY
    Optional params:  summary=, severity=critical|error|warning|info,
                      source=, dedup_key=, component=, group=, class=
    """
    import httpx

    routing_key = os.environ.get("PAGERDUTY_ROUTING_KEY")
    if not routing_key:
        raise ValueError(
            "PAGERDUTY_ROUTING_KEY environment variable not set. "
            "Get your integration key from PagerDuty → Integrations → Events API v2."
        )

    summary  = params.get("summary", msg)[:1024]
    severity = params.get("severity", "error")
    if severity not in ("critical", "error", "warning", "info"):
        severity = "error"

    payload: dict = {
        "routing_key":   routing_key,
        "event_action":  "trigger",
        "payload": {
            "summary":   summary,
            "source":    params.get("source", "praxis"),
            "severity":  severity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
    if params.get("dedup_key"):
        payload["dedup_key"] = params["dedup_key"]
    if params.get("component"):
        payload["payload"]["component"] = params["component"]
    if params.get("group"):
        payload["payload"]["group"] = params["group"]
    if params.get("class"):
        payload["payload"]["class"] = params["class"]

    resp = httpx.post(
        "https://events.pagerduty.com/v2/enqueue",
        json=payload,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "status":    data.get("status", "ok"),
        "dedup_key": data.get("dedup_key", ""),
        "message":   data.get("message", ""),
    }


def _send_jira(msg: str, params: dict) -> dict:
    """OUT.jira — Create a Jira issue via REST API v3.

    Required env vars: JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN
    Optional params:   project=SEC, summary=, description=,
                       issue_type=Bug|Task|Story, priority=Critical|High|Medium|Low,
                       labels= (comma-separated)
    """
    import httpx

    base_url = os.environ.get("JIRA_BASE_URL", "").rstrip("/")
    email    = os.environ.get("JIRA_EMAIL", "")
    token    = os.environ.get("JIRA_API_TOKEN", "")

    if not all([base_url, email, token]):
        raise ValueError(
            "JIRA_BASE_URL, JIRA_EMAIL, and JIRA_API_TOKEN environment variables must be set."
        )

    project    = params.get("project", "SEC")
    summary    = params.get("summary", msg)[:255]
    description = params.get("description", msg)
    issue_type = params.get("issue_type", "Bug")
    priority   = params.get("priority", "High")
    labels_raw = params.get("labels", [])
    labels     = (
        [l.strip() for l in labels_raw.split(",")]
        if isinstance(labels_raw, str)
        else list(labels_raw)
    )

    body = {
        "fields": {
            "project":     {"key": project},
            "summary":     summary,
            "description": {
                "type":    "doc",
                "version": 1,
                "content": [{"type": "paragraph", "content": [{"type": "text", "text": description}]}],
            },
            "issuetype": {"name": issue_type},
            "priority":  {"name": priority},
            "labels":    labels,
        }
    }

    credentials = base64.b64encode(f"{email}:{token}".encode()).decode()
    resp = httpx.post(
        f"{base_url}/rest/api/3/issue",
        json=body,
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type":  "application/json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    key  = data.get("key", "")
    return {
        "id":  data.get("id"),
        "key": key,
        "url": f"{base_url}/browse/{key}",
    }


def out_handler(target: list[str], params: dict, ctx) -> Any:
    """OUT — Send to a named channel.

    Built-in channels:
      OUT.telegram   — sends via Telegram Bot API (reads TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)
      OUT.slack      — sends via Slack incoming webhook (reads SLACK_WEBHOOK_URL)
      OUT.discord    — sends via Discord incoming webhook (reads DISCORD_WEBHOOK_URL)
      OUT.x          — posts a tweet via X API v2 (reads X_API_KEY etc.)
      OUT.console    — prints to stdout (default)

    Extend with register_out_channel() for custom channels.
    """
    channel = target[0] if target else "console"
    msg = params.get("message") or params.get("msg") or (
        str(ctx.last_output) if ctx.last_output is not None else ""
    )

    if channel == "telegram":
        result = _send_telegram(msg, params)
        return {"channel": "telegram", "msg": msg, **result}

    if channel == "slack":
        result = _send_slack(msg, params)
        return {"channel": "slack", "msg": msg, **result}

    if channel == "discord":
        result = _send_discord(msg, params)
        return {"channel": "discord", "msg": msg, **result}

    if channel in ("x", "twitter"):
        result = _send_x(msg, params)
        return {"channel": "x", "msg": msg, **result}

    if channel == "pagerduty":
        result = _send_pagerduty(msg, params)
        return {"channel": "pagerduty", "msg": msg, **result}

    if channel == "jira":
        result = _send_jira(msg, params)
        return {"channel": "jira", "msg": msg, **result}

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
    """RECALL — Retrieve data.

    RECALL.docs(query=..., corpus=name, k=5, provider=local)
      One-step RAG: embed query → search embeddings corpus → return formatted context block.
      Returns: str — "[Source: path]\nchunk text\n\n[Source: ...]..."
      Empty string if corpus has no results.

    Other targets: key/value read from ~/.praxis/kv.db.
    """
    if target and target[0] == "docs":
        query = params.get("query") or (str(ctx.last_output) if ctx.last_output else "")
        if not query:
            raise ValueError("RECALL.docs requires query= parameter")
        corpus = params.get("corpus", "default")
        k = max(1, int(params.get("k", 5)))
        provider = params.get("provider", "local")
        db = EmbeddingsDB(provider=provider)
        results = db.search(query, corpus, k=k)
        if not results:
            return ""
        return "\n\n".join(f"[Source: {r['source']}]\n{r['text']}" for r in results)

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
