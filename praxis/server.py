"""
praxis serve — local web dashboard

Serves a browser UI at http://localhost:7822 with six tabs:
  Dashboard    — stats overview + recent activity
  Programs     — full program memory library with search
  Logs         — live execution history from ~/.praxis/execution.log
  Constitution — view and add constitutional rules
  Schedules    — view, enable/disable, and delete scheduled jobs
  Editor       — write and run Praxis programs in-browser

API routes (all JSON):
  GET  /api/stats
  GET  /api/programs?limit=N&search=q
  GET  /api/programs/{id}
  DELETE /api/programs/{id}
  GET  /api/logs?limit=N
  GET  /api/constitution
  POST /api/constitution/rules   {rule_text, verbs}
  GET  /api/schedules
  DELETE /api/schedules/{id}
  PATCH /api/schedules/{id}      {enabled: bool}
  POST /api/run                  {program, mode}

Start:
  praxis serve                   # default port 7822
  praxis serve --port 8080
  praxis serve --host 0.0.0.0
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from praxis.grammar import parse
from praxis.validator import validate
from praxis.executor import Executor
from praxis.handlers import HANDLERS
from praxis.memory import ProgramMemory
from praxis.constitution import Constitution

_PRAXIS_DIR   = Path.home() / ".praxis"
_LOG_PATH     = _PRAXIS_DIR / "execution.log"
_KV_DB        = _PRAXIS_DIR / "kv.db"
_MEM_DB       = _PRAXIS_DIR / "memory.db"
_SCHEDULE_DB  = _PRAXIS_DIR / "schedule.db"
_WEBHOOK_DB   = _PRAXIS_DIR / "webhooks.db"
_ACTIVITY_LOG = _PRAXIS_DIR / "activity.log"

# ─────────────────────────────────────────────────────────────────────────────
# Singletons
# ─────────────────────────────────────────────────────────────────────────────

_memory: ProgramMemory | None = None
_constitution: Constitution | None = None


def _get_memory() -> ProgramMemory:
    global _memory
    if _memory is None:
        _memory = ProgramMemory()
    return _memory


def _get_constitution() -> Constitution:
    global _constitution
    if _constitution is None:
        _constitution = Constitution()
    return _constitution


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Praxis Dashboard", version="0.7.0", docs_url="/api/docs")


# ─────────────────────────────────────────────────────────────────────────────
# API — stats
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/stats")
def stats() -> dict:
    mem   = _get_memory()
    const = _get_constitution()

    program_count = mem.count()
    rule_count    = len(const)

    # Count log entries and tally outcomes
    log_entries = _read_log(limit=10000)
    total_runs  = len(log_entries)
    errors      = sum(1 for e in log_entries if e.get("status") == "error")
    ok_runs     = total_runs - errors
    success_rate = round(ok_runs / total_runs * 100, 1) if total_runs else 0.0

    # Detect active provider
    provider = (
        "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else
        "openai"    if os.environ.get("OPENAI_API_KEY")    else
        "grok"      if os.environ.get("GROK_API_KEY")      else
        "gemini"    if os.environ.get("GEMINI_API_KEY")    else
        "ollama"
    )

    return {
        "programs":    program_count,
        "rules":       rule_count,
        "log_entries": total_runs,
        "success_rate": success_rate,
        "provider":    provider,
    }


# ─────────────────────────────────────────────────────────────────────────────
# API — programs
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/programs")
def list_programs(limit: int = 50, search: str = "") -> dict:
    mem = _get_memory()
    programs = mem.recent(limit * 3 if search else limit)  # over-fetch when filtering

    if search:
        q = search.lower()
        programs = [p for p in programs if q in p.goal_text.lower()
                    or q in p.shaun_program.lower()]
        programs = programs[:limit]

    return {
        "programs": [
            {
                "id":       p.id,
                "goal":     p.goal_text,
                "program":  p.shaun_program,
                "outcome":  p.outcome,
                "created":  p.created_at,
            }
            for p in programs
        ],
        "total": mem.count(),
    }


@app.get("/api/programs/{program_id}")
def get_program(program_id: str) -> dict:
    mem = _get_memory()
    programs = mem.recent(10000)
    for p in programs:
        if p.id == program_id or p.id.startswith(program_id):
            return {
                "id":      p.id,
                "goal":    p.goal_text,
                "program": p.shaun_program,
                "outcome": p.outcome,
                "created": p.created_at,
            }
    raise HTTPException(status_code=404, detail="Program not found")


@app.delete("/api/programs/{program_id}")
def delete_program(program_id: str) -> dict:
    """Delete a stored program by id prefix."""
    if not _MEM_DB.exists():
        raise HTTPException(status_code=404, detail="Memory database not found")
    conn = sqlite3.connect(_MEM_DB)
    try:
        cur = conn.execute(
            "DELETE FROM programs WHERE id = ? OR id LIKE ?",
            (program_id, program_id + "%"),
        )
        conn.commit()
        deleted = cur.rowcount
    finally:
        conn.close()
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Program not found")
    return {"deleted": deleted}


# ─────────────────────────────────────────────────────────────────────────────
# API — logs
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/logs")
def get_logs(limit: int = 100) -> dict:
    entries = _read_log(limit=limit)
    return {"entries": entries, "total": len(entries)}


# ─────────────────────────────────────────────────────────────────────────────
# API — constitution
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/constitution")
def get_constitution() -> dict:
    const = _get_constitution()
    rules = [
        {
            "verbs": sorted(r.verbs),
            "text":  r.text,
            "source": r.source,
        }
        for r in const
    ]
    return {"rules": rules, "count": len(rules)}


class AddRuleRequest(BaseModel):
    rule_text: str
    verbs: list[str]


@app.post("/api/constitution/rules")
def add_rule(req: AddRuleRequest) -> dict:
    const = _get_constitution()
    added = const.append_rule(req.rule_text, req.verbs, source="dashboard")
    return {"added": added, "total": len(const)}


# ─────────────────────────────────────────────────────────────────────────────
# API — schedules
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/schedules")
def list_schedules() -> dict:
    if not _SCHEDULE_DB.exists():
        return {"schedules": []}
    conn = sqlite3.connect(_SCHEDULE_DB)
    try:
        rows = conn.execute(
            "SELECT id, goal, program_text, interval_seconds, next_run_at, "
            "last_run_at, last_outcome, enabled FROM schedule ORDER BY next_run_at"
        ).fetchall()
    finally:
        conn.close()
    schedules = []
    import time
    now = time.time()
    for row in rows:
        sid, goal, program_text, interval_seconds, next_run_at, last_run_at, last_outcome, enabled = row
        schedules.append({
            "id":               sid,
            "goal":             goal or "",
            "program":          program_text or "",
            "interval_seconds": interval_seconds,
            "next_run_at":      next_run_at,
            "last_run_at":      last_run_at,
            "last_outcome":     last_outcome or "",
            "enabled":          bool(enabled),
            "overdue":          bool(enabled and next_run_at and next_run_at < now),
        })
    return {"schedules": schedules}


@app.delete("/api/schedules/{schedule_id}")
def delete_schedule(schedule_id: str) -> dict:
    if not _SCHEDULE_DB.exists():
        raise HTTPException(status_code=404, detail="Schedule database not found")
    conn = sqlite3.connect(_SCHEDULE_DB)
    try:
        cur = conn.execute("DELETE FROM schedule WHERE id = ?", (schedule_id,))
        conn.commit()
        deleted = cur.rowcount
    finally:
        conn.close()
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return {"deleted": deleted}


class PatchScheduleRequest(BaseModel):
    enabled: bool


@app.patch("/api/schedules/{schedule_id}")
def patch_schedule(schedule_id: str, req: PatchScheduleRequest) -> dict:
    if not _SCHEDULE_DB.exists():
        raise HTTPException(status_code=404, detail="Schedule database not found")
    conn = sqlite3.connect(_SCHEDULE_DB)
    try:
        cur = conn.execute(
            "UPDATE schedule SET enabled = ? WHERE id = ?",
            (1 if req.enabled else 0, schedule_id),
        )
        conn.commit()
        updated = cur.rowcount
    finally:
        conn.close()
    if updated == 0:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return {"id": schedule_id, "enabled": req.enabled}


# ─────────────────────────────────────────────────────────────────────────────
# API — webhooks
# ─────────────────────────────────────────────────────────────────────────────

def _get_webhook_conn() -> sqlite3.Connection:
    _WEBHOOK_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_WEBHOOK_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS webhooks (
            id TEXT PRIMARY KEY,
            name TEXT,
            program_text TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    return conn


class RegisterWebhookRequest(BaseModel):
    name: str
    program_text: str


@app.get("/api/webhooks")
def list_webhooks() -> dict:
    conn = _get_webhook_conn()
    try:
        rows = conn.execute("SELECT id, name, program_text, created_at FROM webhooks").fetchall()
    finally:
        conn.close()
    return {"webhooks": [{"id": r[0], "name": r[1], "program": r[2], "created_at": r[3]} for r in rows]}


@app.post("/api/webhooks")
def register_webhook(req: RegisterWebhookRequest) -> dict:
    import uuid, datetime
    wid = str(uuid.uuid4())[:8]
    now = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    conn = _get_webhook_conn()
    try:
        conn.execute(
            "INSERT INTO webhooks (id, name, program_text, created_at) VALUES (?, ?, ?, ?)",
            (wid, req.name, req.program_text, now),
        )
        conn.commit()
    finally:
        conn.close()
    return {"id": wid, "name": req.name, "url": f"/webhook/{wid}"}


@app.delete("/api/webhooks/{webhook_id}")
def delete_webhook(webhook_id: str) -> dict:
    conn = _get_webhook_conn()
    try:
        cur = conn.execute("DELETE FROM webhooks WHERE id = ?", (webhook_id,))
        conn.commit()
        deleted = cur.rowcount
    finally:
        conn.close()
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return {"deleted": deleted}


@app.post("/webhook/{webhook_id}")
async def trigger_webhook(webhook_id: str, request: Request) -> dict:
    """Trigger a registered webhook program with the incoming payload."""
    conn = _get_webhook_conn()
    try:
        row = conn.execute(
            "SELECT name, program_text FROM webhooks WHERE id = ?", (webhook_id,)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Webhook not found")

    name, program_text = row
    try:
        body = await request.body()
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {"raw": body.decode(errors="replace")}
    except Exception:
        payload = {}

    try:
        from praxis.grammar import parse
        from praxis.validator import validate
        from praxis.executor import Executor
        from praxis.handlers import HANDLERS

        ast = parse(program_text)
        errors = validate(ast)
        if errors:
            return {"ok": False, "error": "\n".join(errors)}

        executor = Executor(handlers=HANDLERS)
        results = executor.execute(ast, initial_variables={"event": payload})
        _append_activity("webhook", f"Webhook '{name}' triggered", {"id": webhook_id, "steps": len(results)})
        return {"ok": True, "steps": len(results), "webhook": name}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# API — activity feed
# ─────────────────────────────────────────────────────────────────────────────

def _append_activity(event_type: str, summary: str, detail: dict | None = None) -> None:
    """Append a plain-English activity event to the activity log."""
    import time as _time
    _ACTIVITY_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "type": event_type,
        "summary": summary,
        "detail": detail or {},
        "ts": _time.time(),
    }
    with open(_ACTIVITY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


@app.get("/api/activity")
def get_activity(limit: int = 50) -> dict:
    if not _ACTIVITY_LOG.exists():
        return {"events": []}
    lines = _ACTIVITY_LOG.read_text(encoding="utf-8").splitlines()
    events = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(events) >= limit:
            break
    return {"events": events}


# ─────────────────────────────────────────────────────────────────────────────
# API — run
# ─────────────────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    program: str
    mode: str = "dev"


@app.post("/api/run")
def run_program(req: RunRequest) -> dict:
    try:
        ast = parse(req.program)
    except Exception as exc:
        return {"ok": False, "error": f"Parse error: {exc}", "steps": []}

    errors = validate(ast, mode=req.mode)
    if errors:
        return {"ok": False, "error": "\n".join(errors), "steps": []}

    try:
        executor = Executor(handlers=HANDLERS, mode=req.mode)
        results  = executor.execute(ast)
        _append_activity("run", f"Program run via Editor ({len(results)} steps)", {"ok": True})
        return {"ok": True, "error": None, "steps": results}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "steps": []}


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard HTML (single-page, no build step)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return HTMLResponse(_HTML)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_log(limit: int = 100) -> list[dict]:
    if not _LOG_PATH.exists():
        return []
    lines = _LOG_PATH.read_text(encoding="utf-8").splitlines()
    entries = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(entries) >= limit:
            break
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# HTML — inline single-page dashboard
# ─────────────────────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Praxis Dashboard</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
    --text: #e2e8f0; --muted: #64748b; --accent: #6366f1;
    --green: #22c55e; --red: #ef4444; --yellow: #f59e0b; --blue: #3b82f6;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 14px; }

  /* Layout */
  nav { background: var(--surface); border-bottom: 1px solid var(--border); padding: 0 24px; display: flex; align-items: center; height: 52px; gap: 24px; position: sticky; top: 0; z-index: 10; }
  nav .logo { font-weight: 700; font-size: 16px; color: var(--accent); letter-spacing: -0.5px; margin-right: 8px; }
  nav .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); display: inline-block; margin-right: 4px; }
  nav .tab { color: var(--muted); cursor: pointer; padding: 4px 0; border-bottom: 2px solid transparent; transition: color .15s; font-size: 13px; white-space: nowrap; }
  nav .tab:hover { color: var(--text); }
  nav .tab.active { color: var(--text); border-bottom-color: var(--accent); }
  .spacer { flex: 1; }
  .provider-badge { background: #1e2130; border: 1px solid var(--border); border-radius: 6px; padding: 3px 10px; font-size: 12px; color: var(--muted); }

  main { max-width: 1100px; margin: 0 auto; padding: 28px 24px; }
  .page { display: none; }
  .page.active { display: block; }

  /* Cards */
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 28px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px; }
  .card .label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .6px; margin-bottom: 10px; }
  .card .value { font-size: 28px; font-weight: 700; }
  .card .sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .card.green .value { color: var(--green); }
  .card.blue  .value { color: var(--blue); }
  .card.yellow .value { color: var(--yellow); }
  .card.accent .value { color: var(--accent); }

  /* Tables */
  .section-title { font-size: 13px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .6px; margin-bottom: 12px; }
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; padding: 8px 12px; border-bottom: 1px solid var(--border); }
  td { padding: 9px 12px; border-bottom: 1px solid #1e2130; font-size: 13px; vertical-align: top; }
  tr:hover td { background: #1e2130; }
  .mono { font-family: var(--font-mono); font-size: 12px; }
  .badge { display: inline-block; border-radius: 4px; padding: 2px 7px; font-size: 11px; font-weight: 600; }
  .badge.ok, .badge.success { background: #14532d; color: var(--green); }
  .badge.error, .badge.failure { background: #450a0a; color: var(--red); }
  .badge.partial { background: #451a03; color: var(--yellow); }
  .badge.planned { background: #1e1b4b; color: var(--accent); }
  .truncate { max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  /* Search */
  .toolbar { display: flex; gap: 10px; margin-bottom: 16px; align-items: center; }
  input[type=text], textarea, select {
    background: var(--surface); border: 1px solid var(--border); color: var(--text);
    border-radius: 7px; padding: 7px 12px; font-size: 13px; outline: none;
    transition: border-color .15s;
  }
  input[type=text]:focus, textarea:focus { border-color: var(--accent); }
  input[type=text] { width: 260px; }
  .btn { background: var(--accent); color: #fff; border: none; border-radius: 7px; padding: 7px 16px; font-size: 13px; cursor: pointer; font-weight: 600; transition: opacity .15s; }
  .btn:hover { opacity: .85; }
  .btn.secondary { background: var(--surface); border: 1px solid var(--border); color: var(--text); font-weight: 500; }
  .btn.danger { background: #7f1d1d; }
  .btn:disabled { opacity: .4; cursor: not-allowed; }

  /* Editor */
  .editor-wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .editor-pane { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
  .pane-header { padding: 10px 14px; border-bottom: 1px solid var(--border); font-size: 12px; color: var(--muted); display: flex; justify-content: space-between; align-items: center; }
  textarea.code { font-family: var(--font-mono); font-size: 13px; line-height: 1.6; padding: 14px; width: 100%; height: 340px; resize: vertical; border: none; border-radius: 0; background: var(--surface); }
  .results-body { padding: 14px; height: 340px; overflow-y: auto; }
  .step-row { display: flex; gap: 10px; align-items: flex-start; padding: 6px 0; border-bottom: 1px solid #1a1d27; font-size: 12px; }
  .step-verb { font-family: var(--font-mono); color: var(--accent); font-weight: 700; min-width: 70px; }
  .step-status { min-width: 50px; }
  .step-out { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 280px; }
  .step-ms { color: var(--muted); min-width: 40px; text-align: right; }
  .error-box { background: #450a0a; border: 1px solid #7f1d1d; border-radius: 7px; padding: 12px 14px; color: #fca5a5; font-family: var(--font-mono); font-size: 12px; margin-top: 10px; white-space: pre-wrap; }

  /* Constitution */
  .rule-list { display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px; }
  .rule-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; display: flex; gap: 14px; align-items: flex-start; }
  .rule-verbs { display: flex; gap: 5px; flex-wrap: wrap; min-width: 80px; }
  .vtag { background: #1e1b4b; color: var(--accent); border-radius: 4px; padding: 1px 6px; font-size: 11px; font-family: var(--font-mono); }
  .rule-text { flex: 1; line-height: 1.5; }
  .rule-source { font-size: 11px; color: var(--muted); }
  .add-rule-form { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px; }
  .add-rule-form label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 5px; margin-top: 12px; }
  .add-rule-form label:first-child { margin-top: 0; }
  .add-rule-form input[type=text] { width: 100%; }

  /* Logs */
  .log-row td:first-child { font-family: var(--font-mono); color: var(--accent); font-weight: 700; }
  .empty { text-align: center; padding: 48px; color: var(--muted); }

  /* Modal */
  .modal-bg { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.7); z-index: 100; align-items: center; justify-content: center; }
  .modal-bg.open { display: flex; }
  .modal { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 24px; width: 600px; max-width: 95vw; }
  .modal h3 { margin-bottom: 14px; }
  .modal pre { background: var(--bg); border: 1px solid var(--border); border-radius: 7px; padding: 14px; font-size: 12px; overflow-x: auto; max-height: 320px; overflow-y: auto; white-space: pre-wrap; }
  .modal-footer { display: flex; justify-content: flex-end; gap: 8px; margin-top: 16px; }
</style>
</head>
<body>

<nav>
  <span class="logo">⬡ Praxis</span>
  <span class="dot"></span>
  <span id="nav-status" style="font-size:12px;color:var(--muted)">connecting…</span>
  <div class="spacer"></div>
  <span id="tab-dashboard" class="tab active" onclick="showTab('dashboard')">Dashboard</span>
  <span id="tab-programs"  class="tab" onclick="showTab('programs')">Programs</span>
  <span id="tab-logs"      class="tab" onclick="showTab('logs')">Logs</span>
  <span id="tab-constitution" class="tab" onclick="showTab('constitution')">Constitution</span>
  <span id="tab-schedules" class="tab" onclick="showTab('schedules')">Schedules</span>
  <span id="tab-webhooks"  class="tab" onclick="showTab('webhooks')">Webhooks</span>
  <span id="tab-activity"  class="tab" onclick="showTab('activity')">Activity</span>
  <span id="tab-editor"    class="tab" onclick="showTab('editor')">Editor</span>
  <div class="spacer"></div>
  <span class="provider-badge" id="provider-badge">—</span>
</nav>

<main>

<!-- ── DASHBOARD ── -->
<div id="page-dashboard" class="page active">
  <div class="cards">
    <div class="card blue">  <div class="label">Programs</div>   <div class="value" id="stat-programs">—</div>  <div class="sub">in memory</div></div>
    <div class="card green"> <div class="label">Success rate</div><div class="value" id="stat-success">—</div>  <div class="sub">of all runs</div></div>
    <div class="card accent"><div class="label">Rules</div>       <div class="value" id="stat-rules">—</div>    <div class="sub">constitutional</div></div>
    <div class="card yellow"><div class="label">Log entries</div> <div class="value" id="stat-logs">—</div>     <div class="sub">recorded</div></div>
  </div>
  <div class="section-title">Recent Programs</div>
  <table>
    <thead><tr><th>Goal</th><th>Outcome</th><th>Stored</th><th></th></tr></thead>
    <tbody id="dash-programs"><tr><td colspan="4" class="empty">Loading…</td></tr></tbody>
  </table>
</div>

<!-- ── PROGRAMS ── -->
<div id="page-programs" class="page">
  <div class="toolbar">
    <input type="text" id="prog-search" placeholder="Search goals or programs…" oninput="searchPrograms()">
    <span style="color:var(--muted);font-size:13px" id="prog-count"></span>
  </div>
  <table>
    <thead><tr><th>ID</th><th>Goal</th><th>Program</th><th>Outcome</th><th>Stored</th><th></th></tr></thead>
    <tbody id="programs-tbody"><tr><td colspan="6" class="empty">Loading…</td></tr></tbody>
  </table>
</div>

<!-- ── LOGS ── -->
<div id="page-logs" class="page">
  <div class="toolbar">
    <span class="section-title" style="margin:0">Execution Log</span>
    <div class="spacer"></div>
    <button class="btn secondary" onclick="loadLogs()">↻ Refresh</button>
    <select id="log-limit" onchange="loadLogs()">
      <option value="50">Last 50</option>
      <option value="100" selected>Last 100</option>
      <option value="250">Last 250</option>
    </select>
  </div>
  <table>
    <thead><tr><th>Verb</th><th>Label / Target</th><th>Status</th><th>Data</th><th>Time</th></tr></thead>
    <tbody id="logs-tbody"><tr><td colspan="5" class="empty">Loading…</td></tr></tbody>
  </table>
</div>

<!-- ── CONSTITUTION ── -->
<div id="page-constitution" class="page">
  <div class="toolbar">
    <span class="section-title" style="margin:0">Constitutional Rules</span>
    <span id="rule-count" style="color:var(--muted);font-size:13px;margin-left:8px"></span>
  </div>
  <div class="rule-list" id="rule-list"></div>
  <div class="section-title">Add Rule</div>
  <div class="add-rule-form">
    <label>Rule text (start with ALWAYS / NEVER / CONSIDER)</label>
    <input type="text" id="new-rule-text" placeholder="ALWAYS validate input before TRN." style="width:100%">
    <label>Verbs (comma-separated)</label>
    <input type="text" id="new-rule-verbs" placeholder="TRN, CLN, VALIDATE">
    <div style="margin-top:14px">
      <button class="btn" onclick="addRule()">Add Rule</button>
      <span id="rule-msg" style="margin-left:12px;font-size:13px;color:var(--muted)"></span>
    </div>
  </div>
</div>

<!-- ── SCHEDULES ── -->
<div id="page-schedules" class="page">
  <div class="toolbar">
    <span class="section-title" style="margin:0">Scheduled Jobs</span>
    <span id="sched-count" style="color:var(--muted);font-size:13px;margin-left:8px"></span>
    <div class="spacer"></div>
    <button class="btn secondary" onclick="loadSchedules()">↻ Refresh</button>
  </div>
  <table>
    <thead><tr><th>ID</th><th>Goal</th><th>Interval</th><th>Next Run</th><th>Last Run</th><th>Last Outcome</th><th>Status</th><th></th></tr></thead>
    <tbody id="schedules-tbody"><tr><td colspan="8" class="empty">Loading…</td></tr></tbody>
  </table>
</div>

<!-- ── WEBHOOKS ── -->
<div id="page-webhooks" class="page">
  <div class="toolbar">
    <span class="section-title" style="margin:0">Webhooks</span>
    <span id="wh-count" style="color:var(--muted);font-size:13px;margin-left:8px"></span>
    <div class="spacer"></div>
    <button class="btn secondary" onclick="loadWebhooks()">↻ Refresh</button>
  </div>
  <table style="margin-bottom:28px">
    <thead><tr><th>ID</th><th>Name</th><th>Trigger URL</th><th>Program</th><th></th></tr></thead>
    <tbody id="webhooks-tbody"><tr><td colspan="5" class="empty">Loading…</td></tr></tbody>
  </table>
  <div class="section-title">Register Webhook</div>
  <div class="add-rule-form">
    <label>Name</label>
    <input type="text" id="wh-name" placeholder="my-slack-bot" style="width:100%">
    <label>Program (Praxis DSL)</label>
    <textarea class="code" id="wh-program" style="height:120px;border:1px solid var(--border);border-radius:7px" placeholder="GEN.reply(input=$event.text) -> OUT.telegram(msg=$reply)"></textarea>
    <div style="margin-top:14px">
      <button class="btn" onclick="registerWebhook()">Register</button>
      <span id="wh-msg" style="margin-left:12px;font-size:13px;color:var(--muted)"></span>
    </div>
  </div>
</div>

<!-- ── ACTIVITY ── -->
<div id="page-activity" class="page">
  <div class="toolbar">
    <span class="section-title" style="margin:0">Activity Feed</span>
    <div class="spacer"></div>
    <button class="btn secondary" onclick="loadActivity()">↻ Refresh</button>
  </div>
  <div id="activity-feed" style="display:flex;flex-direction:column;gap:8px">
    <div class="empty">Loading…</div>
  </div>
</div>

<!-- ── EDITOR ── -->
<div id="page-editor" class="page">
  <div class="toolbar">
    <select id="editor-mode">
      <option value="dev">dev mode</option>
      <option value="prod">prod mode</option>
    </select>
    <button class="btn" onclick="runProgram()" id="run-btn">▶ Run</button>
    <button class="btn secondary" onclick="clearEditor()">Clear</button>
    <span id="run-status" style="font-size:13px;color:var(--muted);margin-left:8px"></span>
  </div>
  <div class="editor-wrap">
    <div class="editor-pane">
      <div class="pane-header">Program <span style="font-size:11px">Praxis DSL</span></div>
      <textarea class="code" id="editor-code" placeholder="// Write a Praxis program&#10;ING.sales -> CLN.null -> SUMM.text(max=200) -> OUT.telegram(msg=&quot;done&quot;)"></textarea>
    </div>
    <div class="editor-pane">
      <div class="pane-header">Results <span id="step-count" style="font-size:11px"></span></div>
      <div class="results-body" id="results-body">
        <div style="color:var(--muted);font-size:13px;margin-top:60px;text-align:center">Run a program to see results here.</div>
      </div>
    </div>
  </div>
</div>

</main>

<!-- Program detail modal -->
<div class="modal-bg" id="prog-modal" onclick="if(event.target===this)closeModal()">
  <div class="modal">
    <h3 id="modal-goal"></h3>
    <pre id="modal-program"></pre>
    <div class="modal-footer">
      <button class="btn secondary" onclick="sendToEditor()">Open in Editor</button>
      <button class="btn danger" onclick="deleteProgram()">Delete</button>
      <button class="btn secondary" onclick="closeModal()">Close</button>
    </div>
  </div>
</div>

<script>
let _currentProgramId = null;

// ── tab switching ──────────────────────────────────────────────────────────
function showTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  document.getElementById('page-' + name).classList.add('active');
  if (name === 'programs') loadPrograms();
  if (name === 'logs') loadLogs();
  if (name === 'constitution') loadConstitution();
  if (name === 'schedules') loadSchedules();
  if (name === 'webhooks') loadWebhooks();
  if (name === 'activity') loadActivity();
}

// ── stats ─────────────────────────────────────────────────────────────────
async function loadStats() {
  try {
    const r = await fetch('/api/stats');
    const d = await r.json();
    document.getElementById('stat-programs').textContent = d.programs;
    document.getElementById('stat-success').textContent  = d.success_rate + '%';
    document.getElementById('stat-rules').textContent    = d.rules;
    document.getElementById('stat-logs').textContent     = d.log_entries;
    document.getElementById('provider-badge').textContent = d.provider;
    document.getElementById('nav-status').textContent = 'connected';
    document.querySelector('.dot').style.background = '#22c55e';
  } catch(e) {
    document.getElementById('nav-status').textContent = 'offline';
    document.querySelector('.dot').style.background = '#ef4444';
  }
}

async function loadDashboardPrograms() {
  const r = await fetch('/api/programs?limit=8');
  const d = await r.json();
  const tbody = document.getElementById('dash-programs');
  if (!d.programs.length) { tbody.innerHTML = '<tr><td colspan="4" class="empty">No programs yet. Run <code>praxis goal "..."</code> to populate.</td></tr>'; return; }
  tbody.innerHTML = d.programs.map(p => `
    <tr>
      <td class="truncate">${esc(p.goal)}</td>
      <td><span class="badge ${p.outcome}">${p.outcome}</span></td>
      <td class="mono" style="color:var(--muted);font-size:11px">${p.created.slice(0,16)}</td>
      <td><button class="btn secondary" style="padding:3px 10px;font-size:11px" onclick="showProgram('${p.id}')">View</button></td>
    </tr>`).join('');
}

// ── programs ──────────────────────────────────────────────────────────────
let _allPrograms = [];
async function loadPrograms() {
  const r = await fetch('/api/programs?limit=200');
  const d = await r.json();
  _allPrograms = d.programs;
  renderPrograms(_allPrograms, d.total);
}

function renderPrograms(programs, total) {
  document.getElementById('prog-count').textContent = `${programs.length} of ${total}`;
  const tbody = document.getElementById('programs-tbody');
  if (!programs.length) { tbody.innerHTML = '<tr><td colspan="6" class="empty">No programs found.</td></tr>'; return; }
  tbody.innerHTML = programs.map(p => `
    <tr>
      <td class="mono" style="font-size:11px;color:var(--muted)">${p.id.slice(0,8)}…</td>
      <td class="truncate">${esc(p.goal)}</td>
      <td class="mono truncate" style="color:var(--muted);font-size:11px">${esc(p.program.split('\\n')[0])}</td>
      <td><span class="badge ${p.outcome}">${p.outcome}</span></td>
      <td class="mono" style="font-size:11px;color:var(--muted)">${p.created.slice(0,16)}</td>
      <td><button class="btn secondary" style="padding:3px 10px;font-size:11px" onclick="showProgram('${p.id}')">View</button></td>
    </tr>`).join('');
}

function searchPrograms() {
  const q = document.getElementById('prog-search').value.toLowerCase();
  const filtered = q ? _allPrograms.filter(p => p.goal.toLowerCase().includes(q) || p.program.toLowerCase().includes(q)) : _allPrograms;
  renderPrograms(filtered, _allPrograms.length);
}

async function showProgram(id) {
  const r = await fetch('/api/programs/' + id);
  const p = await r.json();
  _currentProgramId = p.id;
  document.getElementById('modal-goal').textContent = p.goal;
  document.getElementById('modal-program').textContent = p.program;
  document.getElementById('prog-modal').classList.add('open');
}

function closeModal() { document.getElementById('prog-modal').classList.remove('open'); }

function sendToEditor() {
  const prog = document.getElementById('modal-program').textContent;
  document.getElementById('editor-code').value = prog;
  closeModal();
  showTab('editor');
}

async function deleteProgram() {
  if (!_currentProgramId) return;
  if (!confirm('Delete this program?')) return;
  await fetch('/api/programs/' + _currentProgramId, { method: 'DELETE' });
  closeModal();
  loadPrograms();
  loadStats();
  loadDashboardPrograms();
}

// ── logs ──────────────────────────────────────────────────────────────────
async function loadLogs() {
  const limit = document.getElementById('log-limit')?.value || 100;
  const r = await fetch('/api/logs?limit=' + limit);
  const d = await r.json();
  const tbody = document.getElementById('logs-tbody');
  if (!d.entries.length) { tbody.innerHTML = '<tr><td colspan="5" class="empty">No log entries yet.</td></tr>'; return; }
  tbody.innerHTML = d.entries.map(e => {
    const ts = e.timestamp ? new Date(e.timestamp * 1000).toLocaleTimeString() : '';
    const label = e.label || e.target || '';
    const dataStr = e.data ? JSON.stringify(e.data).slice(0, 60) : '';
    return `<tr class="log-row">
      <td>${esc(e.verb || '—')}</td>
      <td>${esc(label)}</td>
      <td><span class="badge ${e.status || ''}">${e.status || ''}</span></td>
      <td class="mono" style="font-size:11px;color:var(--muted)">${esc(dataStr)}</td>
      <td class="mono" style="font-size:11px;color:var(--muted)">${ts}</td>
    </tr>`;
  }).join('');
}

// ── constitution ──────────────────────────────────────────────────────────
async function loadConstitution() {
  const r = await fetch('/api/constitution');
  const d = await r.json();
  document.getElementById('rule-count').textContent = `${d.count} rule${d.count !== 1 ? 's' : ''}`;
  const list = document.getElementById('rule-list');
  if (!d.rules.length) { list.innerHTML = '<div class="empty">No rules yet.</div>'; return; }
  list.innerHTML = d.rules.map(r => `
    <div class="rule-card">
      <div class="rule-verbs">${r.verbs.map(v => `<span class="vtag">${esc(v)}</span>`).join('')}</div>
      <div class="rule-text">${esc(r.text)}</div>
      <div class="rule-source">${esc(r.source)}</div>
    </div>`).join('');
}

async function addRule() {
  const text  = document.getElementById('new-rule-text').value.trim();
  const verbs = document.getElementById('new-rule-verbs').value.split(',').map(v => v.trim()).filter(Boolean);
  const msg   = document.getElementById('rule-msg');
  if (!text || !verbs.length) { msg.textContent = 'Rule text and at least one verb required.'; return; }
  const r = await fetch('/api/constitution/rules', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ rule_text: text, verbs })
  });
  const d = await r.json();
  if (d.added) {
    msg.style.color = 'var(--green)';
    msg.textContent = 'Rule added.';
    document.getElementById('new-rule-text').value = '';
    document.getElementById('new-rule-verbs').value = '';
    loadConstitution();
    loadStats();
  } else {
    msg.style.color = 'var(--yellow)';
    msg.textContent = 'Duplicate — rule already exists.';
  }
}

// ── schedules ─────────────────────────────────────────────────────────────
function fmtInterval(seconds) {
  if (!seconds) return '—';
  if (seconds < 120)  return seconds + 's';
  if (seconds < 7200) return Math.round(seconds / 60) + 'm';
  if (seconds < 172800) return Math.round(seconds / 3600) + 'h';
  return Math.round(seconds / 86400) + 'd';
}

function fmtTs(ts) {
  if (!ts) return '—';
  return new Date(ts * 1000).toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
}

async function loadSchedules() {
  const r = await fetch('/api/schedules');
  const d = await r.json();
  const schedules = d.schedules || [];
  document.getElementById('sched-count').textContent =
    schedules.length + ' job' + (schedules.length !== 1 ? 's' : '');
  const tbody = document.getElementById('schedules-tbody');
  if (!schedules.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="empty">No scheduled jobs. Use <code>praxis agent</code> and ask to schedule a program.</td></tr>';
    return;
  }
  tbody.innerHTML = schedules.map(s => {
    const enabledBadge = s.enabled
      ? '<span class="badge ok">enabled</span>'
      : '<span class="badge partial">paused</span>';
    const overdueBadge = s.overdue ? ' <span class="badge error" style="font-size:10px">overdue</span>' : '';
    const outcomeBadge = s.last_outcome
      ? `<span class="badge ${s.last_outcome === 'ok' ? 'ok' : 'error'}">${esc(s.last_outcome)}</span>`
      : '<span style="color:var(--muted)">—</span>';
    return `<tr>
      <td class="mono" style="font-size:11px;color:var(--muted)">${esc(s.id)}</td>
      <td class="truncate">${esc(s.goal || '—')}</td>
      <td class="mono">${fmtInterval(s.interval_seconds)}</td>
      <td class="mono" style="font-size:12px">${fmtTs(s.next_run_at)}${overdueBadge}</td>
      <td class="mono" style="font-size:12px">${fmtTs(s.last_run_at)}</td>
      <td>${outcomeBadge}</td>
      <td>${enabledBadge}</td>
      <td style="display:flex;gap:6px;padding:6px 12px">
        <button class="btn secondary" style="padding:3px 10px;font-size:11px"
          onclick="toggleSchedule('${esc(s.id)}', ${!s.enabled})">${s.enabled ? 'Pause' : 'Resume'}</button>
        <button class="btn danger" style="padding:3px 10px;font-size:11px"
          onclick="deleteSchedule('${esc(s.id)}')">Delete</button>
      </td>
    </tr>`;
  }).join('');
}

async function toggleSchedule(id, enable) {
  await fetch('/api/schedules/' + id, {
    method: 'PATCH', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ enabled: enable })
  });
  loadSchedules();
}

async function deleteSchedule(id) {
  if (!confirm('Delete schedule ' + id + '?')) return;
  await fetch('/api/schedules/' + id, { method: 'DELETE' });
  loadSchedules();
}

// ── webhooks ──────────────────────────────────────────────────────────────
async function loadWebhooks() {
  const r = await fetch('/api/webhooks');
  const d = await r.json();
  const hooks = d.webhooks || [];
  document.getElementById('wh-count').textContent = hooks.length + ' registered';
  const tbody = document.getElementById('webhooks-tbody');
  if (!hooks.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty">No webhooks. Register one below.</td></tr>';
    return;
  }
  const origin = window.location.origin;
  tbody.innerHTML = hooks.map(h => `
    <tr>
      <td class="mono" style="font-size:11px;color:var(--muted)">${esc(h.id)}</td>
      <td>${esc(h.name)}</td>
      <td class="mono" style="font-size:11px">
        <a href="${origin}/webhook/${esc(h.id)}" style="color:var(--accent)" target="_blank">${origin}/webhook/${esc(h.id)}</a>
      </td>
      <td class="mono truncate" style="font-size:11px;color:var(--muted)">${esc((h.program||'').split('\\n')[0])}</td>
      <td><button class="btn danger" style="padding:3px 10px;font-size:11px" onclick="deleteWebhook('${esc(h.id)}')">Delete</button></td>
    </tr>`).join('');
}

async function registerWebhook() {
  const name = document.getElementById('wh-name').value.trim();
  const program = document.getElementById('wh-program').value.trim();
  const msg = document.getElementById('wh-msg');
  if (!name || !program) { msg.textContent = 'Name and program required.'; return; }
  const r = await fetch('/api/webhooks', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ name, program_text: program })
  });
  const d = await r.json();
  msg.style.color = 'var(--green)';
  msg.textContent = `Registered. URL: ${window.location.origin}/webhook/${d.id}`;
  document.getElementById('wh-name').value = '';
  document.getElementById('wh-program').value = '';
  loadWebhooks();
}

async function deleteWebhook(id) {
  if (!confirm('Delete webhook ' + id + '?')) return;
  await fetch('/api/webhooks/' + id, { method: 'DELETE' });
  loadWebhooks();
}

// ── activity ───────────────────────────────────────────────────────────────
const _TYPE_ICONS = { webhook: '⚡', schedule: '⏱', run: '▶', system: 'ℹ' };

async function loadActivity() {
  const r = await fetch('/api/activity?limit=50');
  const d = await r.json();
  const feed = document.getElementById('activity-feed');
  if (!d.events || !d.events.length) {
    feed.innerHTML = '<div class="empty">No activity yet. Run a program, trigger a webhook, or let a schedule fire.</div>';
    return;
  }
  feed.innerHTML = d.events.map(e => {
    const icon = _TYPE_ICONS[e.type] || '•';
    const ts = e.ts ? new Date(e.ts * 1000).toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}) : '';
    const detailStr = e.detail && Object.keys(e.detail).length ? JSON.stringify(e.detail) : '';
    return `<div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 16px;display:flex;gap:12px;align-items:flex-start">
      <span style="font-size:16px;min-width:24px">${icon}</span>
      <div style="flex:1">
        <div style="font-size:13px">${esc(e.summary)}</div>
        ${detailStr ? `<div class="mono" style="font-size:11px;color:var(--muted);margin-top:3px">${esc(detailStr.slice(0,120))}</div>` : ''}
      </div>
      <div class="mono" style="font-size:11px;color:var(--muted);white-space:nowrap">${ts}</div>
    </div>`;
  }).join('');
}

// ── editor ────────────────────────────────────────────────────────────────
async function runProgram() {
  const code = document.getElementById('editor-code').value.trim();
  const mode = document.getElementById('editor-mode').value;
  const btn  = document.getElementById('run-btn');
  const body = document.getElementById('results-body');
  const status = document.getElementById('run-status');
  if (!code) return;
  btn.disabled = true; btn.textContent = '⏳ Running…';
  status.textContent = '';
  body.innerHTML = '';
  try {
    const r = await fetch('/api/run', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ program: code, mode })
    });
    const d = await r.json();
    if (!d.ok) {
      body.innerHTML = `<div class="error-box">${esc(d.error)}</div>`;
      status.textContent = '✗ failed';
      status.style.color = 'var(--red)';
    } else {
      document.getElementById('step-count').textContent = `${d.steps.length} step${d.steps.length !== 1 ? 's' : ''}`;
      body.innerHTML = d.steps.map((s, i) => {
        const out = typeof s.output === 'object' ? JSON.stringify(s.output) : String(s.output || '');
        return `<div class="step-row">
          <span style="color:var(--muted);font-size:11px;min-width:22px">${i+1}</span>
          <span class="step-verb">${esc(s.verb)}${s.target?.length ? '.' + s.target.join('.') : ''}</span>
          <span class="step-status"><span class="badge ${s.status}">${s.status}</span></span>
          <span class="step-out" title="${esc(out)}">${esc(out.slice(0, 80))}</span>
          <span class="step-ms">${s.duration_ms}ms</span>
        </div>`;
      }).join('');
      const errs = d.steps.filter(s => s.status === 'error').length;
      status.textContent = errs ? `✗ ${errs} error${errs > 1 ? 's' : ''}` : '✓ ok';
      status.style.color = errs ? 'var(--red)' : 'var(--green)';
      loadStats();
    }
  } catch(e) {
    body.innerHTML = `<div class="error-box">${esc(String(e))}</div>`;
  }
  btn.disabled = false; btn.textContent = '▶ Run';
}

function clearEditor() {
  document.getElementById('editor-code').value = '';
  document.getElementById('results-body').innerHTML = '<div style="color:var(--muted);font-size:13px;margin-top:60px;text-align:center">Run a program to see results here.</div>';
  document.getElementById('step-count').textContent = '';
  document.getElementById('run-status').textContent = '';
}

// ── keyboard shortcut Ctrl/Cmd+Enter to run ───────────────────────────────
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    const activePage = document.querySelector('.page.active');
    if (activePage && activePage.id === 'page-editor') runProgram();
  }
});

// ── utils ─────────────────────────────────────────────────────────────────
function esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── init ──────────────────────────────────────────────────────────────────
(async function init() {
  await loadStats();
  await loadDashboardPrograms();
})();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PORT = 7822
DEFAULT_HOST = "127.0.0.1"


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "uvicorn is required for praxis serve. "
            "Run: pip install 'praxis-lang[bridge]'"
        ) from exc
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    serve()
