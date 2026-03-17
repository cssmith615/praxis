"""
Praxis Scheduler — run stored programs on a timed schedule.

Uses a SQLite database at ~/.praxis/schedule.db.

Concepts
--------
* ScheduledProgram — a Praxis program text + interval + metadata
* Scheduler        — add/remove/list programs; run_pending executes any that are due
* Triage           — optional callable that decides whether a full run is worth doing
                     (pass triage_fn to Scheduler; return False to skip the run)

Triage is how you get Ollama-class cost savings on monitoring loops: the triage
function runs a cheap local model first; if nothing changed, it returns False and
the full LLM-powered program doesn't fire.

Example
-------
    from praxis.scheduler import Scheduler
    from praxis.memory import ProgramMemory
    from praxis.handlers import HANDLERS
    from praxis.executor import Executor

    mem = ProgramMemory()
    exe = Executor(HANDLERS)
    sched = Scheduler(executor=exe, memory=mem)

    sched.add(
        goal="monitor denver flight prices",
        program_text='FETCH.api(url=https://api.prices.io/denver) -> EVAL.price(threshold=200) -> IF.$price < 200 -> OUT.telegram(msg="price drop!")',
        interval_seconds=3600,   # every hour
    )

    # One-shot: run anything due now
    sched.run_pending()

    # Or start a background thread (checks every 60 s by default)
    sched.start()
    ...
    sched.stop()
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

_SCHEDULE_DB = Path.home() / ".praxis" / "schedule.db"

# Module-level import so tests can patch praxis.scheduler.parse
from praxis.grammar import parse


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScheduledProgram:
    id:               str
    goal:             str
    program_text:     str
    interval_seconds: int
    next_run_at:      float
    last_run_at:      Optional[float] = None
    last_outcome:     Optional[str]   = None
    last_output:      Any             = None
    enabled:          bool            = True


@dataclass
class RunResult:
    schedule_id:  str
    goal:         str
    started_at:   float
    duration_ms:  int
    status:       str   # "ok" | "error" | "skipped" | "triage_skip"
    steps:        int   = 0
    error:        Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────────────────────

class Scheduler:
    """
    SQLite-backed interval scheduler for Praxis programs.

    Parameters
    ----------
    executor : Executor | None
        If provided, used to execute programs directly from program_text.
        If None, only schedule management (add/list/remove) works.
    memory : ProgramMemory | None
        Optional — used to store run outcomes back into program memory.
    triage_fn : Callable[[str, Any], bool] | None
        Optional — called before each run with (goal, last_output).
        Return True to proceed, False to skip this cycle (triage_skip).
        Ideal integration point for Ollama-based cheapness check.
    handlers : dict | None
        Handler registry passed to executor if executor is None.
    """

    def __init__(
        self,
        executor=None,
        memory=None,
        triage_fn: Optional[Callable[[str, Any], bool]] = None,
        handlers: Optional[dict] = None,
    ) -> None:
        self.executor   = executor
        self.memory     = memory
        self.triage_fn  = triage_fn
        self.handlers   = handlers
        self._db_path   = _SCHEDULE_DB
        self._stop      = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._init_db()

    # ── DB setup ───────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schedule (
                id               TEXT PRIMARY KEY,
                goal             TEXT NOT NULL,
                program_text     TEXT NOT NULL,
                interval_seconds INTEGER NOT NULL,
                next_run_at      REAL NOT NULL,
                last_run_at      REAL,
                last_outcome     TEXT,
                last_output      TEXT,
                enabled          INTEGER NOT NULL DEFAULT 1
            )
        """)
        conn.commit()
        conn.close()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    # ── CRUD ───────────────────────────────────────────────────────────────────

    def add(
        self,
        goal: str,
        program_text: str,
        interval_seconds: int,
        run_immediately: bool = False,
    ) -> str:
        """
        Add a program to the schedule. Returns the schedule ID.

        Parameters
        ----------
        goal             : human-readable description of what this program does
        program_text     : the Praxis program to execute
        interval_seconds : how often to run (e.g. 3600 = every hour)
        run_immediately  : if True, next_run_at = now (fires on first run_pending call)
        """
        schedule_id  = str(uuid.uuid4())[:8]
        next_run_at  = time.time() if run_immediately else time.time() + interval_seconds
        conn = self._conn()
        conn.execute(
            """INSERT INTO schedule
               (id, goal, program_text, interval_seconds, next_run_at, enabled)
               VALUES (?, ?, ?, ?, ?, 1)""",
            (schedule_id, goal, program_text, interval_seconds, next_run_at),
        )
        conn.commit()
        conn.close()
        return schedule_id

    def remove(self, schedule_id: str) -> bool:
        """Remove a scheduled program. Returns True if it existed."""
        conn = self._conn()
        cur  = conn.execute("DELETE FROM schedule WHERE id = ?", (schedule_id,))
        conn.commit()
        conn.close()
        return cur.rowcount > 0

    def enable(self, schedule_id: str, enabled: bool = True) -> None:
        conn = self._conn()
        conn.execute("UPDATE schedule SET enabled = ? WHERE id = ?",
                     (1 if enabled else 0, schedule_id))
        conn.commit()
        conn.close()

    def list_programs(self) -> list[ScheduledProgram]:
        """Return all scheduled programs sorted by next_run_at."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT id, goal, program_text, interval_seconds, next_run_at, "
            "last_run_at, last_outcome, last_output, enabled FROM schedule ORDER BY next_run_at"
        ).fetchall()
        conn.close()
        return [
            ScheduledProgram(
                id=r[0], goal=r[1], program_text=r[2],
                interval_seconds=r[3], next_run_at=r[4],
                last_run_at=r[5], last_outcome=r[6],
                last_output=json.loads(r[7]) if r[7] else None,
                enabled=bool(r[8]),
            )
            for r in rows
        ]

    # ── Execution ──────────────────────────────────────────────────────────────

    def run_pending(self) -> list[RunResult]:
        """
        Execute all enabled programs whose next_run_at <= now.
        Updates next_run_at and last_run_at after each run.
        Returns a list of RunResult for each program that was checked.
        """
        now     = time.time()
        results = []
        conn    = self._conn()
        due_rows = conn.execute(
            "SELECT id, goal, program_text, interval_seconds, last_output "
            "FROM schedule WHERE enabled = 1 AND next_run_at <= ?",
            (now,),
        ).fetchall()
        conn.close()

        for row in due_rows:
            schedule_id, goal, program_text, interval_seconds, last_output_json = row
            last_output = json.loads(last_output_json) if last_output_json else None
            result = self._run_one(schedule_id, goal, program_text,
                                   interval_seconds, last_output)
            results.append(result)

        return results

    def _run_one(
        self,
        schedule_id: str,
        goal: str,
        program_text: str,
        interval_seconds: int,
        last_output: Any,
    ) -> RunResult:
        started_at = time.time()

        # Triage gate — cheap check before expensive run
        if self.triage_fn is not None:
            try:
                should_run = self.triage_fn(goal, last_output)
            except Exception as e:
                should_run = True  # on triage failure, default to running
                print(f"[SCHEDULER] triage error for '{goal}': {e} — proceeding anyway")

            if not should_run:
                self._update_db(schedule_id, interval_seconds,
                                outcome="triage_skip", output=last_output)
                return RunResult(
                    schedule_id=schedule_id, goal=goal,
                    started_at=started_at, duration_ms=0,
                    status="triage_skip",
                )

        # Execute
        if self.executor is None:
            self._update_db(schedule_id, interval_seconds,
                            outcome="error", output=None)
            return RunResult(
                schedule_id=schedule_id, goal=goal,
                started_at=started_at, duration_ms=0,
                status="error", error="No executor configured",
            )

        try:
            program = parse(program_text)
            exec_results = self.executor.execute(program, memory=self.memory)
            duration_ms  = int((time.time() - started_at) * 1000)
            output = exec_results[-1]["output"] if exec_results else None
            steps  = len(exec_results)

            # Store outcome back into program memory
            if self.memory is not None:
                log_entries = [r["log_entry"] for r in exec_results]
                self.memory.store(goal, program_text, "ok", log_entries)

            self._update_db(schedule_id, interval_seconds,
                            outcome="ok", output=output)
            return RunResult(
                schedule_id=schedule_id, goal=goal,
                started_at=started_at, duration_ms=duration_ms,
                status="ok", steps=steps,
            )

        except Exception as exc:
            duration_ms = int((time.time() - started_at) * 1000)
            self._update_db(schedule_id, interval_seconds,
                            outcome="error", output=None)
            return RunResult(
                schedule_id=schedule_id, goal=goal,
                started_at=started_at, duration_ms=duration_ms,
                status="error", error=str(exc),
            )

    def _update_db(
        self,
        schedule_id: str,
        interval_seconds: int,
        outcome: str,
        output: Any,
    ) -> None:
        conn = self._conn()
        conn.execute(
            """UPDATE schedule SET
               last_run_at  = ?,
               next_run_at  = ?,
               last_outcome = ?,
               last_output  = ?
               WHERE id = ?""",
            (
                time.time(),
                time.time() + interval_seconds,
                outcome,
                json.dumps(output, default=str) if output is not None else None,
                schedule_id,
            ),
        )
        conn.commit()
        conn.close()

    # ── Background thread ──────────────────────────────────────────────────────

    def start(self, check_interval: int = 60) -> None:
        """Start a background thread that calls run_pending every check_interval seconds."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def _loop():
            while not self._stop.wait(check_interval):
                results = self.run_pending()
                for r in results:
                    if r.status == "ok":
                        print(f"[SCHEDULER] ran '{r.goal}' — {r.steps} steps in {r.duration_ms}ms")
                    elif r.status == "triage_skip":
                        pass  # silent skip — that's the point
                    else:
                        print(f"[SCHEDULER] error running '{r.goal}': {r.error}")

        self._thread = threading.Thread(target=_loop, daemon=True, name="praxis-scheduler")
        self._thread.start()
        print(f"[SCHEDULER] started (checking every {check_interval}s)")

    def stop(self) -> None:
        """Signal the background thread to stop."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        print("[SCHEDULER] stopped")
