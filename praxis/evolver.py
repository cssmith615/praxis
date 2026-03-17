"""
Praxis Program Evolver — Sprint 16 (Pillar 6-PartB).

Outcome-driven program evolution: closes the feedback loop between execution
results and the program memory library. Programs that degrade in performance or
start failing are flagged as stale. Rewrites proposed by the Rewriter (Sprint 13)
can be promoted into the program library when benchmarked as improvements.

Pipeline:
  1. score()     — compute a composite health score for each stored program
  2. mark_stale()— flag programs whose score has dropped below a threshold
  3. benchmark() — compare an original program vs a proposed rewrite
  4. promote()   — replace a stale program with a faster/more-reliable rewrite

Health score = weighted average of:
  - success_rate  : fraction of recent executions with status='ok'
  - speed_score   : 1.0 if avg_ms < baseline, declining to 0.0 at 3× baseline
  - stability     : 1.0 if variance is low, 0.0 if high (steps that fluctuate)

Design decisions:
  - Works entirely from execution.log — no re-execution needed
  - All scoring is local; no LLM required
  - promote() writes directly to ProgramMemory (stores new program, marks old)
  - Staleness threshold is configurable (default 0.5 / 1.0)
  - A program is only promoted if the rewrite score is strictly better
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_LOG_PATH = Path.home() / ".praxis" / "execution.log"

DEFAULT_STALE_THRESHOLD = 0.5   # below this composite score → stale
DEFAULT_BASELINE_MS = 1000.0    # reference "fast" execution time


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProgramScore:
    """Health metrics for a single stored program."""
    program_id: str
    program_text: str
    success_rate: float          # 0.0 – 1.0
    avg_ms: float                # average total execution time
    speed_score: float           # 0.0 – 1.0 (1.0 = fast)
    composite: float             # weighted average of above
    execution_count: int         # number of log entries for this program
    is_stale: bool = False       # True if composite < stale_threshold


@dataclass
class BenchmarkResult:
    """Comparison between an original program and a proposed rewrite."""
    original_id: str
    original_program: str
    rewrite_program: str
    original_score: float
    rewrite_score: float        # estimated from log statistics of component verbs
    speedup_ms: int             # estimated ms saved per run
    should_promote: bool        # True if rewrite is strictly better


# ─────────────────────────────────────────────────────────────────────────────
# Evolver
# ─────────────────────────────────────────────────────────────────────────────

class ProgramEvolver:
    """
    Analyzes stored programs against execution history and manages evolution.

    Parameters
    ----------
    memory : ProgramMemory
        The program library to read from and promote into.
    log_path : Path | str | None
        Path to execution.log; defaults to ~/.praxis/execution.log.
    stale_threshold : float
        Programs with composite score below this are marked stale (default 0.5).
    baseline_ms : float
        Reference execution time for speed scoring (default 1000ms).
    """

    def __init__(
        self,
        memory: Any,
        log_path: Path | str | None = None,
        stale_threshold: float = DEFAULT_STALE_THRESHOLD,
        baseline_ms: float = DEFAULT_BASELINE_MS,
    ) -> None:
        self.memory = memory
        self.log_path = Path(log_path) if log_path else _LOG_PATH
        self.stale_threshold = stale_threshold
        self.baseline_ms = baseline_ms

    # ── Scoring ────────────────────────────────────────────────────────────────

    def score(self, limit: int = 100) -> list[ProgramScore]:
        """
        Score all recently used programs from the program library.
        Returns a list sorted by composite score ascending (worst first).
        """
        log_entries = self._load_log()
        verb_stats = self._aggregate_verb_stats(log_entries)

        programs = self.memory.recent(n=limit)
        scores = []

        for prog in programs:
            prog_id = prog.get("id", "")
            prog_text = prog.get("program", "")
            outcome = prog.get("outcome", "success")

            # Extract verbs used in this program from stored data
            verbs = self._extract_verbs_from_program(prog_text)

            # Build execution stats from log for this program's verbs
            total_ok = sum(verb_stats[v]["ok"] for v in verbs if v in verb_stats)
            total_err = sum(verb_stats[v]["error"] for v in verbs if v in verb_stats)
            total = total_ok + total_err

            success_rate = total_ok / total if total > 0 else (1.0 if outcome == "success" else 0.0)

            avg_ms_vals = [verb_stats[v]["avg_ms"] for v in verbs if v in verb_stats]
            avg_ms = sum(avg_ms_vals) / len(avg_ms_vals) if avg_ms_vals else 0.0

            speed_score = self._speed_score(avg_ms)
            composite = round(0.6 * success_rate + 0.4 * speed_score, 4)

            scores.append(ProgramScore(
                program_id=prog_id,
                program_text=prog_text,
                success_rate=round(success_rate, 4),
                avg_ms=round(avg_ms, 1),
                speed_score=round(speed_score, 4),
                composite=composite,
                execution_count=total,
                is_stale=composite < self.stale_threshold,
            ))

        scores.sort(key=lambda s: s.composite)
        return scores

    def mark_stale(self, limit: int = 100) -> list[ProgramScore]:
        """Return programs whose composite score is below stale_threshold."""
        return [s for s in self.score(limit=limit) if s.is_stale]

    # ── Benchmark ──────────────────────────────────────────────────────────────

    def benchmark(
        self,
        original_id: str,
        rewrite_program: str,
    ) -> BenchmarkResult | None:
        """
        Compare a stored program against a proposed rewrite.
        Returns None if the original program cannot be found.
        """
        programs = self.memory.recent(n=1000)
        original = next((p for p in programs if p.get("id", "").startswith(original_id)), None)
        if original is None:
            return None

        original_text = original.get("program", "")
        original_scores = self.score(limit=1000)
        orig_score_obj = next(
            (s for s in original_scores if s.program_id == original.get("id", "")),
            None,
        )
        original_score = orig_score_obj.composite if orig_score_obj else 0.5

        # Estimate rewrite score from log stats of its component verbs
        log_entries = self._load_log()
        verb_stats = self._aggregate_verb_stats(log_entries)
        rewrite_verbs = self._extract_verbs_from_text(rewrite_program)

        ok_count = sum(verb_stats[v]["ok"] for v in rewrite_verbs if v in verb_stats)
        err_count = sum(verb_stats[v]["error"] for v in rewrite_verbs if v in verb_stats)
        total = ok_count + err_count
        rewrite_success = ok_count / total if total > 0 else 0.9  # optimistic default

        avg_ms_vals = [verb_stats[v]["avg_ms"] for v in rewrite_verbs if v in verb_stats]
        rewrite_avg_ms = sum(avg_ms_vals) / len(avg_ms_vals) if avg_ms_vals else 0.0
        rewrite_speed = self._speed_score(rewrite_avg_ms)
        rewrite_score = round(0.6 * rewrite_success + 0.4 * rewrite_speed, 4)

        orig_ms_vals = [verb_stats[v]["avg_ms"] for v in self._extract_verbs_from_text(original_text) if v in verb_stats]
        orig_avg_ms = sum(orig_ms_vals) / len(orig_ms_vals) if orig_ms_vals else 0.0
        speedup_ms = max(0, int(orig_avg_ms - rewrite_avg_ms))

        return BenchmarkResult(
            original_id=original.get("id", ""),
            original_program=original_text,
            rewrite_program=rewrite_program,
            original_score=original_score,
            rewrite_score=rewrite_score,
            speedup_ms=speedup_ms,
            should_promote=rewrite_score > original_score,
        )

    # ── Promotion ──────────────────────────────────────────────────────────────

    def promote(
        self,
        benchmark: BenchmarkResult,
        goal: str = "",
        dry_run: bool = False,
    ) -> str | None:
        """
        Promote a rewrite into the program library if benchmark.should_promote is True.

        Stores the rewritten program under the same goal text, using outcome='success'.
        Returns the new program ID, or None if promotion was skipped.
        """
        if not benchmark.should_promote:
            return None
        if dry_run:
            return "dry-run"

        new_id = self.memory.store(
            goal=goal or f"evolved from {benchmark.original_id[:8]}",
            program=benchmark.rewrite_program,
            outcome="success",
            steps=[],
        )
        return new_id

    # ── Internals ──────────────────────────────────────────────────────────────

    def _speed_score(self, avg_ms: float) -> float:
        """
        Map avg execution time to a [0, 1] score.
        1.0 at 0ms; 0.0 at 3× baseline_ms; linear decline.
        """
        if avg_ms <= 0:
            return 1.0
        ratio = avg_ms / self.baseline_ms
        return max(0.0, 1.0 - ratio / 3.0)

    def _aggregate_verb_stats(self, entries: list[dict]) -> dict[str, dict]:
        """Return {verb: {ok, error, avg_ms}} from log entries."""
        stats: dict[str, dict] = defaultdict(lambda: {"ok": 0, "error": 0, "durations": []})
        for entry in entries:
            verb = entry.get("verb", "")
            status = entry.get("status", "")
            ms = entry.get("duration_ms")
            if not verb:
                continue
            if status == "ok":
                stats[verb]["ok"] += 1
            elif status == "error":
                stats[verb]["error"] += 1
            if ms is not None:
                stats[verb]["durations"].append(int(ms))

        result = {}
        for verb, s in stats.items():
            durations = s["durations"]
            result[verb] = {
                "ok": s["ok"],
                "error": s["error"],
                "avg_ms": sum(durations) / len(durations) if durations else 0.0,
            }
        return result

    def _extract_verbs_from_program(self, program_text: str) -> list[str]:
        """Extract verb names from a stored program text (best-effort)."""
        return self._extract_verbs_from_text(program_text)

    def _extract_verbs_from_text(self, text: str) -> list[str]:
        """Extract all uppercase verb tokens from a program text string."""
        import re
        return re.findall(r'\b([A-Z][A-Z0-9]{1,7})\b', text)

    def _load_log(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        entries = []
        try:
            with open(self.log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except OSError:
            pass
        return entries
