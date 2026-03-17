"""
Sprint 16 tests — Outcome-driven program evolution (Pillar 6-PartB).
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from praxis.evolver import (
    ProgramEvolver, ProgramScore, BenchmarkResult,
    DEFAULT_STALE_THRESHOLD, DEFAULT_BASELINE_MS,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_memory(programs: list[dict]) -> MagicMock:
    mem = MagicMock()
    mem.recent.return_value = programs
    mem.store.return_value = "new-prog-id-abc123"
    return mem


def _write_log(log_path: Path, entries: list[dict]) -> None:
    with open(log_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _evolver(programs, log_entries=None, **kwargs):
    tmp = Path(tempfile.mkdtemp())
    log = tmp / "execution.log"
    if log_entries:
        _write_log(log, log_entries)
    mem = _make_memory(programs)
    return ProgramEvolver(memory=mem, log_path=log, **kwargs), mem


# ══════════════════════════════════════════════════════════════════════════════
# ProgramScore dataclass
# ══════════════════════════════════════════════════════════════════════════════

class TestProgramScore:
    def test_is_stale_false_by_default(self):
        s = ProgramScore("id", "LOG.x", 1.0, 0.0, 1.0, 1.0, 5)
        assert s.is_stale is False

    def test_fields_present(self):
        s = ProgramScore("id", "LOG.x", 0.8, 500.0, 0.5, 0.65, 10, is_stale=True)
        assert s.success_rate == 0.8
        assert s.avg_ms == 500.0
        assert s.speed_score == 0.5
        assert s.composite == 0.65
        assert s.is_stale is True


# ══════════════════════════════════════════════════════════════════════════════
# BenchmarkResult dataclass
# ══════════════════════════════════════════════════════════════════════════════

class TestBenchmarkResult:
    def test_should_promote_when_rewrite_better(self):
        b = BenchmarkResult("id", "LOG.x", "PAR(LOG.x, ANNOTATE.y)",
                            0.4, 0.7, 200, should_promote=True)
        assert b.should_promote is True

    def test_should_not_promote_when_worse(self):
        b = BenchmarkResult("id", "LOG.x", "LOG.x",
                            0.8, 0.5, 0, should_promote=False)
        assert b.should_promote is False


# ══════════════════════════════════════════════════════════════════════════════
# _speed_score
# ══════════════════════════════════════════════════════════════════════════════

class TestSpeedScore:
    def _ev(self):
        return ProgramEvolver(memory=_make_memory([]), log_path="/nonexistent")

    def test_zero_ms_returns_one(self):
        assert self._ev()._speed_score(0.0) == 1.0

    def test_at_baseline_returns_2_thirds(self):
        ev = self._ev()
        # At 1x baseline, ratio=1, score = 1 - 1/3 = 0.6667
        assert abs(ev._speed_score(DEFAULT_BASELINE_MS) - 2/3) < 0.001

    def test_at_3x_baseline_returns_zero(self):
        ev = self._ev()
        assert ev._speed_score(DEFAULT_BASELINE_MS * 3) == 0.0

    def test_above_3x_clamped_to_zero(self):
        ev = self._ev()
        assert ev._speed_score(DEFAULT_BASELINE_MS * 10) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# _extract_verbs_from_text
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractVerbs:
    def _ev(self):
        return ProgramEvolver(memory=_make_memory([]), log_path="/nonexistent")

    def test_extracts_verb(self):
        assert "LOG" in self._ev()._extract_verbs_from_text("LOG.x")

    def test_extracts_multiple_verbs(self):
        result = self._ev()._extract_verbs_from_text("ING.data -> CLN.normalize -> SUMM.text")
        assert "ING" in result
        assert "CLN" in result
        assert "SUMM" in result

    def test_lowercase_not_extracted(self):
        result = self._ev()._extract_verbs_from_text("log.test")
        assert "log" not in result

    def test_par_block_verbs_extracted(self):
        result = self._ev()._extract_verbs_from_text("PAR(LOG.a, ANNOTATE.b)")
        assert "LOG" in result
        assert "ANNOTATE" in result


# ══════════════════════════════════════════════════════════════════════════════
# score()
# ══════════════════════════════════════════════════════════════════════════════

class TestScore:
    def test_empty_memory_returns_empty(self):
        ev, _ = _evolver([])
        assert ev.score() == []

    def test_returns_program_scores(self):
        programs = [{"id": "abc", "program": "LOG.x", "outcome": "success"}]
        ev, _ = _evolver(programs)
        scores = ev.score()
        assert len(scores) == 1
        assert isinstance(scores[0], ProgramScore)

    def test_successful_program_no_log_gets_high_success_rate(self):
        programs = [{"id": "abc", "program": "LOG.x", "outcome": "success"}]
        ev, _ = _evolver(programs)
        s = ev.score()[0]
        assert s.success_rate == 1.0

    def test_failed_program_no_log_gets_low_success_rate(self):
        programs = [{"id": "abc", "program": "LOG.x", "outcome": "error"}]
        ev, _ = _evolver(programs)
        s = ev.score()[0]
        assert s.success_rate == 0.0

    def test_log_entries_affect_success_rate(self):
        programs = [{"id": "abc", "program": "LOG.x", "outcome": "success"}]
        log = [
            {"verb": "LOG", "status": "ok", "duration_ms": 100},
            {"verb": "LOG", "status": "ok", "duration_ms": 100},
            {"verb": "LOG", "status": "error", "duration_ms": 50},
            {"verb": "LOG", "status": "ok", "duration_ms": 100},
        ]
        ev, _ = _evolver(programs, log_entries=log)
        s = ev.score()[0]
        assert abs(s.success_rate - 0.75) < 0.01

    def test_composite_between_0_and_1(self):
        programs = [{"id": "abc", "program": "LOG.x", "outcome": "success"}]
        ev, _ = _evolver(programs)
        s = ev.score()[0]
        assert 0.0 <= s.composite <= 1.0

    def test_sorted_worst_first(self):
        programs = [
            {"id": "a", "program": "LOG.x", "outcome": "success"},
            {"id": "b", "program": "LOG.x", "outcome": "error"},
        ]
        ev, _ = _evolver(programs)
        scores = ev.score()
        assert scores[0].composite <= scores[-1].composite

    def test_stale_flag_set_when_below_threshold(self):
        programs = [{"id": "abc", "program": "LOG.x", "outcome": "error"}]
        log = [
            {"verb": "LOG", "status": "error", "duration_ms": 5000},
            {"verb": "LOG", "status": "error", "duration_ms": 5000},
        ]
        ev, _ = _evolver(programs, log_entries=log, stale_threshold=0.9)
        s = ev.score()[0]
        assert s.is_stale is True


# ══════════════════════════════════════════════════════════════════════════════
# mark_stale()
# ══════════════════════════════════════════════════════════════════════════════

class TestMarkStale:
    def test_no_stale_programs_returns_empty(self):
        programs = [{"id": "abc", "program": "LOG.x", "outcome": "success"}]
        ev, _ = _evolver(programs)
        assert ev.mark_stale() == []

    def test_stale_programs_returned(self):
        programs = [{"id": "abc", "program": "LOG.x", "outcome": "error"}]
        log = [{"verb": "LOG", "status": "error", "duration_ms": 5000}] * 3
        ev, _ = _evolver(programs, log_entries=log, stale_threshold=0.99)
        stale = ev.mark_stale()
        assert len(stale) == 1
        assert stale[0].is_stale is True


# ══════════════════════════════════════════════════════════════════════════════
# benchmark()
# ══════════════════════════════════════════════════════════════════════════════

class TestBenchmark:
    def test_returns_none_for_unknown_id(self):
        programs = [{"id": "abc123", "program": "LOG.x", "outcome": "success"}]
        ev, _ = _evolver(programs)
        result = ev.benchmark("nonexistent", "LOG.x -> ANNOTATE.y")
        assert result is None

    def test_returns_benchmark_result(self):
        programs = [{"id": "abc123", "program": "LOG.x", "outcome": "success"}]
        ev, _ = _evolver(programs)
        result = ev.benchmark("abc123", "LOG.x -> ANNOTATE.y")
        assert isinstance(result, BenchmarkResult)

    def test_benchmark_has_original_id(self):
        programs = [{"id": "abc123", "program": "LOG.x", "outcome": "success"}]
        ev, _ = _evolver(programs)
        result = ev.benchmark("abc123", "PAR(LOG.x, ANNOTATE.y)")
        assert result.original_id == "abc123"

    def test_benchmark_prefix_match(self):
        """benchmark() accepts ID prefixes."""
        programs = [{"id": "abc123def", "program": "LOG.x", "outcome": "success"}]
        ev, _ = _evolver(programs)
        result = ev.benchmark("abc123", "LOG.x")
        assert result is not None


# ══════════════════════════════════════════════════════════════════════════════
# promote()
# ══════════════════════════════════════════════════════════════════════════════

class TestPromote:
    def test_promote_calls_memory_store(self):
        programs = [{"id": "abc", "program": "LOG.x", "outcome": "success"}]
        ev, mem = _evolver(programs)
        bench = BenchmarkResult("abc", "LOG.x", "PAR(LOG.x, OUT.y)",
                                0.4, 0.8, 300, should_promote=True)
        result = ev.promote(bench)
        mem.store.assert_called_once()
        assert result == "new-prog-id-abc123"

    def test_promote_skips_when_should_promote_false(self):
        programs = [{"id": "abc", "program": "LOG.x", "outcome": "success"}]
        ev, mem = _evolver(programs)
        bench = BenchmarkResult("abc", "LOG.x", "LOG.x",
                                0.8, 0.3, 0, should_promote=False)
        result = ev.promote(bench)
        assert result is None
        mem.store.assert_not_called()

    def test_promote_dry_run_does_not_call_store(self):
        programs = [{"id": "abc", "program": "LOG.x", "outcome": "success"}]
        ev, mem = _evolver(programs)
        bench = BenchmarkResult("abc", "LOG.x", "PAR(LOG.x, OUT.y)",
                                0.4, 0.8, 300, should_promote=True)
        result = ev.promote(bench, dry_run=True)
        assert result == "dry-run"
        mem.store.assert_not_called()

    def test_promote_uses_goal_text(self):
        programs = [{"id": "abc", "program": "LOG.x", "outcome": "success"}]
        ev, mem = _evolver(programs)
        bench = BenchmarkResult("abc", "LOG.x", "PAR(LOG.x, OUT.y)",
                                0.4, 0.8, 300, should_promote=True)
        ev.promote(bench, goal="my test goal")
        call_kwargs = mem.store.call_args
        assert "my test goal" in str(call_kwargs)
