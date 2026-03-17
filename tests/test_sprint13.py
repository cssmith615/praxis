"""
Sprint 13 tests — Performance-driven program rewriting (Pillar 6-PartA).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from praxis.rewriter import Rewriter, SlowPattern, RewriteProposal


# ─── helpers ──────────────────────────────────────────────────────────────────

def _write_log(log_path: Path, entries: list[dict]) -> None:
    with open(log_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _slow_entry(verb: str, ms: int, program: str = "") -> dict:
    return {"verb": verb, "status": "ok", "duration_ms": ms, "program": program}


def _make_rw(log_entries: list[dict], slow_ms: int = 500) -> tuple[Rewriter, Path]:
    tmp = Path(tempfile.mkdtemp())
    log = tmp / "execution.log"
    _write_log(log, log_entries)
    return Rewriter(log_path=log, slow_threshold_ms=slow_ms), log


# ══════════════════════════════════════════════════════════════════════════════
# SlowPattern dataclass
# ══════════════════════════════════════════════════════════════════════════════

class TestSlowPattern:
    def test_fields_present(self):
        sp = SlowPattern(verb="ING", count=3, avg_ms=800.0, max_ms=1200)
        assert sp.verb == "ING"
        assert sp.count == 3
        assert sp.avg_ms == 800.0
        assert sp.max_ms == 1200
        assert sp.sample_programs == []


# ══════════════════════════════════════════════════════════════════════════════
# RewriteProposal dataclass
# ══════════════════════════════════════════════════════════════════════════════

class TestRewriteProposal:
    def test_fields_present(self):
        p = RewriteProposal(
            original_program="A.x -> B.y",
            proposed_program="PAR(A.x, B.y)",
            slow_verbs=["A", "B"],
            estimated_speedup_ms=400,
        )
        assert p.source == "performance"
        assert p.estimated_speedup_ms == 400


# ══════════════════════════════════════════════════════════════════════════════
# analyze_slow
# ══════════════════════════════════════════════════════════════════════════════

class TestAnalyzeSlow:
    def test_empty_log_returns_empty(self):
        rw, _ = _make_rw([])
        assert rw.analyze_slow() == []

    def test_fast_steps_not_returned(self):
        rw, _ = _make_rw([_slow_entry("LOG", 100), _slow_entry("LOG", 100)])
        assert rw.analyze_slow() == []

    def test_slow_verb_below_min_count_not_returned(self):
        """A single slow occurrence is not enough — need MIN_SLOW_COUNT."""
        rw, _ = _make_rw([_slow_entry("ING", 1000)])
        assert rw.analyze_slow() == []

    def test_slow_verb_above_min_count_returned(self):
        rw, _ = _make_rw([_slow_entry("ING", 1000), _slow_entry("ING", 800)])
        patterns = rw.analyze_slow()
        assert len(patterns) == 1
        assert patterns[0].verb == "ING"

    def test_correct_count(self):
        rw, _ = _make_rw([_slow_entry("ING", 900)] * 5)
        assert rw.analyze_slow()[0].count == 5

    def test_correct_avg_ms(self):
        rw, _ = _make_rw([_slow_entry("ING", 600), _slow_entry("ING", 1000)])
        assert rw.analyze_slow()[0].avg_ms == 800.0

    def test_correct_max_ms(self):
        rw, _ = _make_rw([_slow_entry("ING", 600), _slow_entry("ING", 1400)])
        assert rw.analyze_slow()[0].max_ms == 1400

    def test_error_entries_excluded(self):
        entries = [
            {"verb": "ING", "status": "error", "duration_ms": 1000},
            {"verb": "ING", "status": "error", "duration_ms": 1000},
        ]
        rw, _ = _make_rw(entries)
        assert rw.analyze_slow() == []

    def test_sample_programs_collected(self):
        entries = [
            _slow_entry("ING", 1000, program="ING.data"),
            _slow_entry("ING", 1000, program="ING.other"),
        ]
        rw, _ = _make_rw(entries)
        patterns = rw.analyze_slow()
        assert len(patterns[0].sample_programs) == 2

    def test_sample_programs_capped_at_3(self):
        entries = [_slow_entry("ING", 1000, program=f"ING.p{i}") for i in range(6)]
        rw, _ = _make_rw(entries)
        assert len(rw.analyze_slow()[0].sample_programs) == 3

    def test_sorted_by_avg_ms_descending(self):
        entries = (
            [_slow_entry("FAST", 600)] * 2 +
            [_slow_entry("SLOW", 2000)] * 2
        )
        rw, _ = _make_rw(entries)
        patterns = rw.analyze_slow()
        assert patterns[0].verb == "SLOW"

    def test_missing_log_returns_empty(self):
        rw = Rewriter(log_path="/nonexistent/path/log.json")
        assert rw.analyze_slow() == []


# ══════════════════════════════════════════════════════════════════════════════
# propose_par
# ══════════════════════════════════════════════════════════════════════════════

class TestProposePar:
    def test_empty_patterns_returns_empty(self):
        rw, _ = _make_rw([])
        assert rw.propose_par([]) == []

    def test_single_slow_verb_no_proposal(self):
        """One slow verb in a chain with no adjacent slow verb → no PAR proposal."""
        pattern = SlowPattern(
            verb="ING", count=2, avg_ms=800.0, max_ms=1000,
            sample_programs=["ING.data -> CLN.normalize"]
        )
        rw, _ = _make_rw([])
        proposals = rw.propose_par([pattern])
        assert proposals == []

    def test_two_adjacent_slow_verbs_produces_proposal(self):
        entries = (
            [_slow_entry("ING", 1000, "ING.data -> XFRM.normalize")] * 2 +
            [_slow_entry("XFRM", 900, "ING.data -> XFRM.normalize")] * 2
        )
        rw, _ = _make_rw(entries)
        patterns = rw.analyze_slow()
        proposals = rw.propose_par(patterns)
        assert len(proposals) >= 1

    def test_proposal_contains_par(self):
        entries = (
            [_slow_entry("ING", 1000, "ING.data -> XFRM.normalize")] * 2 +
            [_slow_entry("XFRM", 900, "ING.data -> XFRM.normalize")] * 2
        )
        rw, _ = _make_rw(entries)
        patterns = rw.analyze_slow()
        proposals = rw.propose_par(patterns)
        assert any("PAR" in p.proposed_program for p in proposals)

    def test_proposal_speedup_is_positive(self):
        entries = (
            [_slow_entry("ING", 1000, "ING.data -> XFRM.normalize")] * 2 +
            [_slow_entry("XFRM", 900, "ING.data -> XFRM.normalize")] * 2
        )
        rw, _ = _make_rw(entries)
        patterns = rw.analyze_slow()
        proposals = rw.propose_par(patterns)
        if proposals:
            assert proposals[0].estimated_speedup_ms > 0

    def test_duplicate_programs_not_proposed_twice(self):
        """Same program text appearing in two patterns only produces one proposal."""
        prog = "ING.data -> XFRM.normalize"
        pattern1 = SlowPattern(verb="ING", count=2, avg_ms=1000.0, max_ms=1000,
                               sample_programs=[prog])
        pattern2 = SlowPattern(verb="XFRM", count=2, avg_ms=900.0, max_ms=900,
                               sample_programs=[prog])
        rw, _ = _make_rw([])
        proposals = rw.propose_par([pattern1, pattern2])
        # At most one proposal for the same original program
        originals = [p.original_program for p in proposals]
        assert len(originals) == len(set(originals))


# ══════════════════════════════════════════════════════════════════════════════
# apply()
# ══════════════════════════════════════════════════════════════════════════════

class TestApply:
    def test_no_slow_verbs_returns_original(self):
        rw, _ = _make_rw([])
        result = rw.apply("LOG.a -> ANNOTATE.b", slow_verbs=set())
        assert "LOG" in result
        assert "ANNOTATE" in result

    def test_explicit_slow_verbs_produces_par(self):
        rw, _ = _make_rw([])
        result = rw.apply("LOG.a -> ANNOTATE.b", slow_verbs={"LOG", "ANNOTATE"})
        assert "PAR" in result

    def test_single_slow_verb_no_par(self):
        rw, _ = _make_rw([])
        result = rw.apply("LOG.a -> ANNOTATE.b", slow_verbs={"LOG"})
        # Only one of the two is slow — can't PAR a group of 1
        assert "PAR" not in result

    def test_unparseable_program_returns_original(self):
        rw, _ = _make_rw([])
        bad = "NOT VALID !!!"
        assert rw.apply(bad, slow_verbs={"LOG"}) == bad

    def test_three_slow_verbs_all_in_par(self):
        rw, _ = _make_rw([])
        result = rw.apply("LOG.a -> ANNOTATE.b -> OUT.c",
                          slow_verbs={"LOG", "ANNOTATE", "OUT"})
        assert "PAR" in result

    def test_order_sensitive_verb_breaks_par(self):
        """SET is order-sensitive — the chain cannot be PAR'd across it."""
        rw, _ = _make_rw([])
        result = rw.apply("LOG.a -> SET.myvar -> ANNOTATE.b",
                          slow_verbs={"LOG", "SET", "ANNOTATE"})
        # SET breaks the group — LOG and ANNOTATE are in different groups of 1
        assert "PAR" not in result


# ══════════════════════════════════════════════════════════════════════════════
# End-to-end: log → analyze → propose → apply
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_full_pipeline(self):
        prog_text = "ING.data -> XFRM.normalize -> SUMM.report"
        entries = (
            [_slow_entry("ING", 1200, prog_text)] * 3 +
            [_slow_entry("XFRM", 1100, prog_text)] * 3 +
            [_slow_entry("SUMM", 900, prog_text)] * 3
        )
        rw, _ = _make_rw(entries, slow_ms=500)
        patterns = rw.analyze_slow()
        assert len(patterns) == 3
        proposals = rw.propose_par(patterns)
        assert len(proposals) >= 1
        assert "PAR" in proposals[0].proposed_program

    def test_rewriter_with_no_log_produces_no_patterns(self):
        rw = Rewriter(log_path="/nonexistent/path/exec.log")
        patterns = rw.analyze_slow()
        assert patterns == []
        proposals = rw.propose_par(patterns)
        assert proposals == []
