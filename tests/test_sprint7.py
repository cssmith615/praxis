"""
Sprint 7 tests — self-improvement loop: Improver, failure analysis,
rule proposal, eval, accept.

All tests are in-process; no Anthropic API calls, no real log files.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from praxis.constitution import Constitution
from praxis.improver import (
    Improver,
    FailurePattern,
    RuleProposal,
    _load_log,
    _extract_verbs,
    _most_common_snippet,
    _similarity,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _write_log(path: Path, entries: list[dict]) -> None:
    """Write JSONL execution log entries to a file."""
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _constitution(tmp_path: Path) -> Constitution:
    """Create a fresh constitution backed by a temp file."""
    p = tmp_path / "constitution.md"
    p.write_text("# Test constitution\n")
    return Constitution(p)


# ══════════════════════════════════════════════════════════════════════════════
# Log loading
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadLog:
    def test_reads_jsonl_entries(self, tmp_path):
        log = tmp_path / "exec.log"
        _write_log(log, [
            {"verb": "ING", "status": "ok"},
            {"verb": "CLN", "status": "error", "error": "null value"},
        ])
        entries = _load_log(log)
        assert len(entries) == 2
        assert entries[1]["verb"] == "CLN"

    def test_missing_log_returns_empty(self, tmp_path):
        entries = _load_log(tmp_path / "nonexistent.log")
        assert entries == []

    def test_skips_malformed_lines(self, tmp_path):
        log = tmp_path / "exec.log"
        log.write_text('{"verb":"ING","status":"ok"}\nNOT JSON\n{"verb":"CLN","status":"ok"}\n')
        entries = _load_log(log)
        assert len(entries) == 2


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestHelpers:
    def test_extract_verbs_from_program(self):
        verbs = _extract_verbs("ING.sales -> CLN.null -> TRN.model")
        assert "ING" in verbs
        assert "CLN" in verbs
        assert "TRN" in verbs

    def test_extract_verbs_ignores_non_verbs(self):
        verbs = _extract_verbs("NOTAVERB.x -> ING.y -> lowercase")
        assert "NOTAVERB" not in verbs
        assert "ING" in verbs

    def test_most_common_snippet_picks_most_frequent(self):
        msgs = ["timeout error", "timeout error", "null value"]
        result = _most_common_snippet(msgs)
        assert result == "timeout error"

    def test_most_common_snippet_empty(self):
        assert _most_common_snippet([]) == ""

    def test_similarity_identical(self):
        assert _similarity("always use CLN", "always use CLN") == 1.0

    def test_similarity_different(self):
        assert _similarity("always use CLN", "never skip GATE") < 0.5

    def test_similarity_partial_overlap(self):
        s = _similarity("always use CLN before TRN", "always include CLN")
        assert 0.0 < s < 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Analyze
# ══════════════════════════════════════════════════════════════════════════════

class TestAnalyze:
    def test_finds_failure_patterns(self, tmp_path):
        log = tmp_path / "exec.log"
        _write_log(log, [
            {"verb": "TRN", "status": "error", "error": "input not clean", "program": "ING.x -> TRN.model"},
            {"verb": "TRN", "status": "error", "error": "input not clean", "program": "ING.y -> TRN.model"},
            {"verb": "TRN", "status": "error", "error": "null tensor",     "program": "ING.z -> TRN.model"},
            {"verb": "ING", "status": "ok",    "error": "",                "program": "ING.sales"},
        ])
        imp = Improver(constitution=_constitution(tmp_path), log_path=log)
        patterns = imp.analyze()
        assert len(patterns) == 1
        assert patterns[0].verb == "TRN"
        assert patterns[0].count == 3

    def test_ignores_below_threshold(self, tmp_path):
        log = tmp_path / "exec.log"
        _write_log(log, [
            {"verb": "OUT", "status": "error", "error": "timeout", "program": "OUT.telegram"},
        ])
        imp = Improver(constitution=_constitution(tmp_path), log_path=log)
        patterns = imp.analyze()
        assert len(patterns) == 0   # only 1 failure, below _MIN_FAILURES=2

    def test_sorts_by_count_descending(self, tmp_path):
        log = tmp_path / "exec.log"
        entries = (
            [{"verb": "OUT", "status": "error", "error": "x", "program": "OUT.x"}] * 2 +
            [{"verb": "TRN", "status": "error", "error": "y", "program": "TRN.y"}] * 5
        )
        _write_log(log, entries)
        imp = Improver(constitution=_constitution(tmp_path), log_path=log)
        patterns = imp.analyze()
        assert patterns[0].verb == "TRN"
        assert patterns[1].verb == "OUT"

    def test_collects_co_occurring_verbs(self, tmp_path):
        log = tmp_path / "exec.log"
        _write_log(log, [
            {"verb": "TRN", "status": "error", "error": "e", "program": "ING.x -> CLN.null -> TRN.model"},
            {"verb": "TRN", "status": "error", "error": "e", "program": "ING.y -> CLN.null -> TRN.model"},
        ])
        imp = Improver(constitution=_constitution(tmp_path), log_path=log)
        patterns = imp.analyze()
        assert "ING" in patterns[0].co_occurring_verbs
        assert "CLN" in patterns[0].co_occurring_verbs

    def test_empty_log_returns_no_patterns(self, tmp_path):
        log = tmp_path / "exec.log"
        log.write_text("")
        imp = Improver(constitution=_constitution(tmp_path), log_path=log)
        assert imp.analyze() == []

    def test_missing_log_returns_no_patterns(self, tmp_path):
        imp = Improver(
            constitution=_constitution(tmp_path),
            log_path=tmp_path / "nonexistent.log",
        )
        assert imp.analyze() == []


# ══════════════════════════════════════════════════════════════════════════════
# Propose (heuristic)
# ══════════════════════════════════════════════════════════════════════════════

class TestProposeHeuristic:
    def _pattern(self, verb: str, count: int = 3, co: list[str] | None = None) -> FailurePattern:
        return FailurePattern(
            verb=verb,
            count=count,
            error_summary="test error",
            sample_programs=[f"ING.x -> {verb}.model"],
            co_occurring_verbs=co or [],
        )

    def test_trn_without_cln_proposes_cln_rule(self, tmp_path):
        imp = Improver(constitution=_constitution(tmp_path))
        imp._log_entries = []
        proposal = imp._propose_heuristic(self._pattern("TRN"))
        assert proposal is not None
        assert "CLN" in proposal.rule_text
        assert "CLN" in proposal.verbs

    def test_write_proposes_gate_rule(self, tmp_path):
        imp = Improver(constitution=_constitution(tmp_path))
        imp._log_entries = []
        proposal = imp._propose_heuristic(self._pattern("WRITE"))
        assert proposal is not None
        assert "GATE" in proposal.rule_text

    def test_out_proposes_log_rule(self, tmp_path):
        imp = Improver(constitution=_constitution(tmp_path))
        imp._log_entries = []
        proposal = imp._propose_heuristic(self._pattern("OUT"))
        assert proposal is not None
        assert "LOG" in proposal.rule_text

    def test_ing_proposes_validate_rule(self, tmp_path):
        imp = Improver(constitution=_constitution(tmp_path))
        imp._log_entries = []
        proposal = imp._propose_heuristic(self._pattern("ING"))
        assert proposal is not None
        assert "VALIDATE" in proposal.rule_text or "CLN" in proposal.rule_text

    def test_propose_returns_list_from_patterns(self, tmp_path):
        imp = Improver(constitution=_constitution(tmp_path))
        imp._log_entries = []
        patterns = [
            self._pattern("TRN"),
            self._pattern("WRITE"),
        ]
        proposals = imp.propose(patterns)
        assert len(proposals) == 2
        assert all(isinstance(p, RuleProposal) for p in proposals)

    def test_proposal_has_source_heuristic(self, tmp_path):
        imp = Improver(constitution=_constitution(tmp_path))
        imp._log_entries = []
        proposal = imp._propose_heuristic(self._pattern("TRN"))
        assert proposal.source == "heuristic"


# ══════════════════════════════════════════════════════════════════════════════
# Eval
# ══════════════════════════════════════════════════════════════════════════════

class TestEvalRule:
    def test_counts_affected_programs(self, tmp_path):
        log = tmp_path / "exec.log"
        _write_log(log, [
            {"verb": "TRN", "status": "ok",    "program": "ING.x -> TRN.model"},
            {"verb": "TRN", "status": "error", "program": "ING.y -> TRN.model"},
            {"verb": "OUT", "status": "ok",    "program": "OUT.telegram"},
        ])
        imp = Improver(constitution=_constitution(tmp_path), log_path=log)
        imp.analyze()   # loads _log_entries

        pattern = FailurePattern("TRN", 1, "", [], [])
        proposal = RuleProposal("ALWAYS CLN before TRN", ["TRN", "CLN"],
                                "heuristic", pattern, 0, 0)
        affected, prevented = imp.eval_rule(proposal)
        assert affected == 2    # both TRN entries
        assert prevented >= 0

    def test_no_affected_when_verbs_absent(self, tmp_path):
        log = tmp_path / "exec.log"
        _write_log(log, [
            {"verb": "ING", "status": "ok", "program": "ING.sales"},
        ])
        imp = Improver(constitution=_constitution(tmp_path), log_path=log)
        imp.analyze()

        pattern = FailurePattern("OUT", 1, "", [], [])
        proposal = RuleProposal("ALWAYS LOG before OUT", ["OUT", "LOG"],
                                "heuristic", pattern, 0, 0)
        affected, prevented = imp.eval_rule(proposal)
        assert affected == 0
        assert prevented == 0


# ══════════════════════════════════════════════════════════════════════════════
# Accept (write to constitution)
# ══════════════════════════════════════════════════════════════════════════════

class TestAccept:
    def _proposal(self, rule_text: str, verbs: list[str], const: Constitution) -> RuleProposal:
        pattern = FailurePattern("TRN", 3, "test", [], [])
        return RuleProposal(rule_text, verbs, "heuristic", pattern, 5, 2)

    def test_accept_appends_rule(self, tmp_path):
        const = _constitution(tmp_path)
        imp = Improver(constitution=const)
        prop = self._proposal("ALWAYS CLN before TRN.", ["CLN", "TRN"], const)
        result = imp.accept(prop)
        assert result is True
        assert "ALWAYS CLN before TRN." in const.path.read_text()

    def test_accept_deduplicates(self, tmp_path):
        const = _constitution(tmp_path)
        imp = Improver(constitution=const)
        prop = self._proposal("ALWAYS CLN before TRN.", ["CLN", "TRN"], const)
        imp.accept(prop)
        result = imp.accept(prop)   # second time
        assert result is False

    def test_accepted_rule_is_retrievable(self, tmp_path):
        const = _constitution(tmp_path)
        imp = Improver(constitution=const)
        prop = self._proposal("ALWAYS VALIDATE after ING.", ["VALIDATE", "ING"], const)
        imp.accept(prop)
        rules = const.get_rules_for_verbs({"ING"})
        assert any("VALIDATE" in r for r in rules)


# ══════════════════════════════════════════════════════════════════════════════
# End-to-end: analyze → propose → accept
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_full_pipeline(self, tmp_path):
        log = tmp_path / "exec.log"
        _write_log(log, [
            {"verb": "TRN", "status": "error", "error": "unnormalized input",
             "program": "ING.dataset -> TRN.model"},
            {"verb": "TRN", "status": "error", "error": "null tensor",
             "program": "ING.raw -> TRN.model"},
            {"verb": "TRN", "status": "error", "error": "shape mismatch",
             "program": "FETCH.api -> TRN.model"},
        ])

        const = _constitution(tmp_path)
        imp = Improver(constitution=const, log_path=log)

        patterns  = imp.analyze()
        assert len(patterns) == 1
        assert patterns[0].verb == "TRN"

        proposals = imp.propose(patterns)
        assert len(proposals) == 1

        accepted = imp.accept(proposals[0])
        assert accepted is True

        # Rule is now in the constitution
        rules = const.get_rules_for_verbs({"TRN"})
        assert len(rules) >= 1
