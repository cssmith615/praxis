"""
Praxis Performance Rewriter — Sprint 13 (Pillar 6-PartA).

Reads ~/.praxis/execution.log, identifies slow steps (steps whose duration_ms
exceeds a configurable threshold), and proposes program rewrites that replace
sequential slow steps with PAR blocks.

This closes the feedback loop for performance:
  execution → slow step detected → PAR proposal → user accept → faster program

Pipeline:
  1. analyze_slow()   — parse log, find verb+step combos that are consistently slow
  2. propose_par()    — for each slow pattern, generate a rewritten program text
  3. apply()          — rewrite a specific program string to parallelize slow steps

Design decisions:
  - Only proposes PAR for steps that are independent (no $var dependencies)
  - Never rewrites order-sensitive verbs (SET/CALL/RETRY/ROLLBACK/SNAP/SPAWN/JOIN)
  - Proposals are text — the user decides whether to promote them
  - No LLM required — all analysis is structural

Usage:
    from praxis.rewriter import Rewriter
    rw = Rewriter()
    slow = rw.analyze_slow()          # list[SlowPattern]
    proposals = rw.propose_par(slow)  # list[RewriteProposal]
    for p in proposals:
        print(p.original_program, "→", p.proposed_program, f"({p.estimated_speedup_ms}ms saved)")
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from praxis.optimizer import _ORDER_SENSITIVE, _collect_var_refs
from praxis.ast_types import VerbAction, Chain, Program, ParBlock


_LOG_PATH = Path.home() / ".praxis" / "execution.log"

# Steps slower than this threshold (ms) are flagged as slow
DEFAULT_SLOW_MS = 500

# A verb must appear slow at least this many times to be considered a pattern
MIN_SLOW_COUNT = 2


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SlowPattern:
    """A verb that is consistently slow in the execution log."""
    verb: str
    count: int                          # how many slow occurrences
    avg_ms: float                       # average duration across slow occurrences
    max_ms: int                         # worst-case duration
    sample_programs: list[str] = field(default_factory=list)  # up to 3 program texts


@dataclass
class RewriteProposal:
    """A proposed PAR rewrite for a program containing slow steps."""
    original_program: str
    proposed_program: str
    slow_verbs: list[str]               # verbs that were parallelized
    estimated_speedup_ms: int           # sum of slow step durations minus the max
    source: str = "performance"         # always "performance" for this pass


# ─────────────────────────────────────────────────────────────────────────────
# Rewriter
# ─────────────────────────────────────────────────────────────────────────────

class Rewriter:
    def __init__(
        self,
        log_path: Path | str | None = None,
        slow_threshold_ms: int = DEFAULT_SLOW_MS,
    ) -> None:
        self.log_path = Path(log_path) if log_path else _LOG_PATH
        self.slow_threshold_ms = slow_threshold_ms

    # ── Analysis ───────────────────────────────────────────────────────────────

    def analyze_slow(self) -> list[SlowPattern]:
        """
        Parse execution.log and return verbs that are consistently slow.
        Only entries with status='ok' are included (errors have other causes).
        """
        entries = self._load_log()
        verb_durations: dict[str, list[int]] = defaultdict(list)
        verb_programs: dict[str, list[str]] = defaultdict(list)

        for entry in entries:
            verb = entry.get("verb", "")
            status = entry.get("status", "")
            duration = entry.get("duration_ms")
            program = entry.get("program", "")

            if not verb or status != "ok" or duration is None:
                continue
            if int(duration) < self.slow_threshold_ms:
                continue

            verb_durations[verb].append(int(duration))
            if program and len(verb_programs[verb]) < 3:
                verb_programs[verb].append(program)

        patterns = []
        for verb, durations in verb_durations.items():
            if len(durations) < MIN_SLOW_COUNT:
                continue
            patterns.append(SlowPattern(
                verb=verb,
                count=len(durations),
                avg_ms=round(sum(durations) / len(durations), 1),
                max_ms=max(durations),
                sample_programs=verb_programs[verb],
            ))

        patterns.sort(key=lambda p: p.avg_ms, reverse=True)
        return patterns

    # ── Proposal ───────────────────────────────────────────────────────────────

    def propose_par(self, patterns: list[SlowPattern]) -> list[RewriteProposal]:
        """
        For each slow pattern, scan sample programs for sequential pairs of slow
        verbs that are independent, and propose a PAR rewrite.
        """
        slow_verbs = {p.verb for p in patterns}
        proposals: list[RewriteProposal] = []
        seen_programs: set[str] = set()

        avg_by_verb = {p.verb: p.avg_ms for p in patterns}

        for pattern in patterns:
            for prog_text in pattern.sample_programs:
                if prog_text in seen_programs:
                    continue
                seen_programs.add(prog_text)

                proposal = self._try_par_rewrite(prog_text, slow_verbs, avg_by_verb)
                if proposal:
                    proposals.append(proposal)

        return proposals

    def _try_par_rewrite(
        self,
        program_text: str,
        slow_verbs: set[str],
        avg_by_verb: dict[str, float],
    ) -> RewriteProposal | None:
        """
        Try to rewrite program_text so that consecutive independent slow verbs
        run in PAR. Returns a RewriteProposal or None if no rewrite is possible.
        """
        try:
            from praxis.grammar import parse
            program = parse(program_text)
        except Exception:
            return None

        rewritten, parallelized_verbs, speedup_ms = self._rewrite_program(
            program, slow_verbs, avg_by_verb
        )

        if not parallelized_verbs:
            return None

        proposed_text = self._ast_to_text(rewritten)
        return RewriteProposal(
            original_program=program_text,
            proposed_program=proposed_text,
            slow_verbs=parallelized_verbs,
            estimated_speedup_ms=int(speedup_ms),
        )

    def apply(
        self,
        program_text: str,
        slow_verbs: set[str] | None = None,
    ) -> str:
        """
        Rewrite program_text to parallelize slow steps. If slow_verbs is None,
        runs analyze_slow() first to determine which verbs to target.
        Returns the rewritten program text (or original if no rewrite possible).
        """
        if slow_verbs is None:
            patterns = self.analyze_slow()
            slow_verbs = {p.verb for p in patterns}
            avg_by_verb = {p.verb: p.avg_ms for p in patterns}
        else:
            avg_by_verb = {v: float(self.slow_threshold_ms) for v in slow_verbs}

        try:
            from praxis.grammar import parse
            program = parse(program_text)
        except Exception:
            return program_text

        rewritten, _, _ = self._rewrite_program(program, slow_verbs, avg_by_verb)
        return self._ast_to_text(rewritten)

    # ── Internal rewriting ─────────────────────────────────────────────────────

    def _rewrite_program(
        self,
        program: Program,
        slow_verbs: set[str],
        avg_by_verb: dict[str, float],
    ) -> tuple[Program, list[str], float]:
        """
        Walk the program AST and group consecutive independent slow VerbActions
        into PAR blocks. Returns (rewritten_program, parallelized_verbs, speedup_ms).
        """
        parallelized: list[str] = []
        speedup = 0.0
        new_stmts = []

        for stmt in program.statements:
            stmt, pv, sp = self._rewrite_node(stmt, slow_verbs, avg_by_verb)
            parallelized.extend(pv)
            speedup += sp
            new_stmts.append(stmt)

        return Program(statements=new_stmts), parallelized, speedup

    def _rewrite_node(
        self, node: Any, slow_verbs: set[str], avg_by_verb: dict[str, float]
    ) -> tuple[Any, list[str], float]:
        if isinstance(node, Chain):
            return self._rewrite_chain(node, slow_verbs, avg_by_verb)
        return node, [], 0.0

    def _rewrite_chain(
        self, chain: Chain, slow_verbs: set[str], avg_by_verb: dict[str, float]
    ) -> tuple[Any, list[str], float]:
        """Group consecutive independent slow steps into PAR blocks."""
        groups: list[list[VerbAction]] = []
        current_slow: list[VerbAction] = []
        written_vars: set[str] = set()

        def flush():
            if current_slow:
                groups.append(list(current_slow))
            current_slow.clear()
            written_vars.clear()

        for step in chain.steps:
            if not isinstance(step, VerbAction):
                flush()
                groups.append([step])
                continue

            is_slow = step.verb in slow_verbs
            is_safe = step.verb not in _ORDER_SENSITIVE
            refs = _collect_var_refs(step)
            has_dep = bool(refs & written_vars)

            if is_slow and is_safe and not has_dep:
                current_slow.append(step)
                if step.verb == "SET" and step.target:
                    written_vars.add(step.target[0])
            else:
                flush()
                groups.append([step])

        flush()

        # Rebuild: groups of 2+ slow VerbActions → PAR
        new_steps = []
        parallelized: list[str] = []
        speedup = 0.0

        for group in groups:
            if len(group) >= 2 and all(isinstance(s, VerbAction) and s.verb in slow_verbs for s in group):
                new_steps.append(ParBlock(branches=list(group)))
                durations = [avg_by_verb.get(s.verb, 0.0) for s in group]
                # Speedup = sum of all - max (we pay only for the slowest)
                speedup += sum(durations) - max(durations)
                parallelized.extend(s.verb for s in group)
            else:
                new_steps.extend(group)

        if not new_steps:
            return chain, [], 0.0

        return Chain(steps=new_steps), parallelized, speedup

    # ── Text serialization (simple, round-trip safe for flat chains) ───────────

    def _ast_to_text(self, program: Program) -> str:
        """Serialize a rewritten program back to Praxis text."""
        parts = []
        for stmt in program.statements:
            parts.append(self._node_to_text(stmt))
        return " -> ".join(p for p in parts if p)

    def _node_to_text(self, node: Any) -> str:
        if isinstance(node, VerbAction):
            target = ".".join(str(t) for t in node.target) if node.target else ""
            base = f"{node.verb}.{target}" if target else node.verb
            if node.params:
                kv = ", ".join(f"{k}={v}" for k, v in node.params.items())
                return f"{base}({kv})"
            return base
        if isinstance(node, ParBlock):
            inner = ", ".join(self._node_to_text(b) for b in node.branches)
            return f"PAR({inner})"
        if isinstance(node, Chain):
            return " -> ".join(self._node_to_text(s) for s in node.steps)
        return ""

    # ── Log loading ────────────────────────────────────────────────────────────

    def _load_log(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        entries = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return entries
