"""
Praxis Self-Improvement Loop — Sprint 7

`praxis improve` reads ~/.praxis/execution.log, finds recurring failure
patterns, proposes constitutional rules, evaluates impact, and appends
accepted rules to praxis-constitution.md.

Pipeline:
  1. analyze()   — parse execution.log, group failures by verb + error type
  2. propose()   — generate rule candidates (heuristic + optional LLM)
  3. eval_rule() — estimate before/after impact using past programs
  4. accept()    — append to constitution file, return ConstitutionalRule

Design decisions:
  - Works without Anthropic API (heuristic mode only)
  - LLM-mode enhances rule text quality but is never required
  - Eval is static (no re-execution): counts programs where verb set
    intersects the rule's verbs, estimates impact from failure rate
  - Deduplication is handled by Constitution.append_rule()
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from praxis.constitution import Constitution
from praxis.validator import VALID_VERBS

_LOG_PATH = Path.home() / ".praxis" / "execution.log"

# Minimum failures before a pattern is considered significant
_MIN_FAILURES = 2

# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FailurePattern:
    """A recurring failure observed in execution.log."""
    verb: str                       # verb that errored most often
    count: int                      # number of occurrences
    error_summary: str              # most common error message snippet
    sample_programs: list[str]      # up to 3 program texts that failed
    co_occurring_verbs: list[str]   # other verbs in the failing programs


@dataclass
class RuleProposal:
    """A proposed constitutional rule derived from a FailurePattern."""
    rule_text: str
    verbs: list[str]
    source: str                     # "heuristic" | "llm"
    pattern: FailurePattern
    affected_programs: int          # how many logged programs this would touch
    estimated_prevented: int        # rough estimate of prevented future failures


# ─────────────────────────────────────────────────────────────────────────────
# Improver
# ─────────────────────────────────────────────────────────────────────────────

class Improver:
    """
    Analyze execution history, propose constitutional rules, and apply
    accepted rules to the constitution file.

    Usage:
        imp = Improver()
        patterns  = imp.analyze()
        proposals = imp.propose(patterns)
        for p in proposals:
            accepted = imp.accept(p)   # appends to constitution

    With LLM enhancement (requires ANTHROPIC_API_KEY):
        imp = Improver(use_llm=True)
    """

    def __init__(
        self,
        constitution: Constitution | None = None,
        log_path: str | Path | None = None,
        use_llm: bool = False,
    ) -> None:
        self.constitution = constitution if constitution is not None else Constitution()
        self.log_path = Path(log_path) if log_path else _LOG_PATH
        self.use_llm = use_llm
        self._log_entries: list[dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self) -> list[FailurePattern]:
        """
        Read execution.log and return significant failure patterns.
        Returns patterns sorted by failure count descending.
        """
        self._log_entries = _load_log(self.log_path)
        if not self._log_entries:
            return []

        # Group error entries by failing verb
        verb_errors: dict[str, list[dict]] = defaultdict(list)
        for entry in self._log_entries:
            if entry.get("status") == "error":
                verb = entry.get("verb", "UNKNOWN")
                if verb in VALID_VERBS:
                    verb_errors[verb].append(entry)

        patterns: list[FailurePattern] = []
        for verb, errors in verb_errors.items():
            if len(errors) < _MIN_FAILURES:
                continue

            # Summarize the most common error message
            msgs = [e.get("error", "") for e in errors if e.get("error")]
            error_summary = _most_common_snippet(msgs)

            # Collect program texts and co-occurring verbs
            programs = []
            co_verb_counter: Counter = Counter()
            for e in errors:
                prog = e.get("program", "")
                if prog and prog not in programs:
                    programs.append(prog)
                verbs_in_prog = _extract_verbs(prog)
                for v in verbs_in_prog:
                    if v != verb:
                        co_verb_counter[v] += 1

            co_occurring = [v for v, _ in co_verb_counter.most_common(5)]

            patterns.append(FailurePattern(
                verb=verb,
                count=len(errors),
                error_summary=error_summary,
                sample_programs=programs[:3],
                co_occurring_verbs=co_occurring,
            ))

        patterns.sort(key=lambda p: p.count, reverse=True)
        return patterns

    def propose(self, patterns: list[FailurePattern]) -> list[RuleProposal]:
        """
        Generate rule proposals from failure patterns.
        Uses LLM if use_llm=True and ANTHROPIC_API_KEY is set, otherwise heuristic.
        """
        proposals: list[RuleProposal] = []
        for pattern in patterns:
            if self.use_llm and _has_api_key():
                try:
                    proposal = self._propose_llm(pattern)
                    proposals.append(proposal)
                    continue
                except Exception:
                    pass  # fall through to heuristic
            proposal = self._propose_heuristic(pattern)
            if proposal:
                proposals.append(proposal)
        return proposals

    def eval_rule(self, proposal: RuleProposal) -> tuple[int, int]:
        """
        Estimate impact of a proposed rule against logged history.

        Returns (affected_programs, estimated_prevented):
          - affected_programs: logged programs whose verb set overlaps the rule's verbs
          - estimated_prevented: estimated failures prevented (affected * failure_rate)
        """
        rule_verbs = set(proposal.verbs)
        affected = 0
        failures_in_affected = 0

        for entry in self._log_entries:
            prog_verbs = _extract_verbs(entry.get("program", ""))
            if prog_verbs & rule_verbs:
                affected += 1
                if entry.get("status") == "error":
                    failures_in_affected += 1

        # Conservative estimate: rule prevents ~60% of failures in affected programs
        estimated = int(failures_in_affected * 0.6)
        return affected, estimated

    def accept(self, proposal: RuleProposal) -> bool:
        """
        Append the proposed rule to the constitution file.
        Returns True if added, False if deduplicated.
        """
        return self.constitution.append_rule(
            rule_text=proposal.rule_text,
            verbs=proposal.verbs,
            source="auto-accepted",
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _propose_heuristic(self, pattern: FailurePattern) -> RuleProposal | None:
        """Generate a rule from structural patterns without LLM."""
        verb = pattern.verb
        co = pattern.co_occurring_verbs

        # Pattern: verb fails often when CLN is absent in a data pipeline
        if verb in {"TRN", "INF", "EVAL"} and "CLN" not in co:
            rule_text = (
                f"ALWAYS run CLN before {verb} to ensure inputs are normalized. "
                f"Raw data passed directly to {verb} causes errors."
            )
            rule_verbs = [verb, "CLN"]

        # Pattern: WRITE/DEP fails — suggest GATE guard
        elif verb in {"WRITE", "DEP", "BUILD"}:
            rule_text = (
                f"ALWAYS precede {verb} with GATE in production mode to require "
                f"human confirmation before irreversible operations."
            )
            rule_verbs = [verb, "GATE"]

        # Pattern: OUT fails — suggest LOG before OUT for diagnostics
        elif verb == "OUT":
            rule_text = (
                "ALWAYS LOG the payload before OUT to preserve a record if "
                "the delivery channel fails."
            )
            rule_verbs = ["OUT", "LOG"]

        # Pattern: ING fails — suggest VALIDATE after ingestion
        elif verb == "ING":
            rule_text = (
                "ALWAYS follow ING with VALIDATE or CLN to catch malformed "
                "input before downstream processing."
            )
            rule_verbs = ["ING", "VALIDATE", "CLN"]

        # Pattern: STORE fails — suggest SNAP before write
        elif verb == "STORE":
            rule_text = (
                "ALWAYS SNAP state before STORE when the stored value is derived "
                "from a multi-step pipeline, to enable ROLLBACK on failure."
            )
            rule_verbs = ["STORE", "SNAP"]

        # Generic: verb fails often — suggest ASSERT guard
        else:
            rule_text = (
                f"CONSIDER adding ASSERT before {verb} to validate preconditions "
                f"when {verb} is preceded by external data ingestion or transformation."
            )
            rule_verbs = [verb, "ASSERT"]

        # Skip if this rule already exists in the constitution
        existing = self.constitution.get_rules_for_verbs(set(rule_verbs))
        for r in existing:
            if verb in r and _similarity(r, rule_text) > 0.6:
                return None

        affected, prevented = self.eval_rule(RuleProposal(
            rule_text=rule_text,
            verbs=rule_verbs,
            source="heuristic",
            pattern=pattern,
            affected_programs=0,
            estimated_prevented=0,
        ))

        return RuleProposal(
            rule_text=rule_text,
            verbs=rule_verbs,
            source="heuristic",
            pattern=pattern,
            affected_programs=affected,
            estimated_prevented=prevented,
        )

    def _propose_llm(self, pattern: FailurePattern) -> RuleProposal:
        """Use Claude to write a constitutional rule from a failure pattern."""
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        sample_text = "\n".join(f"  - {p}" for p in pattern.sample_programs[:2])
        existing_rules = "\n".join(
            f"  - {r}" for r in self.constitution.get_rules_for_verbs({pattern.verb})[:5]
        )

        prompt = f"""\
You are a Praxis constitutional rule author. Praxis is an AI-native workflow language.

A recurring failure has been observed:
  Verb:          {pattern.verb}
  Failures:      {pattern.count} times
  Error summary: {pattern.error_summary}
  Co-occurring verbs in failing programs: {', '.join(pattern.co_occurring_verbs) or 'none'}

Sample failing programs:
{sample_text or '  (none available)'}

Existing rules for this verb:
{existing_rules or '  (none)'}

Write ONE constitutional rule in this exact format:
  ALWAYS/NEVER/CONSIDER <specific guidance about {pattern.verb}>

Rules must be:
- Concrete and actionable (not vague)
- Specific to the failure pattern observed
- One sentence, under 150 characters
- Not a duplicate of existing rules above

Return ONLY the rule text, no explanation, no prefix."""

        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        rule_text = resp.content[0].text.strip().strip('"').strip("'")

        # Extract verbs mentioned in the rule
        rule_verbs = list(_extract_verbs(rule_text) | {pattern.verb})

        affected, prevented = self.eval_rule(RuleProposal(
            rule_text=rule_text,
            verbs=rule_verbs,
            source="llm",
            pattern=pattern,
            affected_programs=0,
            estimated_prevented=0,
        ))

        return RuleProposal(
            rule_text=rule_text,
            verbs=rule_verbs,
            source="llm",
            pattern=pattern,
            affected_programs=affected,
            estimated_prevented=prevented,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_VERB_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,7})\b")


def _load_log(log_path: Path) -> list[dict]:
    """Load all JSON entries from the execution log."""
    if not log_path.exists():
        return []
    entries = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _extract_verbs(text: str) -> set[str]:
    """Extract valid Praxis verb tokens from any text string."""
    return {v for v in _VERB_RE.findall(text) if v in VALID_VERBS}


def _most_common_snippet(messages: list[str], max_len: int = 80) -> str:
    """Return a representative error snippet from a list of error messages."""
    if not messages:
        return ""
    # Find the shortest non-empty message as the representative
    messages = [m[:max_len] for m in messages if m]
    counter: Counter = Counter(messages)
    return counter.most_common(1)[0][0] if counter else ""


def _similarity(a: str, b: str) -> float:
    """Very cheap word-overlap similarity (Jaccard) for dedup check."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))
