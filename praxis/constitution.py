"""
Praxis Constitutional Rules

Loads praxis-constitution.md and provides verb-tag-filtered rule injection
for the planner prompt. Rules grow over time via `shaun improve` (Sprint 7).

Rule format in the markdown file:
  [verb:ING,TRN] NEVER chain TRN directly after ING without CLN.
  [verb:WRITE,DEP] ALWAYS precede WRITE and DEP with GATE in production mode.

Filtering:
  Given the set of verbs in the current planning context (either from a
  retrieved similar program or guessed from the goal text), return only rules
  whose verb tag intersects with that set.  This keeps the planner prompt
  focused and avoids injecting irrelevant rules.

Auto-deduplication:
  append_rule() normalizes whitespace and lowercases before checking for
  duplicates, so minor rephrasing doesn't create duplicate rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Default path: praxis-constitution.md at the repo root (one level above shaun/)
DEFAULT_CONSTITUTION_PATH = Path(__file__).parent.parent / "praxis-constitution.md"


# ──────────────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ConstitutionalRule:
    verbs: frozenset[str]     # e.g. frozenset({"ING", "TRN"})
    text: str                 # the rule sentence
    source: str = "manual"   # "manual" | "auto-proposed" | "auto-accepted"


# ──────────────────────────────────────────────────────────────────────────────
# Constitution
# ──────────────────────────────────────────────────────────────────────────────

class Constitution:
    """
    Load, filter, and append constitutional rules from praxis-constitution.md.

    Usage:
        c = Constitution()                          # default path
        c = Constitution("path/to/constitution.md") # custom path

        rules = c.get_rules_for_verbs({"ING", "TRN", "EVAL"})
        # → ["NEVER chain TRN directly after ING without CLN.", ...]

        c.append_rule("ALWAYS LOG after DEP in production.", verbs=["DEP", "LOG"])
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_CONSTITUTION_PATH
        self._rules: list[ConstitutionalRule] = []
        self.load()

    # ── Loading ────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """(Re)load rules from the constitution file."""
        self._rules = []
        if not self.path.exists():
            return
        text = self.path.read_text(encoding="utf-8")
        for line in text.splitlines():
            rule = _parse_rule_line(line)
            if rule:
                self._rules.append(rule)

    # ── Filtering ──────────────────────────────────────────────────────────────

    def get_rules_for_verbs(self, verbs: set[str]) -> list[str]:
        """
        Return rule texts whose verb tags overlap with the given verb set.
        An empty verbs set returns all rules (useful for first-pass prompts).
        """
        if not verbs:
            return self.get_all_rules()
        return [
            r.text for r in self._rules
            if r.verbs & verbs  # set intersection
        ]

    def get_all_rules(self) -> list[str]:
        return [r.text for r in self._rules]

    def get_rules_for_program(self, shaun_program: str) -> list[str]:
        """
        Extract verbs from a Praxis program string and return matching rules.
        Useful when you have the program text but not yet a parsed AST.
        """
        verbs = _extract_verbs_from_text(shaun_program)
        return self.get_rules_for_verbs(verbs)

    # ── Writing ────────────────────────────────────────────────────────────────

    def append_rule(
        self,
        rule_text: str,
        verbs: list[str],
        source: str = "manual",
    ) -> bool:
        """
        Append a new rule to the constitution file and in-memory list.

        Returns True if the rule was added, False if a near-duplicate was found.
        Deduplication is based on normalized (lowercase + collapsed whitespace)
        text comparison.
        """
        rule_text = rule_text.strip()
        if not rule_text:
            return False

        # Deduplicate
        normalized = _normalize_text(rule_text)
        for existing in self._rules:
            if _normalize_text(existing.text) == normalized:
                return False

        verb_set = frozenset(v.strip().upper() for v in verbs)
        verb_tag = ",".join(sorted(verb_set))
        new_line = f"\n[verb:{verb_tag}] {rule_text}\n"

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(new_line)

        self._rules.append(ConstitutionalRule(
            verbs=verb_set,
            text=rule_text,
            source=source,
        ))
        return True

    # ── Metadata ───────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._rules)

    def __iter__(self):
        return iter(self._rules)

    def rules_by_verb(self) -> dict[str, list[str]]:
        """Return a dict mapping each verb to the rules that reference it."""
        result: dict[str, list[str]] = {}
        for rule in self._rules:
            for verb in rule.verbs:
                result.setdefault(verb, []).append(rule.text)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# Matches: [verb:ING,TRN] rule text here...
_RULE_PATTERN = re.compile(r"^\[verb:([A-Z,\s]+)\]\s+(.+)$")

# Matches all-caps 2-8 char tokens in a Praxis program string
_VERB_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]{1,7})\b")


def _parse_rule_line(line: str) -> ConstitutionalRule | None:
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("<!--"):
        return None
    match = _RULE_PATTERN.match(line)
    if not match:
        return None
    verbs = frozenset(v.strip() for v in match.group(1).split(",") if v.strip())
    text = match.group(2).strip()
    return ConstitutionalRule(verbs=verbs, text=text)


def _extract_verbs_from_text(program_text: str) -> set[str]:
    """Pull all VERB tokens from raw Praxis program text (no parsing needed)."""
    from praxis.validator import VALID_VERBS
    found = _VERB_PATTERN.findall(program_text)
    return {v for v in found if v in VALID_VERBS}


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())
