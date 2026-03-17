"""
Shaun LLM Planner

Translates a natural language goal into a valid, validated Praxis program.

Architecture:
  1. Retrieve similar past programs from ProgramMemory (cosine KNN on goal text)
  2. Load relevant constitutional rules from Constitution (verb-tag filtered)
  3. Call Claude with a structured system prompt containing:
       - Grammar spec (compact)
       - 51-token vocabulary
       - Constitutional rules (verb-filtered)
       - Similar past programs (as few-shot examples or adaptation base)
  4. Parse + validate the response
  5. If invalid, retry with the error message injected (max_attempts=3)
  6. On success: return program text + context used
  7. On persistent failure: raise PlanningFailure

The planner does NOT execute or store — that's the caller's responsibility.
The CLI `praxis goal` command calls plan() then execute() then memory.store().
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from praxis.grammar import parse
from praxis.validator import validate, VALID_VERBS
from praxis.memory import ProgramMemory, StoredProgram
from praxis.constitution import Constitution

if TYPE_CHECKING:
    import anthropic as _anthropic_module

# ──────────────────────────────────────────────────────────────────────────────
# Prompt constants
# ──────────────────────────────────────────────────────────────────────────────

_GRAMMAR_SUMMARY = """\
Shaun v1 — compact grammar reference

Sequential chain:   VERB.target(key=val) -> VERB.target -> VERB
Parallel:           PAR(VERB.a, VERB.b, VERB.c) -> MERGE
Conditional:        IF.condition -> action ELSE -> SKIP
                    IF.$var > 0.9 -> action ELSE -> SKIP
                    IF.func_cond(key=val) -> action ELSE -> SKIP
Loop:               LOOP(action, until=condition)
                    LOOP(action, until=$var > 0.9)
Named plan:         PLAN:name { statement+ }
Call named plan:    CALL.plan_name
Capture output:     VERB -> SET.varname
Use variable:       VERB.target(key=$varname)
Goal label:         GOAL:identifier
Block body:         IF.x -> { VERB.a -> VERB.b -> VERB.c } ELSE -> SKIP

Syntax rules:
  • VERB must be ALL-CAPS, 2–8 chars (e.g. ING, TRN, EVAL, OUT)
  • target / identifier must start with lowercase or underscore
  • Variable references: $varname  (dollar prefix, no spaces)
  • Comparison ops in expressions: > < >= <= == !=
  • Boolean: AND  OR  NOT
  • Function conditions: name(key=val, key=val)
  • List values in params: [item1, item2, item3]
  • Comments: // this is a comment
  • Bare verbs (no target needed): MERGE JOIN ROLLBACK BREAK SKIP WAIT\
"""

_VOCAB_SUMMARY = """\
Data:    ING  CLN  XFRM  FILTER  SORT  MERGE
AI/ML:   TRN  INF  EVAL  SUMM  CLASS  GEN  EMBED  SEARCH
I/O:     READ  WRITE  FETCH  POST  OUT  STORE  RECALL
Agents:  SPAWN  CALL  MSG  WAIT  CAST  JOIN  SIGN  CAP
Deploy:  BUILD  DEP  TEST
Control: GOAL  PLAN  IF  LOOP  SKIP  PAR  FORK  BREAK  SET
Error:   ERR  RETRY  ROLLBACK
Audit:   VALIDATE  ASSERT  LOG  GATE  SNAP  ANNOTATE  ROUTE\
"""

_SYSTEM_TEMPLATE = """\
You are a Praxis planner. Your only job is to translate a natural language goal
into a valid Praxis program.  Output ONLY the Praxis program — no explanation,
no markdown fences, no commentary.  Start directly with GOAL: or the first verb.

━━━ GRAMMAR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{grammar}

━━━ VALID VERBS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{vocab}

━━━ CONSTITUTIONAL RULES (you must follow these) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{rules}

━━━ SIMILAR PAST PROGRAMS (adapt if relevant; use as few-shot if not) ━━━━━━━
{past_programs}\
"""


# ──────────────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────────────

class PlanningFailure(Exception):
    """Raised when the planner cannot produce a valid program after max_attempts."""

    def __init__(self, goal: str, last_error: str, attempts: int) -> None:
        self.goal = goal
        self.last_error = last_error
        self.attempts = attempts
        super().__init__(
            f"Planning failed after {attempts} attempt(s) for goal: {goal!r}\n"
            f"Last error: {last_error}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Planner result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PlanResult:
    program: str                          # validated Praxis program text
    similar: list[StoredProgram]          # programs used as context
    adapted: bool                         # True if based on a similar program
    attempts: int                         # how many LLM calls were needed
    rules_used: list[str]                 # constitutional rules injected


# ──────────────────────────────────────────────────────────────────────────────
# Planner
# ──────────────────────────────────────────────────────────────────────────────

class Planner:
    """
    LLM-powered Praxis program generator.

    Parameters
    ----------
    memory : ProgramMemory
        Program library for retrieval and (post-execution) storage.
    constitution : Constitution
        Constitutional rules for the system prompt.
    model : str
        Anthropic model ID. Defaults to claude-sonnet-4-6.
    max_attempts : int
        Number of generate→validate retry cycles before raising PlanningFailure.
    mode : str
        "dev" or "prod" — passed to the validator.
    client : anthropic.Anthropic | None
        Injected client (for tests / custom setups). If None, created from
        ANTHROPIC_API_KEY env var.
    """

    def __init__(
        self,
        memory: ProgramMemory,
        constitution: Constitution,
        model: str = "claude-sonnet-4-6",
        max_attempts: int = 3,
        mode: str = "dev",
        client=None,
    ) -> None:
        self.memory = memory
        self.constitution = constitution
        self.model = model
        self.max_attempts = max_attempts
        self.mode = mode
        self._client = client  # lazy-created if None

    # ── Public API ─────────────────────────────────────────────────────────────

    def plan(self, goal: str) -> PlanResult:
        """
        Generate a validated Praxis program for goal.

        Returns a PlanResult on success.
        Raises PlanningFailure if all attempts fail.
        """
        adapt, similar = self.memory.should_adapt(goal)

        # First pass: inject all rules (we don't know verbs yet)
        # Subsequent passes: filter by verbs in the last generated program
        rules = self.constitution.get_all_rules()
        last_program_text: str | None = None
        last_error: str | None = None

        for attempt in range(1, self.max_attempts + 1):
            # Refine rules to verbs seen in last attempt's output
            if last_program_text:
                from praxis.constitution import _extract_verbs_from_text
                verbs = _extract_verbs_from_text(last_program_text)
                rules = self.constitution.get_rules_for_verbs(verbs) or rules

            program_text = self._call_llm(
                goal=goal,
                similar=similar,
                rules=rules,
                adapt=adapt,
                last_error=last_error,
            )
            last_program_text = program_text

            # Validate
            error = self._validate(program_text)
            if error is None:
                return PlanResult(
                    program=program_text,
                    similar=similar,
                    adapted=adapt and bool(similar),
                    attempts=attempt,
                    rules_used=rules,
                )

            last_error = f"Attempt {attempt}: {error}"

        raise PlanningFailure(goal, last_error or "unknown", self.max_attempts)

    # ── LLM call ───────────────────────────────────────────────────────────────

    def _call_llm(
        self,
        goal: str,
        similar: list[StoredProgram],
        rules: list[str],
        adapt: bool,
        last_error: str | None,
    ) -> str:
        client = self._get_client()

        system = _SYSTEM_TEMPLATE.format(
            grammar=_GRAMMAR_SUMMARY,
            vocab=_VOCAB_SUMMARY,
            rules=_format_rules(rules),
            past_programs=_format_past_programs(similar),
        )

        user_lines = [f"GOAL: {goal}"]

        if last_error:
            user_lines.append(
                f"\nYour previous attempt produced errors. Fix them:\n{last_error}\n"
                "Output only the corrected Praxis program."
            )
        elif adapt and similar:
            user_lines.append(
                f"\nA similar program exists (similarity: {similar[0].similarity:.2f}). "
                "Adapt it for this goal rather than generating from scratch."
            )

        message = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": "\n".join(user_lines)}],
        )
        return message.content[0].text.strip()

    # ── Validation ─────────────────────────────────────────────────────────────

    def _validate(self, program_text: str) -> str | None:
        """
        Parse and validate program_text.
        Returns an error string if invalid, None if valid.
        """
        try:
            ast = parse(program_text)
        except Exception as exc:
            return f"Parse error: {exc}"

        errors = validate(ast, mode=self.mode)
        if errors:
            return "Validation errors:\n" + "\n".join(f"  • {e}" for e in errors)

        return None

    # ── Client ─────────────────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package required for the planner. "
                "Run: pip install anthropic"
            ) from exc
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "The planner requires a valid API key."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        return self._client


# ──────────────────────────────────────────────────────────────────────────────
# Prompt formatters
# ──────────────────────────────────────────────────────────────────────────────

def _format_rules(rules: list[str]) -> str:
    if not rules:
        return "  (none yet — constitution is empty)"
    return "\n".join(f"  • {r}" for r in rules)


def _format_past_programs(programs: list[StoredProgram]) -> str:
    if not programs:
        return "  (none yet — this is the first program)"
    parts = []
    for i, p in enumerate(programs[:3], 1):
        parts.append(
            f"  [{i}] Goal ({p.similarity:.2f} similarity, {p.outcome}):\n"
            f"       {p.goal_text}\n"
            f"       Program:\n"
            + "\n".join(f"         {line}" for line in p.shaun_program.splitlines())
        )
    return "\n\n".join(parts)
