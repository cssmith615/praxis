"""
Constitution tests — 9 tests.
"""

import pytest
import tempfile
from pathlib import Path
from praxis.constitution import Constitution, ConstitutionalRule, _parse_rule_line


# ── Fixtures ──────────────────────────────────────────────────────────────────

SEED_RULES = """\
# Praxis Constitution

[verb:ING,TRN] NEVER chain TRN directly after ING without CLN.
[verb:ING] ALWAYS PAR independent ING operations — 2-3x faster.
[verb:WRITE,DEP] ALWAYS precede WRITE and DEP with GATE in production mode.
[verb:LOOP] ALWAYS include until= condition on LOOP — open loops are rejected.
[verb:MSG] ALWAYS SIGN messages before sending to another agent.
"""


@pytest.fixture
def constitution_file(tmp_path):
    f = tmp_path / "constitution.md"
    f.write_text(SEED_RULES, encoding="utf-8")
    return f


@pytest.fixture
def c(constitution_file):
    return Constitution(path=constitution_file)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_loads_rules_from_file(c):
    assert len(c) == 5


def test_all_rules_returned(c):
    rules = c.get_all_rules()
    assert len(rules) == 5
    assert any("TRN" in r for r in rules)


def test_get_rules_for_verbs_filters_correctly(c):
    rules = c.get_rules_for_verbs({"ING", "TRN"})
    # Should match [verb:ING,TRN] and [verb:ING]
    assert len(rules) == 2
    assert any("CLN" in r for r in rules)   # the ING,TRN rule
    assert any("PAR" in r for r in rules)   # the ING rule


def test_get_rules_for_unrelated_verbs_returns_empty(c):
    rules = c.get_rules_for_verbs({"SUMM", "GEN"})
    assert rules == []


def test_get_rules_for_empty_verbs_returns_all(c):
    rules = c.get_rules_for_verbs(set())
    assert len(rules) == 5


def test_append_rule_adds_to_file(c, constitution_file):
    added = c.append_rule(
        "ALWAYS LOG after every EVAL step.",
        verbs=["EVAL", "LOG"],
    )
    assert added is True
    assert len(c) == 6

    # Reload and verify it persisted
    c2 = Constitution(path=constitution_file)
    assert len(c2) == 6
    rules = c2.get_rules_for_verbs({"EVAL"})
    assert any("LOG" in r for r in rules)


def test_append_rule_deduplicates(c):
    # Add the same rule twice
    c.append_rule("ALWAYS LOG after every EVAL step.", verbs=["EVAL"])
    result = c.append_rule("ALWAYS LOG after every EVAL step.", verbs=["EVAL"])
    assert result is False
    assert len(c) == 6  # Only one was added


def test_get_rules_for_program_text(c):
    program = "ING.sales.db -> CLN.null -> TRN.lstm -> EVAL.rmse"
    rules = c.get_rules_for_program(program)
    # Should match ING,TRN and ING rules
    assert len(rules) >= 1
    assert any("CLN" in r for r in rules)


def test_rules_by_verb(c):
    by_verb = c.rules_by_verb()
    assert "ING" in by_verb
    assert "TRN" in by_verb
    # ING appears in two rules
    assert len(by_verb["ING"]) == 2


def test_empty_constitution_file(tmp_path):
    f = tmp_path / "empty.md"
    f.write_text("# No rules yet\n", encoding="utf-8")
    c = Constitution(path=f)
    assert len(c) == 0
    assert c.get_all_rules() == []


def test_parse_rule_line_valid():
    rule = _parse_rule_line("[verb:ING,TRN] NEVER chain TRN after ING without CLN.")
    assert rule is not None
    assert rule.verbs == frozenset({"ING", "TRN"})
    assert "CLN" in rule.text


def test_parse_rule_line_comment_ignored():
    assert _parse_rule_line("# This is a comment") is None
    assert _parse_rule_line("") is None
    assert _parse_rule_line("  ") is None
