"""
Validator tests — 10 tests.
"""

import pytest
from praxis.grammar import parse
from praxis.validator import validate, Validator, VALID_VERBS, GATE_REQUIRED_VERBS


def _errors(source: str, mode: str = "dev") -> list[str]:
    return validate(parse(source), mode=mode)


def _valid(source: str, mode: str = "dev") -> bool:
    return len(_errors(source, mode=mode)) == 0


# ──────────────────────────────────────────────────────────────────────────────

def test_valid_program_passes():
    assert _valid("ING.sales.db -> CLN.null -> SUMM.text")


def test_invalid_verb_rejected():
    errors = _errors("BADVERB.something -> CLN.null")
    assert any("BADVERB" in e for e in errors)


def test_all_valid_verbs_accepted():
    # A sample of verbs from each category
    for verb in ["ING", "TRN", "EVAL", "FETCH", "SPAWN", "GATE", "RETRY", "CAP", "SET"]:
        assert verb in VALID_VERBS, f"{verb} missing from VALID_VERBS"


def test_call_nonexistent_plan_rejected():
    errors = _errors("CALL.nonexistent_plan")
    assert any("nonexistent_plan" in e for e in errors)


def test_call_declared_plan_passes():
    source = """
        PLAN:my_plan { ING.data -> CLN.null }
        CALL.my_plan
    """
    assert _valid(source)


def test_set_dot_path_rejected():
    # SET must have a single-segment target
    errors = _errors("TRN.lstm -> SET.score.rmse")
    assert any("SET" in e for e in errors)


def test_dep_without_gate_rejected_in_prod():
    errors = _errors("ING.data -> TRN.lstm -> DEP.api", mode="prod")
    assert any("GATE" in e for e in errors)


def test_dep_without_gate_allowed_in_dev():
    assert _valid("ING.data -> TRN.lstm -> DEP.api", mode="dev")


def test_dep_with_gate_passes_in_prod():
    assert _valid("ING.data -> GATE.confirm -> DEP.api(env=prod)", mode="prod")


def test_loop_nesting_too_deep_rejected():
    source = """
        LOOP(
            LOOP(
                LOOP(
                    LOOP(EVAL.metric, until=done),
                    until=done
                ),
                until=done
            ),
            until=done
        )
    """
    errors = _errors(source)
    assert any("LOOP" in e and "depth" in e for e in errors)


def test_cap_without_role_rejected():
    errors = _errors("CAP.agent(allow=[search, summ])")
    assert any("CAP" in e and "role" in e for e in errors)


def test_cap_with_role_passes():
    assert _valid("CAP.agent(role=worker, allow=[search, summ])")
