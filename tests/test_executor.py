"""
Executor tests — 8 tests.
"""

import pytest
from praxis.grammar import parse
from praxis.validator import validate
from praxis.executor import Executor, ExecutionContext, UnregisteredVerbError
from praxis.handlers import HANDLERS


def _run(source: str, mode: str = "dev") -> list[dict]:
    prog = parse(source)
    errors = validate(prog, mode=mode)
    assert not errors, f"Validation errors: {errors}"
    return Executor(HANDLERS, mode=mode).execute(prog)


# ──────────────────────────────────────────────────────────────────────────────

def test_simple_chain_executes():
    results = _run("ING.sales.db -> CLN.null -> SUMM.text")
    assert len(results) == 3
    assert all(r["status"] == "ok" for r in results)
    verbs = [r["verb"] for r in results]
    assert verbs == ["ING", "CLN", "SUMM"]


def test_execution_result_has_required_fields():
    results = _run("ING.flights")
    r = results[0]
    assert "verb" in r
    assert "target" in r
    assert "params" in r
    assert "output" in r
    assert "status" in r
    assert "duration_ms" in r
    assert "log_entry" in r


def test_parallel_block_executes_all_branches():
    results = _run("PAR(ING.sales, ING.marketing, ING.crm)")
    verbs = [r["verb"] for r in results]
    assert verbs.count("ING") == 3


def test_skip_is_noop():
    results = _run("IF.ready -> OUT.telegram ELSE -> SKIP")
    # condition 'ready' is False (not in ctx.variables) → ELSE branch → SKIP
    assert len(results) == 1
    assert results[0]["verb"] == "SKIP"
    assert results[0]["status"] == "skipped"


def test_if_true_branch_executes():
    # We'll use a program that sets a variable then conditions on it
    results = _run("ING.flights -> SET.data -> IF.$data -> SUMM.text ELSE -> SKIP")
    verbs = [r["verb"] for r in results]
    # ING returns truthy data → SET captures it → IF evaluates $data = truthy → SUMM runs
    assert "SUMM" in verbs
    assert "SKIP" not in verbs


def test_set_stores_variable():
    prog = parse("TRN.lstm -> SET.model")
    errors = validate(prog)
    assert not errors
    ctx = ExecutionContext()
    executor = Executor(HANDLERS)
    results = executor.execute(prog)
    assert results[-1]["verb"] == "SET"
    assert results[-1]["status"] == "ok"


def test_loop_respects_max_depth():
    # until=never means never True — loop runs MAX_LOOP_DEPTH times and stops
    results = _run("LOOP(LOG.tick, until=never)")
    from praxis.executor import MAX_LOOP_DEPTH
    assert len(results) == MAX_LOOP_DEPTH


def test_unregistered_verb_raises():
    # Bypass validator to test executor directly with unregistered verb
    prog = parse("ING.data")  # valid parse
    # Remove ING from handlers copy
    handlers_copy = dict(HANDLERS)
    del handlers_copy["ING"]
    executor = Executor(handlers_copy)
    with pytest.raises(UnregisteredVerbError):
        executor.execute(prog)
