"""
Planner tests — 9 tests.

The Anthropic API is mocked throughout — no API key or network required.
Tests verify:
  - Valid programs are returned on first attempt
  - Invalid programs trigger retry with error context
  - PlanningFailure raised after max_attempts
  - Constitutional rules are injected into the system prompt
  - Similar programs are retrieved and injected
  - Adapted programs are flagged correctly
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from praxis.memory import ProgramMemory, _normalize
from praxis.constitution import Constitution
from praxis.planner import Planner, PlanningFailure, PlanResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_embedder(dim: int = 32):
    def embed(text: str) -> np.ndarray:
        seed = abs(hash(text)) % (2**31)
        r = np.random.default_rng(seed=seed)
        return _normalize(r.random(dim).astype(np.float32))
    return embed


def _make_mock_client(responses: list[str]):
    """Return a mock Anthropic client that cycles through given response strings."""
    client = MagicMock()
    call_count = [0]

    def create(**kwargs):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        msg = MagicMock()
        msg.content = [MagicMock(text=responses[idx])]
        return msg

    client.messages.create.side_effect = create
    return client


VALID_PROGRAM = "GOAL:flight_monitor\nING.flights(dest=denver) -> EVAL.price(threshold=200) -> IF.$price < 200 -> OUT.telegram(msg=\"drop!\") ELSE -> SKIP"

INVALID_PROGRAM = "BADVERB.something -> CLN.null"

ANOTHER_VALID = "GOAL:sales_report\nING.sales.db -> CLN.null -> SUMM.text(max=200) -> OUT.notion"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mem(tmp_path):
    return ProgramMemory(db_path=tmp_path / "test.db", embedder=_mock_embedder())


@pytest.fixture
def constitution_file(tmp_path):
    f = tmp_path / "constitution.md"
    f.write_text(
        "[verb:ING,TRN] NEVER chain TRN directly after ING without CLN.\n"
        "[verb:WRITE,DEP] ALWAYS precede WRITE and DEP with GATE in production mode.\n",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def c(constitution_file):
    return Constitution(path=constitution_file)


def _planner(mem, c, responses: list[str], mode: str = "dev") -> Planner:
    return Planner(
        memory=mem,
        constitution=c,
        mode=mode,
        client=_make_mock_client(responses),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_plan_returns_valid_program_on_first_attempt(mem, c):
    planner = _planner(mem, c, [VALID_PROGRAM])
    result = planner.plan("check denver flight prices under $200")

    assert isinstance(result, PlanResult)
    assert result.program == VALID_PROGRAM
    assert result.attempts == 1
    assert result.adapted is False


def test_plan_retries_on_invalid_program(mem, c):
    # First response is invalid, second is valid
    planner = _planner(mem, c, [INVALID_PROGRAM, VALID_PROGRAM])
    result = planner.plan("check denver flight prices")

    assert result.attempts == 2
    assert result.program == VALID_PROGRAM


def test_planning_failure_raised_after_max_attempts(mem, c):
    planner = _planner(mem, c, [INVALID_PROGRAM, INVALID_PROGRAM, INVALID_PROGRAM])
    with pytest.raises(PlanningFailure) as exc_info:
        planner.plan("some goal that keeps failing")

    assert exc_info.value.attempts == 3
    assert "BADVERB" in exc_info.value.last_error or "Unknown verb" in exc_info.value.last_error


def test_planning_failure_contains_goal(mem, c):
    planner = _planner(mem, c, [INVALID_PROGRAM] * 3)
    with pytest.raises(PlanningFailure) as exc_info:
        planner.plan("my specific goal")
    assert "my specific goal" in exc_info.value.goal


def test_constitutional_rules_injected_into_prompt(mem, c):
    client = _make_mock_client([VALID_PROGRAM])
    planner = Planner(memory=mem, constitution=c, client=client)
    planner.plan("train a model on sales data")

    call_args = client.messages.create.call_args
    system_prompt = call_args.kwargs["system"]
    # Should contain at least one constitutional rule
    assert "CLN" in system_prompt or "GATE" in system_prompt


def test_similar_programs_injected_when_available(mem, c):
    # Store a program in memory
    mem.store(
        "check denver flight prices",
        "ING.flights(dest=denver) -> EVAL.price -> OUT.telegram",
        "success",
        [],
    )
    client = _make_mock_client([VALID_PROGRAM])
    planner = Planner(memory=mem, constitution=c, client=client)
    result = planner.plan("check denver flight prices")

    call_args = client.messages.create.call_args
    system_prompt = call_args.kwargs["system"]
    # Similar program's text should appear in the prompt
    assert "denver" in system_prompt.lower() or "flights" in system_prompt.lower()


def test_adapted_flag_true_when_similarity_high(mem, c):
    goal = "check denver flight prices and alert me if under $200"
    mem.store(goal, VALID_PROGRAM, "success", [])

    client = _make_mock_client([VALID_PROGRAM])
    planner = Planner(memory=mem, constitution=c, client=client)
    result = planner.plan(goal)

    # Identical goal → similarity = 1.0 → should_adapt = True
    assert result.adapted is True


def test_adapted_flag_false_when_no_memory(mem, c):
    planner = _planner(mem, c, [VALID_PROGRAM])
    result = planner.plan("check denver flight prices")
    assert result.adapted is False


def test_plan_result_contains_similar_programs(mem, c):
    mem.store("flight check", VALID_PROGRAM, "success", [])
    planner = _planner(mem, c, [VALID_PROGRAM])
    result = planner.plan("check flight prices")
    assert isinstance(result.similar, list)


def test_planner_raises_on_missing_anthropic_key(mem, c):
    """AnthropicProvider raises EnvironmentError when key is absent."""
    from praxis.providers import AnthropicProvider
    provider = AnthropicProvider(api_key=None)
    provider._api_key = None   # ensure not set
    planner = Planner(memory=mem, constitution=c, provider=provider)
    import os
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            planner.plan("test goal")
