"""
Memory tests — 9 tests.

Uses a mock embedder (random normalized vectors) so no sentence-transformers
model is downloaded during CI. A temporary SQLite file is used per test.
"""

import tempfile
import numpy as np
import pytest
from pathlib import Path
from praxis.memory import ProgramMemory, StoredProgram, ADAPT_THRESHOLD, _normalize


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _mock_embedder(dim: int = 32):
    """Returns a deterministic mock embedder for tests."""
    rng = np.random.default_rng(seed=42)

    def embed(text: str) -> np.ndarray:
        # Hash the text to get a consistent seed, then generate a fixed vector
        seed = abs(hash(text)) % (2**31)
        r = np.random.default_rng(seed=seed)
        vec = r.random(dim).astype(np.float32)
        return _normalize(vec)

    return embed


@pytest.fixture
def mem(tmp_path):
    """ProgramMemory backed by a temp file with mock embedder."""
    return ProgramMemory(
        db_path=tmp_path / "test_programs.db",
        embedder=_mock_embedder(),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_empty_memory_returns_no_similar(mem):
    results = mem.retrieve_similar("check flight prices")
    assert results == []


def test_store_returns_uuid(mem):
    pid = mem.store("check flight prices", "ING.flights -> EVAL.price", "success", [])
    assert len(pid) == 36  # UUID format
    assert pid.count("-") == 4


def test_count_increments_on_store(mem):
    assert mem.count() == 0
    mem.store("goal 1", "ING.data", "success", [])
    assert mem.count() == 1
    mem.store("goal 2", "ING.data -> CLN.null", "success", [])
    assert mem.count() == 2


def test_retrieve_returns_stored_program(mem):
    mem.store("check denver flight prices", "ING.flights(dest=denver) -> EVAL.price", "success", [])
    results = mem.retrieve_similar("check denver flight prices")
    assert len(results) == 1
    assert isinstance(results[0], StoredProgram)
    assert results[0].goal_text == "check denver flight prices"
    assert results[0].shaun_program == "ING.flights(dest=denver) -> EVAL.price"
    assert results[0].outcome == "success"


def test_retrieve_respects_k(mem):
    for i in range(5):
        mem.store(f"goal number {i}", f"ING.data{i}", "success", [])
    results = mem.retrieve_similar("some goal", k=3)
    assert len(results) <= 3


def test_similarity_is_between_zero_and_one(mem):
    mem.store("flight price check", "ING.flights -> EVAL.price", "success", [])
    results = mem.retrieve_similar("check flight prices")
    assert 0.0 <= results[0].similarity <= 1.0


def test_identical_goal_has_high_similarity(mem):
    goal = "check denver flight prices and alert me if under $200"
    mem.store(goal, "ING.flights -> EVAL.price -> OUT.telegram", "success", [])
    results = mem.retrieve_similar(goal)
    # Identical text → same embedding → cosine similarity = 1.0
    assert results[0].similarity > 0.99


def test_should_adapt_true_above_threshold(mem):
    goal = "check denver flight prices and alert me if under $200"
    mem.store(goal, "ING.flights -> EVAL.price -> OUT.telegram", "success", [])
    adapt, similar = mem.should_adapt(goal)
    assert adapt is True
    assert len(similar) >= 1


def test_should_adapt_false_when_empty(mem):
    adapt, similar = mem.should_adapt("check flight prices")
    assert adapt is False
    assert similar == []


def test_recent_returns_latest(mem):
    for i in range(5):
        mem.store(f"goal {i}", f"ING.data", "success", [])
    recent = mem.recent(3)
    assert len(recent) == 3


def test_delete_removes_program(mem):
    pid = mem.store("temporary goal", "ING.data", "success", [])
    assert mem.count() == 1
    deleted = mem.delete(pid)
    assert deleted is True
    assert mem.count() == 0


def test_execution_log_stored_and_retrieved(mem):
    log = [{"verb": "ING", "status": "ok", "output": [{"id": 1}]}]
    mem.store("test goal", "ING.data", "success", log)
    results = mem.retrieve_similar("test goal")
    assert results[0].execution_log[0]["verb"] == "ING"
