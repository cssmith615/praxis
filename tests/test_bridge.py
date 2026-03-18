"""
Bridge tests — 10 tests.

Uses FastAPI TestClient (httpx) — no running server or API key required.
Shaun singletons (_memory, _planner) are patched before each relevant test.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from praxis.bridge import app
from praxis.memory import ProgramMemory, _normalize
from praxis.planner import PlanResult, PlanningFailure


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_embedder(dim: int = 32):
    def embed(text: str) -> np.ndarray:
        seed = abs(hash(text)) % (2**31)
        r = np.random.default_rng(seed=seed)
        return _normalize(r.random(dim).astype(np.float32))
    return embed


def _make_memory(tmp_path: Path) -> ProgramMemory:
    return ProgramMemory(db_path=tmp_path / "test.db", embedder=_mock_embedder())


VALID_PROGRAM = (
    "GOAL:flight_monitor\n"
    "ING.flights(dest=denver) -> EVAL.price(threshold=200) -> OUT.telegram(msg=done)"
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


# ── /health ───────────────────────────────────────────────────────────────────

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── /plan ─────────────────────────────────────────────────────────────────────

def test_plan_success(client, tmp_path):
    mock_memory = _make_memory(tmp_path)
    mock_result = PlanResult(
        program=VALID_PROGRAM,
        similar=[],
        adapted=False,
        attempts=1,
        rules_used=[],
    )
    mock_planner = MagicMock()
    mock_planner.plan.return_value = mock_result

    with patch("praxis.bridge._get_memory", return_value=mock_memory), \
         patch("praxis.bridge._get_planner", return_value=mock_planner):
        resp = client.post("/plan", json={"goal": "monitor denver flights"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["program"] == VALID_PROGRAM
    assert data["adapted"] is False
    assert data["attempts"] == 1


def test_plan_failure_returns_error(client, tmp_path):
    mock_planner = MagicMock()
    mock_planner.plan.side_effect = PlanningFailure(
        goal="some goal",
        last_error="Unknown verb BADVERB",
        attempts=3,
    )

    with patch("praxis.bridge._get_planner", return_value=mock_planner):
        resp = client.post("/plan", json={"goal": "some goal"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False
    assert "BADVERB" in data["error"]
    assert data["attempts"] == 3


def test_plan_unexpected_exception_returns_error(client):
    mock_planner = MagicMock()
    mock_planner.plan.side_effect = RuntimeError("unexpected crash")

    with patch("praxis.bridge._get_planner", return_value=mock_planner):
        resp = client.post("/plan", json={"goal": "test"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False
    assert "unexpected crash" in data["error"]


# ── /execute ──────────────────────────────────────────────────────────────────

def test_execute_valid_program(client):
    resp = client.post("/execute", json={"program": VALID_PROGRAM})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert isinstance(data["results"], list)
    assert len(data["results"]) > 0
    assert all(r["status"] in {"ok", "error", "skipped"} for r in data["results"])


def test_execute_parse_error(client):
    resp = client.post("/execute", json={"program": "BADVERB.something -> CLN"})
    assert resp.status_code == 200
    data = resp.json()
    # Either parse error or validation error
    assert data["ok"] is False
    assert len(data["errors"]) > 0


def test_execute_validation_error(client):
    # GATE_REQUIRED: DEP without GATE in prod mode (must be in a chain to trigger)
    resp = client.post("/execute", json={
        "program": "GOAL:deploy_test\nING.data -> TRN.lstm -> DEP.prod",
        "mode": "prod",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False
    assert any("GATE" in e or "DEP" in e for e in data["errors"])


# ── /memory/store ─────────────────────────────────────────────────────────────

def test_memory_store(client, tmp_path):
    mock_memory = _make_memory(tmp_path)

    with patch("praxis.bridge._get_memory", return_value=mock_memory):
        resp = client.post("/memory/store", json={
            "goal": "monitor denver flights",
            "program": VALID_PROGRAM,
            "outcome": "success",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["id"] is not None


# ── /memory/retrieve ──────────────────────────────────────────────────────────

def test_memory_retrieve_empty(client, tmp_path):
    mock_memory = _make_memory(tmp_path)

    with patch("praxis.bridge._get_memory", return_value=mock_memory):
        resp = client.post("/memory/retrieve", json={"goal": "check flights", "k": 3})

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["programs"] == []


def test_memory_retrieve_returns_stored(client, tmp_path):
    mock_memory = _make_memory(tmp_path)
    mock_memory.store("monitor denver flights", VALID_PROGRAM, "success", [])

    with patch("praxis.bridge._get_memory", return_value=mock_memory):
        resp = client.post("/memory/retrieve", json={"goal": "monitor denver flights", "k": 3})

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert len(data["programs"]) == 1
    assert data["programs"][0]["outcome"] == "success"
    # Allow tiny float32 rounding above 1.0
    assert -0.01 <= data["programs"][0]["similarity"] <= 1.01
