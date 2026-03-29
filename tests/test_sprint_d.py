"""
Sprint D tests — Corpora tab API + EmbeddingsDB.sources().

Uses the mock embedder from test_rag.py (same pattern — no model download).
FastAPI TestClient for the three new endpoints.
"""

from __future__ import annotations

import json
import numpy as np
import pytest
from unittest.mock import patch

from praxis.embeddings import EmbeddingsDB, _normalize


# ── Mock embedder ─────────────────────────────────────────────────────────────

def _mock_embed(text: str) -> np.ndarray:
    seed = abs(hash(text)) % (2**31)
    vec = np.random.default_rng(seed=seed).random(32).astype(np.float32)
    return _normalize(vec)


# ── EmbeddingsDB.sources() ────────────────────────────────────────────────────

@pytest.fixture
def edb(tmp_path):
    return EmbeddingsDB(db_path=tmp_path / "emb.db", embedder=_mock_embed)


def test_sources_empty_corpus_returns_empty(edb):
    assert edb.sources("no-such-corpus") == []


def test_sources_returns_distinct_paths(edb):
    chunks = [
        {"id": "a0", "text": "chunk from file A", "source": "/docs/a.md", "chunk_index": 0},
        {"id": "a1", "text": "more from file A",  "source": "/docs/a.md", "chunk_index": 1},
        {"id": "b0", "text": "chunk from file B", "source": "/docs/b.md", "chunk_index": 0},
    ]
    edb.store_chunks(chunks, corpus="docs")
    sources = edb.sources("docs")
    assert set(sources) == {"/docs/a.md", "/docs/b.md"}


def test_sources_respects_corpus_isolation(edb):
    edb.store_chunks(
        [{"id": "x", "text": "content", "source": "/a/x.md", "chunk_index": 0}],
        corpus="A",
    )
    edb.store_chunks(
        [{"id": "y", "text": "content", "source": "/b/y.md", "chunk_index": 0}],
        corpus="B",
    )
    assert edb.sources("A") == ["/a/x.md"]
    assert edb.sources("B") == ["/b/y.md"]


# ── API endpoints via TestClient ──────────────────────────────────────────────

@pytest.fixture
def client(tmp_path):
    """FastAPI TestClient with an in-memory EmbeddingsDB injected."""
    from fastapi.testclient import TestClient
    from praxis.server import app
    import praxis.server as server_module

    mock_db = EmbeddingsDB(db_path=tmp_path / "api.db", embedder=_mock_embed)
    server_module._embeddings_db = mock_db
    yield TestClient(app), mock_db
    server_module._embeddings_db = None


@pytest.fixture
def client_with_data(client):
    test_client, db = client
    chunks = [
        {"id": "c0", "text": "JWT tokens are short-lived", "source": "/arch.md", "chunk_index": 0},
        {"id": "c1", "text": "PostgreSQL via Supabase",     "source": "/arch.md", "chunk_index": 1},
        {"id": "c2", "text": "React frontend with TypeScript", "source": "/stack.md", "chunk_index": 0},
    ]
    db.store_chunks(chunks, corpus="project_docs")
    return test_client, db


def test_list_corpora_empty(client):
    test_client, _ = client
    r = test_client.get("/api/corpora")
    assert r.status_code == 200
    assert r.json()["corpora"] == []


def test_list_corpora_returns_name_chunks_sources(client_with_data):
    test_client, _ = client_with_data
    r = test_client.get("/api/corpora")
    assert r.status_code == 200
    corpora = r.json()["corpora"]
    assert len(corpora) == 1
    c = corpora[0]
    assert c["name"] == "project_docs"
    assert c["chunks"] == 3
    assert set(c["sources"]) == {"/arch.md", "/stack.md"}


def test_search_corpus_returns_results(client_with_data):
    test_client, _ = client_with_data
    r = test_client.post("/api/corpora/project_docs/search", json={"query": "JWT auth", "k": 2})
    assert r.status_code == 200
    d = r.json()
    assert d["corpus"] == "project_docs"
    assert len(d["results"]) >= 1
    assert "similarity" in d["results"][0]
    assert "text" in d["results"][0]


def test_search_corpus_respects_k(client_with_data):
    test_client, _ = client_with_data
    r = test_client.post("/api/corpora/project_docs/search", json={"query": "tech stack", "k": 1})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 1


def test_search_corpus_empty_returns_empty(client):
    test_client, _ = client
    r = test_client.post("/api/corpora/empty/search", json={"query": "anything"})
    assert r.status_code == 200
    assert r.json()["results"] == []


def test_reindex_missing_corpus_returns_404(client):
    test_client, _ = client
    r = test_client.post("/api/corpora/ghost/reindex")
    assert r.status_code == 404


def test_reindex_file_corpus(client_with_data, tmp_path):
    test_client, db = client_with_data
    # Write a real file at the source path won't work (absolute paths in mock data),
    # so instead store chunks with a real tmp_path source and verify the endpoint
    # handles file-not-found gracefully (reports in errors, doesn't crash)
    doc = tmp_path / "real.md"
    doc.write_text("Praxis is an AI-native language.\n\nBuilt for agentic workflows.")
    db.store_chunks(
        [{"id": "r0", "text": "Praxis content", "source": str(doc), "chunk_index": 0}],
        corpus="real_corpus",
    )
    r = test_client.post("/api/corpora/real_corpus/reindex")
    assert r.status_code == 200
    d = r.json()
    assert d["corpus"] == "real_corpus"
    assert d["sources_indexed"] >= 1
    assert "errors" in d
