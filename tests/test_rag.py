"""
Sprint A RAG tests — 14 tests.

Uses a deterministic mock embedder so no model is downloaded during CI.
Each test gets a fresh temp SQLite file via fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from praxis.embeddings import EmbeddingsDB, _normalize
from praxis.executor import ExecutionContext
from praxis.handlers import HANDLERS


# ── Mock embedder (deterministic, no model download) ─────────────────────────

def _mock_embed(text: str) -> np.ndarray:
    seed = abs(hash(text)) % (2**31)
    vec = np.random.default_rng(seed=seed).random(32).astype(np.float32)
    return _normalize(vec)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def edb(tmp_path):
    return EmbeddingsDB(db_path=tmp_path / "emb.db", embedder=_mock_embed)


@pytest.fixture
def sample_chunks():
    return [
        {"id": "c0", "text": "Authentication uses short-lived JWT tokens", "source": "arch.md", "chunk_index": 0},
        {"id": "c1", "text": "Database layer uses PostgreSQL via Supabase", "source": "arch.md", "chunk_index": 1},
        {"id": "c2", "text": "Frontend built with React and TypeScript",   "source": "arch.md", "chunk_index": 2},
    ]


# ── EmbeddingsDB unit tests ───────────────────────────────────────────────────

def test_empty_db_returns_no_results(edb):
    assert edb.search("anything", corpus="docs") == []


def test_store_and_count(edb, sample_chunks):
    stored = edb.store_chunks(sample_chunks, corpus="docs")
    assert stored == 3
    assert edb.count("docs") == 3


def test_search_returns_results(edb, sample_chunks):
    edb.store_chunks(sample_chunks, corpus="docs")
    results = edb.search("JWT authentication", corpus="docs", k=2)
    assert len(results) == 2
    assert all("text" in r and "similarity" in r and "source" in r for r in results)


def test_search_respects_k(edb, sample_chunks):
    edb.store_chunks(sample_chunks, corpus="docs")
    results = edb.search("technology", corpus="docs", k=1)
    assert len(results) == 1


def test_threshold_filters_low_similarity(edb, sample_chunks):
    edb.store_chunks(sample_chunks, corpus="docs")
    results = edb.search("JWT authentication", corpus="docs", threshold=0.99)
    # No chunk should have cosine sim >= 0.99 with a different query
    assert all(r["similarity"] >= 0.99 for r in results)


def test_identical_text_similarity_near_one(edb):
    text = "Praxis is an AI-native intermediate language for agentic workflows"
    edb.store_chunks([{"id": "x", "text": text, "source": "t.md", "chunk_index": 0}], corpus="c")
    results = edb.search(text, corpus="c")
    assert results[0]["similarity"] > 0.99


def test_corpus_isolation(edb):
    edb.store_chunks([{"id": "a", "text": "corpus A content", "source": "a.md", "chunk_index": 0}], corpus="A")
    edb.store_chunks([{"id": "b", "text": "corpus B content", "source": "b.md", "chunk_index": 0}], corpus="B")
    assert edb.count("A") == 1
    assert edb.count("B") == 1
    results = edb.search("content", corpus="A")
    assert all(r["source"] == "a.md" for r in results)


def test_reindex_replaces_not_duplicates(edb):
    chunk = {"id": "stable-id", "text": "original text", "source": "doc.md", "chunk_index": 0}
    edb.store_chunks([chunk], corpus="docs")
    chunk["text"] = "updated text"
    edb.store_chunks([chunk], corpus="docs")
    assert edb.count("docs") == 1


# ── ING.docs handler tests ────────────────────────────────────────────────────

def test_ing_docs_chunks_markdown(tmp_path):
    doc = tmp_path / "readme.md"
    doc.write_text("# Praxis\n\nAn AI-native language.\n\nBuilt for agentic workflows.")
    ctx = ExecutionContext()
    chunks = HANDLERS["ING"](["docs"], {"src": str(doc)}, ctx)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    for c in chunks:
        assert all(k in c for k in ("id", "text", "source", "chunk_index", "char_count"))
        assert c["char_count"] > 0


def test_ing_docs_deterministic_ids(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("First paragraph.\n\nSecond paragraph.")
    ctx = ExecutionContext()
    run1 = HANDLERS["ING"](["docs"], {"src": str(doc)}, ctx)
    run2 = HANDLERS["ING"](["docs"], {"src": str(doc)}, ctx)
    assert [c["id"] for c in run1] == [c["id"] for c in run2]


def test_ing_docs_missing_file_raises(tmp_path):
    ctx = ExecutionContext()
    with pytest.raises(FileNotFoundError):
        HANDLERS["ING"](["docs"], {"src": str(tmp_path / "ghost.md")}, ctx)


def test_ing_docs_missing_src_raises():
    ctx = ExecutionContext()
    with pytest.raises(ValueError):
        HANDLERS["ING"](["docs"], {}, ctx)


def test_ing_docs_indexes_directory(tmp_path):
    (tmp_path / "a.md").write_text("File A paragraph one.\n\nFile A paragraph two.")
    (tmp_path / "b.md").write_text("File B paragraph one.\n\nFile B paragraph two.")
    ctx = ExecutionContext()
    chunks = HANDLERS["ING"](["docs"], {"src": str(tmp_path)}, ctx)
    assert len(chunks) >= 2
    sources = {c["source"] for c in chunks}
    assert len(sources) == 2  # both files contributed chunks


def test_ing_docs_json_chuck_decision(tmp_path):
    import json
    decision = {
        "id": "dec_use_jwt_auth",
        "decision": "Use short-lived JWT tokens for authentication",
        "rejected": ["session cookies", "API keys"],
        "reason": "Stateless scaling — no Redis dependency",
        "constraints": ["no Redis", "stateless API"],
        "tags": ["auth", "security"],
        "date": "2026-03-29",
        "status": "active",
    }
    f = tmp_path / "dec_use_jwt_auth.json"
    f.write_text(json.dumps(decision))
    ctx = ExecutionContext()
    chunks = HANDLERS["ING"](["docs"], {"src": str(f)}, ctx)
    assert len(chunks) >= 1
    text = chunks[0]["text"]
    assert "JWT" in text
    assert "session cookies" in text  # rejected alternatives are searchable


# ── EMBED.text → SEARCH.semantic round-trip ───────────────────────────────────

def test_embed_search_round_trip(tmp_path, sample_chunks):
    mock_db = EmbeddingsDB(db_path=tmp_path / "e.db", embedder=_mock_embed)

    ctx = ExecutionContext()
    ctx.last_output = sample_chunks
    with patch("praxis.handlers.ai_ml.EmbeddingsDB", return_value=mock_db):
        embed_result = HANDLERS["EMBED"](["text"], {"corpus": "arch", "provider": "local"}, ctx)

    assert embed_result["chunks_stored"] == 3
    assert embed_result["corpus"] == "arch"

    ctx2 = ExecutionContext()
    with patch("praxis.handlers.ai_ml.EmbeddingsDB", return_value=mock_db):
        results = HANDLERS["SEARCH"](["semantic"], {"query": "JWT tokens", "corpus": "arch", "k": "2"}, ctx2)

    assert isinstance(results, list)
    assert len(results) >= 1
    assert "text" in results[0] and "similarity" in results[0]


# ── RECALL.docs one-step RAG ──────────────────────────────────────────────────

def test_recall_docs_returns_formatted_context(tmp_path, sample_chunks):
    mock_db = EmbeddingsDB(db_path=tmp_path / "r.db", embedder=_mock_embed)
    mock_db.store_chunks(sample_chunks, corpus="arch")

    ctx = ExecutionContext()
    with patch("praxis.handlers.io.EmbeddingsDB", return_value=mock_db):
        context_block = HANDLERS["RECALL"](["docs"], {"query": "database", "corpus": "arch", "k": "2"}, ctx)

    assert isinstance(context_block, str)
    assert "[Source:" in context_block
    assert len(context_block) > 0


def test_recall_docs_empty_corpus_returns_empty_string(tmp_path):
    mock_db = EmbeddingsDB(db_path=tmp_path / "empty.db", embedder=_mock_embed)

    ctx = ExecutionContext()
    with patch("praxis.handlers.io.EmbeddingsDB", return_value=mock_db):
        result = HANDLERS["RECALL"](["docs"], {"query": "anything", "corpus": "empty"}, ctx)

    assert result == ""
