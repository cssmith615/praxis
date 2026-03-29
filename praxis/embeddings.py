"""
Document Embeddings DB — Sprint A RAG foundation.

SQLite-backed store for document chunk embeddings.
Path: ~/.praxis/embeddings.db

Table: embeddings(id, text, source, chunk_index, embedding BLOB, corpus, created_at)

Providers:
  local   — sentence-transformers all-MiniLM-L6-v2  (pip install praxis-lang[memory])
  voyage  — Voyage AI    (VOYAGE_API_KEY env var,   pip install voyageai)
  openai  — OpenAI       (OPENAI_API_KEY env var,   pip install openai)

The embedder callable is injectable so tests run without loading any model.
"""

from __future__ import annotations

import os
import sqlite3
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

DEFAULT_DB_PATH = Path.home() / ".praxis" / "embeddings.db"


class EmbeddingsDB:
    def __init__(
        self,
        db_path: str | Path | None = None,
        embedder: Callable[[str], np.ndarray] | None = None,
        provider: str = "local",
    ) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._embedder_fn = embedder
        self._provider = provider
        self._st_model = None
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id          TEXT PRIMARY KEY,
                    text        TEXT NOT NULL,
                    source      TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    embedding   BLOB NOT NULL,
                    corpus      TEXT NOT NULL,
                    created_at  TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_emb_corpus ON embeddings (corpus)"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _embed(self, text: str) -> np.ndarray:
        if self._embedder_fn is not None:
            return _normalize(np.array(self._embedder_fn(text), dtype=np.float32))
        if self._provider == "voyage":
            return self._embed_voyage(text)
        if self._provider == "openai":
            return self._embed_openai(text)
        return self._embed_local(text)

    def _embed_local(self, text: str) -> np.ndarray:
        if self._st_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                raise ImportError(
                    "Local embeddings require sentence-transformers: "
                    "pip install praxis-lang[memory]"
                )
        vec = self._st_model.encode(text, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

    def _embed_voyage(self, text: str) -> np.ndarray:
        try:
            import voyageai
        except ImportError:
            raise ImportError("Voyage embeddings require: pip install voyageai")
        key = os.environ.get("VOYAGE_API_KEY", "")
        if not key:
            raise RuntimeError("VOYAGE_API_KEY env var required for provider=voyage")
        client = voyageai.Client(api_key=key)
        result = client.embed([text], model="voyage-2")
        return _normalize(np.array(result.embeddings[0], dtype=np.float32))

    def _embed_openai(self, text: str) -> np.ndarray:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI embeddings require: pip install openai")
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY env var required for provider=openai")
        client = OpenAI(api_key=key)
        resp = client.embeddings.create(input=text, model="text-embedding-3-small")
        return _normalize(np.array(resp.data[0].embedding, dtype=np.float32))

    def store_chunks(self, chunks: list[dict], corpus: str) -> int:
        """Embed and store chunks. Returns count stored. Uses INSERT OR REPLACE."""
        now = datetime.now(timezone.utc).isoformat()
        stored = 0
        with self._conn() as conn:
            for chunk in chunks:
                vec = self._embed(chunk["text"])
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk.get("id", str(uuid.uuid4())),
                        chunk["text"],
                        chunk.get("source", ""),
                        int(chunk.get("chunk_index", 0)),
                        _to_blob(vec),
                        corpus,
                        now,
                    ),
                )
                stored += 1
        return stored

    def search(
        self, query: str, corpus: str, k: int = 5, threshold: float = 0.0
    ) -> list[dict]:
        """Cosine similarity search within a corpus. Returns top-k above threshold."""
        query_vec = self._embed(query)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, text, source, chunk_index, embedding "
                "FROM embeddings WHERE corpus = ?",
                (corpus,),
            ).fetchall()

        if not rows:
            return []

        scored = []
        for row in rows:
            stored_vec = _from_blob(row[4])
            if len(stored_vec) != len(query_vec):
                continue
            sim = float(np.dot(query_vec, stored_vec))
            if sim >= threshold:
                scored.append((sim, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": r[0],
                "text": r[1],
                "source": r[2],
                "chunk_index": r[3],
                "similarity": sim,
            }
            for sim, r in scored[:k]
        ]

    def count(self, corpus: str | None = None) -> int:
        with self._conn() as conn:
            if corpus:
                return conn.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE corpus = ?", (corpus,)
                ).fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    def corpora(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT corpus FROM embeddings ORDER BY corpus"
            ).fetchall()
        return [r[0] for r in rows]

    def sources(self, corpus: str) -> list[str]:
        """Return distinct source file paths stored in a corpus."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT source FROM embeddings WHERE corpus = ? ORDER BY source",
                (corpus,),
            ).fetchall()
        return [r[0] for r in rows]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _to_blob(vec: np.ndarray) -> bytes:
    arr = vec.astype(np.float32)
    return struct.pack(f"{len(arr)}f", *arr.tolist())


def _from_blob(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)
