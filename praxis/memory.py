"""
Shaun Program Memory

SQLite-backed library of past Praxis programs. Every successful (or failed)
execution is stored with its goal embedding so future goals can retrieve and
adapt similar programs rather than generating from scratch.

Storage format:
  ~/.praxis/programs.db — default location
  Table: programs (id, goal_text, goal_embedding BLOB, shaun_program,
                   outcome, execution_log JSON, created_at)

Retrieval:
  - Embed the new goal with sentence-transformers (all-MiniLM-L6-v2)
  - Cosine similarity against all stored goal embeddings (numpy dot product
    on normalized vectors — O(n) scan; fine until ~50k rows, index later)
  - Return top-k sorted by similarity

Adaptation threshold:
  similarity >= 0.85 → adapt existing program (planner told to reuse structure)
  similarity <  0.85 → generate fresh (planner given as few-shot examples only)

The embedder is injectable so tests can pass a mock without loading the
384-MB sentence-transformers model.
"""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────────────

ADAPT_THRESHOLD = 0.85   # similarity >= this → tell planner to adapt
DEFAULT_DB_PATH = Path.home() / ".shaun" / "programs.db"


@dataclass
class StoredProgram:
    id: str
    goal_text: str
    shaun_program: str
    outcome: str                      # "success" | "failure" | "partial"
    execution_log: list[dict]
    created_at: str
    similarity: float = 0.0           # filled by retrieve_similar


# ──────────────────────────────────────────────────────────────────────────────
# Memory
# ──────────────────────────────────────────────────────────────────────────────

class ProgramMemory:
    """
    Program library backed by SQLite.

    Parameters
    ----------
    db_path : str | Path | None
        Path to the SQLite file.  Defaults to ~/.praxis/programs.db.
    embedder : Callable[[str], np.ndarray] | None
        Function that maps a text string to a normalized float32 numpy vector.
        If None, a sentence-transformers model is loaded on first use (lazy).
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        embedder: Callable[[str], np.ndarray] | None = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._embedder_fn = embedder          # injected (for tests / custom models)
        self._st_model = None                 # sentence-transformers, lazy-loaded
        self._init_db()

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS programs (
                    id           TEXT PRIMARY KEY,
                    goal_text    TEXT NOT NULL,
                    goal_embedding BLOB,
                    shaun_program  TEXT NOT NULL,
                    outcome      TEXT NOT NULL,
                    execution_log TEXT,
                    created_at   TEXT NOT NULL
                )
            """)
            # Index on created_at for chronological queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_programs_created
                ON programs (created_at)
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    # ── Embedding ──────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        if self._embedder_fn is not None:
            vec = self._embedder_fn(text)
            return _normalize(np.array(vec, dtype=np.float32))

        # Lazy-load sentence-transformers
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._st_model.encode(text, normalize_embeddings=True)

    # ── Write ──────────────────────────────────────────────────────────────────

    def store(
        self,
        goal: str,
        program: str,
        outcome: str,
        log: list[dict],
    ) -> str:
        """
        Embed the goal and persist the program.  Returns the new program id.
        outcome should be "success" | "failure" | "partial".
        """
        pid = str(uuid.uuid4())
        embedding = self._embed(goal)
        blob = _vec_to_blob(embedding)
        now = datetime.now(timezone.utc).isoformat()

        with self._conn() as conn:
            conn.execute(
                "INSERT INTO programs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (pid, goal, blob, program, outcome, json.dumps(log), now),
            )
        return pid

    # ── Read ───────────────────────────────────────────────────────────────────

    def retrieve_similar(self, goal: str, k: int = 3) -> list[StoredProgram]:
        """
        Return up to k stored programs most similar to goal, sorted by
        cosine similarity descending.
        """
        query_vec = self._embed(goal)

        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, goal_text, goal_embedding, shaun_program, "
                "outcome, execution_log, created_at FROM programs"
            ).fetchall()

        if not rows:
            return []

        scored: list[tuple[float, Any]] = []
        for row in rows:
            if row[2] is None:
                continue
            stored_vec = _blob_to_vec(row[2])
            if len(stored_vec) != len(query_vec):
                continue
            similarity = float(np.dot(query_vec, stored_vec))
            scored.append((similarity, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[StoredProgram] = []
        for similarity, row in scored[:k]:
            results.append(StoredProgram(
                id=row[0],
                goal_text=row[1],
                shaun_program=row[3],
                outcome=row[4],
                execution_log=json.loads(row[5]) if row[5] else [],
                created_at=row[6],
                similarity=similarity,
            ))
        return results

    def should_adapt(self, goal: str) -> tuple[bool, list[StoredProgram]]:
        """
        Retrieve similar programs and decide whether to adapt or generate fresh.

        Returns:
            (adapt: bool, similar: list[StoredProgram])
            adapt=True  → top match similarity >= ADAPT_THRESHOLD
            adapt=False → generate fresh (similar programs used as few-shot only)
        """
        similar = self.retrieve_similar(goal, k=3)
        adapt = bool(similar and similar[0].similarity >= ADAPT_THRESHOLD)
        return adapt, similar

    # ── Metadata ───────────────────────────────────────────────────────────────

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM programs").fetchone()[0]

    def recent(self, n: int = 10) -> list[StoredProgram]:
        """Return the n most recently stored programs."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, goal_text, goal_embedding, shaun_program, "
                "outcome, execution_log, created_at FROM programs "
                "ORDER BY created_at DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [
            StoredProgram(
                id=r[0], goal_text=r[1], shaun_program=r[3],
                outcome=r[4],
                execution_log=json.loads(r[5]) if r[5] else [],
                created_at=r[6],
            )
            for r in rows
        ]

    def delete(self, program_id: str) -> bool:
        """Delete a stored program by id. Returns True if a row was deleted."""
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM programs WHERE id = ?", (program_id,)
            )
            return cursor.rowcount > 0


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _vec_to_blob(vec: np.ndarray) -> bytes:
    arr = vec.astype(np.float32)
    return struct.pack(f"{len(arr)}f", *arr.tolist())


def _blob_to_vec(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)
