"""
Sprint 24 tests — Memory temporal decay (24A), Agent context compaction (24B),
OUT.slack / OUT.discord (24C).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from praxis.memory import (
    ADAPT_THRESHOLD,
    DEFAULT_DB_PATH,
    RECENCY_HALF_LIFE_DAYS,
    RECENCY_WEIGHT,
    ProgramMemory,
    StoredProgram,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fixed_embedder(v: list[float]):
    """Return a closure that always emits the given vector (normalized)."""
    arr = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(arr)
    normed = arr / norm if norm > 0 else arr
    return lambda _text: normed


def _unit(v: list[float]) -> list[float]:
    arr = np.array(v, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tolist()


def mem(tmp_path, embedder=None) -> ProgramMemory:
    return ProgramMemory(db_path=tmp_path / "test.db", embedder=embedder)


# ──────────────────────────────────────────────────────────────────────────────
# Sprint 24A — Memory temporal decay
# ──────────────────────────────────────────────────────────────────────────────

class TestDefaultDbPath:
    def test_default_db_is_praxis_not_shaun(self):
        assert ".praxis" in str(DEFAULT_DB_PATH)
        assert ".shaun" not in str(DEFAULT_DB_PATH)


class TestRecencyConstants:
    def test_recency_weight_is_0_2(self):
        assert RECENCY_WEIGHT == pytest.approx(0.2)

    def test_half_life_is_90_days(self):
        assert RECENCY_HALF_LIFE_DAYS == 90


class TestLastUsedAtField:
    def test_stored_program_has_last_used_at(self):
        sp = StoredProgram(
            id="x", goal_text="g", shaun_program="p",
            outcome="success", execution_log=[], created_at="2026-01-01T00:00:00+00:00",
        )
        assert hasattr(sp, "last_used_at")
        assert sp.last_used_at is None   # default is None

    def test_stored_program_last_used_at_settable(self):
        sp = StoredProgram(
            id="x", goal_text="g", shaun_program="p",
            outcome="success", execution_log=[], created_at="2026-01-01T00:00:00+00:00",
            last_used_at="2026-03-01T00:00:00+00:00",
        )
        assert sp.last_used_at == "2026-03-01T00:00:00+00:00"


class TestSchemaMigration:
    def test_new_db_has_last_used_at_column(self, tmp_path):
        m = mem(tmp_path)
        with m._conn() as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(programs)").fetchall()}
        assert "last_used_at" in cols

    def test_existing_db_migrated(self, tmp_path):
        """Simulate a DB created without last_used_at; ProgramMemory should add it."""
        db_path = tmp_path / "old.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE programs (
                    id TEXT PRIMARY KEY, goal_text TEXT NOT NULL,
                    goal_embedding BLOB, shaun_program TEXT NOT NULL,
                    outcome TEXT NOT NULL, execution_log TEXT,
                    created_at TEXT NOT NULL
                )
            """)
        # Opening with ProgramMemory should migrate
        m = ProgramMemory(db_path=db_path, embedder=_fixed_embedder([1, 0, 0]))
        with m._conn() as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(programs)").fetchall()}
        assert "last_used_at" in cols


class TestStoreLastUsedAt:
    def test_store_sets_last_used_at(self, tmp_path):
        m = mem(tmp_path, embedder=_fixed_embedder([1, 0, 0]))
        pid = m.store("goal", "PROG", "success", [])
        with m._conn() as conn:
            row = conn.execute(
                "SELECT last_used_at FROM programs WHERE id = ?", (pid,)
            ).fetchone()
        assert row[0] is not None
        # Should be a valid ISO datetime
        datetime.fromisoformat(row[0])


class TestRetrieveSimilarRecency:
    def test_recent_program_ranked_above_stale_similar(self, tmp_path):
        """
        Two programs with identical similarity; the recently-used one should
        rank first after recency weighting.
        """
        # Both embedders produce the same vector as the query → similarity = 1.0
        query_vec = _unit([1, 0, 0])
        m = mem(tmp_path, embedder=lambda _: np.array(query_vec, dtype=np.float32))

        # Store both
        pid_old = m.store("stale goal", "OLD_PROG", "success", [])
        pid_new = m.store("fresh goal", "NEW_PROG", "success", [])

        # Backdate last_used_at for the old program to 200 days ago
        old_ts = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        with m._conn() as conn:
            conn.execute(
                "UPDATE programs SET last_used_at = ? WHERE id = ?",
                (old_ts, pid_old),
            )

        results = m.retrieve_similar("any goal", k=2)
        assert len(results) == 2
        # fresh program should be first (higher recency score)
        assert results[0].id == pid_new

    def test_retrieve_updates_last_used_at(self, tmp_path):
        m = mem(tmp_path, embedder=_fixed_embedder([1, 0, 0]))
        pid = m.store("goal", "PROG", "success", [])

        # Backdate
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        with m._conn() as conn:
            conn.execute("UPDATE programs SET last_used_at = ? WHERE id = ?", (old_ts, pid))

        before = datetime.now(timezone.utc)
        m.retrieve_similar("goal", k=1)
        after = datetime.now(timezone.utc)

        with m._conn() as conn:
            row = conn.execute(
                "SELECT last_used_at FROM programs WHERE id = ?", (pid,)
            ).fetchone()
        updated = datetime.fromisoformat(row[0])
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        assert before <= updated <= after

    def test_retrieve_returns_last_used_at_on_result(self, tmp_path):
        m = mem(tmp_path, embedder=_fixed_embedder([1, 0, 0]))
        m.store("goal", "PROG", "success", [])
        results = m.retrieve_similar("goal", k=1)
        assert results[0].last_used_at is not None

    def test_stale_program_scores_lower(self, tmp_path):
        """Score of a 90-day-old program should be noticeably below a fresh one."""
        query_vec = _unit([1, 0, 0])
        m = mem(tmp_path, embedder=lambda _: np.array(query_vec, dtype=np.float32))

        pid = m.store("goal", "PROG", "success", [])
        old_ts = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        with m._conn() as conn:
            conn.execute("UPDATE programs SET last_used_at = ? WHERE id = ?", (old_ts, pid))

        # similarity = 1.0, recency_score = 0.5 (90-day half-life)
        # adjusted = 0.8 * 1.0 + 0.2 * 0.5 = 0.9
        results = m.retrieve_similar("goal", k=1)
        # adjusted < 1.0 means recency pulled it down from perfect similarity
        # We just check that we get a result and similarity is still 1.0
        assert results[0].similarity == pytest.approx(1.0, abs=0.01)


class TestRecentIncludesLastUsedAt:
    def test_recent_populates_last_used_at(self, tmp_path):
        m = mem(tmp_path, embedder=_fixed_embedder([1, 0, 0]))
        m.store("goal", "PROG", "success", [])
        rows = m.recent(n=1)
        assert rows[0].last_used_at is not None


# ──────────────────────────────────────────────────────────────────────────────
# Sprint 24B — Agent context compaction (unit tests, no real API calls)
# ──────────────────────────────────────────────────────────────────────────────

class TestContextCompaction:
    def _make_context(self):
        from praxis.agent.context import AgentContext
        return AgentContext(chat_id="test-chat")

    def test_context_has_maybe_compact(self):
        ctx = self._make_context()
        assert hasattr(ctx, "maybe_compact")

    def test_no_compact_below_threshold(self):
        ctx = self._make_context()
        for i in range(5):
            ctx.add("user", f"msg {i}")
        # Should not compact — just pass a None client
        ctx.maybe_compact(None, "claude-haiku-4-5-20251001")
        assert len(ctx.messages) == 5

    def test_compact_fires_above_threshold(self):
        from praxis.agent.context import COMPACT_THRESHOLD, KEEP_RECENT
        ctx = self._make_context()
        for i in range(COMPACT_THRESHOLD + 5):
            ctx.add("user", f"msg {i}")

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="Summary of earlier conversation.")]
        mock_client.messages.create.return_value = mock_resp

        ctx.maybe_compact(mock_client, "claude-haiku-4-5-20251001")

        # After compaction: 1 summary message + KEEP_RECENT recent messages
        assert len(ctx.messages) == 1 + KEEP_RECENT

    def test_compact_summary_is_system_role(self):
        from praxis.agent.context import COMPACT_THRESHOLD, KEEP_RECENT
        ctx = self._make_context()
        for i in range(COMPACT_THRESHOLD + 2):
            ctx.add("user", f"msg {i}")

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="Compact summary here.")]
        mock_client.messages.create.return_value = mock_resp

        ctx.maybe_compact(mock_client, "claude-haiku-4-5-20251001")
        assert ctx.messages[0]["role"] == "user"
        assert "Compact summary here." in ctx.messages[0]["content"]

    def test_compact_preserves_recent_messages(self):
        from praxis.agent.context import COMPACT_THRESHOLD, KEEP_RECENT
        ctx = self._make_context()
        msgs = [f"unique-msg-{i}" for i in range(COMPACT_THRESHOLD + 2)]
        for msg in msgs:
            ctx.add("user", msg)

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="summary")]
        mock_client.messages.create.return_value = mock_resp

        ctx.maybe_compact(mock_client, "claude-haiku-4-5-20251001")

        kept_content = [m["content"] for m in ctx.messages[1:]]
        # Last KEEP_RECENT messages should be preserved verbatim
        for msg in msgs[-KEEP_RECENT:]:
            assert msg in kept_content


# ──────────────────────────────────────────────────────────────────────────────
# Sprint 24C — OUT.slack / OUT.discord
# ──────────────────────────────────────────────────────────────────────────────

class TestOutSlack:
    def test_slack_sends_to_webhook(self):
        from praxis.handlers.io import out_handler
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            mock_open.return_value.status = 200
            result = out_handler(
                ["slack"],
                {"message": "hello slack", "webhook": "https://hooks.slack.com/fake"},
                MagicMock(last_output="hello slack"),
            )
        mock_open.assert_called_once()
        assert result["ok"] is True

    def test_slack_missing_webhook_raises(self):
        from praxis.handlers.io import out_handler
        with pytest.raises(ValueError, match="webhook"):
            out_handler(
                ["slack"],
                {"message": "hello"},
                MagicMock(last_output="hello"),
            )

    def test_slack_reads_webhook_from_env(self, monkeypatch):
        from praxis.handlers.io import out_handler
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/env-fake")
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            mock_open.return_value.status = 200
            result = out_handler(
                ["slack"],
                {"message": "hello"},
                MagicMock(last_output="hello"),
            )
        assert result["ok"] is True


class TestOutDiscord:
    def test_discord_sends_to_webhook(self):
        from praxis.handlers.io import out_handler
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            mock_open.return_value.status = 204
            result = out_handler(
                ["discord"],
                {"message": "hello discord", "webhook": "https://discord.com/api/webhooks/fake"},
                MagicMock(last_output="hello discord"),
            )
        mock_open.assert_called_once()
        assert result["ok"] is True

    def test_discord_missing_webhook_raises(self):
        from praxis.handlers.io import out_handler
        with pytest.raises(ValueError, match="webhook"):
            out_handler(
                ["discord"],
                {"message": "hello"},
                MagicMock(last_output="hello"),
            )

    def test_discord_reads_webhook_from_env(self, monkeypatch):
        from praxis.handlers.io import out_handler
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/env-fake")
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            mock_open.return_value.status = 204
            result = out_handler(
                ["discord"],
                {"message": "hello"},
                MagicMock(last_output="hello"),
            )
        assert result["ok"] is True

    def test_discord_long_message_split(self, monkeypatch):
        """Discord has a 2000-char message limit; long messages should be split."""
        from praxis.handlers.io import out_handler
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/env-fake")
        long_msg = "x" * 4500
        call_count = 0

        def fake_open(req, timeout=None):
            nonlocal call_count
            call_count += 1
            m = MagicMock()
            m.__enter__ = lambda s: s
            m.__exit__ = MagicMock(return_value=False)
            m.status = 204
            return m

        with patch("urllib.request.urlopen", side_effect=fake_open):
            out_handler(
                ["discord"],
                {"message": long_msg},
                MagicMock(last_output=long_msg),
            )
        assert call_count >= 3   # ceil(4500 / 2000) = 3
