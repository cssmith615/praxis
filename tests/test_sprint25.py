"""
Sprint 25 tests — Multi-tier model routing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from praxis.agent.router import (
    FAST_MAX_CHARS,
    ModelRouter,
    RouteDecision,
    _DEFAULT_FAST_MODEL,
    _DEFAULT_FULL_MODEL,
)


# ──────────────────────────────────────────────────────────────────────────────
# ModelRouter — defaults
# ──────────────────────────────────────────────────────────────────────────────

class TestRouterDefaults:
    def test_default_fast_model(self):
        r = ModelRouter()
        assert r.fast_model == _DEFAULT_FAST_MODEL

    def test_default_full_model(self):
        r = ModelRouter()
        assert r.full_model == _DEFAULT_FULL_MODEL

    def test_fast_model_is_haiku(self):
        assert "haiku" in _DEFAULT_FAST_MODEL

    def test_full_model_is_sonnet(self):
        assert "sonnet" in _DEFAULT_FULL_MODEL

    def test_custom_models(self):
        r = ModelRouter(fast_model="my-fast", full_model="my-full")
        assert r.fast_model == "my-fast"
        assert r.full_model == "my-full"

    def test_env_var_fast_model(self, monkeypatch):
        monkeypatch.setenv("PRAXIS_FAST_MODEL", "env-fast-model")
        r = ModelRouter()
        assert r.fast_model == "env-fast-model"

    def test_env_var_full_model(self, monkeypatch):
        monkeypatch.setenv("PRAXIS_FULL_MODEL", "env-full-model")
        r = ModelRouter()
        assert r.full_model == "env-full-model"


# ──────────────────────────────────────────────────────────────────────────────
# ModelRouter — disabled
# ──────────────────────────────────────────────────────────────────────────────

class TestRouterDisabled:
    def test_disabled_always_returns_full(self):
        r = ModelRouter(enabled=False)
        for msg in ["run LOG.msg", "hello", "plan a complex workflow with multiple steps"]:
            d = r.route(msg)
            assert d.tier == "full"
            assert d.model == r.full_model

    def test_disabled_reason(self):
        r = ModelRouter(enabled=False)
        d = r.route("run LOG.msg")
        assert "disabled" in d.reason


# ──────────────────────────────────────────────────────────────────────────────
# ModelRouter — fast tier (simple commands)
# ──────────────────────────────────────────────────────────────────────────────

class TestFastRouting:
    def setup_method(self):
        self.r = ModelRouter()

    def _fast(self, msg: str) -> RouteDecision:
        d = self.r.route(msg)
        assert d.tier == "fast", f"Expected fast for: {msg!r}, got: {d.reason}"
        return d

    def test_run_prefix(self):
        self._fast("run LOG.msg(msg=hello)")

    def test_validate_prefix(self):
        self._fast("validate ING.sales.db -> SUMM.text")

    def test_list_prefix(self):
        self._fast("list my schedules")

    def test_recall_prefix(self):
        self._fast("recall programs similar to sales")

    def test_remove_schedule_prefix(self):
        self._fast("remove schedule sched-abc123")

    def test_cancel_schedule_prefix(self):
        self._fast("cancel schedule sched-xyz")

    def test_hello(self):
        self._fast("hello")

    def test_hi(self):
        self._fast("hi there")

    def test_short_message_no_keywords(self):
        self._fast("what did my last program do?")

    def test_returns_fast_model(self):
        d = self._fast("run LOG.msg")
        assert d.model == self.r.fast_model


# ──────────────────────────────────────────────────────────────────────────────
# ModelRouter — full tier (complex requests)
# ──────────────────────────────────────────────────────────────────────────────

class TestFullRouting:
    def setup_method(self):
        self.r = ModelRouter()

    def _full(self, msg: str) -> RouteDecision:
        d = self.r.route(msg)
        assert d.tier == "full", f"Expected full for: {msg!r}, got: {d.reason}"
        return d

    def test_goal_keyword(self):
        self._full("plan a goal: fetch top news and summarize")

    def test_plan_keyword(self):
        self._full("plan a workflow that checks my email daily")

    def test_schedule_keyword(self):
        self._full("schedule this every morning at 8am")

    def test_every_keyword(self):
        self._full("run this every hour")

    def test_daily_keyword(self):
        self._full("send me a daily brief")

    def test_long_message(self):
        long = "what is the status of " + "x" * (FAST_MAX_CHARS + 10)
        self._full(long)

    def test_create_keyword(self):
        self._full("create a workflow that fetches data from the API")

    def test_returns_full_model(self):
        d = self._full("plan goal: summarize my sales pipeline")
        assert d.model == self.r.full_model


# ──────────────────────────────────────────────────────────────────────────────
# RouteDecision shape
# ──────────────────────────────────────────────────────────────────────────────

class TestRouteDecision:
    def test_has_model_tier_reason(self):
        r = ModelRouter()
        d = r.route("run LOG.msg")
        assert hasattr(d, "model")
        assert hasattr(d, "tier")
        assert hasattr(d, "reason")
        assert d.tier in ("fast", "full")
        assert d.model
        assert d.reason


# ──────────────────────────────────────────────────────────────────────────────
# PraxisAgent integration — router wired in
# ──────────────────────────────────────────────────────────────────────────────

class TestAgentUsesRouter:
    def _make_agent(self, **kwargs):
        from praxis.agent.core import PraxisAgent
        with patch("anthropic.Anthropic"):
            agent = PraxisAgent(
                model="claude-sonnet-4-6",
                api_key="test",
                **kwargs,
            )
        return agent

    def test_agent_has_router(self):
        agent = self._make_agent()
        assert hasattr(agent, "_router")

    def test_agent_router_enabled_by_default(self):
        agent = self._make_agent()
        assert agent._router.enabled is True

    def test_agent_router_disabled_with_flag(self):
        agent = self._make_agent(router_enabled=False)
        assert agent._router.enabled is False

    def test_agent_custom_fast_model(self):
        agent = self._make_agent(fast_model="my-fast-model")
        assert agent._router.fast_model == "my-fast-model"

    def test_simple_message_routed_to_fast_model(self):
        from praxis.agent.context import AgentContext

        agent = self._make_agent(
            fast_model="haiku-test",
        )

        # Track which model was called
        called_models = []

        def fake_create(**kwargs):
            called_models.append(kwargs.get("model"))
            resp = MagicMock()
            resp.stop_reason = "end_turn"
            resp.content = [MagicMock(text="ok", spec=["text"])]
            return resp

        agent._client.messages.create = fake_create

        ctx = AgentContext(chat_id="test")
        agent.chat("run LOG.msg", ctx)

        assert called_models[0] == "haiku-test"

    def test_complex_message_routed_to_full_model(self):
        from praxis.agent.context import AgentContext

        agent = self._make_agent(
            fast_model="haiku-test",
        )
        agent.model = "sonnet-test"
        agent._router.full_model = "sonnet-test"

        called_models = []

        def fake_create(**kwargs):
            called_models.append(kwargs.get("model"))
            resp = MagicMock()
            resp.stop_reason = "end_turn"
            resp.content = [MagicMock(text="ok", spec=["text"])]
            return resp

        agent._client.messages.create = fake_create

        ctx = AgentContext(chat_id="test")
        agent.chat("plan goal: fetch and summarize my sales pipeline data", ctx)

        assert called_models[0] == "sonnet-test"
