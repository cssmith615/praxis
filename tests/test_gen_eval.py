"""
Sprint C tests — GEN and EVAL.sufficient handlers.

All LLM calls are patched at _llm_call level — no network, no API key required.
Tests cover: provider routing, YES/NO parsing, stub fallback, error conditions.
"""

from __future__ import annotations

import os
import pytest
from unittest.mock import patch, MagicMock

from praxis.executor import ExecutionContext
from praxis.handlers import HANDLERS
from praxis.handlers.ai_ml import _llm_call, _llm_claude, _llm_openai, _llm_local


# ── GEN handler ───────────────────────────────────────────────────────────────

def test_gen_with_prompt_calls_llm():
    ctx = ExecutionContext()
    with patch("praxis.handlers.ai_ml._llm_call", return_value="Paris") as mock_llm:
        result = HANDLERS["GEN"](["answer"], {"prompt": "What is the capital of France?"}, ctx)
    assert result == "Paris"
    mock_llm.assert_called_once_with(
        "What is the capital of France?", "claude", None, 1024
    )


def test_gen_no_prompt_returns_stub():
    ctx = ExecutionContext()
    result = HANDLERS["GEN"](["content"], {"template": "email"}, ctx)
    assert isinstance(result, str)
    assert "Generated" in result
    assert "email" in result


def test_gen_respects_provider_and_model():
    ctx = ExecutionContext()
    with patch("praxis.handlers.ai_ml._llm_call", return_value="result") as mock_llm:
        HANDLERS["GEN"](
            ["answer"],
            {"prompt": "hi", "provider": "openai", "model": "gpt-4o", "max_tokens": "512"},
            ctx,
        )
    mock_llm.assert_called_once_with("hi", "openai", "gpt-4o", 512)


def test_gen_max_param_alias():
    """max= should work as an alias for max_tokens=."""
    ctx = ExecutionContext()
    with patch("praxis.handlers.ai_ml._llm_call", return_value="ok") as mock_llm:
        HANDLERS["GEN"](["answer"], {"prompt": "hi", "max": "256"}, ctx)
    _, _, _, max_tokens = mock_llm.call_args[0]
    assert max_tokens == 256


def test_gen_followup_query_same_impl():
    """GEN.followup_query uses the same code path as GEN.answer."""
    ctx = ExecutionContext()
    with patch("praxis.handlers.ai_ml._llm_call", return_value="search for X") as mock_llm:
        result = HANDLERS["GEN"](["followup_query"], {"prompt": "What else?"}, ctx)
    assert result == "search for X"


# ── EVAL.sufficient handler ───────────────────────────────────────────────────

def test_eval_sufficient_yes():
    ctx = ExecutionContext()
    with patch("praxis.handlers.ai_ml._llm_call", return_value="YES"):
        result = HANDLERS["EVAL"](
            ["sufficient"],
            {"prompt": "Is this enough context? YES or NO"},
            ctx,
        )
    assert result == "YES"


def test_eval_sufficient_no():
    ctx = ExecutionContext()
    with patch("praxis.handlers.ai_ml._llm_call", return_value="NO"):
        result = HANDLERS["EVAL"](
            ["sufficient"],
            {"prompt": "Is this enough context? YES or NO"},
            ctx,
        )
    assert result == "NO"


def test_eval_sufficient_parses_verbose_yes():
    """LLMs often reply with explanations — must still extract YES."""
    ctx = ExecutionContext()
    with patch("praxis.handlers.ai_ml._llm_call", return_value="YES, the context is sufficient."):
        result = HANDLERS["EVAL"](["sufficient"], {"prompt": "check"}, ctx)
    assert result == "YES"


def test_eval_sufficient_parses_verbose_no():
    ctx = ExecutionContext()
    with patch("praxis.handlers.ai_ml._llm_call", return_value="NO, more context needed."):
        result = HANDLERS["EVAL"](["sufficient"], {"prompt": "check"}, ctx)
    assert result == "NO"


def test_eval_sufficient_defaults_no_on_ambiguous():
    """If the LLM returns something unexpected, default to NO (safe for agentic loops)."""
    ctx = ExecutionContext()
    with patch("praxis.handlers.ai_ml._llm_call", return_value="Maybe"):
        result = HANDLERS["EVAL"](["sufficient"], {"prompt": "check"}, ctx)
    assert result == "NO"


def test_eval_sufficient_builds_default_prompt_from_context():
    """When no prompt= given, EVAL.sufficient builds one from ctx.last_output."""
    ctx = ExecutionContext()
    ctx.last_output = "Some context block"
    with patch("praxis.handlers.ai_ml._llm_call", return_value="YES") as mock_llm:
        HANDLERS["EVAL"](["sufficient"], {}, ctx)
    prompt_used = mock_llm.call_args[0][0]
    assert "Some context block" in prompt_used


def test_eval_sufficient_max_tokens_is_small():
    """Sufficiency check should request very few tokens — 10 by default."""
    ctx = ExecutionContext()
    with patch("praxis.handlers.ai_ml._llm_call", return_value="YES") as mock_llm:
        HANDLERS["EVAL"](["sufficient"], {"prompt": "check"}, ctx)
    _, _, _, max_tokens = mock_llm.call_args[0]
    assert max_tokens <= 10


def test_eval_other_target_returns_mock():
    """Non-sufficient EVAL targets keep mock behaviour — no LLM call."""
    ctx = ExecutionContext()
    result = HANDLERS["EVAL"](["accuracy"], {"threshold": "0.9"}, ctx)
    assert "metric" in result
    assert "value" in result


# ── _llm_call provider routing ────────────────────────────────────────────────

def test_llm_call_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        _llm_call("hi", "groq", None, 100)


def test_llm_claude_missing_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        _llm_claude("hi", "claude-haiku-4-5-20251001", 10)


def test_llm_openai_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # openai may not be installed in dev — patch the import to isolate the key check
    mock_openai_module = MagicMock()
    mock_openai_module.OpenAI = MagicMock()
    with patch.dict("sys.modules", {"openai": mock_openai_module}):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            _llm_openai("hi", "gpt-4o-mini", 10)


def test_llm_local_connection_error():
    import httpx
    with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(ConnectionError, match="localhost:11434"):
            _llm_local("hi", "llama3", 100)


def test_llm_claude_calls_anthropic_sdk(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="Hello from Claude")]
    )
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client
    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        result = _llm_claude("Say hello", "claude-haiku-4-5-20251001", 50)
    assert result == "Hello from Claude"
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
    assert call_kwargs["max_tokens"] == 50
