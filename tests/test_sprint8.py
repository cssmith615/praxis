"""
Sprint 8 tests — provider abstraction: Provider interface, all concrete
providers, resolve_provider(), and Planner integration.

No real API calls are made — providers are tested with mocked HTTP and
injected clients.
"""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from praxis.providers import (
    Provider,
    AnthropicProvider,
    OpenAIProvider,
    OllamaProvider,
    GrokProvider,
    GeminiProvider,
    resolve_provider,
    _PROVIDER_ALIASES,
)
from praxis.planner import Planner
from praxis.memory import ProgramMemory
from praxis.constitution import Constitution


# ─── helpers ─────────────────────────────────────────────────────────────────

class _EchoProvider(Provider):
    """Test provider that echoes the user message back."""
    def __init__(self, model: str = "echo-1"):
        self._model = model

    @property
    def model_id(self) -> str:
        return self._model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        return f"ECHO: {user[:40]}"


def _mock_response(text: str):
    """Build a mock httpx response with a chat completion payload."""
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "choices": [{"message": {"content": text}}]
    }
    return mock


def _mock_ollama_response(text: str):
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {"message": {"content": text}}
    return mock


# ══════════════════════════════════════════════════════════════════════════════
# Provider interface
# ══════════════════════════════════════════════════════════════════════════════

class TestProviderInterface:
    def test_echo_provider_is_provider(self):
        p = _EchoProvider()
        assert isinstance(p, Provider)

    def test_echo_provider_complete(self):
        p = _EchoProvider()
        result = p.complete("sys", "hello")
        assert "ECHO" in result

    def test_provider_repr(self):
        p = _EchoProvider("my-model")
        assert "my-model" in repr(p)

    def test_abstract_provider_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Provider()


# ══════════════════════════════════════════════════════════════════════════════
# AnthropicProvider
# ══════════════════════════════════════════════════════════════════════════════

class TestAnthropicProvider:
    def test_model_id(self):
        p = AnthropicProvider(model="claude-haiku-4-5-20251001", api_key="test")
        assert p.model_id == "claude-haiku-4-5-20251001"

    def test_complete_calls_client(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="LOG.step1")]
        )
        p = AnthropicProvider(api_key="test-key")
        p._client = mock_client

        result = p.complete("sys prompt", "user msg")
        assert result == "LOG.step1"
        mock_client.messages.create.assert_called_once()

    def test_complete_passes_system_and_user(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="ING.x")]
        )
        p = AnthropicProvider(api_key="test-key")
        p._client = mock_client

        p.complete("my system", "my user", max_tokens=512)
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "my system"
        assert call_kwargs["messages"][0]["content"] == "my user"
        assert call_kwargs["max_tokens"] == 512

    def test_missing_api_key_raises(self):
        p = AnthropicProvider(api_key=None)
        p._api_key = None  # ensure not set
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises((EnvironmentError, ImportError)):
                p.complete("sys", "user")

    def test_reads_api_key_from_env(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}):
            p = AnthropicProvider()
            assert p._api_key == "env-key"


# ══════════════════════════════════════════════════════════════════════════════
# OpenAIProvider
# ══════════════════════════════════════════════════════════════════════════════

class TestOpenAIProvider:
    def test_model_id(self):
        p = OpenAIProvider(model="gpt-4o-mini", api_key="x")
        assert p.model_id == "gpt-4o-mini"

    def test_default_base_url(self):
        p = OpenAIProvider(api_key="x")
        assert "openai.com" in p._base_url

    def test_custom_base_url(self):
        p = OpenAIProvider(api_key="x", base_url="http://localhost:1234/v1")
        assert p._base_url == "http://localhost:1234/v1"

    def test_base_url_from_env(self):
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://api.groq.com/openai/v1"}):
            p = OpenAIProvider(api_key="x")
            assert "groq.com" in p._base_url

    def test_complete_posts_to_chat_completions(self):
        p = OpenAIProvider(model="gpt-4o", api_key="test-key")
        with patch("httpx.post", return_value=_mock_response("LOG.step")) as mock_post:
            result = p.complete("sys", "user")
        assert result == "LOG.step"
        url = mock_post.call_args[0][0]
        assert "/chat/completions" in url

    def test_complete_sends_bearer_auth(self):
        p = OpenAIProvider(model="gpt-4o", api_key="sk-abc")
        with patch("httpx.post", return_value=_mock_response("x")) as mock_post:
            p.complete("s", "u")
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer sk-abc"

    def test_complete_sends_system_and_user(self):
        p = OpenAIProvider(model="gpt-4o", api_key="test")
        with patch("httpx.post", return_value=_mock_response("x")) as mock_post:
            p.complete("my-sys", "my-user", max_tokens=256)
        payload = mock_post.call_args[1]["json"]
        msgs = payload["messages"]
        assert msgs[0] == {"role": "system", "content": "my-sys"}
        assert msgs[1] == {"role": "user",   "content": "my-user"}
        assert payload["max_tokens"] == 256

    def test_reads_api_key_from_env(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-openai-key"}):
            p = OpenAIProvider()
            assert p._api_key == "env-openai-key"


# ══════════════════════════════════════════════════════════════════════════════
# OllamaProvider
# ══════════════════════════════════════════════════════════════════════════════

class TestOllamaProvider:
    def test_model_id(self):
        p = OllamaProvider(model="mistral")
        assert p.model_id == "mistral"

    def test_default_base_url(self):
        p = OllamaProvider()
        assert "localhost:11434" in p._base_url

    def test_base_url_from_env(self):
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://192.168.1.5:11434"}):
            p = OllamaProvider()
            assert "192.168.1.5" in p._base_url

    def test_complete_posts_to_api_chat(self):
        p = OllamaProvider(model="llama3.2")
        with patch("httpx.post", return_value=_mock_ollama_response("ING.x")) as mock_post:
            result = p.complete("sys", "user")
        assert result == "ING.x"
        url = mock_post.call_args[0][0]
        assert "/api/chat" in url

    def test_complete_sends_system_and_user(self):
        p = OllamaProvider(model="llama3.2")
        with patch("httpx.post", return_value=_mock_ollama_response("x")) as mock_post:
            p.complete("sys-msg", "user-msg")
        payload = mock_post.call_args[1]["json"]
        msgs = payload["messages"]
        assert msgs[0] == {"role": "system", "content": "sys-msg"}
        assert msgs[1] == {"role": "user",   "content": "user-msg"}

    def test_no_stream_in_payload(self):
        p = OllamaProvider()
        with patch("httpx.post", return_value=_mock_ollama_response("x")) as mock_post:
            p.complete("s", "u")
        payload = mock_post.call_args[1]["json"]
        assert payload["stream"] is False

    def test_list_models_returns_names(self):
        p = OllamaProvider()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "llama3.2"}, {"name": "mistral"}]
        }
        with patch("httpx.get", return_value=mock_resp):
            models = p.list_models()
        assert "llama3.2" in models
        assert "mistral" in models

    def test_list_models_returns_empty_on_failure(self):
        p = OllamaProvider()
        with patch("httpx.get", side_effect=Exception("connection refused")):
            assert p.list_models() == []


# ══════════════════════════════════════════════════════════════════════════════
# GrokProvider
# ══════════════════════════════════════════════════════════════════════════════

class TestGrokProvider:
    def test_model_id(self):
        p = GrokProvider(model="grok-3", api_key="test")
        assert p.model_id == "grok-3"

    def test_default_model(self):
        p = GrokProvider(api_key="test")
        assert "grok" in p.model_id

    def test_uses_xai_base_url(self):
        p = GrokProvider(api_key="test")
        assert "x.ai" in p._base_url

    def test_reads_grok_api_key_from_env(self):
        with patch.dict(os.environ, {"GROK_API_KEY": "grok-secret"}):
            p = GrokProvider()
            assert p._api_key == "grok-secret"

    def test_reads_xai_api_key_from_env(self):
        with patch.dict(os.environ, {"XAI_API_KEY": "xai-secret"}, clear=False):
            p = GrokProvider()
            assert p._api_key == "xai-secret"

    def test_complete_delegates_to_openai_provider(self):
        p = GrokProvider(api_key="test")
        with patch("httpx.post", return_value=_mock_response("LOG.x")) as mock_post:
            result = p.complete("s", "u")
        assert result == "LOG.x"


# ══════════════════════════════════════════════════════════════════════════════
# GeminiProvider
# ══════════════════════════════════════════════════════════════════════════════

class TestGeminiProvider:
    def test_model_id(self):
        p = GeminiProvider(model="gemini-1.5-pro", api_key="test")
        assert p.model_id == "gemini-1.5-pro"

    def test_default_model(self):
        p = GeminiProvider(api_key="test")
        assert "gemini" in p.model_id

    def test_uses_google_base_url(self):
        p = GeminiProvider(api_key="test")
        assert "googleapis.com" in p._base_url

    def test_reads_gemini_api_key_from_env(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-secret"}):
            p = GeminiProvider()
            assert p._api_key == "gemini-secret"

    def test_complete_delegates_to_openai_provider(self):
        p = GeminiProvider(api_key="test")
        with patch("httpx.post", return_value=_mock_response("ING.x")) as mock_post:
            result = p.complete("s", "u")
        assert result == "ING.x"


# ══════════════════════════════════════════════════════════════════════════════
# resolve_provider
# ══════════════════════════════════════════════════════════════════════════════

class TestResolveProvider:
    def _clean_env(self):
        """Return a dict with all provider keys removed."""
        return {k: v for k, v in os.environ.items()
                if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                             "GROK_API_KEY", "XAI_API_KEY", "GEMINI_API_KEY",
                             "PRAXIS_PROVIDER")}

    def test_explicit_anthropic(self):
        p = resolve_provider("anthropic", model="claude-haiku-4-5-20251001")
        assert isinstance(p, AnthropicProvider)
        assert p.model_id == "claude-haiku-4-5-20251001"

    def test_explicit_openai(self):
        p = resolve_provider("openai", model="gpt-4o-mini")
        assert isinstance(p, OpenAIProvider)

    def test_explicit_ollama(self):
        p = resolve_provider("ollama", model="mistral")
        assert isinstance(p, OllamaProvider)
        assert p.model_id == "mistral"

    def test_explicit_grok(self):
        p = resolve_provider("grok")
        assert isinstance(p, GrokProvider)

    def test_explicit_gemini(self):
        p = resolve_provider("gemini")
        assert isinstance(p, GeminiProvider)

    def test_alias_claude(self):
        p = resolve_provider("claude")
        assert isinstance(p, AnthropicProvider)

    def test_alias_gpt(self):
        p = resolve_provider("gpt")
        assert isinstance(p, OpenAIProvider)

    def test_alias_local(self):
        p = resolve_provider("local")
        assert isinstance(p, OllamaProvider)

    def test_alias_xai(self):
        p = resolve_provider("xai")
        assert isinstance(p, GrokProvider)

    def test_alias_google(self):
        p = resolve_provider("google")
        assert isinstance(p, GeminiProvider)

    def test_auto_detects_anthropic_from_env(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}, clear=False):
            p = resolve_provider()
        assert isinstance(p, AnthropicProvider)

    def test_auto_detects_openai_from_env(self):
        env = self._clean_env()
        env["OPENAI_API_KEY"] = "test"
        with patch.dict(os.environ, env, clear=True):
            p = resolve_provider()
        assert isinstance(p, OpenAIProvider)

    def test_auto_detects_grok_from_env(self):
        env = self._clean_env()
        env["GROK_API_KEY"] = "test"
        with patch.dict(os.environ, env, clear=True):
            p = resolve_provider()
        assert isinstance(p, GrokProvider)

    def test_auto_detects_gemini_from_env(self):
        env = self._clean_env()
        env["GEMINI_API_KEY"] = "test"
        with patch.dict(os.environ, env, clear=True):
            p = resolve_provider()
        assert isinstance(p, GeminiProvider)

    def test_falls_back_to_ollama(self):
        env = self._clean_env()
        with patch.dict(os.environ, env, clear=True):
            p = resolve_provider()
        assert isinstance(p, OllamaProvider)

    def test_praxis_provider_env_var(self):
        with patch.dict(os.environ, {"PRAXIS_PROVIDER": "ollama"}, clear=False):
            p = resolve_provider()
        assert isinstance(p, OllamaProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            resolve_provider("badprovider")

    def test_model_override(self):
        p = resolve_provider("ollama", model="phi4")
        assert p.model_id == "phi4"


# ══════════════════════════════════════════════════════════════════════════════
# Planner integration
# ══════════════════════════════════════════════════════════════════════════════

class TestPlannerProviderIntegration:
    def _planner(self, provider, tmp_path):
        mem = ProgramMemory(db_path=str(tmp_path / "mem.db"))
        const = Constitution(tmp_path / "const.md")
        (tmp_path / "const.md").write_text("# test\n")
        return Planner(memory=mem, constitution=const, provider=provider)

    @pytest.fixture(autouse=True)
    def _no_embeddings(self):
        """Patch out sentence_transformers so CI (no [memory] extra) works."""
        with patch("praxis.memory.ProgramMemory.should_adapt", return_value=(False, [])):
            yield

    def test_planner_accepts_provider(self, tmp_path):
        p = _EchoProvider()
        planner = self._planner(p, tmp_path)
        assert planner._provider is p

    def test_planner_model_property_reflects_provider(self, tmp_path):
        p = _EchoProvider("my-custom-model")
        planner = self._planner(p, tmp_path)
        assert planner.model == "my-custom-model"

    def test_planner_uses_provider_complete(self, tmp_path):
        # Provider returns a valid minimal program
        class _FixedProvider(Provider):
            @property
            def model_id(self): return "fixed"
            def complete(self, system, user, max_tokens=1024):
                return "LOG.test"

        planner = self._planner(_FixedProvider(), tmp_path)
        result = planner.plan("log something")
        assert result.program == "LOG.test"
        assert result.attempts == 1

    def test_planner_retries_on_invalid_program(self, tmp_path):
        attempts = []

        class _RetryProvider(Provider):
            @property
            def model_id(self): return "retry"
            def complete(self, system, user, max_tokens=1024):
                attempts.append(1)
                if len(attempts) < 2:
                    return "NOT VALID !!!"   # bad first attempt
                return "LOG.fixed"

        planner = self._planner(_RetryProvider(), tmp_path)
        result = planner.plan("anything")
        assert result.program == "LOG.fixed"
        assert result.attempts == 2

    def test_planner_legacy_client_compat(self, tmp_path):
        """Passing client= (old API) still works via _LegacyClientProvider."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="LOG.legacy")]
        )
        mem = ProgramMemory(db_path=str(tmp_path / "mem.db"))
        const = Constitution(tmp_path / "const.md")
        (tmp_path / "const.md").write_text("# test\n")

        planner = Planner(memory=mem, constitution=const, client=mock_client)
        result = planner.plan("test goal")
        assert result.program == "LOG.legacy"

    def test_planner_openai_provider(self, tmp_path):
        p = OpenAIProvider(model="gpt-4o", api_key="test")
        with patch("httpx.post", return_value=_mock_response("ING.data -> SUMM.text")):
            planner = self._planner(p, tmp_path)
            result = planner.plan("summarize data")
        assert "ING" in result.program

    def test_planner_ollama_provider(self, tmp_path):
        p = OllamaProvider(model="llama3.2")
        with patch("httpx.post", return_value=_mock_ollama_response("LOG.result")):
            planner = self._planner(p, tmp_path)
            result = planner.plan("log something")
        assert result.program == "LOG.result"
