"""
Praxis Provider Abstraction — Sprint 8

A thin interface so the Planner can use any LLM backend without
changing planning logic.

Available providers
-------------------
AnthropicProvider    — Anthropic API (claude-*)
                       Requires: pip install anthropic
                       Key env:  ANTHROPIC_API_KEY

OpenAIProvider       — OpenAI or any OpenAI-compatible endpoint
                       (OpenAI, Groq, Azure OpenAI, LM Studio, Grok, Gemini, etc.)
                       Uses httpx — no `openai` package required.
                       Key env:  OPENAI_API_KEY  (or pass api_key=)
                       URL env:  OPENAI_BASE_URL (default: api.openai.com/v1)

OllamaProvider       — Local Ollama server, no API key.
                       Uses httpx — zero extra dependencies.
                       URL env:  OLLAMA_BASE_URL (default: localhost:11434)

GrokProvider         — xAI Grok (OpenAI-compatible, convenience wrapper)
                       Key env:  GROK_API_KEY  (or XAI_API_KEY)

GeminiProvider       — Google Gemini (OpenAI-compatible endpoint, convenience wrapper)
                       Key env:  GEMINI_API_KEY

Auto-resolution
---------------
resolve_provider() picks a provider in priority order:
  1. Explicit `provider=` argument
  2. PRAXIS_PROVIDER env var  ("anthropic" | "openai" | "ollama")
  3. First available API key  (ANTHROPIC_API_KEY > OPENAI_API_KEY)
  4. Ollama as local fallback

Usage
-----
    from praxis.providers import resolve_provider

    provider = resolve_provider()                          # auto
    provider = resolve_provider("ollama", model="llama3.2")
    provider = resolve_provider("openai", model="gpt-4o")
    provider = resolve_provider("anthropic", model="claude-haiku-4-5-20251001")

    text = provider.complete(system="...", user="...")

Planner integration
-------------------
    from praxis.planner import Planner
    from praxis.providers import resolve_provider

    planner = Planner(memory=..., constitution=..., provider=resolve_provider())
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class Provider(ABC):
    """
    Minimal interface every LLM backend must implement.

    complete() takes a system prompt and a user message and returns
    the assistant's response as a plain string.
    """

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Call the model and return the response text."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Human-readable model identifier for logging / display."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model_id!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic
# ─────────────────────────────────────────────────────────────────────────────

class AnthropicProvider(Provider):
    """
    Anthropic Messages API.

    Parameters
    ----------
    model : str
        Any claude-* model ID.
    api_key : str | None
        Falls back to ANTHROPIC_API_KEY env var.
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def model_id(self) -> str:
        return self._model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        client = self._get_client()
        message = client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text.strip()

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package required. Run: pip install 'praxis-lang[ai]'"
            ) from exc
        if not self._api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. "
                "Export the key or pass api_key= to AnthropicProvider."
            )
        self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible  (OpenAI, Groq, Azure, LM Studio, …)
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIProvider(Provider):
    """
    OpenAI Chat Completions API — or any compatible endpoint.

    Works with:
      - OpenAI          base_url="https://api.openai.com/v1"
      - Groq            base_url="https://api.groq.com/openai/v1"
      - Azure OpenAI    base_url="https://<resource>.openai.azure.com/openai/deployments/<model>"
      - LM Studio       base_url="http://localhost:1234/v1"  api_key="lm-studio"
      - Any OpenAI-compatible local server

    Uses httpx directly — the `openai` Python package is NOT required.

    Parameters
    ----------
    model : str
        Model ID (e.g. "gpt-4o", "llama-3.1-70b-versatile" on Groq).
    api_key : str | None
        Falls back to OPENAI_API_KEY env var.
    base_url : str | None
        API base URL. Falls back to OPENAI_BASE_URL env var, then OpenAI default.
    """

    DEFAULT_MODEL   = "gpt-4o"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model    = model
        self._api_key  = api_key  or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = (base_url or os.environ.get("OPENAI_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")

    @property
    def model_id(self) -> str:
        return self._model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError("httpx is required. Run: pip install httpx") from exc

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        payload = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }

        resp = httpx.post(
            f"{self._base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Ollama  (local, zero extra deps)
# ─────────────────────────────────────────────────────────────────────────────

class OllamaProvider(Provider):
    """
    Local Ollama server — no API key, no extra packages.

    Requires Ollama running: https://ollama.com
    Pull a model first:  ollama pull llama3.2

    Parameters
    ----------
    model : str
        Ollama model name (e.g. "llama3.2", "mistral", "gemma3", "phi4").
    base_url : str
        Ollama server URL. Falls back to OLLAMA_BASE_URL env var,
        then "http://localhost:11434".
    """

    DEFAULT_MODEL   = "llama3.2"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
    ) -> None:
        self._model    = model
        self._base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")

    @property
    def model_id(self) -> str:
        return self._model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError("httpx is required. Run: pip install httpx") from exc

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }

        resp = httpx.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=120.0,   # local models can be slow on first run
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()

    def list_models(self) -> list[str]:
        """Return model names available on the local Ollama server."""
        try:
            import httpx
            resp = httpx.get(f"{self._base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []


# ─────────────────────────────────────────────────────────────────────────────
# Grok  (xAI — OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────

class GrokProvider(OpenAIProvider):
    """
    xAI Grok — convenience wrapper around OpenAIProvider.

    xAI's API is OpenAI-compatible; this sets the right base URL and
    reads GROK_API_KEY (or XAI_API_KEY) automatically.

    Parameters
    ----------
    model : str
        Grok model ID (e.g. "grok-3", "grok-3-mini").
    api_key : str | None
        Falls back to GROK_API_KEY or XAI_API_KEY env vars.
    """

    DEFAULT_MODEL    = "grok-3-mini"
    DEFAULT_BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> None:
        key = (
            api_key
            or os.environ.get("GROK_API_KEY")
            or os.environ.get("XAI_API_KEY", "")
        )
        super().__init__(model=model, api_key=key, base_url=self.DEFAULT_BASE_URL)


# ─────────────────────────────────────────────────────────────────────────────
# Gemini  (Google — OpenAI-compatible endpoint)
# ─────────────────────────────────────────────────────────────────────────────

class GeminiProvider(OpenAIProvider):
    """
    Google Gemini via its OpenAI-compatible REST endpoint.

    No extra packages needed — uses the same httpx-based OpenAIProvider.
    Requires a Gemini API key from https://aistudio.google.com/

    Parameters
    ----------
    model : str
        Gemini model ID (e.g. "gemini-2.0-flash", "gemini-1.5-pro").
    api_key : str | None
        Falls back to GEMINI_API_KEY env var.
    """

    DEFAULT_MODEL    = "gemini-2.0-flash"
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        super().__init__(model=model, api_key=key, base_url=self.DEFAULT_BASE_URL)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-resolution
# ─────────────────────────────────────────────────────────────────────────────

_PROVIDER_ALIASES: dict[str, str] = {
    "anthropic": "anthropic",
    "claude":    "anthropic",
    "openai":    "openai",
    "oai":       "openai",
    "gpt":       "openai",
    "ollama":    "ollama",
    "local":     "ollama",
    "grok":      "grok",
    "xai":       "grok",
    "gemini":    "gemini",
    "google":    "gemini",
}


def resolve_provider(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Provider:
    """
    Resolve and instantiate a Provider.

    Priority:
      1. Explicit `provider` argument (or PRAXIS_PROVIDER env var)
      2. First detected API key  (ANTHROPIC_API_KEY > OPENAI_API_KEY)
      3. Ollama as local fallback

    Parameters
    ----------
    provider : str | None
        Provider name: "anthropic", "openai", "ollama" (and aliases).
        If None, reads PRAXIS_PROVIDER env var, then auto-detects.
    model : str | None
        Override default model for the chosen provider.
    api_key : str | None
        API key override (passed through to provider).
    base_url : str | None
        Base URL override (used for OpenAI-compatible and Ollama providers).
    """
    name = provider or os.environ.get("PRAXIS_PROVIDER")
    if name:
        name = _PROVIDER_ALIASES.get(name.lower(), name.lower())
    else:
        # Auto-detect from environment — first key wins
        if os.environ.get("ANTHROPIC_API_KEY") or api_key:
            name = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            name = "openai"
        elif os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY"):
            name = "grok"
        elif os.environ.get("GEMINI_API_KEY"):
            name = "gemini"
        else:
            name = "ollama"

    if name == "anthropic":
        return AnthropicProvider(
            model=model or AnthropicProvider.DEFAULT_MODEL,
            api_key=api_key,
        )
    if name == "openai":
        return OpenAIProvider(
            model=model or OpenAIProvider.DEFAULT_MODEL,
            api_key=api_key,
            base_url=base_url,
        )
    if name == "ollama":
        return OllamaProvider(
            model=model or OllamaProvider.DEFAULT_MODEL,
            base_url=base_url,
        )
    if name == "grok":
        return GrokProvider(
            model=model or GrokProvider.DEFAULT_MODEL,
            api_key=api_key,
        )
    if name == "gemini":
        return GeminiProvider(
            model=model or GeminiProvider.DEFAULT_MODEL,
            api_key=api_key,
        )

    raise ValueError(
        f"Unknown provider {name!r}. "
        f"Choose from: anthropic, openai, ollama, grok, gemini"
    )
