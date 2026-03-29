"""
AI/ML handlers: TRN, INF, EVAL, SUMM, CLASS, GEN, EMBED, SEARCH

Sprint 1: stubs for TRN, INF, EVAL, SUMM, CLASS, GEN.
Sprint A: EMBED.text and SEARCH.semantic wired to EmbeddingsDB.
          Other targets fall through to stubs.
Sprint C: GEN (all targets) and EVAL.sufficient wired to real LLM providers.
          Provider routing: claude (default) | openai | local (Ollama).
          API keys sourced from env vars only — never from params.
"""

import os
from typing import Any

from praxis.embeddings import EmbeddingsDB


# ── LLM provider routing ──────────────────────────────────────────────────────

_DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_DEFAULT_LOCAL_MODEL  = "llama3"
_DEFAULT_MAX_TOKENS   = 1024


def _llm_call(prompt: str, provider: str, model: str | None, max_tokens: int) -> str:
    """Route a text-generation call to the configured LLM provider."""
    if provider == "claude":
        return _llm_claude(prompt, model or _DEFAULT_CLAUDE_MODEL, max_tokens)
    if provider == "openai":
        return _llm_openai(prompt, model or _DEFAULT_OPENAI_MODEL, max_tokens)
    if provider == "local":
        return _llm_local(prompt, model or _DEFAULT_LOCAL_MODEL, max_tokens)
    raise ValueError(
        f"Unknown LLM provider: {provider!r}. "
        "Use provider=claude (default), provider=openai, or provider=local (Ollama)."
    )


def _llm_claude(prompt: str, model: str, max_tokens: int) -> str:
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "GEN/EVAL requires the Anthropic SDK: pip install praxis-lang[ai]"
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Set it before running GEN or EVAL.sufficient."
        )
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _llm_openai(prompt: str, model: str, max_tokens: int) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "GEN with provider=openai requires the OpenAI SDK: pip install openai"
        )
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it before running GEN or EVAL.sufficient with provider=openai."
        )
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _llm_local(prompt: str, model: str, max_tokens: int) -> str:
    """Call a local Ollama instance (http://localhost:11434)."""
    import httpx  # already a core dep
    try:
        resp = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except httpx.ConnectError:
        raise ConnectionError(
            "Cannot reach Ollama at localhost:11434. "
            "Start it with: ollama serve"
        )


def trn_handler(target: list[str], params: dict, ctx) -> Any:
    """TRN — Train model. Returns mock model artifact."""
    arch = target[0] if target else "model"
    epochs = params.get("ep", params.get("epochs", 10))
    return {"model": arch, "epochs": epochs, "rmse": 0.15, "status": "trained"}


def inf_handler(target: list[str], params: dict, ctx) -> Any:
    """INF — Run inference. Returns mock prediction."""
    return {"predictions": [0.82, 0.91, 0.74], "model": ".".join(target)}


def eval_handler(target: list[str], params: dict, ctx) -> Any:
    """EVAL — Evaluate / score.

    EVAL.sufficient(prompt="...", provider=claude, model=..., max_tokens=10)
      LLM-graded sufficiency check for agentic RAG loops.
      Returns "YES" or "NO" (always uppercase, trimmed).

    Other targets: returns mock metric dict.
    """
    if target and target[0] == "sufficient":
        prompt = params.get("prompt")
        if not prompt:
            context = str(ctx.last_output) if ctx.last_output else ""
            prompt = (
                "Is the following context sufficient to fully answer the question?\n"
                f"Context: {context}\n"
                "Reply with only YES or NO."
            )
        provider  = params.get("provider", "claude")
        model     = params.get("model") or None
        max_tokens = int(params.get("max_tokens", 10))
        raw = _llm_call(prompt, provider, model, max_tokens).strip().upper()
        return "YES" if "YES" in raw else "NO"

    # Existing mock behaviour for other EVAL targets
    metric = target[0] if target else "score"
    threshold = params.get("threshold")
    value = 0.88
    result: dict = {"metric": metric, "value": value}
    if threshold is not None:
        result["passed"] = value >= float(str(threshold).rstrip("%")) / 100 \
            if "%" in str(threshold) else value >= float(threshold)
    return result


def summ_handler(target: list[str], params: dict, ctx) -> Any:
    """SUMM — Summarize. Returns mock summary string."""
    data = ctx.last_output
    max_len = params.get("max", 200)
    if isinstance(data, str):
        return data[:max_len]
    return f"Summary of {'.'.join(target)}: {str(data)[:max_len]}"


def class_handler(target: list[str], params: dict, ctx) -> Any:
    """CLASS — Classify. Returns mock class label."""
    return {"label": "positive", "confidence": 0.91}


def gen_handler(target: list[str], params: dict, ctx) -> Any:
    """GEN — Generate text via LLM.

    GEN.<target>(prompt="...", provider=claude, model=..., max_tokens=1024)
      Sends prompt to the configured provider and returns the generated string.
      The executor interpolates $variables in params before this handler runs,
      so prompts arrive fully resolved.

    If no prompt= is given, falls back to a stub string (supports legacy tests).
    """
    prompt = params.get("prompt")
    if not prompt:
        template = params.get("template", "default")
        return f"[Generated content — template={template}, target={'.'.join(target)}]"

    provider   = params.get("provider", "claude")
    model      = params.get("model") or None
    max_tokens = int(params.get("max_tokens", params.get("max", _DEFAULT_MAX_TOKENS)))
    return _llm_call(prompt, provider, model, max_tokens)


def embed_handler(target: list[str], params: dict, ctx) -> Any:
    """EMBED — Create embeddings.

    EMBED.text(corpus=name, provider=local|voyage|openai)
      Reads ctx.last_output (list of chunks from ING.docs), embeds each chunk,
      and stores them in ~/.praxis/embeddings.db.
      Returns: {corpus, chunks_stored, provider}

    Other targets: stub.
    """
    if target and target[0] == "text":
        chunks = ctx.last_output
        if not isinstance(chunks, list):
            raise ValueError("EMBED.text expects a list of chunks — pipe from ING.docs")
        corpus = params.get("corpus", "default")
        provider = params.get("provider", "local")
        db = EmbeddingsDB(provider=provider)
        stored = db.store_chunks(chunks, corpus)
        return {"corpus": corpus, "chunks_stored": stored, "provider": provider}

    return {"dims": 768, "vector": [0.1] * 768, "source": ".".join(target)}


def search_handler(target: list[str], params: dict, ctx) -> Any:
    """SEARCH — Semantic search.

    SEARCH.semantic(query=..., corpus=name, k=5, threshold=0.0, provider=local)
      Cosine similarity search over an embeddings corpus.
      Returns: [{id, text, source, chunk_index, similarity}, ...]

    Other targets: stub.
    """
    if target and target[0] == "semantic":
        query = params.get("query") or (str(ctx.last_output) if ctx.last_output else "")
        if not query:
            raise ValueError("SEARCH.semantic requires query= parameter")
        corpus = params.get("corpus", "default")
        k = max(1, int(params.get("k", 5)))
        threshold = float(params.get("threshold", 0.0))
        provider = params.get("provider", "local")
        db = EmbeddingsDB(provider=provider)
        return db.search(query, corpus, k=k, threshold=threshold)

    query = params.get("q", "")
    return [
        {"score": 0.95, "text": f"Result 1 for '{query}'"},
        {"score": 0.87, "text": f"Result 2 for '{query}'"},
    ]
