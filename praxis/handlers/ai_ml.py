"""
AI/ML handlers: TRN, INF, EVAL, SUMM, CLASS, GEN, EMBED, SEARCH

Sprint 1: stubs for TRN, INF, EVAL, SUMM, CLASS, GEN.
Sprint A: EMBED.text and SEARCH.semantic wired to EmbeddingsDB.
          Other targets fall through to stubs.
"""

from typing import Any

from praxis.embeddings import EmbeddingsDB


def trn_handler(target: list[str], params: dict, ctx) -> Any:
    """TRN — Train model. Returns mock model artifact."""
    arch = target[0] if target else "model"
    epochs = params.get("ep", params.get("epochs", 10))
    return {"model": arch, "epochs": epochs, "rmse": 0.15, "status": "trained"}


def inf_handler(target: list[str], params: dict, ctx) -> Any:
    """INF — Run inference. Returns mock prediction."""
    return {"predictions": [0.82, 0.91, 0.74], "model": ".".join(target)}


def eval_handler(target: list[str], params: dict, ctx) -> Any:
    """EVAL — Evaluate / score. Returns mock metric."""
    metric = target[0] if target else "score"
    threshold = params.get("threshold")
    value = 0.88
    result = {"metric": metric, "value": value}
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
    """GEN — Generate text/content. Returns mock generated content."""
    template = params.get("template", "default")
    return f"[Generated content — template={template}, target={'.'.join(target)}]"


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
