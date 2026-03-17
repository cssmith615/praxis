"""
Data handlers: ING, CLN, XFRM, FILTER, SORT, MERGE

Sprint 1: ING reads from a file path or returns mock data.
          All others are pass-through stubs that log and return input.
Sprint 4: Real implementations replace these.
"""

from __future__ import annotations
import csv
import json
import os
from typing import Any


def ing_handler(target: list[str], params: dict, ctx) -> Any:
    """
    ING — Ingest / load data.

    target:  dot-path describing the data source
             e.g. ['sales', 'db'] | ['flights'] | ['api', 'weather']

    params:
      path=   file path to ingest (optional; falls back to mock data)
      format= 'csv' | 'json' | 'auto' (default: auto)

    Returns:
      list[dict] if CSV/JSON, or a mock dataset if no real source.
    """
    source = ".".join(target)
    path: str | None = params.get("path")
    fmt: str = params.get("format", "auto")

    if path and os.path.exists(path):
        return _read_file(path, fmt)

    # Mock data keyed by source name so tests get predictable results
    mock_data = {
        "sales.db":   [{"id": 1, "amount": 100.0}, {"id": 2, "amount": 250.0}],
        "flights":    [{"dest": "denver", "price": 189.0}, {"dest": "austin", "price": 145.0}],
        "marketing":  [{"campaign": "q1", "clicks": 5000}],
        "crm":        [{"customer": "acme", "revenue": 12000}],
    }
    data = mock_data.get(source, [{"source": source, "stub": True}])
    ctx.last_output = data
    return data


def cln_handler(target: list[str], params: dict, ctx) -> Any:
    """
    CLN — Clean / normalize data.

    target: cleaning strategy  e.g. ['null'] | ['dedupe'] | ['normalize']
    Input:  ctx.last_output (expected: list[dict])
    Returns: cleaned list[dict]
    """
    data = ctx.last_output
    if data is None:
        return []
    strategy = target[0] if target else "null"

    if strategy == "null" and isinstance(data, list):
        # Remove rows where any value is None
        return [row for row in data if all(v is not None for v in row.values())]

    if strategy == "dedupe" and isinstance(data, list):
        seen = set()
        result = []
        for row in data:
            key = str(sorted(row.items()))
            if key not in seen:
                seen.add(key)
                result.append(row)
        return result

    # Passthrough for unknown strategies
    return data


def xfrm_handler(target: list[str], params: dict, ctx) -> Any:
    """XFRM — Transform data. Stub: returns input unchanged."""
    return ctx.last_output


def filter_handler(target: list[str], params: dict, ctx) -> Any:
    """FILTER — Filter rows. Stub: returns input unchanged."""
    return ctx.last_output


def sort_handler(target: list[str], params: dict, ctx) -> Any:
    """SORT — Sort data. Stub: returns input unchanged."""
    return ctx.last_output


def merge_handler(target: list[str], params: dict, ctx) -> Any:
    """MERGE — Merge datasets. Stub: returns combined list."""
    data = ctx.last_output
    if isinstance(data, list):
        return data
    return [data] if data is not None else []


# ── Internal helpers ──────────────────────────────────────────────────────────

def _read_file(path: str, fmt: str) -> list[dict]:
    ext = os.path.splitext(path)[1].lower()
    if fmt == "auto":
        fmt = "csv" if ext == ".csv" else "json"

    if fmt == "csv":
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    if fmt == "json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]

    return [{"path": path, "raw": open(path).read()}]
