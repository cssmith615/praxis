"""
Praxis Program Registry — Sprint 30.

Provides `praxis install`, `praxis search`, and `praxis publish` commands
for sharing community-built .px programs.

Registry index format (JSON):
  {
    "version": "1",
    "programs": [
      {
        "name": "news-brief",
        "description": "Fetch top HN stories and summarize",
        "author": "cssmith615",
        "version": "1.0.0",
        "url": "https://raw.githubusercontent.com/cssmith615/praxis-programs/main/news-brief.px",
        "tags": ["news", "summarize", "hacker-news"]
      }
    ]
  }

The default registry is hosted at REGISTRY_URL. You can override with the
PRAXIS_REGISTRY_URL environment variable or --registry CLI flag.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

# ── Registry URL ──────────────────────────────────────────────────────────────

REGISTRY_URL = os.environ.get(
    "PRAXIS_REGISTRY_URL",
    "https://raw.githubusercontent.com/cssmith615/praxis/main/registry/index.json",
)

_LOCAL_REGISTRY = Path(__file__).parent.parent / "registry" / "index.json"


# ── Types ────────────────────────────────────────────────────────────────────

class RegistryProgram:
    def __init__(self, data: dict[str, Any]) -> None:
        self.name        = data.get("name", "")
        self.description = data.get("description", "")
        self.author      = data.get("author", "")
        self.version     = data.get("version", "1.0.0")
        self.url         = data.get("url", "")
        self.tags        = data.get("tags", [])
        self.program     = data.get("program", "")  # inline program text (alternative to url)

    def __repr__(self) -> str:
        return f"RegistryProgram(name={self.name!r}, description={self.description!r})"


class RegistryError(Exception):
    pass


# ── Fetch registry ────────────────────────────────────────────────────────────

def fetch_registry(registry_url: str = REGISTRY_URL) -> list[RegistryProgram]:
    """
    Fetch the program registry index and return a list of RegistryProgram objects.

    Falls back to the bundled local registry if the remote fetch fails.
    """
    data = None

    # Try remote first
    try:
        req = urllib.request.Request(
            registry_url,
            headers={"User-Agent": "praxis-lang/1.3"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
    except Exception:
        pass

    # Fall back to local bundled registry
    if data is None and _LOCAL_REGISTRY.exists():
        data = json.loads(_LOCAL_REGISTRY.read_text(encoding="utf-8"))

    if data is None:
        raise RegistryError(
            "Could not fetch registry. Check your internet connection or set PRAXIS_REGISTRY_URL."
        )

    programs = data.get("programs", [])
    return [RegistryProgram(p) for p in programs]


# ── Search ───────────────────────────────────────────────────────────────────

def search_registry(query: str, registry_url: str = REGISTRY_URL) -> list[RegistryProgram]:
    """
    Search the registry for programs matching the query string.

    Matches against name, description, and tags (case-insensitive).
    Returns all programs if query is empty.
    """
    programs = fetch_registry(registry_url)
    if not query:
        return programs
    q = query.lower()
    return [
        p for p in programs
        if q in p.name.lower()
        or q in p.description.lower()
        or any(q in tag.lower() for tag in p.tags)
    ]


# ── Install ───────────────────────────────────────────────────────────────────

def install_program(
    name: str,
    memory=None,
    registry_url: str = REGISTRY_URL,
) -> RegistryProgram:
    """
    Fetch a program by name from the registry and store it in program memory.

    Parameters
    ----------
    name:
        Exact name of the program in the registry.
    memory:
        A ProgramMemory instance. If None, creates one with default path.
    registry_url:
        Registry index URL.

    Returns the installed RegistryProgram.
    Raises RegistryError if not found or fetch fails.
    """
    programs = fetch_registry(registry_url)
    match = next((p for p in programs if p.name.lower() == name.lower()), None)
    if match is None:
        available = ", ".join(p.name for p in programs[:10])
        raise RegistryError(
            f"Program '{name}' not found in registry. "
            f"Available: {available}" + (" and more..." if len(programs) > 10 else "")
        )

    # Fetch program text
    program_text = match.program
    if not program_text and match.url:
        try:
            req = urllib.request.Request(
                match.url, headers={"User-Agent": "praxis-lang/1.3"}
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                program_text = r.read().decode()
        except Exception as exc:
            raise RegistryError(f"Failed to fetch program from {match.url}: {exc}") from exc

    if not program_text:
        raise RegistryError(f"Program '{name}' has no program text or URL in the registry.")

    # Validate it parses
    try:
        from praxis.grammar import parse
        parse(program_text)
    except Exception as exc:
        raise RegistryError(f"Registry program '{name}' failed to parse: {exc}") from exc

    # Store in memory
    if memory is None:
        from praxis.memory import ProgramMemory
        memory = ProgramMemory()

    memory.store(
        goal=match.description or match.name,
        program=program_text,
        outcome="planned",
        log=[{"source": "registry", "registry_name": match.name, "author": match.author}],
    )

    return match


# ── Publish ───────────────────────────────────────────────────────────────────

def publish_program(
    program_text: str,
    name: str,
    description: str,
    tags: list[str] | None = None,
    author: str = "",
    output_path: Path | None = None,
) -> dict:
    """
    Package a program for publishing to the registry.

    Creates a metadata JSON file alongside the .px file, suitable for
    submitting as a pull request to the praxis-programs registry.

    Returns the metadata dict.
    """
    metadata = {
        "name": name,
        "description": description,
        "author": author,
        "version": "1.0.0",
        "tags": tags or [],
        "program": program_text,
    }

    if output_path is None:
        output_path = Path(f"{name}.px")

    output_path.write_text(program_text, encoding="utf-8")
    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata
