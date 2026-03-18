"""
Praxis MCP Server — Sprint 31.

Exposes Praxis as a Model Context Protocol server so Claude Code,
Cursor, Zed, and any MCP-aware tool can call into the Praxis runtime
directly — no subprocess, no REST bridge required.

Tools exposed:
  run_program       — execute a .px program, return step results
  validate_program  — check syntax + semantics, return errors or "valid"
  plan_goal         — natural language → .px program (requires ANTHROPIC_API_KEY)
  recall_similar    — search program memory by goal text
  search_registry   — search the community program registry
  install_program   — fetch a registry program into local memory
  get_constitution  — return all active constitutional rules

Resources exposed:
  praxis://constitution   — all constitutional rules
  praxis://programs       — recent stored programs (last 20)

Usage (Claude Code):
  Add to your Claude Code settings.json:

  {
    "mcpServers": {
      "praxis": {
        "command": "praxis",
        "args": ["mcp"]
      }
    }
  }

Or run directly:
  praxis mcp
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

# ── Tool handler implementations ──────────────────────────────────────────────
# These are plain functions, testable without the MCP protocol layer.


def tool_run_program(program: str, mode: str = "dev") -> dict[str, Any]:
    """Execute a Praxis program and return step results."""
    from praxis.grammar import parse
    from praxis.validator import validate
    from praxis.executor import Executor
    from praxis.handlers import HANDLERS

    try:
        ast = parse(program)
    except Exception as exc:
        return {"ok": False, "error": f"Parse error: {exc}", "results": []}

    errors = validate(ast, mode=mode)
    if errors:
        return {"ok": False, "error": "Validation errors: " + "; ".join(errors), "results": []}

    results = Executor(handlers=HANDLERS).execute(ast)
    return {
        "ok": all(r["status"] != "error" for r in results),
        "results": results,
        "steps": len(results),
    }


def tool_validate_program(program: str, mode: str = "dev") -> dict[str, Any]:
    """Validate a Praxis program. Returns errors list or empty list if valid."""
    from praxis.grammar import parse
    from praxis.validator import validate

    try:
        ast = parse(program)
    except Exception as exc:
        return {"valid": False, "errors": [f"Parse error: {exc}"]}

    errors = validate(ast, mode=mode)
    return {"valid": not errors, "errors": errors}


def tool_plan_goal(goal: str, mode: str = "dev") -> dict[str, Any]:
    """Generate a Praxis program from a natural-language goal."""
    try:
        from praxis.planner import Planner
        from praxis.providers import resolve_provider
        from praxis.memory import ProgramMemory
        from praxis.constitution import Constitution
    except ImportError as exc:
        return {"ok": False, "error": f"Missing dependency: {exc}"}

    try:
        mem = ProgramMemory()
        const = Constitution()
        provider = resolve_provider()
        planner = Planner(memory=mem, constitution=const, provider=provider)
        program, _attempts = planner.plan(goal, mode=mode)
        return {"ok": True, "program": program, "goal": goal}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "program": None}


def tool_recall_similar(goal: str, k: int = 5) -> dict[str, Any]:
    """Search program memory for programs similar to the given goal."""
    from praxis.memory import ProgramMemory

    try:
        mem = ProgramMemory()
        results = mem.retrieve_similar(goal, k=k)
        return {
            "ok": True,
            "matches": [
                {
                    "id": r.id,
                    "goal_text": r.goal_text,
                    "program": r.shaun_program,
                    "outcome": r.outcome,
                    "similarity": round(r.similarity, 3),
                }
                for r in results
            ],
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "matches": []}


def tool_search_registry(query: str) -> dict[str, Any]:
    """Search the Praxis program registry."""
    from praxis.registry import search_registry, RegistryError

    try:
        programs = search_registry(query)
        return {
            "ok": True,
            "results": [
                {
                    "name": p.name,
                    "description": p.description,
                    "author": p.author,
                    "tags": p.tags,
                    "version": p.version,
                }
                for p in programs
            ],
        }
    except RegistryError as exc:
        return {"ok": False, "error": str(exc), "results": []}


def tool_install_program(name: str) -> dict[str, Any]:
    """Install a program from the registry into local program memory."""
    from praxis.registry import install_program, RegistryError
    from praxis.memory import ProgramMemory

    try:
        mem = ProgramMemory()
        prog = install_program(name, memory=mem)
        return {
            "ok": True,
            "name": prog.name,
            "description": prog.description,
            "author": prog.author,
            "tags": prog.tags,
            "message": f"Installed '{prog.name}' into program memory.",
        }
    except RegistryError as exc:
        return {"ok": False, "error": str(exc)}


def tool_get_constitution() -> dict[str, Any]:
    """Return all active constitutional rules."""
    from praxis.constitution import Constitution

    try:
        const = Constitution()
        rules = const.get_rules()
        return {
            "ok": True,
            "rules": [
                {
                    "text": r.text,
                    "verbs": r.verbs,
                    "tags": getattr(r, "tags", []),
                }
                for r in rules
            ],
            "count": len(rules),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "rules": []}


# ── Resource helpers ───────────────────────────────────────────────────────────


def resource_constitution() -> str:
    """Return constitution as formatted text."""
    result = tool_get_constitution()
    if not result["ok"]:
        return f"Error: {result['error']}"
    lines = [f"# Praxis Constitution ({result['count']} rules)\n"]
    for i, r in enumerate(result["rules"], 1):
        verbs = ", ".join(r["verbs"]) if r["verbs"] else "all"
        lines.append(f"{i}. [{verbs}] {r['text']}")
    return "\n".join(lines)


def resource_programs() -> str:
    """Return recent programs as formatted text."""
    from praxis.memory import ProgramMemory

    try:
        mem = ProgramMemory()
        results = mem.retrieve_similar("", k=20)
        if not results:
            return "No programs stored yet."
        lines = [f"# Praxis Program Memory ({len(results)} recent)\n"]
        for r in results:
            lines.append(f"## {r.goal_text}")
            lines.append(f"outcome: {r.outcome}")
            lines.append(f"```")
            lines.append(r.shaun_program or "(no program text)")
            lines.append(f"```\n")
        return "\n".join(lines)
    except Exception as exc:
        return f"Error: {exc}"


# ── MCP server ────────────────────────────────────────────────────────────────

_TOOL_SCHEMAS = [
    {
        "name": "run_program",
        "description": (
            "Execute a Praxis (.px) program and return step-by-step results. "
            "Each step shows verb, target, status (ok/error), output preview, and duration."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "program": {"type": "string", "description": "The Praxis program source to run"},
                "mode": {
                    "type": "string",
                    "enum": ["dev", "prod"],
                    "default": "dev",
                    "description": "dev = permissive; prod = enforces GATE checks",
                },
            },
            "required": ["program"],
        },
    },
    {
        "name": "validate_program",
        "description": "Check a Praxis program for syntax and semantic errors without running it.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "program": {"type": "string", "description": "The Praxis program source to validate"},
                "mode": {"type": "string", "enum": ["dev", "prod"], "default": "dev"},
            },
            "required": ["program"],
        },
    },
    {
        "name": "plan_goal",
        "description": (
            "Generate a Praxis program from a natural-language goal using the configured LLM. "
            "Requires ANTHROPIC_API_KEY (or another provider env var) to be set."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "Natural language description of what you want to do"},
                "mode": {"type": "string", "enum": ["dev", "prod"], "default": "dev"},
            },
            "required": ["goal"],
        },
    },
    {
        "name": "recall_similar",
        "description": "Search program memory for previously stored programs similar to a goal.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "Goal text to search for"},
                "k": {"type": "integer", "default": 5, "description": "Max results to return"},
            },
            "required": ["goal"],
        },
    },
    {
        "name": "search_registry",
        "description": "Search the Praxis community program registry by name, description, or tag.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (empty string returns all)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "install_program",
        "description": "Install a program from the Praxis registry into local program memory by exact name.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Exact program name from the registry"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "get_constitution",
        "description": "Return all active constitutional rules that guide the Praxis planner.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]

_TOOL_HANDLERS = {
    "run_program": lambda args: tool_run_program(
        args["program"], args.get("mode", "dev")
    ),
    "validate_program": lambda args: tool_validate_program(
        args["program"], args.get("mode", "dev")
    ),
    "plan_goal": lambda args: tool_plan_goal(
        args["goal"], args.get("mode", "dev")
    ),
    "recall_similar": lambda args: tool_recall_similar(
        args["goal"], int(args.get("k", 5))
    ),
    "search_registry": lambda args: tool_search_registry(args["query"]),
    "install_program": lambda args: tool_install_program(args["name"]),
    "get_constitution": lambda args: tool_get_constitution(),
}


def run_server() -> None:
    """Start the Praxis MCP server on stdio."""
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp import types as mcp_types
    except ImportError:
        raise SystemExit(
            "MCP SDK not installed. Run: pip install praxis-lang[mcp]\n"
            "or: pip install mcp"
        )

    import asyncio

    server = Server("praxis")

    @server.list_tools()
    async def list_tools() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                name=s["name"],
                description=s["description"],
                inputSchema=s["inputSchema"],
            )
            for s in _TOOL_SCHEMAS
        ]

    @server.list_resources()
    async def list_resources() -> list[mcp_types.Resource]:
        return [
            mcp_types.Resource(
                uri="praxis://constitution",
                name="Praxis Constitution",
                description="All active constitutional rules",
                mimeType="text/plain",
            ),
            mcp_types.Resource(
                uri="praxis://programs",
                name="Praxis Program Memory",
                description="Recent stored programs",
                mimeType="text/plain",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        if uri == "praxis://constitution":
            return resource_constitution()
        if uri == "praxis://programs":
            return resource_programs()
        raise ValueError(f"Unknown resource: {uri}")

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict | None
    ) -> list[mcp_types.TextContent]:
        args = arguments or {}
        handler = _TOOL_HANDLERS.get(name)
        if handler is None:
            return [mcp_types.TextContent(type="text", text=f"Unknown tool: {name}")]
        try:
            result = handler(args)
            return [mcp_types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        except Exception:
            return [
                mcp_types.TextContent(
                    type="text",
                    text=json.dumps({"ok": False, "error": traceback.format_exc()}),
                )
            ]

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(_run())
