"""
Praxis Agent — native tool definitions for the Anthropic tool-use loop.

Tools are pure Python calls into the Praxis runtime — no subprocess, no HTTP,
no translation layer. Claude receives results in the same turn they execute.

Tool catalogue
--------------
1. run_program      — execute a .px program text, return step results
2. validate_program — syntax + semantic check without running
3. plan_goal        — natural-language goal → .px program via Planner
4. schedule_task    — add a program to the Scheduler (cron interval)
5. list_schedules   — show all scheduled programs
6. remove_schedule  — cancel a scheduled program by id
7. recall_similar   — semantic search over ProgramMemory by topic
"""

from __future__ import annotations

import json
import traceback
from typing import Any

from praxis.agent.context import AgentContext
from praxis.grammar import parse
from praxis.validator import Validator

# ──────────────────────────────────────────────────────────────────────────────
# Anthropic tool schemas
# ──────────────────────────────────────────────────────────────────────────────

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "run_program",
        "description": (
            "Execute a Praxis (.px) program directly in the runtime. "
            "Returns step-by-step results including outputs, statuses, and timing. "
            "Use this whenever the user wants to run a workflow or automation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "program": {
                    "type": "string",
                    "description": "The Praxis program text to execute (e.g. 'LOG.msg -> SUMM.text').",
                },
                "mode": {
                    "type": "string",
                    "enum": ["dev", "prod"],
                    "description": "Execution mode. Default: dev.",
                },
            },
            "required": ["program"],
        },
    },
    {
        "name": "validate_program",
        "description": (
            "Check a Praxis program for syntax and semantic errors without running it. "
            "Returns 'valid' or a list of error messages. "
            "Always validate before scheduling a program."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "program": {
                    "type": "string",
                    "description": "The Praxis program text to validate.",
                },
            },
            "required": ["program"],
        },
    },
    {
        "name": "plan_goal",
        "description": (
            "Convert a natural-language goal into a Praxis program using the AI planner. "
            "The planner checks ProgramMemory for similar past programs and adapts them. "
            "Returns the generated program text ready to run or schedule."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "Natural-language description of what the program should do.",
                },
            },
            "required": ["goal"],
        },
    },
    {
        "name": "schedule_task",
        "description": (
            "Add a Praxis program to the scheduler so it runs automatically on a cron interval. "
            "Validate the program first. "
            "Returns the schedule id you can use to cancel it later."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "program": {
                    "type": "string",
                    "description": "The validated Praxis program text to schedule.",
                },
                "goal": {
                    "type": "string",
                    "description": "Human-readable description of what this schedule does.",
                },
                "interval_seconds": {
                    "type": "integer",
                    "description": "How often to run the program, in seconds.",
                    "minimum": 60,
                },
            },
            "required": ["program", "goal", "interval_seconds"],
        },
    },
    {
        "name": "list_schedules",
        "description": "List all currently scheduled Praxis programs with their ids, goals, and intervals.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "remove_schedule",
        "description": "Cancel and remove a scheduled program by its schedule id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "schedule_id": {
                    "type": "string",
                    "description": "The schedule id returned by schedule_task or list_schedules.",
                },
            },
            "required": ["schedule_id"],
        },
    },
    {
        "name": "recall_similar",
        "description": (
            "Search ProgramMemory for past Praxis programs similar to a topic or goal. "
            "Useful for understanding what has been run before, or as context before planning. "
            "Returns up to 5 similar programs with their goals and outcomes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic or goal text to search for similar programs.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1–10). Default: 5.",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["topic"],
        },
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Tool executors
# ──────────────────────────────────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: dict, ctx: AgentContext) -> str:
    """Dispatch a tool call to the appropriate executor. Returns a JSON string."""
    try:
        if tool_name == "run_program":
            return _run_program(tool_input, ctx)
        if tool_name == "validate_program":
            return _validate_program(tool_input, ctx)
        if tool_name == "plan_goal":
            return _plan_goal(tool_input, ctx)
        if tool_name == "schedule_task":
            return _schedule_task(tool_input, ctx)
        if tool_name == "list_schedules":
            return _list_schedules(ctx)
        if tool_name == "remove_schedule":
            return _remove_schedule(tool_input, ctx)
        if tool_name == "recall_similar":
            return _recall_similar(tool_input, ctx)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    except Exception as exc:
        return json.dumps({"error": str(exc), "trace": traceback.format_exc(limit=5)})


# ── Individual tool implementations ───────────────────────────────────────────

def _run_program(inp: dict, ctx: AgentContext) -> str:
    program_text = inp["program"]
    mode = inp.get("mode", ctx.mode)

    validator = Validator()
    try:
        program = parse(program_text)
        errors = validator.validate(program)
        if errors:
            return json.dumps({"status": "validation_error", "errors": errors})
    except Exception as exc:
        return json.dumps({"status": "parse_error", "error": str(exc)})

    results = ctx.executor.execute(program, memory=ctx.memory)

    # Summarise for context window efficiency
    summary: list[dict] = []
    for r in results:
        summary.append({
            "verb": r.get("verb"),
            "target": r.get("target"),
            "status": r.get("status"),
            "duration_ms": r.get("duration_ms"),
            "output": _truncate(r.get("output")),
        })

    # Store last program for reference
    ctx.state["last_program"] = program_text
    ctx.state["last_results"] = summary

    return json.dumps({"status": "ok", "steps": summary})


def _validate_program(inp: dict, ctx: AgentContext) -> str:
    program_text = inp["program"]

    try:
        program = parse(program_text)
    except Exception as exc:
        return json.dumps({"valid": False, "errors": [str(exc)]})

    errors = Validator().validate(program)
    if errors:
        return json.dumps({"valid": False, "errors": errors})
    return json.dumps({"valid": True})


def _plan_goal(inp: dict, ctx: AgentContext) -> str:
    goal = inp["goal"]

    if ctx.planner is None:
        return json.dumps({
            "error": "No LLM provider configured. Start the agent with --provider to enable planning."
        })

    try:
        result = ctx.planner.plan(goal, memory=ctx.memory)
        return json.dumps({
            "program": result.program,
            "adapted": result.adapted,
            "attempts": result.attempts,
        })
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _schedule_task(inp: dict, ctx: AgentContext) -> str:
    if ctx.scheduler is None:
        return json.dumps({"error": "Scheduler not enabled. Start the agent with --schedule."})

    program = inp["program"]
    goal = inp["goal"]
    interval = int(inp["interval_seconds"])

    schedule_id = ctx.scheduler.add(
        goal=goal,
        program_text=program,
        interval_seconds=interval,
    )
    return json.dumps({"schedule_id": schedule_id, "goal": goal, "interval_seconds": interval})


def _list_schedules(ctx: AgentContext) -> str:
    if ctx.scheduler is None:
        return json.dumps({"error": "Scheduler not enabled."})

    programs = ctx.scheduler.list_programs()
    rows: list[dict] = []
    for p in programs:
        rows.append({
            "id": p.id,
            "goal": p.goal,
            "interval_seconds": p.interval_seconds,
            "enabled": p.enabled,
            "last_run": p.last_run,
            "run_count": p.run_count,
        })
    return json.dumps({"schedules": rows})


def _remove_schedule(inp: dict, ctx: AgentContext) -> str:
    if ctx.scheduler is None:
        return json.dumps({"error": "Scheduler not enabled."})

    schedule_id = inp["schedule_id"]
    removed = ctx.scheduler.remove(schedule_id)
    if removed:
        return json.dumps({"removed": True, "schedule_id": schedule_id})
    return json.dumps({"removed": False, "error": f"No schedule with id {schedule_id}"})


def _recall_similar(inp: dict, ctx: AgentContext) -> str:
    if ctx.memory is None:
        return json.dumps({"error": "Memory not enabled. Install sentence-transformers and restart."})

    topic = inp["topic"]
    top_k = int(inp.get("top_k", 5))

    try:
        similar = ctx.memory.search(topic, top_k=top_k)
        rows: list[dict] = []
        for entry in similar:
            rows.append({
                "id": entry.id,
                "goal": entry.goal_text,
                "program": entry.shaun_program,
                "outcome": entry.outcome,
                "similarity": round(entry.similarity, 3) if hasattr(entry, "similarity") else None,
            })
        return json.dumps({"results": rows})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _truncate(value: Any, max_chars: int = 500) -> Any:
    """Truncate large outputs so tool results stay token-efficient."""
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + f"… [{len(value) - max_chars} chars truncated]"
    if isinstance(value, (dict, list)):
        serialised = json.dumps(value)
        if len(serialised) > max_chars:
            return serialised[:max_chars] + "… [truncated]"
    return value
