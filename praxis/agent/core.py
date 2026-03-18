"""
PraxisAgent — Anthropic SDK tool-use loop.

Flow per turn
-------------
1.  Receive text message from a Channel
2.  Append to AgentContext.messages as a user turn
3.  Call anthropic.messages.create() with TOOL_DEFINITIONS
4.  If the response has tool_use blocks: execute each tool, append
    tool_result, loop back to step 3
5.  When the response is stop_reason == "end_turn" with a text block:
    return the assistant reply to the caller
6.  Caller (AgentRunner) forwards the reply to the Channel

The loop is intentionally synchronous per conversation turn so the
context array stays coherent. Parallel tool execution within one
response is handled by executing tool_use blocks in order (safe because
Praxis's own PAR/threading handles any internal concurrency).
"""

from __future__ import annotations

import os
from typing import Any

from praxis.agent.context import AgentContext
from praxis.agent.tools import TOOL_DEFINITIONS, execute_tool

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

_SYSTEM_PROMPT = """\
You are a Praxis AI agent — a native runtime for the Praxis workflow language (.px).

You have direct access to the Praxis execution engine through your tools. No translation
layer, no subprocess. When the user asks you to run an automation, build a workflow, or
schedule a task, use your tools to do it immediately.

Praxis program syntax primer:
  VERB.target(param=value) -> VERB.target -> PAR(VERB.a, VERB.b)
  IF $var == value THEN ... END
  LOOP ... UNTIL $done == true END
  SET.varname, CAP.self(role=x, allow=[verb,...]), ASSERT.check, GATE.cond

Key rules:
- Verbs are UPPERCASE (LOG, SUMM, EVAL, FETCH, GEN, CLN, XFRM, ING, OUT, SET, CAP, ASSERT, GATE, ROLLBACK, RETRY, SPAWN, MSG, CAST, JOIN, SNAP, RECALL, SEARCH, WAIT, BREAK, SKIP)
- Targets are lowercase identifiers
- Parameters use key=value syntax inside ()
- Use validate_program before schedule_task
- Use plan_goal when the user gives a natural-language description and has no program yet
- Use recall_similar to check if a similar program was run before

Respond concisely. Show program output in code blocks. If a tool call fails, explain why
and suggest a fix. You are Praxis-native: reason in terms of verbs, chains, and PAR blocks.
"""

_MAX_TOOL_ROUNDS = 10  # prevent infinite loops


class PraxisAgent:
    """
    Wraps the Anthropic SDK tool-use loop for a single conversation.

    Parameters
    ----------
    model:
        Claude model id. Defaults to claude-sonnet-4-6.
    api_key:
        Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
    max_tokens:
        Max tokens per assistant response. Default: 2048.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        max_tokens: int = 2048,
    ) -> None:
        if anthropic is None:
            raise ImportError(
                "anthropic package is required for PraxisAgent. "
                "Install it: pip install praxis-lang[agent]"
            )

        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self.model = model
        self.max_tokens = max_tokens

    # ── Public API ─────────────────────────────────────────────────────────

    def chat(self, user_message: str, ctx: AgentContext) -> str:
        """
        Process one user message and return the agent's reply.

        Mutates ctx.messages in-place. Thread-safe per-context via ctx._lock.
        """
        with ctx._lock:
            ctx.add_user_message(user_message)
            reply = self._run_tool_loop(ctx)
            return reply

    # ── Internal tool-use loop ─────────────────────────────────────────────

    def _run_tool_loop(self, ctx: AgentContext) -> str:
        """Keep calling the API until we get an end_turn with a text block."""
        for _round in range(_MAX_TOOL_ROUNDS):
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=ctx.messages,
            )

            # Append assistant turn to context
            ctx.add_assistant_message(response.content)

            if response.stop_reason == "end_turn":
                # Extract text blocks — join in case there are multiple
                texts = [
                    block.text
                    for block in response.content
                    if hasattr(block, "text")
                ]
                return "\n".join(texts) if texts else "(no reply)"

            if response.stop_reason == "tool_use":
                # Execute all tool calls in this response
                for block in response.content:
                    if block.type == "tool_use":
                        result = execute_tool(block.name, block.input, ctx)
                        ctx.add_tool_result(block.id, result)
                # Loop — send tool results back to the model
                continue

            # Unexpected stop reason (e.g. max_tokens, stop_sequence)
            break

        # Fallback: extract any text we have
        last_msg = ctx.messages[-1] if ctx.messages else {}
        content = last_msg.get("content", [])
        if isinstance(content, list):
            texts = [b.get("text", "") for b in content if isinstance(b, dict) and "text" in b]
            if texts:
                return "\n".join(texts)
        return "(agent reached max tool rounds — check logs)"
