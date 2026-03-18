"""
Sprint 21 — Praxis Agent tests.

Coverage:
  TestAgentContext         — context state, message management, lazy singletons
  TestToolDefinitions      — schema structure, required fields
  TestToolExecutors        — run_program, validate_program, plan_goal, schedule_task,
                             list_schedules, remove_schedule, recall_similar
  TestPraxisAgentLoop      — tool-use loop: single reply, tool call, multi-round
  TestTelegramChannel      — message extraction, whitelist, trigger, split_message
  TestAgentRunner          — context creation, message dispatch, graceful error handling
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, call

import pytest

from praxis.agent.context import AgentContext
from praxis.agent.tools import (
    TOOL_DEFINITIONS,
    execute_tool,
    _run_program,
    _validate_program,
    _plan_goal,
    _schedule_task,
    _list_schedules,
    _remove_schedule,
    _recall_similar,
    _truncate,
)
from praxis.agent.channels.base import InboundMessage
from praxis.agent.channels.telegram import TelegramChannel, _split_message


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _make_ctx(**kwargs) -> AgentContext:
    return AgentContext(chat_id="test-chat", mode="dev", **kwargs)


def _make_mock_executor():
    exe = MagicMock()
    exe.execute.return_value = [
        {"verb": "LOG", "target": ["msg"], "params": {}, "output": "hello",
         "status": "ok", "duration_ms": 5, "log_entry": ""}
    ]
    return exe


def _make_mock_scheduler():
    sched = MagicMock()
    sched.add.return_value = "sched-001"
    mock_prog = MagicMock()
    mock_prog.id = "sched-001"
    mock_prog.goal = "test goal"
    mock_prog.interval_seconds = 3600
    mock_prog.enabled = True
    mock_prog.last_run = None
    mock_prog.run_count = 0
    sched.list_programs.return_value = [mock_prog]
    sched.remove.return_value = True
    return sched


def _make_mock_memory():
    mem = MagicMock()
    entry = MagicMock()
    entry.id = "prog-001"
    entry.goal_text = "test goal"
    entry.shaun_program = "LOG.msg"
    entry.outcome = "success"
    entry.similarity = 0.92
    mem.search.return_value = [entry]
    return mem


def _make_mock_planner(program="LOG.msg"):
    planner = MagicMock()
    result = MagicMock()
    result.program = program
    result.adapted = False
    result.attempts = 1
    planner.plan.return_value = result
    return planner


# ──────────────────────────────────────────────────────────────────────────────
# TestAgentContext
# ──────────────────────────────────────────────────────────────────────────────

class TestAgentContext:
    def test_initial_state(self):
        ctx = _make_ctx()
        assert ctx.chat_id == "test-chat"
        assert ctx.mode == "dev"
        assert ctx.messages == []
        assert ctx.state == {}

    def test_add_user_message(self):
        ctx = _make_ctx()
        ctx.add_user_message("hello")
        assert ctx.messages == [{"role": "user", "content": "hello"}]

    def test_add_assistant_message(self):
        ctx = _make_ctx()
        ctx.add_assistant_message("hi there")
        assert ctx.messages[0]["role"] == "assistant"

    def test_add_tool_result(self):
        ctx = _make_ctx()
        ctx.add_tool_result("tool-123", '{"ok": true}')
        msg = ctx.messages[0]
        assert msg["role"] == "user"
        assert msg["content"][0]["type"] == "tool_result"
        assert msg["content"][0]["tool_use_id"] == "tool-123"

    def test_clear(self):
        ctx = _make_ctx()
        ctx.add_user_message("test")
        ctx.state["x"] = 1
        ctx.clear()
        assert ctx.messages == []
        assert ctx.state == {}

    def test_lazy_executor(self):
        ctx = _make_ctx()
        assert ctx._executor is None
        with patch("praxis.agent.context.Executor") as MockExe:
            with patch("praxis.agent.context.HANDLERS", {}):
                _ = ctx.executor
                MockExe.assert_called_once()

    def test_memory_none_when_not_set(self):
        ctx = _make_ctx()
        assert ctx.memory is None

    def test_planner_none_when_not_set(self):
        ctx = _make_ctx()
        assert ctx.planner is None

    def test_lock_is_reentrant_safe(self):
        """Verify that Lock doesn't deadlock when held (non-reentrant, tests acquire/release)."""
        ctx = _make_ctx()
        acquired = ctx._lock.acquire(blocking=False)
        assert acquired
        ctx._lock.release()


# ──────────────────────────────────────────────────────────────────────────────
# TestToolDefinitions
# ──────────────────────────────────────────────────────────────────────────────

class TestToolDefinitions:
    _EXPECTED_NAMES = {
        "run_program", "validate_program", "plan_goal",
        "schedule_task", "list_schedules", "remove_schedule", "recall_similar",
    }

    def test_all_tools_present(self):
        names = {t["name"] for t in TOOL_DEFINITIONS}
        assert names == self._EXPECTED_NAMES

    def test_each_tool_has_description(self):
        for tool in TOOL_DEFINITIONS:
            assert len(tool["description"]) > 10, f"{tool['name']} description too short"

    def test_each_tool_has_input_schema(self):
        for tool in TOOL_DEFINITIONS:
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_run_program_requires_program(self):
        schema = next(t for t in TOOL_DEFINITIONS if t["name"] == "run_program")
        assert "program" in schema["input_schema"]["required"]

    def test_schedule_task_required_fields(self):
        schema = next(t for t in TOOL_DEFINITIONS if t["name"] == "schedule_task")
        required = set(schema["input_schema"]["required"])
        assert required == {"program", "goal", "interval_seconds"}

    def test_recall_similar_requires_topic(self):
        schema = next(t for t in TOOL_DEFINITIONS if t["name"] == "recall_similar")
        assert "topic" in schema["input_schema"]["required"]


# ──────────────────────────────────────────────────────────────────────────────
# TestToolExecutors
# ──────────────────────────────────────────────────────────────────────────────

class TestRunProgram:
    def test_valid_program_ok(self):
        ctx = _make_ctx()
        mock_exe = _make_mock_executor()
        ctx._executor = mock_exe

        with patch("praxis.agent.tools.parse") as mock_parse, \
             patch("praxis.agent.tools.Validator") as MockValidator:
            mock_ast = MagicMock()
            mock_parse.return_value = mock_ast
            MockValidator.return_value.validate.return_value = []

            result = json.loads(_run_program({"program": "LOG.msg"}, ctx))
            assert result["status"] == "ok"
            assert len(result["steps"]) == 1
            assert result["steps"][0]["verb"] == "LOG"
            mock_exe.execute.assert_called_once_with(mock_ast, memory=None)

    def test_validation_error_returned(self):
        ctx = _make_ctx()
        ctx._executor = MagicMock()

        with patch("praxis.agent.tools.parse") as mock_parse, \
             patch("praxis.agent.tools.Validator") as MockValidator:
            mock_parse.return_value = MagicMock()
            MockValidator.return_value.validate.return_value = ["Unknown verb BADVERB"]

            result = json.loads(_run_program({"program": "BADVERB.x"}, ctx))
            assert result["status"] == "validation_error"
            assert "Unknown verb" in result["errors"][0]

    def test_parse_error_returned(self):
        ctx = _make_ctx()
        with patch("praxis.agent.tools.parse", side_effect=Exception("syntax error")):
            result = json.loads(_run_program({"program": "!!!"}, ctx))
            assert result["status"] == "parse_error"

    def test_stores_last_program_in_state(self):
        ctx = _make_ctx()
        ctx._executor = _make_mock_executor()

        with patch("praxis.agent.tools.parse") as mock_parse, \
             patch("praxis.agent.tools.Validator") as MockValidator:
            mock_parse.return_value = MagicMock()
            MockValidator.return_value.validate.return_value = []

            _run_program({"program": "LOG.msg"}, ctx)
            assert ctx.state["last_program"] == "LOG.msg"


class TestValidateProgram:
    def test_valid_returns_true(self):
        ctx = _make_ctx()
        with patch("praxis.agent.tools.parse") as mock_parse, \
             patch("praxis.agent.tools.Validator") as MockValidator:
            mock_parse.return_value = MagicMock()
            MockValidator.return_value.validate.return_value = []

            result = json.loads(_validate_program({"program": "LOG.msg"}, ctx))
            assert result["valid"] is True

    def test_invalid_returns_errors(self):
        ctx = _make_ctx()
        with patch("praxis.agent.tools.parse") as mock_parse, \
             patch("praxis.agent.tools.Validator") as MockValidator:
            mock_parse.return_value = MagicMock()
            MockValidator.return_value.validate.return_value = ["error one"]

            result = json.loads(_validate_program({"program": "X"}, ctx))
            assert result["valid"] is False
            assert "error one" in result["errors"]

    def test_parse_exception_returns_errors(self):
        ctx = _make_ctx()
        with patch("praxis.agent.tools.parse", side_effect=Exception("bad syntax")):
            result = json.loads(_validate_program({"program": "!!"}, ctx))
            assert result["valid"] is False


class TestPlanGoal:
    def test_no_planner_returns_error(self):
        ctx = _make_ctx()  # _planner is None
        result = json.loads(_plan_goal({"goal": "do something"}, ctx))
        assert "error" in result

    def test_planner_called_with_goal(self):
        ctx = _make_ctx()
        ctx._planner = _make_mock_planner("LOG.result")
        result = json.loads(_plan_goal({"goal": "log a result"}, ctx))
        assert result["program"] == "LOG.result"
        assert result["adapted"] is False
        assert result["attempts"] == 1

    def test_planner_exception_returns_error(self):
        ctx = _make_ctx()
        ctx._planner = MagicMock()
        ctx._planner.plan.side_effect = Exception("LLM error")
        result = json.loads(_plan_goal({"goal": "x"}, ctx))
        assert "error" in result


class TestScheduleTask:
    def test_no_scheduler_returns_error(self):
        ctx = _make_ctx()
        result = json.loads(_schedule_task(
            {"program": "LOG.msg", "goal": "test", "interval_seconds": 300}, ctx
        ))
        assert "error" in result

    def test_adds_to_scheduler(self):
        ctx = _make_ctx()
        ctx._scheduler = _make_mock_scheduler()
        result = json.loads(_schedule_task(
            {"program": "LOG.msg", "goal": "test goal", "interval_seconds": 3600}, ctx
        ))
        assert result["schedule_id"] == "sched-001"
        ctx._scheduler.add.assert_called_once()

    def test_list_schedules(self):
        ctx = _make_ctx()
        ctx._scheduler = _make_mock_scheduler()
        result = json.loads(_list_schedules(ctx))
        assert len(result["schedules"]) == 1
        assert result["schedules"][0]["id"] == "sched-001"

    def test_list_schedules_no_scheduler(self):
        ctx = _make_ctx()
        result = json.loads(_list_schedules(ctx))
        assert "error" in result

    def test_remove_schedule_found(self):
        ctx = _make_ctx()
        ctx._scheduler = _make_mock_scheduler()
        result = json.loads(_remove_schedule({"schedule_id": "sched-001"}, ctx))
        assert result["removed"] is True

    def test_remove_schedule_not_found(self):
        ctx = _make_ctx()
        sched = _make_mock_scheduler()
        sched.remove.return_value = False
        ctx._scheduler = sched
        result = json.loads(_remove_schedule({"schedule_id": "bad-id"}, ctx))
        assert result["removed"] is False


class TestRecallSimilar:
    def test_no_memory_returns_error(self):
        ctx = _make_ctx()
        result = json.loads(_recall_similar({"topic": "test"}, ctx))
        assert "error" in result

    def test_returns_results(self):
        ctx = _make_ctx()
        ctx._memory = _make_mock_memory()
        result = json.loads(_recall_similar({"topic": "test goal"}, ctx))
        assert len(result["results"]) == 1
        assert result["results"][0]["goal"] == "test goal"
        assert result["results"][0]["similarity"] == 0.92

    def test_top_k_passed_to_memory(self):
        ctx = _make_ctx()
        ctx._memory = _make_mock_memory()
        _recall_similar({"topic": "x", "top_k": 3}, ctx)
        ctx._memory.search.assert_called_with("x", top_k=3)


class TestExecuteToolDispatch:
    def test_unknown_tool_returns_error(self):
        ctx = _make_ctx()
        result = json.loads(execute_tool("nonexistent_tool", {}, ctx))
        assert "error" in result

    def test_exception_in_tool_returns_error(self):
        ctx = _make_ctx()
        with patch("praxis.agent.tools._run_program", side_effect=RuntimeError("boom")):
            result = json.loads(execute_tool("run_program", {"program": "x"}, ctx))
            assert "error" in result


class TestTruncate:
    def test_short_string_unchanged(self):
        assert _truncate("hello") == "hello"

    def test_long_string_truncated(self):
        long = "x" * 600
        result = _truncate(long, max_chars=500)
        assert len(result) < 600
        assert "truncated" in result

    def test_dict_truncated_when_large(self):
        big = {"key": "v" * 600}
        result = _truncate(big, max_chars=100)
        assert "truncated" in result

    def test_none_unchanged(self):
        assert _truncate(None) is None


# ──────────────────────────────────────────────────────────────────────────────
# TestPraxisAgentLoop
# ──────────────────────────────────────────────────────────────────────────────

class TestPraxisAgentLoop:
    def _make_agent(self, mock_client):
        with patch("praxis.agent.core.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            from praxis.agent.core import PraxisAgent
            agent = PraxisAgent.__new__(PraxisAgent)
            agent._client = mock_client
            agent.model = "claude-sonnet-4-6"
            agent.max_tokens = 2048
            return agent

    def test_simple_text_reply(self):
        from praxis.agent.core import PraxisAgent

        mock_client = MagicMock()
        text_block = MagicMock()
        text_block.text = "Hello!"
        response = MagicMock()
        response.stop_reason = "end_turn"
        response.content = [text_block]
        mock_client.messages.create.return_value = response

        with patch("praxis.agent.core.anthropic"):
            agent = PraxisAgent.__new__(PraxisAgent)
            agent._client = mock_client
            agent.model = "claude-sonnet-4-6"
            agent.max_tokens = 2048

        ctx = _make_ctx()
        reply = agent.chat("hi", ctx)
        assert reply == "Hello!"
        assert len(ctx.messages) == 2  # user + assistant

    def test_tool_use_round_trip(self):
        from praxis.agent.core import PraxisAgent

        mock_client = MagicMock()

        # Round 1: tool_use
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "validate_program"
        tool_block.id = "tu-001"
        tool_block.input = {"program": "LOG.msg"}
        round1 = MagicMock()
        round1.stop_reason = "tool_use"
        round1.content = [tool_block]

        # Round 2: end_turn
        text_block = MagicMock()
        text_block.text = "Program is valid!"
        round2 = MagicMock()
        round2.stop_reason = "end_turn"
        round2.content = [text_block]

        mock_client.messages.create.side_effect = [round1, round2]

        with patch("praxis.agent.core.anthropic"), \
             patch("praxis.agent.core.execute_tool", return_value='{"valid": true}'):
            agent = PraxisAgent.__new__(PraxisAgent)
            agent._client = mock_client
            agent.model = "claude-sonnet-4-6"
            agent.max_tokens = 2048

        ctx = _make_ctx()
        reply = agent.chat("validate LOG.msg", ctx)
        assert reply == "Program is valid!"
        assert mock_client.messages.create.call_count == 2

    def test_max_tool_rounds_fallback(self):
        from praxis.agent.core import PraxisAgent, _MAX_TOOL_ROUNDS

        mock_client = MagicMock()

        # Always return tool_use — should hit max rounds
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "list_schedules"
        tool_block.id = "tu-x"
        tool_block.input = {}
        resp = MagicMock()
        resp.stop_reason = "tool_use"
        resp.content = [tool_block]
        mock_client.messages.create.return_value = resp

        with patch("praxis.agent.core.anthropic"), \
             patch("praxis.agent.core.execute_tool", return_value='{"schedules": []}'):
            agent = PraxisAgent.__new__(PraxisAgent)
            agent._client = mock_client
            agent.model = "claude-sonnet-4-6"
            agent.max_tokens = 2048

        ctx = _make_ctx()
        reply = agent.chat("loop forever", ctx)
        assert "max tool rounds" in reply
        assert mock_client.messages.create.call_count == _MAX_TOOL_ROUNDS


# ──────────────────────────────────────────────────────────────────────────────
# TestTelegramChannel
# ──────────────────────────────────────────────────────────────────────────────

class TestTelegramChannel:
    def _make_update(self, text, chat_id="999", user_id="111", update_id=1):
        return {
            "update_id": update_id,
            "message": {
                "message_id": 1,
                "from": {"id": int(user_id), "first_name": "Test"},
                "chat": {"id": int(chat_id), "type": "private"},
                "text": text,
            },
        }

    def test_extract_basic_message(self):
        ch = TelegramChannel(token="tok")
        update = self._make_update("hello world")
        msg = ch._extract_message(update)
        assert msg is not None
        assert msg.text == "hello world"
        assert msg.chat_id == "999"

    def test_whitelist_blocks_unknown_chat(self):
        ch = TelegramChannel(token="tok", allowed_chat_ids={"allowed-only"})
        update = self._make_update("hi", chat_id="999")
        assert ch._extract_message(update) is None

    def test_whitelist_allows_known_chat(self):
        ch = TelegramChannel(token="tok", allowed_chat_ids={"999"})
        update = self._make_update("hi", chat_id="999")
        msg = ch._extract_message(update)
        assert msg is not None

    def test_trigger_word_strips_prefix(self):
        ch = TelegramChannel(token="tok", trigger_word="@bot")
        update = self._make_update("@bot run LOG.msg")
        msg = ch._extract_message(update)
        assert msg is not None
        assert msg.text == "run LOG.msg"

    def test_trigger_word_blocks_non_matching(self):
        ch = TelegramChannel(token="tok", trigger_word="@bot")
        update = self._make_update("random message")
        assert ch._extract_message(update) is None

    def test_empty_text_skipped(self):
        ch = TelegramChannel(token="tok")
        update = self._make_update("")
        assert ch._extract_message(update) is None

    def test_no_message_key_skipped(self):
        ch = TelegramChannel(token="tok")
        assert ch._extract_message({"update_id": 1}) is None

    def test_split_message_short(self):
        assert _split_message("hello") == ["hello"]

    def test_split_message_long(self):
        long = "x" * 5000
        chunks = _split_message(long, limit=4096)
        assert len(chunks) == 2
        assert all(len(c) <= 4096 for c in chunks)

    def test_stop_sets_flag(self):
        ch = TelegramChannel(token="tok")
        assert ch._running is True
        ch.stop()
        assert ch._running is False

    def test_send_calls_api(self):
        ch = TelegramChannel(token="tok")
        with patch.object(ch, "_api_call") as mock_api:
            ch.send("999", "hello")
            mock_api.assert_called_once()
            args = mock_api.call_args[0]
            assert args[0] == "sendMessage"
            assert args[1]["chat_id"] == "999"

    def test_send_splits_long_message(self):
        ch = TelegramChannel(token="tok")
        long_text = "x" * 5000
        with patch.object(ch, "_api_call") as mock_api:
            ch.send("999", long_text)
            assert mock_api.call_count == 2

    def test_send_typing_no_exception_on_failure(self):
        ch = TelegramChannel(token="tok")
        with patch.object(ch, "_api_call", side_effect=RuntimeError("network")):
            ch.send_typing("999")  # must not raise


# ──────────────────────────────────────────────────────────────────────────────
# TestAgentRunner
# ──────────────────────────────────────────────────────────────────────────────

class TestAgentRunner:
    def _make_runner(self):
        from praxis.agent.runner import AgentRunner
        mock_agent = MagicMock()
        mock_agent.chat.return_value = "agent reply"
        mock_channel = MagicMock()
        runner = AgentRunner(
            agent=mock_agent,
            channel=mock_channel,
            mode="dev",
        )
        return runner, mock_agent, mock_channel

    def test_creates_context_per_chat(self):
        runner, _, _ = self._make_runner()
        ctx1 = runner._get_or_create_context("chat-1")
        ctx2 = runner._get_or_create_context("chat-2")
        ctx1b = runner._get_or_create_context("chat-1")

        assert ctx1 is not ctx2
        assert ctx1 is ctx1b  # same object on second call

    def test_handle_message_calls_send(self):
        runner, mock_agent, mock_channel = self._make_runner()
        msg = InboundMessage(chat_id="999", user_id="111", text="hello", raw={})
        runner._handle_message(msg)

        mock_agent.chat.assert_called_once()
        mock_channel.send.assert_called_once_with("999", "agent reply")

    def test_handle_message_sends_typing(self):
        runner, _, mock_channel = self._make_runner()
        msg = InboundMessage(chat_id="999", user_id="111", text="hello", raw={})
        runner._handle_message(msg)
        mock_channel.send_typing.assert_called_once_with("999")

    def test_handle_message_error_sends_error_reply(self):
        runner, mock_agent, mock_channel = self._make_runner()
        mock_agent.chat.side_effect = RuntimeError("tool exploded")

        msg = InboundMessage(chat_id="999", user_id="111", text="crash me", raw={})
        runner._handle_message(msg)  # should not raise

        # Error message should be sent
        mock_channel.send.assert_called_once()
        sent_text = mock_channel.send.call_args[0][1]
        assert "error" in sent_text.lower() or "Internal" in sent_text

    def test_context_has_correct_mode(self):
        from praxis.agent.runner import AgentRunner
        mock_agent = MagicMock()
        mock_channel = MagicMock()
        runner = AgentRunner(agent=mock_agent, channel=mock_channel, mode="prod")
        ctx = runner._get_or_create_context("chat-1")
        assert ctx.mode == "prod"

    def test_stop_sets_running_false(self):
        runner, _, mock_channel = self._make_runner()
        runner._running = True
        runner.stop()
        assert runner._running is False
        mock_channel.stop.assert_called_once()


# ──────────────────────────────────────────────────────────────────────────────
# TestFetchFanOut
# ──────────────────────────────────────────────────────────────────────────────

class TestFetchFanOut:
    """Tests for FETCH fan-out: $item substitution when last_output is a list."""

    def _make_ctx_with_output(self, last_output):
        """Return a minimal ctx-like object with last_output set."""
        ctx = MagicMock()
        ctx.last_output = last_output
        return ctx

    def _mock_response(self, data, content_type="application/json"):
        resp = MagicMock()
        resp.headers = {"content-type": content_type}
        resp.json.return_value = data
        resp.text = str(data)
        return resp

    def test_fanout_iterates_list(self):
        """Fan-out fetches once per item, substituting $item into the URL."""
        from praxis.handlers.io import fetch_handler

        ctx = self._make_ctx_with_output([1, 2, 3])
        responses = [
            self._mock_response({"id": 1, "title": "Story One"}),
            self._mock_response({"id": 2, "title": "Story Two"}),
            self._mock_response({"id": 3, "title": "Story Three"}),
        ]

        with patch("praxis.handlers.io.httpx") as mock_httpx:
            mock_httpx.get.side_effect = responses
            result = fetch_handler(
                [],
                {"src": "https://api.example.com/item/$item.json"},
                ctx,
            )

        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["title"] == "Story One"
        assert result[2]["title"] == "Story Three"

    def test_fanout_substitutes_item_in_url(self):
        """Each call uses the correct substituted URL."""
        from praxis.handlers.io import fetch_handler

        ctx = self._make_ctx_with_output([42, 99])
        resp = self._mock_response({"id": 0})

        with patch("praxis.handlers.io.httpx") as mock_httpx:
            mock_httpx.get.return_value = resp
            fetch_handler([], {"src": "https://api.example.com/$item"}, ctx)

        calls = mock_httpx.get.call_args_list
        assert calls[0][0][0] == "https://api.example.com/42"
        assert calls[1][0][0] == "https://api.example.com/99"

    def test_fanout_returns_list_same_length_as_input(self):
        """Result list length matches input list length."""
        from praxis.handlers.io import fetch_handler

        items = list(range(5))
        ctx = self._make_ctx_with_output(items)

        with patch("praxis.handlers.io.httpx") as mock_httpx:
            mock_httpx.get.return_value = self._mock_response({"ok": True})
            result = fetch_handler([], {"src": "https://api.example.com/$item"}, ctx)

        assert len(result) == 5

    def test_no_fanout_when_last_output_is_not_list(self):
        """Without a list in last_output, FETCH does a single request normally."""
        from praxis.handlers.io import fetch_handler

        ctx = self._make_ctx_with_output("not a list")
        resp = self._mock_response({"result": "single"})

        with patch("praxis.handlers.io.httpx") as mock_httpx:
            mock_httpx.get.return_value = resp
            result = fetch_handler([], {"src": "https://api.example.com/$item"}, ctx)

        assert mock_httpx.get.call_count == 1
        assert result == {"result": "single"}

    def test_no_fanout_when_url_has_no_item_placeholder(self):
        """Without $item in URL, FETCH does a single request even if last_output is a list."""
        from praxis.handlers.io import fetch_handler

        ctx = self._make_ctx_with_output([1, 2, 3])
        resp = self._mock_response([10, 20, 30])

        with patch("praxis.handlers.io.httpx") as mock_httpx:
            mock_httpx.get.return_value = resp
            result = fetch_handler([], {"src": "https://api.example.com/topstories.json"}, ctx)

        assert mock_httpx.get.call_count == 1

    def test_fanout_src_param_alias(self):
        """Fan-out works with src= param (alias for url=)."""
        from praxis.handlers.io import fetch_handler

        ctx = self._make_ctx_with_output(["a", "b"])

        with patch("praxis.handlers.io.httpx") as mock_httpx:
            mock_httpx.get.return_value = self._mock_response({"ok": True})
            result = fetch_handler([], {"src": "https://api.example.com/$item/data"}, ctx)

        assert len(result) == 2

    def test_fanout_url_param_alias(self):
        """Fan-out also works with url= param."""
        from praxis.handlers.io import fetch_handler

        ctx = self._make_ctx_with_output([7, 8])

        with patch("praxis.handlers.io.httpx") as mock_httpx:
            mock_httpx.get.return_value = self._mock_response({"id": 7})
            result = fetch_handler([], {"url": "https://api.example.com/item/$item"}, ctx)

        assert len(result) == 2

    def test_fanout_empty_list_returns_empty(self):
        """Fan-out with an empty list returns an empty list without making any requests."""
        from praxis.handlers.io import fetch_handler

        ctx = self._make_ctx_with_output([])

        with patch("praxis.handlers.io.httpx") as mock_httpx:
            result = fetch_handler([], {"src": "https://api.example.com/$item"}, ctx)

        mock_httpx.get.assert_not_called()
        assert result == []
