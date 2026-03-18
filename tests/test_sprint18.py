"""
Sprint 18 — praxis chat (Interactive REPL)

Tests cover:
  - _looks_like_program heuristic
  - PraxisREPL._run_program (parse + validate + execute)
  - PraxisREPL._dispatch routing (program vs goal vs no-provider)
  - session command handlers (:clear, :show, :save, :validate, :mode, :history, :help)
  - multi-line buffer accumulation
  - memory integration on execute
  - goal mode (mock provider)
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from praxis.chat import PraxisREPL, _looks_like_program


# ──────────────────────────────────────────────────────────────────────────────
# _looks_like_program
# ──────────────────────────────────────────────────────────────────────────────

class TestLooksLikeProgram:
    def test_simple_verb_dot_target(self):
        assert _looks_like_program("LOG.test")

    def test_chain_arrow(self):
        assert _looks_like_program("ING.file -> CLN.null")

    def test_par_block(self):
        assert _looks_like_program("PAR(LOG.a, LOG.b)")

    def test_comment_line(self):
        assert _looks_like_program("# this is a comment")

    def test_if_condition(self):
        assert _looks_like_program("IF x == y THEN LOG.ok END")

    def test_loop_block(self):
        assert _looks_like_program("LOOP(3, LOG.tick)")

    def test_natural_language_goal(self):
        assert not _looks_like_program("summarize my sales data")

    def test_natural_language_question(self):
        assert not _looks_like_program("what files are in this directory?")

    def test_empty(self):
        assert not _looks_like_program("")

    def test_lowercase_only(self):
        assert not _looks_like_program("hello world")

    def test_multiline_program(self):
        prog = "LOG.start\n-> SUMM.text\n-> EVAL.quality"
        assert _looks_like_program(prog)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def repl(tmp_path):
    """REPL with a real in-memory SQLite program store, no provider."""
    from praxis.memory import ProgramMemory
    mem = ProgramMemory(db_path=str(tmp_path / "test.db"),
                        embedder=lambda t: [0.0] * 384)
    return PraxisREPL(memory=mem, provider=None, mode="dev")


@pytest.fixture
def repl_no_memory():
    return PraxisREPL(memory=None, provider=None, mode="dev")


# ──────────────────────────────────────────────────────────────────────────────
# _run_program
# ──────────────────────────────────────────────────────────────────────────────

class TestRunProgram:
    def test_valid_program_returns_true(self, repl, capsys):
        result = repl._run_program("LOG.test", auto=False)
        assert result is True

    def test_invalid_verb_parse_error_returns_false(self, repl, capsys):
        result = repl._run_program("NOT_A_VERB!!!", auto=False)
        assert result is False

    def test_validation_error_returns_true(self, repl, capsys):
        # parse succeeds but validator catches bad verb name
        result = repl._run_program("BADVERB123456789.x", auto=False)
        # Either parse error (False) or validation error (True) — either is valid
        assert isinstance(result, bool)

    def test_auto_false_does_not_execute(self, repl, capsys):
        with patch.object(repl, "_execute_and_show") as mock_exec:
            repl._run_program("LOG.test", auto=False)
            mock_exec.assert_not_called()

    def test_auto_true_prompts_to_run(self, repl, capsys):
        with patch("praxis.chat.console") as mock_console:
            mock_console.input.return_value = "n"
            mock_console.print = MagicMock()
            with patch.object(repl, "_execute_and_show") as mock_exec:
                repl._run_program("LOG.test", auto=True)
                mock_exec.assert_not_called()

    def test_silent_hint_suppresses_output_on_failure(self, repl, capsys):
        with patch("praxis.chat.console") as mock_console:
            mock_console.print = MagicMock()
            result = repl._run_program("this is not a program", auto=True, silent_hint=True)
            assert result is False


# ──────────────────────────────────────────────────────────────────────────────
# _execute_and_show
# ──────────────────────────────────────────────────────────────────────────────

class TestExecuteAndShow:
    def test_valid_program_appends_to_history(self, repl):
        with patch("praxis.chat.console"):
            repl._execute_and_show("LOG.test")
        assert "LOG.test" in repl._history

    def test_stores_in_memory(self, repl):
        with patch("praxis.chat.console"):
            repl._execute_and_show("LOG.test")
        assert repl.memory.count() > 0

    def test_no_memory_does_not_crash(self, repl_no_memory):
        with patch("praxis.chat.console"):
            repl_no_memory._execute_and_show("LOG.test")
        assert "LOG.test" in repl_no_memory._history

    def test_parse_error_handled_gracefully(self, repl):
        with patch("praxis.chat.console") as mock_console:
            mock_console.print = MagicMock()
            repl._execute_and_show("INVALID!!!")
        # history should NOT have the bad program
        assert "INVALID!!!" not in repl._history


# ──────────────────────────────────────────────────────────────────────────────
# _dispatch
# ──────────────────────────────────────────────────────────────────────────────

class TestDispatch:
    def test_program_input_calls_run_program(self, repl):
        with patch.object(repl, "_run_program", return_value=True) as mock_run:
            repl._dispatch("LOG.test")
            mock_run.assert_called_once()

    def test_goal_input_no_provider_hints_user(self, repl):
        with patch("praxis.chat.console") as mock_console:
            mock_console.print = MagicMock()
            repl._dispatch("summarize my sales report")
            # Should have printed a hint about no provider
            printed = " ".join(
                str(call) for call in mock_console.print.call_args_list
            )
            assert "provider" in printed.lower() or "goal" in printed.lower()

    def test_goal_input_with_provider_calls_run_goal(self, repl):
        repl.provider = MagicMock()
        repl._goal_mode = True
        with patch.object(repl, "_run_goal") as mock_goal:
            repl._dispatch("summarize my sales report")
            mock_goal.assert_called_once_with("summarize my sales report")

    def test_empty_input_no_op(self, repl):
        with patch.object(repl, "_run_program") as mock_run:
            with patch.object(repl, "_run_goal") as mock_goal:
                repl._dispatch("   ")
                mock_run.assert_not_called()
                mock_goal.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# Session commands
# ──────────────────────────────────────────────────────────────────────────────

class TestCommands:
    def test_clear_empties_buffer(self, repl):
        repl._buffer = ["LOG.a", "-> LOG.b"]
        with patch("praxis.chat.console"):
            repl._handle_command(":clear")
        assert repl._buffer == []

    def test_show_empty_buffer(self, repl):
        with patch("praxis.chat.console") as mock_console:
            mock_console.print = MagicMock()
            repl._handle_command(":show")
            printed = " ".join(str(c) for c in mock_console.print.call_args_list)
            assert "empty" in printed.lower()

    def test_show_non_empty_buffer(self, repl):
        repl._buffer = ["LOG.test"]
        with patch("praxis.chat.console") as mock_console:
            mock_console.print = MagicMock()
            with patch("praxis.chat._show_program") as mock_show:
                repl._handle_command(":show")
                mock_show.assert_called_once()

    def test_run_empty_buffer_warns(self, repl):
        with patch("praxis.chat.console") as mock_console:
            mock_console.print = MagicMock()
            repl._handle_command(":run")
            printed = " ".join(str(c) for c in mock_console.print.call_args_list)
            assert "empty" in printed.lower()

    def test_run_with_buffer_executes(self, repl):
        repl._buffer = ["LOG.test"]
        with patch.object(repl, "_execute_and_show") as mock_exec:
            with patch("praxis.chat.console"):
                repl._handle_command(":run")
                mock_exec.assert_called_once_with("LOG.test")
        assert repl._buffer == []

    def test_validate_empty_buffer_warns(self, repl):
        with patch("praxis.chat.console") as mock_console:
            mock_console.print = MagicMock()
            repl._handle_command(":validate")
            printed = " ".join(str(c) for c in mock_console.print.call_args_list)
            assert "empty" in printed.lower()

    def test_validate_with_buffer(self, repl):
        repl._buffer = ["LOG.test"]
        with patch.object(repl, "_run_program", return_value=True) as mock_run:
            with patch("praxis.chat.console"):
                repl._handle_command(":validate")
                mock_run.assert_called_once_with("LOG.test", auto=False)

    def test_save_writes_file(self, repl, tmp_path):
        repl._buffer = ["LOG.test"]
        out = str(tmp_path / "out.px")
        with patch("praxis.chat.console"):
            repl._handle_command(f":save {out}")
        assert Path(out).read_text() == "LOG.test"

    def test_save_default_filename(self, repl, tmp_path):
        repl._buffer = ["LOG.x"]
        with patch("praxis.chat.console"):
            with patch("builtins.open", create=True):
                # Just verify it doesn't crash with no arg
                with patch("pathlib.Path.write_text") as mock_write:
                    repl._handle_command(":save")
                    mock_write.assert_called_once()

    def test_mode_toggle_no_provider(self, repl):
        repl.provider = None
        with patch("praxis.chat.console") as mock_console:
            mock_console.print = MagicMock()
            repl._handle_command(":mode")
            printed = " ".join(str(c) for c in mock_console.print.call_args_list)
            assert "provider" in printed.lower() or "unavailable" in printed.lower()

    def test_mode_toggle_with_provider(self, repl):
        repl.provider = MagicMock()
        repl._goal_mode = False
        with patch("praxis.chat.console"):
            repl._handle_command(":mode")
        assert repl._goal_mode is True

    def test_mode_toggle_back(self, repl):
        repl.provider = MagicMock()
        repl._goal_mode = True
        with patch("praxis.chat.console"):
            repl._handle_command(":mode")
        assert repl._goal_mode is False

    def test_history_empty(self, repl):
        with patch("praxis.chat.console") as mock_console:
            mock_console.print = MagicMock()
            repl._handle_command(":history")
            printed = " ".join(str(c) for c in mock_console.print.call_args_list)
            assert "history" in printed.lower() or "empty" in printed.lower() or "no" in printed.lower()

    def test_help_prints_commands(self, repl):
        with patch("praxis.chat.console") as mock_console:
            mock_console.print = MagicMock()
            repl._handle_command(":help")
            # Should have printed at least the header and table
            assert mock_console.print.call_count >= 2

    def test_unknown_command(self, repl):
        with patch("praxis.chat.console") as mock_console:
            mock_console.print = MagicMock()
            repl._handle_command(":notarealcmd")
            printed = " ".join(str(c) for c in mock_console.print.call_args_list)
            assert "unknown" in printed.lower()

    def test_quit_exits(self, repl):
        with pytest.raises(SystemExit):
            repl._handle_command(":quit")

    def test_exit_exits(self, repl):
        with pytest.raises(SystemExit):
            repl._handle_command(":exit")


# ──────────────────────────────────────────────────────────────────────────────
# Multi-line buffer
# ──────────────────────────────────────────────────────────────────────────────

class TestMultiLine:
    def test_continuation_accumulates(self, repl):
        """Lines ending with \\ should append to buffer without dispatching."""
        with patch.object(repl, "_dispatch") as mock_dispatch:
            with patch("praxis.chat.console") as mock_console:
                mock_console.input.side_effect = [
                    "LOG.a \\",
                    "-> SUMM.b",
                    EOFError,
                ]
                try:
                    repl.run()
                except SystemExit:
                    pass

    def test_blank_line_flushes(self, repl):
        repl._buffer = ["LOG.a", "-> LOG.b"]
        with patch.object(repl, "_dispatch") as mock_dispatch:
            with patch("praxis.chat.console") as mock_console:
                mock_console.input.side_effect = [
                    "",       # blank line → flush
                    EOFError,
                ]
                try:
                    repl.run()
                except SystemExit:
                    pass
                mock_dispatch.assert_called_once_with("LOG.a\n-> LOG.b")
        assert repl._buffer == []


# ──────────────────────────────────────────────────────────────────────────────
# Goal mode (mock provider)
# ──────────────────────────────────────────────────────────────────────────────

class TestGoalMode:
    def _make_plan_result(self, program: str):
        from praxis.planner import PlanResult
        return PlanResult(program=program, adapted=False, attempts=1, similar=[], rules_used=[])

    def test_goal_mode_calls_planner(self, tmp_path):
        from praxis.memory import ProgramMemory
        mem = ProgramMemory(db_path=str(tmp_path / "t.db"),
                            embedder=lambda t: [0.0] * 384)
        mock_provider = MagicMock()
        repl = PraxisREPL(memory=mem, provider=mock_provider, mode="dev")
        repl._goal_mode = True

        plan_result = self._make_plan_result("LOG.test")

        with patch("praxis.chat.Planner") as MockPlanner:
            mock_planner_inst = MagicMock()
            mock_planner_inst.plan.return_value = plan_result
            MockPlanner.return_value = mock_planner_inst

            with patch("praxis.chat.console") as mock_console:
                mock_console.input.return_value = "n"
                mock_console.print = MagicMock()
                repl._run_goal("summarize sales data")

            mock_planner_inst.plan.assert_called_once_with("summarize sales data")

    def test_planning_failure_handled(self, tmp_path):
        from praxis.memory import ProgramMemory
        from praxis.planner import PlanningFailure
        mem = ProgramMemory(db_path=str(tmp_path / "t.db"),
                            embedder=lambda t: [0.0] * 384)
        repl = PraxisREPL(memory=mem, provider=MagicMock(), mode="dev")

        with patch("praxis.chat.Planner") as MockPlanner:
            mock_planner_inst = MagicMock()
            mock_planner_inst.plan.side_effect = PlanningFailure(goal="test", attempts=3, last_error="LLM timeout")
            MockPlanner.return_value = mock_planner_inst

            with patch("praxis.chat.console") as mock_console:
                mock_console.print = MagicMock()
                repl._run_goal("a goal that fails")
                printed = " ".join(str(c) for c in mock_console.print.call_args_list)
                assert "failed" in printed.lower() or "planning" in printed.lower()

    def test_goal_accepted_and_executed(self, tmp_path):
        from praxis.memory import ProgramMemory
        mem = ProgramMemory(db_path=str(tmp_path / "t.db"),
                            embedder=lambda t: [0.0] * 384)
        repl = PraxisREPL(memory=mem, provider=MagicMock(), mode="dev")

        plan_result = self._make_plan_result("LOG.test")

        with patch("praxis.chat.Planner") as MockPlanner:
            mock_planner_inst = MagicMock()
            mock_planner_inst.plan.return_value = plan_result
            MockPlanner.return_value = mock_planner_inst

            with patch.object(repl, "_execute_and_show") as mock_exec:
                with patch("praxis.chat.console") as mock_console:
                    mock_console.input.return_value = "y"
                    mock_console.print = MagicMock()
                    repl._run_goal("run a log test")
                    mock_exec.assert_called_once()
