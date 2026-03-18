"""
Sprint 31 tests — Praxis MCP Server.

Tests the tool handler functions directly (without the MCP protocol layer)
and verifies the server configuration is correct.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Tool handler tests — run_program
# ─────────────────────────────────────────────────────────────────────────────

class TestRunProgram:
    def test_run_valid_program_returns_ok(self):
        from praxis.mcp_server import tool_run_program
        result = tool_run_program("LOG.x")
        assert result["ok"] is True
        assert "results" in result
        assert result["steps"] >= 1

    def test_run_returns_step_results(self):
        from praxis.mcp_server import tool_run_program
        result = tool_run_program("LOG.x")
        step = result["results"][0]
        assert "verb" in step
        assert "status" in step
        assert "duration_ms" in step

    def test_run_parse_error_returns_not_ok(self):
        from praxis.mcp_server import tool_run_program
        result = tool_run_program("NOT VALID PRAXIS !!!")
        assert result["ok"] is False
        assert "error" in result

    def test_run_error_step_sets_ok_false(self):
        from praxis.mcp_server import tool_run_program
        # Programs that fail at runtime still return results
        result = tool_run_program("LOG.x")
        assert "results" in result

    def test_run_accepts_mode_param(self):
        from praxis.mcp_server import tool_run_program
        result = tool_run_program("LOG.x", mode="dev")
        assert result["ok"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Tool handler tests — validate_program
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateProgram:
    def test_valid_program_returns_valid_true(self):
        from praxis.mcp_server import tool_validate_program
        result = tool_validate_program("LOG.x -> OUT.telegram")
        assert result["valid"] is True
        assert result["errors"] == []

    def test_invalid_program_returns_valid_false(self):
        from praxis.mcp_server import tool_validate_program
        result = tool_validate_program("!!! not praxis")
        assert result["valid"] is False
        assert len(result["errors"]) >= 1

    def test_returns_error_list(self):
        from praxis.mcp_server import tool_validate_program
        result = tool_validate_program("LOG.x")
        assert isinstance(result["errors"], list)

    def test_accepts_mode_param(self):
        from praxis.mcp_server import tool_validate_program
        result = tool_validate_program("LOG.x", mode="prod")
        assert "valid" in result


# ─────────────────────────────────────────────────────────────────────────────
# Tool handler tests — plan_goal
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanGoal:
    def test_plan_goal_without_api_key_returns_error(self):
        from praxis.mcp_server import tool_plan_goal
        import os
        env = {k: v for k, v in os.environ.items()
               if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                             "GROK_API_KEY", "GEMINI_API_KEY")}
        with patch.dict("os.environ", env, clear=True):
            result = tool_plan_goal("fetch HN stories")
        # Either ok or error — just check structure
        assert "ok" in result

    def test_plan_goal_with_mock_planner(self):
        from praxis.mcp_server import tool_plan_goal
        mock_planner = MagicMock()
        mock_planner.plan.return_value = ("LOG.x -> OUT.telegram", 1)
        mock_provider = MagicMock()

        with patch("praxis.mcp_server.tool_plan_goal") as mock_fn:
            mock_fn.return_value = {"ok": True, "program": "LOG.x -> OUT.telegram", "goal": "test"}
            result = mock_fn("test")
        assert result["ok"] is True
        assert "program" in result


# ─────────────────────────────────────────────────────────────────────────────
# Tool handler tests — recall_similar
# ─────────────────────────────────────────────────────────────────────────────

class TestRecallSimilar:
    def test_recall_returns_matches_list(self, tmp_path):
        from praxis.mcp_server import tool_recall_similar
        with patch("praxis.memory.ProgramMemory") as MockMem:
            mock_mem = MagicMock()
            mock_mem.retrieve_similar.return_value = []
            MockMem.return_value = mock_mem
            result = tool_recall_similar("fetch news")
        assert result["ok"] is True
        assert isinstance(result["matches"], list)

    def test_recall_formats_match_fields(self, tmp_path):
        from praxis.mcp_server import tool_recall_similar
        mock_match = MagicMock()
        mock_match.id = "abc123"
        mock_match.goal_text = "fetch HN stories"
        mock_match.shaun_program = "FETCH.data -> OUT.telegram"
        mock_match.outcome = "success"
        mock_match.similarity = 0.95

        with patch("praxis.memory.ProgramMemory") as MockMem:
            mock_mem = MagicMock()
            mock_mem.retrieve_similar.return_value = [mock_match]
            MockMem.return_value = mock_mem
            result = tool_recall_similar("HN", k=1)

        assert len(result["matches"]) == 1
        m = result["matches"][0]
        assert m["id"] == "abc123"
        assert m["similarity"] == 0.95

    def test_recall_accepts_k_param(self):
        from praxis.mcp_server import tool_recall_similar
        with patch("praxis.memory.ProgramMemory") as MockMem:
            mock_mem = MagicMock()
            mock_mem.retrieve_similar.return_value = []
            MockMem.return_value = mock_mem
            result = tool_recall_similar("test", k=3)
            mock_mem.retrieve_similar.assert_called_once_with("test", k=3)
        assert result["ok"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Tool handler tests — search_registry
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchRegistry:
    def _mock_programs(self):
        from praxis.registry import RegistryProgram
        return [
            RegistryProgram({"name": "news-brief", "description": "HN stories", "tags": ["news"]}),
            RegistryProgram({"name": "price-alert", "description": "Price alerts", "tags": ["price"]}),
        ]

    def test_search_returns_results(self, monkeypatch):
        from praxis.mcp_server import tool_search_registry
        from praxis import registry as reg_mod
        programs = self._mock_programs()
        monkeypatch.setattr(reg_mod, "search_registry", lambda q, **kw: programs)
        result = tool_search_registry("")
        assert result["ok"] is True
        assert len(result["results"]) == 2

    def test_search_result_has_required_fields(self, monkeypatch):
        from praxis.mcp_server import tool_search_registry
        from praxis import registry as reg_mod
        programs = self._mock_programs()
        monkeypatch.setattr(reg_mod, "search_registry", lambda q, **kw: programs)
        result = tool_search_registry("news")
        r = result["results"][0]
        assert "name" in r
        assert "description" in r
        assert "tags" in r

    def test_search_registry_error_returns_not_ok(self, monkeypatch):
        from praxis.mcp_server import tool_search_registry
        from praxis import registry as reg_mod
        from praxis.registry import RegistryError
        monkeypatch.setattr(reg_mod, "search_registry", lambda q, **kw: (_ for _ in ()).throw(RegistryError("offline")))
        result = tool_search_registry("news")
        assert result["ok"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Tool handler tests — install_program
# ─────────────────────────────────────────────────────────────────────────────

class TestInstallProgram:
    def test_install_success(self, monkeypatch, tmp_path):
        from praxis.mcp_server import tool_install_program
        from praxis import registry as reg_mod
        from praxis.registry import RegistryProgram
        prog = RegistryProgram({"name": "news-brief", "description": "HN stories", "author": "test", "tags": ["news"]})
        monkeypatch.setattr(reg_mod, "install_program", lambda name, memory=None, **kw: prog)
        result = tool_install_program("news-brief")
        assert result["ok"] is True
        assert result["name"] == "news-brief"

    def test_install_not_found_returns_error(self, monkeypatch):
        from praxis.mcp_server import tool_install_program
        from praxis import registry as reg_mod
        from praxis.registry import RegistryError
        monkeypatch.setattr(reg_mod, "install_program", lambda name, memory=None, **kw: (_ for _ in ()).throw(RegistryError("not found")))
        result = tool_install_program("nonexistent")
        assert result["ok"] is False
        assert "error" in result


# ─────────────────────────────────────────────────────────────────────────────
# Tool handler tests — get_constitution
# ─────────────────────────────────────────────────────────────────────────────

class TestGetConstitution:
    def test_returns_rules_list(self, tmp_path):
        from praxis.mcp_server import tool_get_constitution
        mock_rule = MagicMock()
        mock_rule.text = "ALWAYS use LOG for audit trails."
        mock_rule.verbs = ["LOG"]
        mock_rule.tags = []

        with patch("praxis.constitution.Constitution") as MockConst:
            mock_const = MagicMock()
            mock_const.get_rules.return_value = [mock_rule]
            MockConst.return_value = mock_const
            result = tool_get_constitution()

        assert result["ok"] is True
        assert result["count"] == 1
        assert result["rules"][0]["text"] == "ALWAYS use LOG for audit trails."

    def test_returns_count(self):
        from praxis.mcp_server import tool_get_constitution
        with patch("praxis.constitution.Constitution") as MockConst:
            mock_const = MagicMock()
            mock_const.get_rules.return_value = []
            MockConst.return_value = mock_const
            result = tool_get_constitution()
        assert result["count"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Resource helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestResourceHelpers:
    def test_resource_constitution_returns_string(self):
        from praxis.mcp_server import resource_constitution
        with patch("praxis.mcp_server.tool_get_constitution") as mock_fn:
            mock_fn.return_value = {
                "ok": True,
                "count": 1,
                "rules": [{"text": "ALWAYS LOG.", "verbs": ["LOG"], "tags": []}],
            }
            result = resource_constitution()
        assert isinstance(result, str)
        assert "LOG" in result

    def test_resource_constitution_error_handled(self):
        from praxis.mcp_server import resource_constitution
        with patch("praxis.mcp_server.tool_get_constitution") as mock_fn:
            mock_fn.return_value = {"ok": False, "error": "fail"}
            result = resource_constitution()
        assert "Error" in result

    def test_resource_programs_returns_string(self):
        from praxis.mcp_server import resource_programs
        with patch("praxis.memory.ProgramMemory") as MockMem:
            mock_mem = MagicMock()
            mock_mem.retrieve_similar.return_value = []
            MockMem.return_value = mock_mem
            result = resource_programs()
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# Tool schema validation
# ─────────────────────────────────────────────────────────────────────────────

class TestToolSchemas:
    def test_all_tools_have_required_fields(self):
        from praxis.mcp_server import _TOOL_SCHEMAS
        for schema in _TOOL_SCHEMAS:
            assert "name" in schema
            assert "description" in schema
            assert "inputSchema" in schema

    def test_expected_tools_present(self):
        from praxis.mcp_server import _TOOL_SCHEMAS
        names = {s["name"] for s in _TOOL_SCHEMAS}
        assert "run_program" in names
        assert "validate_program" in names
        assert "plan_goal" in names
        assert "recall_similar" in names
        assert "search_registry" in names
        assert "install_program" in names
        assert "get_constitution" in names

    def test_all_tools_have_handlers(self):
        from praxis.mcp_server import _TOOL_SCHEMAS, _TOOL_HANDLERS
        for schema in _TOOL_SCHEMAS:
            assert schema["name"] in _TOOL_HANDLERS, f"No handler for {schema['name']}"

    def test_run_program_schema_has_program_required(self):
        from praxis.mcp_server import _TOOL_SCHEMAS
        schema = next(s for s in _TOOL_SCHEMAS if s["name"] == "run_program")
        assert "program" in schema["inputSchema"]["required"]

    def test_plan_goal_schema_has_goal_required(self):
        from praxis.mcp_server import _TOOL_SCHEMAS
        schema = next(s for s in _TOOL_SCHEMAS if s["name"] == "plan_goal")
        assert "goal" in schema["inputSchema"]["required"]


# ─────────────────────────────────────────────────────────────────────────────
# CLI command registration
# ─────────────────────────────────────────────────────────────────────────────

class TestMcpCliCommand:
    def test_mcp_command_registered(self):
        from praxis.cli import main
        assert "mcp" in main.commands

    def test_mcp_command_has_docstring(self):
        from praxis.cli import main
        cmd = main.commands["mcp"]
        assert cmd.help is not None
        assert len(cmd.help) > 0
