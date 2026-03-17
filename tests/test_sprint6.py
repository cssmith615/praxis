"""
Sprint 6 tests — multi-agent: SPAWN, MSG, CAST, JOIN, SIGN, cycle detection.
All tests are in-process; no Redis, no network.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from praxis.agent_registry import AgentRegistry, Worker, sign_message, verify_message
from praxis.executor import Executor, ExecutionContext
from praxis.grammar import parse
from praxis.handlers import HANDLERS
from praxis.handlers.agents import (
    spawn_handler, msg_handler, cast_handler, join_handler, sign_handler, cap_handler,
)
from praxis.validator import validate


# ─── helpers ─────────────────────────────────────────────────────────────────

def _ctx(**kwargs):
    ctx = ExecutionContext()
    for k, v in kwargs.items():
        setattr(ctx, k, v)
    return ctx


def _exe(extra=None):
    h = dict(HANDLERS)
    if extra:
        h.update(extra)
    return Executor(h)


# ══════════════════════════════════════════════════════════════════════════════
# AgentRegistry
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentRegistry:
    def test_register_and_get(self):
        reg    = AgentRegistry()
        worker = Worker("w1", "data", ["ING", "CLN"], executor=_exe())
        reg.register(worker)
        assert reg.get("w1") is worker

    def test_get_missing_returns_none(self):
        reg = AgentRegistry()
        assert reg.get("nonexistent") is None

    def test_route_by_verb(self):
        reg = AgentRegistry()
        reg.register(Worker("data", "data", ["ING", "CLN", "XFRM"], executor=_exe()))
        reg.register(Worker("analysis", "analysis", ["SUMM", "EVAL"], executor=_exe()))
        assert reg.route("ING").agent_id == "data"
        assert reg.route("SUMM").agent_id == "analysis"
        assert reg.route("UNKNOWN") is None

    def test_route_is_case_insensitive(self):
        reg = AgentRegistry()
        reg.register(Worker("w", "data", ["ing"], executor=_exe()))
        assert reg.route("ING") is not None

    def test_remove(self):
        reg = AgentRegistry()
        reg.register(Worker("w1", "data", ["ING"], executor=_exe()))
        assert reg.remove("w1") is True
        assert reg.get("w1") is None

    def test_remove_missing(self):
        reg = AgentRegistry()
        assert reg.remove("ghost") is False

    def test_capability_map(self):
        reg = AgentRegistry()
        reg.register(Worker("a", "data",     ["ING", "CLN"], executor=_exe()))
        reg.register(Worker("b", "analysis", ["SUMM"],       executor=_exe()))
        cap = reg.capability_map()
        assert "ING"  in cap and "a" in cap["ING"]
        assert "SUMM" in cap and "b" in cap["SUMM"]


# ══════════════════════════════════════════════════════════════════════════════
# Worker execution
# ══════════════════════════════════════════════════════════════════════════════

class TestWorker:
    def test_worker_executes_valid_program(self):
        worker = Worker("w", "test", ["LOG"], executor=_exe())
        result = worker.execute("LOG.step1")
        assert result["status"] == "ok"
        assert result["agent_id"] == "w"
        assert result["steps"] >= 1

    def test_worker_handles_parse_error_gracefully(self):
        worker = Worker("w", "test", [], executor=_exe())
        result = worker.execute("NOT VALID PRAXIS !!!")
        assert result["status"] == "error"
        assert "error" in result

    def test_worker_returns_last_output(self):
        worker = Worker("w", "test", ["STORE"], executor=_exe())
        # STORE returns {"stored": key, "db": ...}; last step output is that dict
        result = worker.execute("STORE.x(key=testkey, value=42)")
        assert result["status"] == "ok"
        assert result["output"] is not None


# ══════════════════════════════════════════════════════════════════════════════
# SPAWN handler
# ══════════════════════════════════════════════════════════════════════════════

class TestSpawn:
    def test_spawn_creates_worker_in_registry(self):
        ctx = _ctx(last_output=None)
        result = spawn_handler(["data_worker"], {"role": "data", "verbs": ["ING", "CLN"]}, ctx)
        assert result["status"] == "spawned"
        assert result["agent_id"] == "data_worker"
        assert ctx.agent_registry is not None
        worker = ctx.agent_registry.get("data_worker")
        assert worker is not None
        assert worker.role == "data"
        assert "ING" in worker.verbs

    def test_spawn_creates_registry_if_not_present(self):
        ctx = _ctx(last_output=None)
        assert not hasattr(ctx, "agent_registry") or ctx.agent_registry is None
        spawn_handler(["w"], {"role": "test", "verbs": ["LOG"]}, ctx)
        assert ctx.agent_registry is not None

    def test_spawn_multiple_workers(self):
        ctx = _ctx(last_output=None)
        spawn_handler(["data"], {"role": "data", "verbs": ["ING"]}, ctx)
        spawn_handler(["analysis"], {"role": "analysis", "verbs": ["SUMM"]}, ctx)
        assert len(ctx.agent_registry.all_workers()) == 2

    def test_spawn_accepts_comma_separated_verbs_string(self):
        ctx = _ctx(last_output=None)
        spawn_handler(["w"], {"role": "test", "verbs": "ING,CLN,XFRM"}, ctx)
        w = ctx.agent_registry.get("w")
        assert "ING" in w.verbs and "CLN" in w.verbs and "XFRM" in w.verbs


# ══════════════════════════════════════════════════════════════════════════════
# MSG handler
# ══════════════════════════════════════════════════════════════════════════════

class TestMsg:
    def test_msg_dispatches_to_worker(self):
        ctx = _ctx(last_output=None)
        spawn_handler(["log_worker"], {"role": "logger", "verbs": ["LOG"]}, ctx)
        result = msg_handler(["log_worker"], {"program": "LOG.step1"}, ctx)
        assert result["dispatched"] is True
        assert result["to"] == "log_worker"
        assert any(k.startswith("msg_") for k in ctx.pending_futures)

    def test_msg_raises_when_no_registry(self):
        ctx = _ctx(last_output=None)
        with pytest.raises(RuntimeError, match="no AgentRegistry"):
            msg_handler(["ghost"], {"program": "LOG.x"}, ctx)

    def test_msg_raises_when_worker_not_found(self):
        ctx = _ctx(last_output=None)
        spawn_handler(["w"], {"role": "test", "verbs": ["LOG"]}, ctx)
        with pytest.raises(RuntimeError, match="no worker registered"):
            msg_handler(["nonexistent"], {"program": "LOG.x"}, ctx)

    def test_msg_uses_last_output_when_no_program_param(self):
        ctx = _ctx(last_output="LOG.auto")
        spawn_handler(["w"], {"role": "test", "verbs": ["LOG"]}, ctx)
        result = msg_handler(["w"], {}, ctx)
        assert result["program"] == "LOG.auto"


# ══════════════════════════════════════════════════════════════════════════════
# JOIN handler
# ══════════════════════════════════════════════════════════════════════════════

class TestJoin:
    def test_join_collects_results(self):
        ctx = _ctx(last_output=None)
        spawn_handler(["w1"], {"role": "test", "verbs": ["LOG"]}, ctx)
        spawn_handler(["w2"], {"role": "test", "verbs": ["ANNOTATE"]}, ctx)
        msg_handler(["w1"], {"program": "LOG.step"}, ctx)
        msg_handler(["w2"], {"program": "ANNOTATE.step"}, ctx)
        result = join_handler([], {"timeout": 10}, ctx)
        assert result["count"] == 2
        assert result["success"] is True
        assert len(ctx.pending_futures) == 0   # cleared after JOIN

    def test_join_empty_returns_zero(self):
        ctx = _ctx(last_output=None)
        ctx.pending_futures = {}
        result = join_handler([], {}, ctx)
        assert result["count"] == 0
        assert result["joined"] == []

    def test_join_reports_worker_errors(self):
        ctx = _ctx(last_output=None)
        spawn_handler(["bad_worker"], {"role": "test", "verbs": ["LOG"]}, ctx)
        msg_handler(["bad_worker"], {"program": "NOT VALID !!!"}, ctx)
        result = join_handler([], {"timeout": 5}, ctx)
        assert result["count"] == 1
        assert result["errors"]  # should have at least one error
        assert result["success"] is False


# ══════════════════════════════════════════════════════════════════════════════
# CAST handler
# ══════════════════════════════════════════════════════════════════════════════

class TestCast:
    def test_cast_dispatches_to_all_workers(self):
        ctx = _ctx(last_output=None)
        spawn_handler(["w1"], {"role": "a", "verbs": ["LOG"]}, ctx)
        spawn_handler(["w2"], {"role": "b", "verbs": ["ANNOTATE"]}, ctx)
        result = cast_handler([], {"program": "LOG.broadcast"}, ctx)
        assert set(result["broadcast_to"]) == {"w1", "w2"}
        assert result["pending"] == 2
        assert len(ctx.pending_futures) == 2

    def test_cast_then_join(self):
        ctx = _ctx(last_output=None)
        spawn_handler(["w1"], {"role": "a", "verbs": ["LOG"]}, ctx)
        spawn_handler(["w2"], {"role": "b", "verbs": ["ANNOTATE"]}, ctx)
        cast_handler([], {"program": "LOG.x"}, ctx)
        result = join_handler([], {"timeout": 10}, ctx)
        assert result["count"] == 2


# ══════════════════════════════════════════════════════════════════════════════
# SIGN handler + HMAC utilities
# ══════════════════════════════════════════════════════════════════════════════

class TestSign:
    def test_sign_returns_hmac_signature(self):
        ctx = _ctx(last_output="secret payload")
        result = sign_handler([], {}, ctx)
        assert "signature" in result
        assert len(result["signature"]) == 64  # sha256 hex = 64 chars
        assert result["algorithm"] == "hmac-sha256"

    def test_sign_stores_signature_in_ctx(self):
        ctx = _ctx(last_output="payload")
        sign_handler([], {}, ctx)
        assert "_signature" in ctx.variables

    def test_sign_verify_roundtrip(self):
        import os
        key = os.urandom(32)
        sig = sign_message("hello world", key)
        assert verify_message("hello world", sig, key) is True
        assert verify_message("tampered",   sig, key) is False

    def test_sign_with_explicit_key_param(self):
        import os
        key = os.urandom(32)
        ctx = _ctx(last_output="data")
        result = sign_handler([], {"key": key.hex(), "payload": "data"}, ctx)
        assert verify_message("data", result["signature"], key)


# ══════════════════════════════════════════════════════════════════════════════
# CAP handler
# ══════════════════════════════════════════════════════════════════════════════

class TestCap:
    def test_cap_stores_capabilities_in_ctx(self):
        ctx = _ctx(last_output=None)
        result = cap_handler(["my_agent"], {"role": "worker", "allow": ["ING", "CLN"]}, ctx)
        assert result["role"] == "worker"
        assert "ING" in result["capabilities"]
        assert ctx.variables["_capabilities"]["my_agent"]["role"] == "worker"

    def test_cap_accepts_comma_string_allow(self):
        ctx = _ctx(last_output=None)
        cap_handler(["a"], {"role": "r", "allow": "ING,SUMM,OUT"}, ctx)
        assert "ING" in ctx.variables["_capabilities"]["a"]["capabilities"]


# ══════════════════════════════════════════════════════════════════════════════
# Validator — MSG cycle detection
# ══════════════════════════════════════════════════════════════════════════════

class TestMsgCycleDetection:
    def test_self_msg_flagged_in_plan(self):
        # A PLAN that MSGs itself — cycle
        program = parse("""
PLAN:coordinator {
    MSG.coordinator(program="LOG.x")
}
CALL.coordinator
        """)
        errors = validate(program)
        assert any("MSG cycle" in e for e in errors)

    def test_no_cycle_for_different_targets(self):
        program = parse('SPAWN.worker(role=test, verbs="LOG") -> MSG.worker(program="LOG.x") -> JOIN')
        errors = validate(program)
        assert not any("MSG cycle" in e for e in errors)

    def test_msg_without_target_passes_cycle_check(self):
        program = parse("MSG.worker(program=\"LOG.x\")")
        errors = validate(program)
        assert not any("MSG cycle" in e for e in errors)


# ══════════════════════════════════════════════════════════════════════════════
# End-to-end: full coordinator/worker flow via executor
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_spawn_msg_join_full_flow(self):
        program = parse(
            'SPAWN.logger(role=log, verbs="LOG,ANNOTATE") -> '
            'MSG.logger(program="LOG.step1") -> '
            "JOIN(timeout=10)"
        )
        exe = _exe()
        results = exe.execute(program)

        verbs = [r["verb"] for r in results]
        assert "SPAWN"  in verbs
        assert "MSG"    in verbs
        assert "JOIN"   in verbs

        join_r = next(r for r in results if r["verb"] == "JOIN")
        assert join_r["output"]["count"] == 1
        assert join_r["output"]["success"] is True

    def test_multi_worker_parallel_dispatch(self):
        program = parse(
            'SPAWN.w1(role=a, verbs="LOG") -> '
            'SPAWN.w2(role=b, verbs="ANNOTATE") -> '
            'PAR(MSG.w1(program="LOG.step"), MSG.w2(program="ANNOTATE.step")) -> '
            "JOIN(timeout=10)"
        )
        exe = _exe()
        results = exe.execute(program)
        join_r = next(r for r in results if r["verb"] == "JOIN")
        assert join_r["output"]["count"] == 2
        assert join_r["output"]["success"] is True
