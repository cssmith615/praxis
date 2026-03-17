"""
Sprint 17 tests — WASM code generator + praxis compile --target wasm (Pillar 2-PartB).
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from praxis import parse
from praxis.codegen.wasm import WasmGenerator
from praxis.ast_types import (
    Program, Chain, VerbAction, ParBlock, IfStmt, Block,
    LoopStmt, NamedCond, Comparison, Break, Wait,
)
from praxis.cli import main


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gen(text: str) -> str:
    return WasmGenerator().generate(parse(text))


def _prog(*verbs) -> Program:
    return Program(statements=[Chain(steps=[
        VerbAction(verb=v, target=["x"], params={}) for v in verbs
    ])])


# ══════════════════════════════════════════════════════════════════════════════
# Module structure
# ══════════════════════════════════════════════════════════════════════════════

class TestWatModuleStructure:
    def test_returns_string(self):
        assert isinstance(_gen("LOG.test"), str)

    def test_starts_with_module(self):
        assert "(module" in _gen("LOG.test")

    def test_ends_with_closing_paren(self):
        wat = _gen("LOG.test").strip()
        assert wat.endswith(")")

    def test_contains_run_export(self):
        assert 'export "run"' in _gen("LOG.test")

    def test_contains_dispatch_import(self):
        assert "$dispatch" in _gen("LOG.test")

    def test_contains_dispatch_params(self):
        assert "i32 i32 i32" in _gen("LOG.test")

    def test_contains_step_count_global(self):
        assert "$step_count" in _gen("LOG.test")

    def test_contains_memory_export(self):
        # Memory is only emitted when there are strings
        wat = _gen("LOG.test")
        assert "(memory" in wat

    def test_contains_data_section(self):
        wat = _gen("LOG.test")
        assert "(data" in wat

    def test_header_comment(self):
        assert "praxis compile" in _gen("LOG.test").lower() or "Generated" in _gen("LOG.test") or "Praxis WAT" in _gen("LOG.test")


# ══════════════════════════════════════════════════════════════════════════════
# Verb dispatch
# ══════════════════════════════════════════════════════════════════════════════

class TestWatVerbDispatch:
    def test_verb_call_dispatch_present(self):
        assert "call $dispatch" in _gen("LOG.test")

    def test_verb_name_in_data_section(self):
        wat = _gen("LOG.test")
        assert '"LOG"' in wat or "LOG" in wat

    def test_step_count_incremented(self):
        wat = _gen("LOG.test")
        assert "i32.add" in wat

    def test_multiple_steps_multiple_dispatches(self):
        wat = _gen("LOG.a -> ANNOTATE.b -> OUT.c")
        assert wat.count("call $dispatch") == 3

    def test_set_verb_calls_set_var(self):
        steps = [
            VerbAction(verb="LOG", target=["x"], params={}),
            VerbAction(verb="SET", target=["myvar"], params={}),
        ]
        prog = Program(statements=[Chain(steps=steps)])
        wat = WasmGenerator().generate(prog)
        assert "call $set_var" in wat

    def test_target_interned_in_data_section(self):
        wat = _gen("LOG.test")
        assert "test" in wat


# ══════════════════════════════════════════════════════════════════════════════
# PAR block (sequential in WAT)
# ══════════════════════════════════════════════════════════════════════════════

class TestWatParBlock:
    def test_par_emits_both_verbs(self):
        par = ParBlock(branches=[
            VerbAction(verb="LOG", target=["a"], params={}),
            VerbAction(verb="ANNOTATE", target=["b"], params={}),
        ])
        wat = WasmGenerator().generate(Program(statements=[par]))
        assert wat.count("call $dispatch") == 2

    def test_par_has_sequential_comment(self):
        par = ParBlock(branches=[
            VerbAction(verb="LOG", target=["a"], params={}),
            VerbAction(verb="OUT", target=["b"], params={}),
        ])
        wat = WasmGenerator().generate(Program(statements=[par]))
        assert "sequential" in wat or "PAR" in wat


# ══════════════════════════════════════════════════════════════════════════════
# IF statement
# ══════════════════════════════════════════════════════════════════════════════

class TestWatIfStatement:
    def test_if_emits_wasm_if(self):
        cond = NamedCond(name="flag")
        then = VerbAction(verb="LOG", target=["x"], params={})
        stmt = IfStmt(condition=cond, then_body=then, else_body=None)
        wat = WasmGenerator().generate(Program(statements=[stmt]))
        assert "(if" in wat

    def test_then_block_present(self):
        cond = NamedCond(name="flag")
        then = VerbAction(verb="LOG", target=["x"], params={})
        stmt = IfStmt(condition=cond, then_body=then, else_body=None)
        wat = WasmGenerator().generate(Program(statements=[stmt]))
        assert "(then" in wat

    def test_else_block_present_when_provided(self):
        cond = NamedCond(name="flag")
        then = VerbAction(verb="LOG", target=["x"], params={})
        else_ = VerbAction(verb="ANNOTATE", target=["y"], params={})
        stmt = IfStmt(condition=cond, then_body=then, else_body=else_)
        wat = WasmGenerator().generate(Program(statements=[stmt]))
        assert "(else" in wat

    def test_comparison_operator_emitted(self):
        cond = Comparison(left=5, op=">", right=3)
        then = VerbAction(verb="LOG", target=["x"], params={})
        stmt = IfStmt(condition=cond, then_body=then, else_body=None)
        wat = WasmGenerator().generate(Program(statements=[stmt]))
        assert "i32.gt_s" in wat

    def test_eq_maps_to_i32_eq(self):
        cond = Comparison(left=1, op="==", right=1)
        then = VerbAction(verb="LOG", target=["x"], params={})
        stmt = IfStmt(condition=cond, then_body=then, else_body=None)
        wat = WasmGenerator().generate(Program(statements=[stmt]))
        assert "i32.eq" in wat


# ══════════════════════════════════════════════════════════════════════════════
# LOOP
# ══════════════════════════════════════════════════════════════════════════════

class TestWatLoop:
    def test_loop_emits_block_and_loop(self):
        until = NamedCond(name="done")
        body = VerbAction(verb="LOG", target=["x"], params={})
        stmt = LoopStmt(until=until, body=body)
        wat = WasmGenerator().generate(Program(statements=[stmt]))
        assert "(block" in wat
        assert "(loop" in wat

    def test_loop_has_back_edge(self):
        until = NamedCond(name="done")
        body = VerbAction(verb="LOG", target=["x"], params={})
        stmt = LoopStmt(until=until, body=body)
        wat = WasmGenerator().generate(Program(statements=[stmt]))
        assert "br $loop_inner" in wat


# ══════════════════════════════════════════════════════════════════════════════
# String interning
# ══════════════════════════════════════════════════════════════════════════════

class TestStringInterning:
    def test_same_string_gets_same_index(self):
        gen = WasmGenerator()
        i1 = gen._intern("LOG")
        i2 = gen._intern("LOG")
        assert i1 == i2

    def test_different_strings_get_different_indices(self):
        gen = WasmGenerator()
        i1 = gen._intern("LOG")
        i2 = gen._intern("ANNOTATE")
        assert i1 != i2

    def test_indices_are_sequential(self):
        gen = WasmGenerator()
        i0 = gen._intern("A")
        i1 = gen._intern("B")
        i2 = gen._intern("C")
        assert i1 == i0 + 1
        assert i2 == i1 + 1


# ══════════════════════════════════════════════════════════════════════════════
# CLI: praxis compile --target wasm
# ══════════════════════════════════════════════════════════════════════════════

class TestCompileWasmCLI:
    def test_wasm_target_outputs_wat(self):
        runner = CliRunner()
        result = runner.invoke(main, ["compile", "--target", "wasm", "LOG.test"])
        assert result.exit_code == 0
        assert "(module" in result.output

    def test_wasm_target_contains_dispatch(self):
        runner = CliRunner()
        result = runner.invoke(main, ["compile", "--target", "wasm", "LOG.test"])
        assert result.exit_code == 0
        assert "dispatch" in result.output

    def test_wasm_target_chain(self):
        runner = CliRunner()
        result = runner.invoke(main, ["compile", "--target", "wasm", "LOG.a -> ANNOTATE.b"])
        assert result.exit_code == 0
        assert result.output.count("call $dispatch") == 2

    def test_typescript_still_works_after_wasm_addition(self):
        runner = CliRunner()
        result = runner.invoke(main, ["compile", "--target", "typescript", "LOG.test"])
        assert result.exit_code == 0
        assert "async function run" in result.output

    def test_invalid_program_exits_nonzero(self):
        runner = CliRunner()
        result = runner.invoke(main, ["compile", "--target", "wasm", "NOT VALID !!!"])
        assert result.exit_code != 0
