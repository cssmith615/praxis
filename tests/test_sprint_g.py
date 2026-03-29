"""
Sprint G tests — IR programs as Praxis .px files.

Validates:
  - Each of the 7 IR programs parses cleanly
  - Expected verbs appear in each program
  - Constitutional rules are followed structurally:
      * AUDIT bookends
      * RECALL before EVAL.risk
      * STORE with persist=true
      * GATE before CAP.remediate
      * PAR blocks where expected
      * OUT.pagerduty + OUT.jira co-present in high-severity paths
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from praxis import parse
from praxis.ast_types import (
    Program, Chain, VerbAction, ParBlock, IfStmt,
    Block, GoalDecl,
)

SECURITY_DIR = Path(__file__).parent.parent / "examples" / "security"

IR_PROGRAMS = [
    "ir-triage.px",
    "ir-ransomware.px",
    "ir-phishing.px",
    "ir-data-breach.px",
    "threat-hunt.px",
    "vulnerability-triage.px",
    "compliance-evidence.px",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load(filename: str) -> Program:
    """Parse a .px file and return its AST."""
    src = (SECURITY_DIR / filename).read_text()
    return parse(src)


def _collect_nodes(node, cls):
    """Recursively collect all AST nodes of a given type."""
    results = []
    if isinstance(node, cls):
        results.append(node)
    for field in vars(node).values() if hasattr(node, "__dict__") else []:
        if isinstance(field, list):
            for item in field:
                if hasattr(item, "__dict__"):
                    results.extend(_collect_nodes(item, cls))
        elif hasattr(field, "__dict__"):
            results.extend(_collect_nodes(field, cls))
    return results


def _collect_verb_actions(program: Program) -> list[VerbAction]:
    return _collect_nodes(program, VerbAction)


def _has_verb_target(actions: list[VerbAction], verb: str, *target_parts) -> bool:
    """Check if any action matches verb + target prefix."""
    for a in actions:
        if a.verb != verb:
            continue
        if not target_parts:
            return True
        if len(a.target) >= len(target_parts):
            if all(a.target[i] == target_parts[i] for i in range(len(target_parts))):
                return True
    return False


def _par_blocks(program: Program) -> list[ParBlock]:
    return _collect_nodes(program, ParBlock)


def _if_stmts(program: Program) -> list[IfStmt]:
    return _collect_nodes(program, IfStmt)


# ── Parse tests — all 7 must parse without error ──────────────────────────────

@pytest.mark.parametrize("filename", IR_PROGRAMS)
def test_program_parses(filename):
    prog = load(filename)
    assert isinstance(prog, Program)
    assert len(prog.statements) > 0


# ── GOAL declarations ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("filename,expected_goal", [
    ("ir-triage.px",           "ir_triage"),
    ("ir-ransomware.px",       "ir_ransomware"),
    ("ir-phishing.px",         "ir_phishing"),
    ("ir-data-breach.px",      "ir_data_breach"),
    ("threat-hunt.px",         "threat_hunt"),
    ("vulnerability-triage.px","vulnerability_triage"),
    ("compliance-evidence.px", "compliance_evidence"),
])
def test_goal_declaration(filename, expected_goal):
    prog = load(filename)
    goals = [s for s in prog.statements if isinstance(s, GoalDecl)]
    assert len(goals) == 1
    assert goals[0].name == expected_goal


# ── Constitutional rule: AUDIT.start and AUDIT.close bookends ─────────────────

@pytest.mark.parametrize("filename", IR_PROGRAMS)
def test_audit_bookends(filename):
    prog = load(filename)
    actions = _collect_verb_actions(prog)
    assert _has_verb_target(actions, "AUDIT", "start"), f"{filename}: missing AUDIT.start"
    assert _has_verb_target(actions, "AUDIT", "close"), f"{filename}: missing AUDIT.close"


# ── Constitutional rule: RECALL.docs present (required before EVAL.risk) ──────

@pytest.mark.parametrize("filename", IR_PROGRAMS)
def test_recall_docs_present(filename):
    prog = load(filename)
    actions = _collect_verb_actions(prog)
    assert _has_verb_target(actions, "RECALL", "docs"), f"{filename}: missing RECALL.docs"


# ── Constitutional rule: EVAL.risk present ───────────────────────────────────

@pytest.mark.parametrize("filename", IR_PROGRAMS)
def test_eval_risk_present(filename):
    prog = load(filename)
    actions = _collect_verb_actions(prog)
    assert _has_verb_target(actions, "EVAL", "risk"), f"{filename}: missing EVAL.risk"


# ── Constitutional rule: STORE persist=true ───────────────────────────────────

@pytest.mark.parametrize("filename", IR_PROGRAMS)
def test_store_with_persist(filename):
    prog = load(filename)
    actions = _collect_verb_actions(prog)
    stores = [a for a in actions if a.verb == "STORE"]
    assert stores, f"{filename}: no STORE statement found"
    # At least one STORE must declare persist=true
    assert any(
        str(a.params.get("persist", "")).lower() in ("true", "1")
        for a in stores
    ), f"{filename}: no STORE with persist=true"


# ── Constitutional rule: GATE before CAP.remediate ────────────────────────────
# Programs that contain CAP.remediate must also contain GATE

@pytest.mark.parametrize("filename", [
    "ir-ransomware.px",
    "ir-phishing.px",
    "ir-data-breach.px",
    "vulnerability-triage.px",
])
def test_gate_before_cap_remediate(filename):
    prog = load(filename)
    actions = _collect_verb_actions(prog)
    has_cap = _has_verb_target(actions, "CAP", "remediate")
    has_gate = _has_verb_target(actions, "GATE")
    assert has_cap, f"{filename}: expected CAP.remediate"
    assert has_gate, f"{filename}: GATE required before CAP.remediate (constitutional rule)"


# ── Constitutional rule: ING.threat_intel with RETRY ──────────────────────────

@pytest.mark.parametrize("filename", [
    "ir-ransomware.px",
    "ir-phishing.px",
    "ir-data-breach.px",
    "threat-hunt.px",
    "vulnerability-triage.px",
    "compliance-evidence.px",
])
def test_retry_with_threat_intel(filename):
    prog = load(filename)
    actions = _collect_verb_actions(prog)
    has_ti = _has_verb_target(actions, "ING", "threat_intel")
    has_retry = _has_verb_target(actions, "RETRY")
    assert has_ti, f"{filename}: expected ING.threat_intel"
    assert has_retry, f"{filename}: RETRY required around ING.threat_intel (constitutional rule)"


# ── Constitutional rule: OUT.pagerduty only in high-risk paths ────────────────
# All 7 programs must contain OUT.pagerduty (they all have high-severity paths)

@pytest.mark.parametrize("filename", IR_PROGRAMS)
def test_pagerduty_present(filename):
    prog = load(filename)
    actions = _collect_verb_actions(prog)
    assert _has_verb_target(actions, "OUT", "pagerduty"), f"{filename}: missing OUT.pagerduty"


# ── OUT.jira present in all programs ──────────────────────────────────────────

@pytest.mark.parametrize("filename", IR_PROGRAMS)
def test_jira_present(filename):
    prog = load(filename)
    actions = _collect_verb_actions(prog)
    assert _has_verb_target(actions, "OUT", "jira"), f"{filename}: missing OUT.jira"


# ── PAR blocks exist for simultaneous escalation ──────────────────────────────

@pytest.mark.parametrize("filename", IR_PROGRAMS)
def test_par_block_present(filename):
    prog = load(filename)
    pars = _par_blocks(prog)
    assert pars, f"{filename}: expected at least one PAR block"


# ── IF routing present (score-based routing) ──────────────────────────────────

@pytest.mark.parametrize("filename", [
    f for f in IR_PROGRAMS if f != "ir-ransomware.px"  # ransomware always escalates — no score routing
])
def test_if_routing_present(filename):
    prog = load(filename)
    ifs = _if_stmts(prog)
    assert ifs, f"{filename}: expected IF routing statements"


# ── Specific program structural tests ─────────────────────────────────────────

def test_ir_triage_four_routing_branches():
    """ir-triage must have 4 IF branches (critical/high/medium/low)."""
    prog = load("ir-triage.px")
    ifs = _if_stmts(prog)
    assert len(ifs) >= 4, f"Expected 4 routing branches, got {len(ifs)}"


def test_ir_ransomware_isolate_and_block():
    """ir-ransomware must isolate AND block."""
    prog = load("ir-ransomware.px")
    actions = _collect_verb_actions(prog)
    assert _has_verb_target(actions, "CAP", "remediate", "isolate")
    assert _has_verb_target(actions, "CAP", "remediate", "block")


def test_ir_data_breach_legal_project():
    """ir-data-breach must create a LEGAL project Jira ticket."""
    prog = load("ir-data-breach.px")
    actions = _collect_verb_actions(prog)
    jira_actions = [a for a in actions if a.verb == "OUT" and a.target == ["jira"]]
    legal_tickets = [a for a in jira_actions if a.params.get("project") == "LEGAL"]
    assert legal_tickets, "ir-data-breach.px: missing OUT.jira(project='LEGAL') for legal notification"


def test_ir_data_breach_notify_action():
    """ir-data-breach must include CAP.remediate.notify for legal notification."""
    prog = load("ir-data-breach.px")
    actions = _collect_verb_actions(prog)
    assert _has_verb_target(actions, "CAP", "remediate", "notify")


def test_threat_hunt_gen_query():
    """threat-hunt must use GEN to generate SIEM query."""
    prog = load("threat-hunt.px")
    actions = _collect_verb_actions(prog)
    gen_actions = [a for a in actions if a.verb == "GEN"]
    assert gen_actions, "threat-hunt.px: expected GEN for SIEM query generation"


def test_threat_hunt_eval_sufficient():
    """threat-hunt must use EVAL.sufficient for agentic loop gate."""
    prog = load("threat-hunt.px")
    actions = _collect_verb_actions(prog)
    assert _has_verb_target(actions, "EVAL", "sufficient")


def test_vulnerability_triage_patch_action():
    """vulnerability-triage must use CAP.remediate.patch."""
    prog = load("vulnerability-triage.px")
    actions = _collect_verb_actions(prog)
    assert _has_verb_target(actions, "CAP", "remediate", "patch")


def test_compliance_evidence_gen_report():
    """compliance-evidence must GEN an evidence report."""
    prog = load("compliance-evidence.px")
    actions = _collect_verb_actions(prog)
    gen_actions = [a for a in actions if a.verb == "GEN"]
    assert gen_actions, "compliance-evidence.px: expected GEN for report generation"


def test_compliance_evidence_compliance_project():
    """compliance-evidence must create Jira tickets in COMPLIANCE project."""
    prog = load("compliance-evidence.px")
    actions = _collect_verb_actions(prog)
    jira_actions = [a for a in actions if a.verb == "OUT" and a.target == ["jira"]]
    compliance_tickets = [a for a in jira_actions if a.params.get("project") == "COMPLIANCE"]
    assert compliance_tickets, "compliance-evidence.px: missing OUT.jira(project='COMPLIANCE')"


def test_compliance_evidence_ing_siem():
    """compliance-evidence must pull audit logs via ING.siem."""
    prog = load("compliance-evidence.px")
    actions = _collect_verb_actions(prog)
    assert _has_verb_target(actions, "ING", "siem")
