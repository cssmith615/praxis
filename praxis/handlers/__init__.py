"""
Handler registry — maps every valid VERB to a handler function.

Handler signature:
    def handler(target: list[str], params: dict, ctx: ExecutionContext) -> Any

All 51 verbs are registered here. Stubs log what they would do and return
typed mock data. Real implementations replace stubs incrementally starting
in Sprint 4.

Verbs handled internally by the Executor (not dispatched here):
  SET   — captured by executor._exec_verb; no registry entry needed
  CALL  — executor resolves PLAN declarations directly
  SKIP  — executor returns skipped result; no handler needed
  BREAK — raises BreakSignal internally
  WAIT  — executor returns stub result
  IF    — executor evaluates condition natively
  LOOP  — executor manages loop iterations natively
  PAR   — executor fans out to ThreadPoolExecutor natively
  GOAL  — declaration, not executed
  PLAN  — declaration, not executed
"""

from praxis.handlers.data import (
    ing_handler, cln_handler, xfrm_handler,
    filter_handler, sort_handler, merge_handler,
)
from praxis.handlers.ai_ml import (
    trn_handler, inf_handler, eval_handler, summ_handler,
    class_handler, gen_handler, embed_handler, search_handler,
)
from praxis.handlers.io import (
    read_handler, write_handler, fetch_handler, post_handler,
    out_handler, store_handler, recall_handler,
)
from praxis.handlers.agents import (
    spawn_handler, msg_handler, cast_handler,
    join_handler, sign_handler, cap_handler,
)
from praxis.handlers.deploy import (
    build_handler, dep_handler, test_handler,
)
from praxis.handlers.control import (
    fork_handler,
)
from praxis.handlers.error import err_handler  # RETRY/ROLLBACK are executor-native
from praxis.handlers.audit import (
    validate_handler, assert_handler, log_handler, gate_handler,
    snap_handler, annotate_handler, route_handler,
)

HANDLERS: dict = {
    # ── Data ───────────────────────────────────────────────────────────────────
    "ING":      ing_handler,
    "CLN":      cln_handler,
    "XFRM":     xfrm_handler,
    "FILTER":   filter_handler,
    "SORT":     sort_handler,
    "MERGE":    merge_handler,

    # ── AI/ML ──────────────────────────────────────────────────────────────────
    "TRN":      trn_handler,
    "INF":      inf_handler,
    "EVAL":     eval_handler,
    "SUMM":     summ_handler,
    "CLASS":    class_handler,
    "GEN":      gen_handler,
    "EMBED":    embed_handler,
    "SEARCH":   search_handler,

    # ── I/O ────────────────────────────────────────────────────────────────────
    "READ":     read_handler,
    "WRITE":    write_handler,
    "FETCH":    fetch_handler,
    "POST":     post_handler,
    "OUT":      out_handler,
    "STORE":    store_handler,
    "RECALL":   recall_handler,

    # ── Agents ─────────────────────────────────────────────────────────────────
    "SPAWN":    spawn_handler,
    "MSG":      msg_handler,
    "CAST":     cast_handler,
    "JOIN":     join_handler,
    "SIGN":     sign_handler,
    "CAP":      cap_handler,

    # ── Deploy ─────────────────────────────────────────────────────────────────
    "BUILD":    build_handler,
    "DEP":      dep_handler,
    "TEST":     test_handler,

    # ── Control ────────────────────────────────────────────────────────────────
    # RETRY and ROLLBACK are handled natively by the Executor (_exec_retry / _exec_rollback)
    "FORK":     fork_handler,
    "ERR":      err_handler,

    # ── Audit ──────────────────────────────────────────────────────────────────
    "VALIDATE": validate_handler,
    "ASSERT":   assert_handler,
    "LOG":      log_handler,
    "GATE":     gate_handler,
    "SNAP":     snap_handler,
    "ANNOTATE": annotate_handler,
    "ROUTE":    route_handler,
}
