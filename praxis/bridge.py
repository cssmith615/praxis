"""
Praxis Bridge — FastAPI sidecar for NanoClaw integration.

Endpoints:
  GET  /health                        — liveness probe
  POST /plan                          — goal → Praxis program  (via Planner)
  POST /execute                       — Praxis program → list[ExecutionResult]
  POST /memory/store                  — persist a program to ProgramMemory
  POST /memory/retrieve               — fetch similar programs by cosine KNN

Distributed worker hub endpoints:
  POST   /workers/register            — remote worker announces itself
  GET    /workers                     — list all registered workers
  GET    /workers/{agent_id}          — get one worker
  POST   /workers/{agent_id}/heartbeat — worker keeps its registration alive
  DELETE /workers/{agent_id}          — deregister a worker
  POST   /workers/dispatch/{agent_id} — hub proxies program to worker /execute

Start:
  python -m praxis.bridge                         # default port 7821
  SHAUN_BRIDGE_PORT=8000 python -m praxis.bridge
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from praxis.grammar import parse
from praxis.validator import validate
from praxis.executor import Executor
from praxis.handlers import HANDLERS
from praxis.memory import ProgramMemory
from praxis.constitution import Constitution
from praxis.planner import Planner, PlanningFailure
from praxis.distributed import RemoteWorkerHub

# ─────────────────────────────────────────────────────────────────────────────
# Shared singletons (lazy — sentence-transformers loads on first /plan call)
# ─────────────────────────────────────────────────────────────────────────────

_memory: ProgramMemory | None = None
_planner: Planner | None = None
_worker_hub = RemoteWorkerHub()


def _get_memory() -> ProgramMemory:
    global _memory
    if _memory is None:
        _memory = ProgramMemory()
    return _memory


def _get_planner(mode: str = "dev") -> Planner:
    global _planner
    if _planner is None:
        _planner = Planner(
            memory=_get_memory(),
            constitution=Constitution(),
            mode=mode,
        )
    return _planner


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Praxis Bridge", version="0.1.0")


# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


# ── /plan ─────────────────────────────────────────────────────────────────────

class PlanRequest(BaseModel):
    goal: str
    mode: str = "dev"


class SimilarEntry(BaseModel):
    id: str
    goal: str
    program: str
    outcome: str
    similarity: float


class PlanResponse(BaseModel):
    ok: bool
    program: str | None = None
    adapted: bool = False
    attempts: int = 0
    similar: list[SimilarEntry] = []
    error: str | None = None


@app.post("/plan", response_model=PlanResponse)
def plan(req: PlanRequest) -> PlanResponse:
    try:
        result = _get_planner(req.mode).plan(req.goal)
        return PlanResponse(
            ok=True,
            program=result.program,
            adapted=result.adapted,
            attempts=result.attempts,
            similar=[
                SimilarEntry(
                    id=p.id,
                    goal=p.goal_text,
                    program=p.shaun_program,
                    outcome=p.outcome,
                    similarity=p.similarity,
                )
                for p in result.similar
            ],
        )
    except PlanningFailure as exc:
        return PlanResponse(ok=False, error=exc.last_error, attempts=exc.attempts)
    except Exception as exc:
        return PlanResponse(ok=False, error=str(exc))


# ── /execute ──────────────────────────────────────────────────────────────────

class ExecuteRequest(BaseModel):
    program: str
    mode: str = "dev"


class StepResult(BaseModel):
    verb: str
    target: list[str]
    params: dict[str, Any]
    output: Any
    status: str
    duration_ms: int
    log_entry: str


class ExecuteResponse(BaseModel):
    ok: bool
    results: list[StepResult] = []
    errors: list[str] = []


@app.post("/execute", response_model=ExecuteResponse)
def execute(req: ExecuteRequest) -> ExecuteResponse:
    try:
        program = parse(req.program)
    except Exception as exc:
        return ExecuteResponse(ok=False, errors=[f"Parse error: {exc}"])

    errors = validate(program, mode=req.mode)
    if errors:
        return ExecuteResponse(ok=False, errors=errors)

    try:
        executor = Executor(handlers=HANDLERS, mode=req.mode)
        results = executor.execute(program)
        return ExecuteResponse(
            ok=True,
            results=[
                StepResult(
                    verb=r["verb"],
                    target=r["target"],
                    params=r["params"],
                    output=r["output"],
                    status=r["status"],
                    duration_ms=r["duration_ms"],
                    log_entry=r["log_entry"],
                )
                for r in results
            ],
        )
    except Exception as exc:
        return ExecuteResponse(ok=False, errors=[str(exc)])


# ── /memory/store ─────────────────────────────────────────────────────────────

class MemoryStoreRequest(BaseModel):
    goal: str
    program: str
    outcome: str = "success"
    log: list[dict[str, Any]] = []


class MemoryStoreResponse(BaseModel):
    ok: bool
    id: str | None = None
    error: str | None = None


@app.post("/memory/store", response_model=MemoryStoreResponse)
def memory_store(req: MemoryStoreRequest) -> MemoryStoreResponse:
    try:
        id_ = _get_memory().store(req.goal, req.program, req.outcome, req.log)
        return MemoryStoreResponse(ok=True, id=id_)
    except Exception as exc:
        return MemoryStoreResponse(ok=False, error=str(exc))


# ── /memory/retrieve ──────────────────────────────────────────────────────────

class MemoryRetrieveRequest(BaseModel):
    goal: str
    k: int = 3


class MemoryRetrieveResponse(BaseModel):
    ok: bool
    programs: list[SimilarEntry] = []
    error: str | None = None


@app.post("/memory/retrieve", response_model=MemoryRetrieveResponse)
def memory_retrieve(req: MemoryRetrieveRequest) -> MemoryRetrieveResponse:
    try:
        programs = _get_memory().retrieve_similar(req.goal, k=req.k)
        return MemoryRetrieveResponse(
            ok=True,
            programs=[
                SimilarEntry(
                    id=p.id,
                    goal=p.goal_text,
                    program=p.shaun_program,
                    outcome=p.outcome,
                    similarity=p.similarity,
                )
                for p in programs
            ],
        )
    except Exception as exc:
        return MemoryRetrieveResponse(ok=False, error=str(exc))


# ── /workers — distributed worker hub ────────────────────────────────────────

class WorkerRegisterRequest(BaseModel):
    agent_id: str
    role: str
    verbs: list[str]
    url: str


class WorkerRegisterResponse(BaseModel):
    ok: bool
    agent_id: str | None = None
    error: str | None = None


class WorkerEntry(BaseModel):
    agent_id: str
    role: str
    verbs: list[str]
    url: str
    registered_at: str
    last_seen: str
    stale: bool


class WorkerDispatchRequest(BaseModel):
    program: str
    mode: str = "dev"


@app.post("/workers/register", response_model=WorkerRegisterResponse)
def workers_register(req: WorkerRegisterRequest) -> WorkerRegisterResponse:
    try:
        reg = _worker_hub.register(
            agent_id=req.agent_id,
            role=req.role,
            verbs=req.verbs,
            url=req.url,
        )
        return WorkerRegisterResponse(ok=True, agent_id=reg.agent_id)
    except Exception as exc:
        return WorkerRegisterResponse(ok=False, error=str(exc))


@app.get("/workers", response_model=list[WorkerEntry])
def workers_list() -> list[WorkerEntry]:
    return [WorkerEntry(**r.to_dict()) for r in _worker_hub.list_all()]


@app.get("/workers/{agent_id}", response_model=WorkerEntry)
def workers_get(agent_id: str):
    reg = _worker_hub.get(agent_id)
    if reg is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Worker '{agent_id}' not found")
    return WorkerEntry(**reg.to_dict())


@app.post("/workers/{agent_id}/heartbeat")
def workers_heartbeat(agent_id: str) -> dict:
    ok = _worker_hub.heartbeat(agent_id)
    return {"ok": ok, "agent_id": agent_id}


@app.delete("/workers/{agent_id}")
def workers_deregister(agent_id: str) -> dict:
    ok = _worker_hub.deregister(agent_id)
    return {"ok": ok, "agent_id": agent_id}


@app.post("/workers/dispatch/{agent_id}")
def workers_dispatch(agent_id: str, req: WorkerDispatchRequest) -> dict:
    return _worker_hub.dispatch(agent_id=agent_id, program=req.program, mode=req.mode)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PORT = 7821


def main() -> None:
    port = int(os.environ.get("SHAUN_BRIDGE_PORT", DEFAULT_PORT))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
