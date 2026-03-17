# Changelog

All notable changes to Praxis will be documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [0.1.0] — 2026-03-17

### Added

**Language core**
- Lark Earley parser with full EBNF grammar for the 51-token Praxis language
- Typed AST dataclasses: `Program`, `Chain`, `VerbAction`, `ParBlock`, `IfStmt`, `LoopStmt`, `Block`, `GoalDecl`, `PlanDecl`, `Skip`, `Break`, `Wait`, `VarRef`, `Comparison`, expression types
- Semantic validator: unknown verb detection, GATE enforcement in prod mode, SET path validation, CALL/PLAN resolution, CAP role requirement, LOOP depth limit (≤ 3)
- Runtime executor: sequential chains, parallel branches via `ThreadPoolExecutor`, IF/ELSE branching, LOOP with break signal, variable resolution via `$varname`, SET capture
- 51-verb handler registry with stub implementations across 8 categories: data, ai_ml, io, agents, deploy, control, error, audit

**AI planning**
- `Planner` class: goal → validate → retry loop (up to 3 attempts) → `PlanResult`
- Constitutional rule injection: filters rules by verbs in the plan
- `PlanningFailure` exception with full error context
- Claude (Anthropic) as the default provider

**Program memory**
- SQLite-backed `ProgramMemory` with cosine KNN (normalized float32 BLOBs)
- Injectable embedder: default uses `sentence-transformers/all-MiniLM-L6-v2`, any callable accepted
- `store`, `retrieve_similar`, `should_adapt`, `recent`, `delete` operations
- Adaptation threshold: 0.85 similarity triggers adapt-vs-generate decision

**Constitutional rules**
- `[verb:X,Y,Z]` tagged rule format in `praxis-constitution.md`
- Verb-intersection filter for prompt injection
- `append_rule` with deduplication
- 8 seed rules covering common patterns

**REST bridge**
- FastAPI sidecar (`python -m praxis.bridge`, default port 7821)
- Endpoints: `GET /health`, `POST /plan`, `POST /execute`, `POST /memory/store`, `POST /memory/retrieve`
- `PRAXIS_BRIDGE_PORT` environment variable override

**CLI**
- `praxis run <file>` — parse, validate, execute with Rich table output
- `praxis validate <file>` — grammar and semantic check only
- `praxis parse <file>` — print AST as JSON
- `praxis goal "<text>"` — plan → execute → store workflow
- `praxis memory` — browse recent program library

**Tests**
- 87 tests across parser (23), validator (12), executor (8), memory (12), constitution (12), planner (10), bridge (10)
- All tests are API-free: Anthropic client mocked, memory uses deterministic mock embedder

[0.1.0]: https://github.com/qwibitai/praxis/releases/tag/v0.1.0
