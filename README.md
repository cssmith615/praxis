# Praxis

> An AI-native intermediate language and runtime for agentic workflows.

[![CI](https://github.com/cssmith615/praxis/actions/workflows/ci.yml/badge.svg)](https://github.com/cssmith615/praxis/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/praxis-lang.svg)](https://pypi.org/project/praxis-lang/)

---

## What is Praxis?

Most AI agent frameworks give you two options: write everything in natural language and hope the LLM figures it out, or write everything in Python/YAML and lose the flexibility of language models. Neither is right.

**Praxis is the middle layer.** It is a 51-token symbolic language that sits between what you say and what executes. You describe a goal in plain English. Praxis generates a structured, auditable program. That program runs.

```
"check denver flight prices and alert me if under $200"
        ↓  praxis goal
ING.flights(dest=denver) -> EVAL.price(threshold=200) -> IF.$price < 200 -> OUT.telegram(msg="drop!")
        ↓  praxis run
Step 1  ING.flights    ok   42ms   fetched 14 flights
Step 2  EVAL.price     ok   3ms    $187 — threshold met
Step 3  IF branch      ok   0ms    condition true
Step 4  OUT.telegram   ok   210ms  message sent
```

Every step is logged. Every variable is traceable. Every program is stored and reused.

---

## The Architecture

```
┌─────────────────────────────────────────────┐
│  Natural Language (your goal)               │
│  "check denver flights under $200"          │
└─────────────────────┬───────────────────────┘
                      │  praxis goal
                      ▼
┌─────────────────────────────────────────────┐
│  Praxis Language Layer                      │
│  • LLM planner → generates program         │
│  • Grammar + validator → ensures it's legal│
│  • Executor → runs it step by step         │
│  • Program memory → learns from past runs  │
│  • Constitution → enforces invariants      │
└─────────────────────┬───────────────────────┘
                      │  structured execution
                      ▼
┌─────────────────────────────────────────────┐
│  Your Execution Layer                       │
│  (container runner, agent platform, API)    │
└─────────────────────────────────────────────┘
```

Praxis is **language-agnostic at the execution boundary**. A FastAPI bridge (included) lets any platform — TypeScript, Go, Ruby, anything — send a goal and get back a Praxis program, then execute it.

---

## Prerequisites

| Feature | Requires |
|---------|---------|
| Parse, validate, run programs | Python 3.11+ |
| AI goal planning | `ANTHROPIC_API_KEY` (Claude) |
| Semantic program memory | `pip install praxis-lang[memory]` (~700 MB, downloads PyTorch via sentence-transformers) |
| REST bridge | `pip install praxis-lang[bridge]` |
| Everything | `pip install praxis-lang[all]` |

> **Memory without PyTorch:** You can inject your own embedder function — any model that returns a normalized `np.ndarray` works. See [Bring Your Own Embedder](#bring-your-own-embedder).

---

## Install

```bash
# Core only — parse, validate, run
pip install praxis-lang

# With AI planning
pip install praxis-lang[ai]

# With semantic memory (requires PyTorch)
pip install praxis-lang[memory]

# With REST bridge
pip install praxis-lang[bridge]

# Everything
pip install praxis-lang[all]
```

---

## Quick Start

### 1. Run a program directly

Create `hello.px`:

```
GOAL:hello_world
ING.api(url=https://api.example.com/data) -> CLN.null -> SUMM.text(max=200) -> OUT.print
```

```bash
praxis run hello.px
```

### 2. Validate before running

```bash
praxis validate hello.px        # grammar + semantic check
praxis parse hello.px           # print the AST as JSON
```

### 3. Generate a program from a goal (requires API key)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
praxis goal "fetch the top 5 HN stories and summarize them"
```

Praxis plans, validates, executes, and stores the program. Next time you run a similar goal it retrieves and adapts the stored version instead of generating from scratch.

### 4. Browse your program library

```bash
praxis memory
```

### 5. Start the REST bridge (for integration with other platforms)

```bash
python -m praxis.bridge         # starts on http://127.0.0.1:7821
```

```bash
# Plan
curl -X POST http://localhost:7821/plan \
  -H "Content-Type: application/json" \
  -d '{"goal": "summarize this week'\''s sales data"}'

# Execute
curl -X POST http://localhost:7821/execute \
  -H "Content-Type: application/json" \
  -d '{"program": "ING.sales.db -> CLN.null -> SUMM.text(max=200) -> OUT.notion"}'
```

---

## The Language

### Syntax at a glance

```
GOAL:name
VERB.target(param=value) -> VERB.target -> VERB.target
```

- **VERB** — all-caps, 2–8 characters: `ING`, `CLN`, `TRN`, `OUT`, `EVAL`, `FETCH`
- **target** — dot-separated path: `ING.flights`, `ING.sales.db`
- **params** — key=value pairs: `(dest=denver, threshold=200)`
- **`$varname`** — variable reference: `SET.price -> IF.$price < 200`
- **`->`** — chain (sequential)
- **`PAR(...)`** — parallel execution
- **`IF.condition -> body ELSE -> body`** — branching
- **`LOOP.condition(until=done) { body }`** — looping

### Full example

```
GOAL:weekly_report

PAR(
  ING.sales.db(period=week),
  ING.support.tickets(period=week),
  ING.marketing.metrics(period=week)
) -> MERGE ->
CLN.null ->
SET.raw_data ->
SUMM.text(max=500) ->
SET.summary ->
IF.$summary != "" ->
  OUT.notion(page=weekly_report, content=$summary)
ELSE ->
  OUT.slack(channel=alerts, msg="Weekly report failed — no data")
```

### The 51 Verbs

| Category | Verbs |
|----------|-------|
| **Data** | `ING` `CLN` `TRN` `NORM` `MERGE` `JOIN` `SPLIT` `FILTER` `SORT` `SAMPLE` |
| **AI/ML** | `TRN` `EVAL` `PRED` `RANK` `CLUST` `EMBED` `GEN` `SUMM` `CLASS` `SCORE` |
| **I/O** | `READ` `WRITE` `FETCH` `POST` `OUT` `STORE` `RECALL` `SEARCH` |
| **Agents** | `SPAWN` `MSG` `SYNC` `CAP` `SIGN` `VERIFY` `CALL` `SET` |
| **Deploy** | `BUILD` `DEP` `TEST` `ROLLBACK` `GATE` |
| **Control** | `IF` `LOOP` `PAR` `GOAL` `PLAN` `SKIP` `BREAK` `WAIT` |
| **Error** | `RETRY` `FALLBACK` `ALERT` |
| **Audit** | `LOG` `AUDIT` `TRACE` |

---

## Constitutional Rules

Praxis programs operate under a **constitution** — a set of named, tagged rules that the planner and validator enforce. Rules are stored in `praxis-constitution.md` and tagged by verb:

```markdown
[verb:ING,TRN] NEVER chain TRN directly after ING without CLN.
[verb:WRITE,DEP] ALWAYS precede WRITE and DEP with GATE in production mode.
[verb:LOOP] ALWAYS include an until= condition on LOOP — open loops are rejected.
```

The planner injects only the rules relevant to the verbs it's about to use. You can extend the constitution:

```python
from praxis.constitution import Constitution

c = Constitution()
c.append_rule("ALWAYS LOG after every EVAL step.", verbs=["EVAL", "LOG"])
```

---

## Program Memory

Every program you run is stored in a local SQLite database (`~/.praxis/programs.db`) with a vector embedding of its goal text. When you run a new goal, Praxis retrieves the most similar past programs and injects them as context — the planner adapts existing programs instead of generating from scratch.

```python
from praxis.memory import ProgramMemory

mem = ProgramMemory()
similar = mem.retrieve_similar("check flight prices", k=3)
for p in similar:
    print(f"{p.similarity:.2f}  {p.goal_text}")
```

### Bring Your Own Embedder

If you don't want the PyTorch dependency, supply any embedding function:

```python
import numpy as np
from praxis.memory import ProgramMemory

def my_embedder(text: str) -> np.ndarray:
    # use OpenAI, Ollama, or any model
    ...

mem = ProgramMemory(embedder=my_embedder)
```

---

## REST Bridge

The bridge turns Praxis into a language-agnostic planning and execution service. Start it once and call it from any platform:

```bash
python -m praxis.bridge
# or with a custom port:
PRAXIS_BRIDGE_PORT=8000 python -m praxis.bridge
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `POST` | `/plan` | `{goal, mode}` → Praxis program |
| `POST` | `/execute` | `{program, mode}` → step results |
| `POST` | `/memory/store` | `{goal, program, outcome}` → persist |
| `POST` | `/memory/retrieve` | `{goal, k}` → similar programs |

### TypeScript / JavaScript example

```typescript
const res = await fetch("http://localhost:7821/plan", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ goal: "summarize this week's sales" }),
});
const { ok, program } = await res.json();

if (ok) {
  const exec = await fetch("http://localhost:7821/execute", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ program }),
  });
  const { results } = await exec.json();
}
```

---

## Python API

```python
from praxis.grammar import parse
from praxis.validator import validate
from praxis.executor import Executor
from praxis.handlers import HANDLERS

source = "ING.sales.db -> CLN.null -> SUMM.text(max=200) -> OUT.print"

program = parse(source)
errors = validate(program, mode="dev")   # or "prod"
assert not errors

results = Executor(handlers=HANDLERS).execute(program)
for step in results:
    print(f"{step['verb']:8}  {step['status']}  {step['duration_ms']}ms")
```

---

## Production Mode

Set `mode="prod"` to enforce safety gates:

```
# This will be rejected in prod mode — DEP requires GATE
ING.build -> DEP.api

# This is accepted — GATE pauses for human confirmation
ING.build -> GATE -> DEP.api
```

---

## Provider Abstraction

v0.6 decouples the planner from any single LLM backend. Five providers ship out of the box — swap with one flag or env var:

| Provider | Flag | Key env var | Default model |
|----------|------|-------------|---------------|
| **Anthropic** | `--provider anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` |
| **OpenAI** | `--provider openai` | `OPENAI_API_KEY` | `gpt-4o` |
| **Grok** (xAI) | `--provider grok` | `GROK_API_KEY` | `grok-3-mini` |
| **Gemini** (Google) | `--provider gemini` | `GEMINI_API_KEY` | `gemini-2.0-flash` |
| **Ollama** (local) | `--provider ollama` | _(none)_ | `llama3.2` |

Auto-detection: Praxis reads your env vars in priority order and picks the first available provider. Set `PRAXIS_PROVIDER` to override.

```bash
# Use Anthropic (auto-detected from env)
praxis goal "summarize sales data"

# Use Ollama with a local model
praxis goal "summarize sales data" --provider ollama --model mistral

# Use Grok
praxis goal "summarize sales data" --provider grok --model grok-3

# Use Gemini
praxis goal "summarize sales data" --provider gemini --model gemini-2.0-flash

# OpenAI-compatible endpoint (Groq, LM Studio, Azure, etc.)
OPENAI_BASE_URL=https://api.groq.com/openai/v1 \
OPENAI_API_KEY=your-groq-key \
praxis goal "..." --provider openai --model llama-3.1-70b-versatile
```

Python API:

```python
from praxis.providers import resolve_provider
from praxis.planner import Planner

# Explicit
provider = resolve_provider("ollama", model="phi4")

# Auto-detect from env
provider = resolve_provider()

planner = Planner(memory=..., constitution=..., provider=provider)
```

`praxis improve --llm` also accepts `--provider` and `--model` to control which backend writes the proposed constitutional rules.

---

## Multi-Agent Coordination

v0.4 ships structured multi-agent coordination without natural language passing between agents. Coordinators dispatch typed Praxis programs to specialist workers. Workers run in parallel. Results are collected with `JOIN`.

```
// Declare coordinator capabilities
CAP.coordinator(role=coordinator, allow=[SPAWN,MSG,JOIN,CAST])

// Spawn specialist workers
SPAWN.data_worker(role=data, verbs=[ING,CLN,XFRM]) ->
SPAWN.analysis_worker(role=analysis, verbs=[SUMM,EVAL,GEN]) ->

// Dispatch in parallel
PAR(
  MSG.data_worker(program="ING.sales(period=week) -> CLN.null -> XFRM.normalize"),
  MSG.analysis_worker(program="SUMM.text(max=300) -> EVAL.sentiment(threshold=0.5)")
) ->

// Collect and synthesize
JOIN(timeout=60) -> MERGE -> SUMM.text(max=200) -> OUT.telegram(msg="Analysis ready")
```

Key properties:
- **No natural language between agents** — workers receive Praxis programs, not prose
- **Parallel dispatch** — `PAR(MSG..., MSG...)` submits all at once; `JOIN` collects
- **HMAC-SHA256 signing** — `SIGN` and `verify_message()` authenticate inter-agent messages
- **Cycle detection** — validator catches `MSG` self-loops at parse time
- **Capability declaration** — `CAP` annotates agent scope for observability and enforcement

See `examples/swarm_analysis.px` for the full reference program.

---

## Self-Improvement Loop

v0.5 introduces `praxis improve` — the runtime watches itself run and proposes rules that prevent recurrence of observed failures.

```
$ praxis improve

Analyzing execution history...
Found 2 failure pattern(s):

  TRN  5 failures  input not clean
  OUT  3 failures  delivery channel timeout

Proposing 2 rule(s):

Rule 1/2  source: heuristic
  Pattern: TRN failed 5×
  Verbs:   CLN, TRN
  Impact:  ~8 programs affected, ~3 failures prevented

  Proposed rule:
  ALWAYS run CLN before TRN to ensure inputs are normalized.

  Accept this rule? [Y/n]: Y
  ✓ Rule added to constitution.

Done. 2 rule(s) added, 0 skipped.
Constitution updated: praxis-constitution.md
```

With `ANTHROPIC_API_KEY` set, use `--llm` for Claude-written rule text:

```
praxis improve --llm
```

Other options:

```
praxis improve --dry-run          # preview without writing
praxis improve --yes              # accept all without prompting
praxis improve --log path/to.log  # custom log file
```

The improvement loop closes the feedback cycle: programs run → failures are logged → `praxis improve` proposes rules → rules guide the planner → fewer failures.

---

## Development Roadmap

| Version | Status | Focus |
|---------|--------|-------|
| **v0.1** | ✅ Released | Language, runtime, planner, memory, constitution, REST bridge |
| **v0.2** | ✅ Released | I/O & audit verbs: `FETCH`, `POST`, `WRITE`, `STORE`, `RECALL`, `ASSERT`, `GATE`, `SNAP`, `LOG`, `ROUTE`, `VALIDATE`; deploy verbs: `BUILD`, `DEP`, `TEST` |
| **v0.3** | ✅ Released | Error recovery: `ERR`, `RETRY` (backoff), `ROLLBACK`; `Scheduler` with triage hook for zero-cost monitoring loops |
| **v0.4** | ✅ Released | Multi-agent coordination: `SPAWN`, `MSG`, `CAST`, `JOIN`, `SIGN`, `CAP`; `AgentRegistry`; HMAC-SHA256 message signing; MSG cycle detection |
| **v0.5** | ✅ Released | Self-improvement loop: `praxis improve` analyzes execution log, proposes constitutional rules, accepts to constitution |
| **v0.6** | ✅ Released | Provider abstraction: Anthropic, OpenAI, Ollama, Grok, Gemini — swap backends with one flag |
| **v0.7** | Planned | `.px` file format, VS Code extension with syntax highlighting |

---

## Contributing

Praxis is in early development. The most useful contributions right now are:

- **Handler implementations** — remaining verb stubs in `praxis/handlers/` (see `v0.3+` roadmap for what's next)
- **Constitutional rules** — add rules to `praxis-constitution.md` that encode best practices for your domain
- **Real-world programs** — share `.px` files that solve real problems
- **Embedder integrations** — examples using Ollama, OpenAI embeddings, etc.

---

## License

MIT — see [LICENSE](LICENSE).
