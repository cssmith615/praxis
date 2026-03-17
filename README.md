# Praxis

> An AI-native intermediate language and runtime for agentic workflows.

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

## Development Roadmap

| Version | Focus |
|---------|-------|
| **v0.1** (current) | Language, runtime, planner, memory, constitution, bridge |
| **v0.2** | I/O verb implementations: `FETCH`, `POST`, `WRITE`, `STORE`, `RECALL` |
| **v0.3** | Provider abstraction: Ollama, OpenAI, local models alongside Anthropic |
| **v0.4** | Multi-agent coordination: `SPAWN`, `MSG`, `SYNC` verb implementations |
| **v0.5** | `.px` file format, VS Code extension with syntax highlighting |

---

## Contributing

Praxis is in early development. The most useful contributions right now are:

- **Handler implementations** — the 51 verb stubs in `praxis/handlers/` need real implementations (see `v0.2` roadmap)
- **Constitutional rules** — add rules to `praxis-constitution.md` that encode best practices for your domain
- **Real-world programs** — share `.px` files that solve real problems
- **Embedder integrations** — examples using Ollama, OpenAI embeddings, etc.

---

## License

MIT — see [LICENSE](LICENSE).
