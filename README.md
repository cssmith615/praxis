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
| RAG — document retrieval | `pip install praxis-lang[rag]` (includes sentence-transformers + pdfplumber) |
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

# With RAG — document ingestion, embeddings, retrieval
pip install praxis-lang[rag]

# With Voyage AI embeddings
pip install praxis-lang[rag-voyage]

# With OpenAI embeddings
pip install praxis-lang[rag-openai]

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

### 3. Start the interactive REPL

```bash
praxis chat                    # program mode — type .px directly
praxis chat --provider anthropic  # enables natural-language goal mode
```

Type `.px` programs or natural-language goals.  Session commands: `:run`, `:validate`, `:save <file>`, `:history`, `:mode`, `:help`, `:quit`.

### 4. Generate a program from a goal (requires API key)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
praxis goal "fetch the top 5 HN stories and summarize them"
```

### Send output to Telegram, Slack, or Discord

Set your credentials once and any program can push results to your preferred channel:

**Telegram**
```bash
export TELEGRAM_BOT_TOKEN=your-token-from-botfather
export TELEGRAM_CHAT_ID=your-chat-id
```

```
FETCH.data(src="https://hacker-news.firebaseio.com/v0/topstories.json") ->
XFRM.slice(limit=5) ->
FETCH.data(src="https://hacker-news.firebaseio.com/v0/item/$item.json") ->
XFRM.pluck(field=title) ->
XFRM.join(sep="\n") ->
OUT.telegram
```

**Slack** (incoming webhook — no bot setup required)
```bash
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```
```
SUMM.text(max=500) -> OUT.slack
```
Or pass the webhook inline: `OUT.slack(webhook="https://hooks.slack.com/...")`

**Discord** (incoming webhook, same pattern)
```bash
export DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR/WEBHOOK
```
```
SUMM.text(max=500) -> OUT.discord
```

**X / Twitter** (post to your timeline)
```bash
export X_API_KEY=...
export X_API_SECRET=...
export X_ACCESS_TOKEN=...
export X_ACCESS_TOKEN_SECRET=...
```
```
GEN.post(topic=$topic, max=280) -> OUT.x
```
`OUT.x` auto-truncates to 280 characters. Works via `tweepy` if installed, or falls back to stdlib OAuth 1.0a — no extra dependency required.

All channels split long messages automatically (Telegram: 4096 chars, Discord: 2000 chars). `OUT.slack`, `OUT.discord`, and `OUT.x` use stdlib only — no extra dependencies.

| Channel | Env var | Param override | Max chunk |
|---------|---------|---------------|-----------|
| `OUT.telegram` | `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` | `token=`, `chat_id=` | 4096 chars |
| `OUT.slack` | `SLACK_WEBHOOK_URL` | `webhook=` | unlimited (Slack handles it) |
| `OUT.discord` | `DISCORD_WEBHOOK_URL` | `webhook=` | 2000 chars |
| `OUT.x` | `X_API_KEY` + `X_API_SECRET` + `X_ACCESS_TOKEN` + `X_ACCESS_TOKEN_SECRET` | — | 280 chars (auto-truncated) |

Praxis plans, validates, executes, and stores the program. Next time you run a similar goal it retrieves and adapts the stored version instead of generating from scratch.

### 5. Browse your program library

```bash
praxis memory
```

### 6. Start the REST bridge (for integration with other platforms)

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

### FETCH fan-out

When a URL template contains `$item` and the previous step returned a list, `FETCH` automatically iterates over each item, substitutes it into the URL, and returns a list of responses. This is how you turn a list of IDs into a list of full objects in a single chain step.

**Live example — top 5 Hacker News stories:**

```
FETCH.data(src="https://hacker-news.firebaseio.com/v0/topstories.json") ->
XFRM.slice(limit=5) ->
FETCH.data(src="https://hacker-news.firebaseio.com/v0/item/$item.json") ->
XFRM.pluck(field=title) ->
XFRM.join(sep="\n") ->
LOG.msg
```

```
Step 1  FETCH.data   ok  160ms  fetched ID list (500 items)
Step 2  XFRM.slice   ok    0ms  trimmed to first 5 IDs
Step 3  FETCH.data   ok  669ms  fetched each item by $item ID (5 requests)
Step 4  XFRM.pluck   ok    0ms  extracted title field from each object
Step 5  XFRM.join    ok    0ms  joined with \n separator
Step 6  LOG.msg      ok    2ms  emitted final output
```

No loops, no SET, no manual iteration. `$item` does it.

### XFRM sub-targets

`XFRM` operates on the previous step's output in-place and passes the result to the next step:

| Target | Parameters | What it does |
|--------|------------|--------------|
| `XFRM.slice` | `limit`, `offset` | Trim a list to the first N items |
| `XFRM.pluck` | `field` | Extract one field from each object in a list |
| `XFRM.join` | `sep` | Join a list of strings into one string |
| `XFRM.flatten` | — | Flatten a nested list one level |
| `XFRM.keys` | — | Return the keys of a dict |

`FILTER` and `SORT` work similarly:

```
FILTER.field(name=status, value=active)      # keep items where status == active
FILTER.field(name=score, gt=100)             # keep items where score > 100
SORT.field(field=score, order=desc)          # sort list by field descending
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

## RAG — Document Retrieval

`pip install praxis-lang[rag]`

Three verbs give you a complete retrieval-augmented generation pipeline. Index once, query anywhere.

### ING.docs — Ingest documents into chunks

```
ING.docs(src=./docs/, chunk_size=400, overlap=50)
```

Accepts `.txt`, `.md`, `.pdf` files and `https://` URLs. Returns a list of `{id, text, source, chunk_index, char_count}` chunks. Chunk IDs are deterministic — re-indexing the same file updates existing entries rather than creating duplicates.

### EMBED.text — Embed and store chunks

```
ING.docs(src=./docs/) -> EMBED.text(corpus=project_docs, provider=local)
```

Embeds each chunk and stores it in `~/.praxis/embeddings.db`. Providers: `local` (sentence-transformers, default), `voyage` (`VOYAGE_API_KEY`), `openai` (`OPENAI_API_KEY`).

### RECALL.docs — One-step RAG

```
RECALL.docs(query=$question, k=5, corpus=project_docs) -> SET.context
```

Embeds the query, retrieves the top-k matching chunks, and returns a formatted context block ready to inject into a `GEN` prompt. Returns an empty string if the corpus has no matching results.

### Full pipeline

```
// index.px — run once to build the corpus
GOAL:index_docs

ING.docs(src=./docs/, chunk_size=400, overlap=50) ->
EMBED.text(corpus=project_docs, provider=local) ->
OUT.console(msg="Indexed corpus")
```

```
// query.px — ask a question against the corpus
GOAL:rag_query

SET.question(value="How does authentication work?") ->
RECALL.docs(query=$question, k=5, corpus=project_docs) ->
SET.context ->
GEN.answer(prompt="Answer using ONLY the context below.\n\nQuestion: $question\n\nContext:\n$context\n\nAnswer:") ->
OUT.console
```

For multi-hop agentic retrieval, use `SEARCH.semantic` to retrieve, evaluate sufficiency with `EVAL`, and loop:

```
GOAL:agentic_query

SET.question(value="How is rate limiting implemented?") ->
SEARCH.semantic(query=$question, k=5, corpus=project_docs) ->
SET.results ->
EVAL.sufficient(threshold=0.8) ->
IF.$sufficient == false ->
  SEARCH.semantic(query="middleware request handling", k=3, corpus=project_docs) ->
  MERGE ->
GEN.answer(prompt="Answer: $question\n\nContext: $last_output") ->
OUT.console
```

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

### Memory Temporal Decay

Retrieval is **recency-weighted** — programs you use often stay at the top; programs that haven't been touched in months gradually stop crowding out fresher alternatives. The scoring formula is:

```
adjusted_score = 0.8 × cosine_similarity  +  0.2 × 0.5^(days_since_last_use / 90)
```

- At 0 days old, the recency bonus is `0.2 × 1.0 = 0.2` (full weight).
- At 90 days old, the recency bonus halves to `0.2 × 0.5 = 0.1`.
- At 180 days old it halves again to `0.2 × 0.25 = 0.05`.

Similarity still dominates (80%), so a very relevant old program still beats a slightly-similar new one. Recency just breaks ties and prevents your library from being dominated by ancient programs. The `last_used_at` timestamp is updated automatically every time a program is retrieved.

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
| `POST` | `/workers/register` | `{agent_id, role, verbs, url}` → register remote worker |
| `GET` | `/workers` | List all registered workers (includes stale flag) |
| `GET` | `/workers/{id}` | Get one worker |
| `POST` | `/workers/{id}/heartbeat` | Keep worker registration alive |
| `DELETE` | `/workers/{id}` | Deregister worker |
| `POST` | `/workers/dispatch/{id}` | Hub proxies `{program, mode}` to worker `/execute` |

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

## Web Dashboard

`praxis serve` is a local dashboard that makes everything visible in one place.

```bash
praxis serve              # http://localhost:7822
praxis serve --port 8080
praxis serve --open       # auto-opens browser
```

Nine tabs:

| Tab | What it shows |
|-----|---------------|
| **Dashboard** | Stats overview (programs, success rate, rule count) + recent activity |
| **Programs** | Full program memory library with search, view, open in editor, delete |
| **Logs** | Live execution history from `~/.praxis/execution.log`, filterable |
| **Constitution** | All constitutional rules with inline add form |
| **Editor** | Write and run Praxis programs in-browser, step-by-step results, Ctrl+Enter to run |
| **Schedules** | Active cron schedules — view, pause, delete, see last run time |
| **Webhooks** | Registered webhook triggers — create, view, delete, copy endpoint URL |
| **Activity** | Live activity feed of every run and webhook trigger (verb, status, timing) |
| **Audit** | Constitutional audit reports — verbs used, rules checked, violations per run |

The dashboard is a single self-contained HTML page served by FastAPI — no build step, no npm, no external CDN. It works completely offline.

---

## Persistent Variables

Variables in a Praxis program live only for the duration of that run by default. Use `persist=true` on `SET` to write a value to a local key-value store, then `LOAD` to read it back in any future run — cross-session state without a database schema.

```
# Run 1 — store the last price seen
FETCH.price(src=$url) -> SET.last_price(persist=true)

# Run 2 (hours later) — compare against it
LOAD.last_price -> EVAL.threshold(value=$price, min=$last_price) -> OUT.telegram
```

Values are stored in `~/.praxis/kv.db` (SQLite). `SET` without `persist=true` behaves exactly as before — in-memory only, no side effects.

---

## Webhook Triggers

Register a program against an HTTP endpoint and fire it from any external service — Slack, GitHub, Zapier, or a plain `curl`:

```bash
# Register a webhook
curl -X POST http://localhost:7822/api/webhooks \
  -H "Content-Type: application/json" \
  -d '{"name": "github-push", "program_text": "LOG.event -> OUT.telegram"}'
# → {"id": "wh-abc123", "url": "/webhook/wh-abc123"}

# Fire it
curl -X POST http://localhost:7822/webhook/wh-abc123 \
  -H "Content-Type: application/json" \
  -d '{"ref": "refs/heads/main", "pusher": "cssmith615"}'
```

The incoming payload is available as `$event` in the program. Webhook runs appear in the Activity feed and generate a constitutional audit record exactly like `praxis run` does.

---

## Plugin Handlers

Drop a Python file into `~/.praxis/handlers/` and Praxis loads it automatically at startup — no config, no import statements. This is the "Lua for AI apps" extension point.

```python
# ~/.praxis/handlers/my_handler.py

VERB_NAME = "NOTIFY"   # the verb this handler owns

def handle(target, params, context):
    # target: the dot-path after VERB (e.g. "slack.channel")
    # params: dict of key=value params from the program
    # context: ExecutionContext — read/write variables, last_output
    send_notification(params.get("msg", context.last_output))
    return {"status": "ok", "log_entry": "notification sent"}
```

After saving the file, restart `praxis` (or the `praxis serve` server) and your new verb is live:

```
LOG.data -> NOTIFY.slack(msg=$data)
```

Handlers that fail to load are logged and skipped — they never crash the runtime.

---

## Program Registry

Search, install, and publish community programs with three commands:

```bash
# Search the registry
praxis search news
# → news-brief  Fetch top Hacker News stories and summarize to Telegram  [news, summarize, hacker-news, telegram]

# Install a program into your local memory
praxis install news-brief
# → Installed: news-brief (Fetch top Hacker News stories and summarize to Telegram)

# Publish your own program
praxis publish my-workflow.px --name "price-alert" \
  --description "Alert on price drops" \
  --tags "price,alert,telegram" \
  --author cssmith615
# → Created price-alert.px and price-alert.json
```

The registry index is a JSON file (`registry/index.json` in the repo). It falls back to the bundled local copy if the remote fetch fails — `praxis search` and `praxis install` work completely offline.

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

## Distributed Workers

v1.1 extends `SPAWN` to span separate machines. Add a `url=` param pointing at any running Praxis bridge and that worker executes over HTTP — `MSG`, `CAST`, and `JOIN` work identically whether the worker is local or remote.

```
# On machine A — start a data worker
praxis worker --port 7823 --hub http://hub:7821 --role data --verbs ING,CLN,XFRM

# On machine B — start an analysis worker
praxis worker --port 7824 --hub http://hub:7821 --role analysis --verbs SUMM,EVAL,GEN

# In your coordinator program
SPAWN.data(role=data, verbs=[ING,CLN], url=http://machineA:7823) ->
SPAWN.analysis(role=analysis, verbs=[SUMM,EVAL], url=http://machineB:7824) ->
PAR(
  MSG.data(program="ING.sales.db -> CLN.null -> XFRM.normalize"),
  MSG.analysis(program="SUMM.text(max=300) -> EVAL.sentiment")
) ->
JOIN(timeout=60) -> MERGE -> OUT.telegram
```

Workers register with the hub on startup, send heartbeats every 30 seconds, and deregister cleanly on exit. Workers that miss heartbeats for 120 seconds are automatically marked stale and skipped by routing.

```python
from praxis.distributed import WorkerClient

client = WorkerClient("http://hub:7821")
workers = client.discover()            # all non-stale workers
worker  = client.discover(role="data") # filter by role
```

No new dependencies — all HTTP calls use Python's stdlib `urllib`.

---

## Praxis Agent

v1.2 ships a native Praxis agent that replaces NanoClaw — a Claude-powered conversational agent with direct `.px` execution built in. No subprocess, no translation layer. The agent lives inside Praxis and knows every verb natively.

### Quick start

```bash
pip install praxis-lang[agent]

export ANTHROPIC_API_KEY=sk-ant-...
export TELEGRAM_BOT_TOKEN=your-token-from-botfather
export TELEGRAM_CHAT_IDS=123456789   # your chat id — whitelist recommended

praxis agent
```

The agent listens on Telegram and can:
- **Run programs** — `run LOG.msg -> SUMM.text` executes immediately
- **Validate** — checks syntax before you schedule anything
- **Plan goals** — natural language → `.px` program via the planner (requires `--provider`)
- **Schedule** — add programs to the cron Scheduler with an interval
- **List / remove schedules** — manage what's running
- **Recall** — search ProgramMemory for similar past programs

### Multi-tier model routing

The agent automatically routes each turn to the most cost-effective model:

| Turn type | Routed to | Examples |
|-----------|-----------|---------|
| Simple commands | `claude-haiku-4-5` (~20× cheaper) | `run LOG.msg`, `validate ...`, `list my schedules`, `recall ...` |
| Complex requests | `claude-sonnet-4-6` (full capability) | `plan goal: ...`, `schedule this every morning`, `create a workflow...` |

This is completely automatic — no config needed. Override with `--fast-model` or set `PRAXIS_FAST_MODEL`. Disable with `--fast-model off`.

```bash
praxis agent --fast-model claude-haiku-4-5-20251001   # explicit (this is the default)
praxis agent --fast-model off                          # always use full model
```

### Context compaction

Long-running agent sessions accumulate conversation history that inflates API costs over time. Praxis automatically compacts the conversation when it exceeds 20 messages: the older portion is summarised using `claude-haiku-4-5` (cheap), and the last 10 messages are kept verbatim. The summary is injected as a single context message so the agent never loses thread.

This is transparent — you don't need to configure anything. The agent always keeps recent messages intact, so in-progress tasks aren't interrupted by compaction.

### Docker (production)

```bash
# Copy and fill in your keys
cp praxis/agent/.env.example .env

# Start
docker compose -f praxis/agent/docker-compose.yml up -d

# Logs
docker compose -f praxis/agent/docker-compose.yml logs -f
```

Persistent data (memory.db, schedule.db, execution.log) is stored in the `praxis_data` named volume.

### VPS deployment (24/7)

To run the agent permanently on a cheap Linux server (~$4–6/month), see **[DEPLOY.md](DEPLOY.md)** for the full guide. One-shot setup on a fresh Ubuntu VPS:

```bash
ssh root@your-server-ip
curl -fsSL https://raw.githubusercontent.com/cssmith615/praxis/main/scripts/setup-vps.sh | bash
# then: edit /opt/praxis/.env and docker compose up -d --build
```

### CLI options

```bash
praxis agent --help

# Run locally with goal planning
praxis agent \
  --token $TELEGRAM_BOT_TOKEN \
  --chat-id 123456789 \
  --provider anthropic \
  --schedule \
  --memory \
  --mode prod
```

### Security

The agent runs Praxis programs in the same sandboxing layers that guard all other execution paths:
1. **CAP enforcement** — verb allow-lists per program (Sprint 11)
2. **SandboxedExecutor** — subprocess isolation with `allowed_paths` + timeouts (Sprint 15)
3. **Resource limits** — `ResourceLimitExceeded` on CPU/memory/steps (Sprint 10)
4. **Chat whitelist** — set `--chat-id` (or `TELEGRAM_CHAT_IDS`) to restrict who can use the agent

For production deployments the Docker container adds a fourth OS-level layer on top of these.

### Provider support

The agent conversation loop always uses Claude (via `ANTHROPIC_API_KEY`). The Planner for goal→program translation is independently configurable:

```bash
praxis agent --provider ollama --model phi4   # plan with local Ollama, converse with Claude
praxis agent --provider openai                # plan with GPT-4o
```

---

## Chuck Integration

If you use [Chuck](https://github.com/cssmith615/chuck) for Claude Code context management, Praxis ships as a built-in pack:

```bash
chuck add praxis
```

This installs 14 rules covering Praxis best practices into your Chuck domain. They activate automatically when Claude is working with `.px` files, workflows, or pipelines. The Chuck monitor also validates `.px` files on write and surfaces `praxis validate` errors inline before Claude continues.

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
| **v0.7** | ✅ Released | `praxis serve` — local web dashboard: programs, logs, constitution, live editor |
| **v0.8** | ✅ Released | Resource limits: per-step timeout, wall-clock budget, output size cap enforced in executor |
| **v0.9** | ✅ Released | CAP enforcement at runtime; optimizer (parallelization, dead step elimination, constant folding); performance rewriter; TypeScript + WASM code generators; process isolation sandbox; outcome-driven program evolution |
| **v1.0** | ✅ Released | Interactive REPL (`praxis chat`); VS Code extension with syntax highlighting, inline validation, and run commands; Chuck integration (`chuck add praxis`) |
| **v1.1** | ✅ Released | Distributed workers: `SPAWN` with `url=` routes over HTTP; hub registration/heartbeat/dispatch on bridge; `praxis worker` CLI; `WorkerClient` discovery |
| **v1.2** | ✅ Released | Praxis Agent: native Claude tool-use loop with 7 Praxis tools; Telegram channel (urllib, no new deps); `praxis agent` CLI; Docker-ready; replaces NanoClaw. Full XFRM/FILTER/SORT handler implementations; FETCH fan-out (`$item` substitution over lists); `src=` param alias; `OUT.telegram` built-in channel. **Sprint 24:** `OUT.slack` + `OUT.discord` (incoming webhook, no extra deps); memory temporal decay (recency-weighted retrieval, `last_used_at` tracking); agent context compaction (auto-summarise at 20 messages, keep last 10 verbatim); [SHIELD.md](SHIELD.md) security policy. **Sprint 25:** Multi-tier model routing — simple requests auto-routed to Haiku (~20× cheaper), complex planning/scheduling stays on Sonnet; `--fast-model` CLI flag. 750 tests passing. |
| **v1.3** | ✅ Released | **Sprint 26:** Schedules tab in `praxis serve` dashboard. **Sprint 27:** Persistent `SET`/`LOAD` — cross-run key-value state in `~/.praxis/kv.db`; `initial_variables` for executor pre-loading. **Sprint 28:** Webhook triggers (`POST /webhook/{id}` fires registered programs with `$event` payload); `OUT.x` (X/Twitter posting via tweepy or stdlib OAuth 1.0a); plugin handler auto-loader (`~/.praxis/handlers/*.py`); Activity feed tab in dashboard. **Sprint 29:** Constitutional Audit Reports — post-run verb extraction, rule matching, violation detection, plain-English summary; Audit tab in dashboard. **Sprint 30:** Program Registry — `praxis install`, `praxis search`, `praxis publish`; bundled `registry/index.json` with 8 starter programs; remote fetch with local fallback. 850 tests passing. |

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
