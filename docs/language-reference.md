# Praxis Language Reference

## Syntax

### Program structure

```
// Optional comment
GOAL:identifier        // optional — names the program

statement              // one or more statements
```

### Verb action

```
VERB                           // bare verb (MERGE, JOIN, ROLLBACK, etc.)
VERB.target                    // verb with target
VERB.target.subpath            // multi-segment target
VERB.target(param=value, ...)  // verb with params
```

**VERB:** all-caps, 2–8 characters, `[A-Z][A-Z0-9]{1,7}`

**target:** lowercase identifier(s) separated by `.`

**params:** comma-separated `key=value` pairs. Values can be:
- string literals: `msg="hello"`
- numbers: `threshold=200`
- variable refs: `model=$myvar`
- lists: `allow=[FETCH, POST]`

### Chain (sequential)

```
VERB.a -> VERB.b -> VERB.c
```

Each step receives the previous step's output as `ctx.last_output`.

### Parallel block

```
PAR(VERB.a, VERB.b, VERB.c) -> MERGE
```

All branches run concurrently. Results are ordered by branch position.

### Variables

```
VERB.something -> SET.varname    // capture last output into $varname
VERB.other(input=$varname)       // use a variable in params
```

### Conditional

```
IF.condition -> body
IF.condition -> body ELSE -> body

// With comparison
IF.$price < 200 -> OUT.telegram(msg="drop!")

// With function condition
IF.state_changed -> OUT.notify

// With block body (multiple steps)
IF.$score < 0.15 -> {
  GATE ->
  DEP.api(model=$model)
}
```

### Loop

```
LOOP.check(until=done) -> VERB.body

// With block body
LOOP.poll(until=ready, interval=5s) -> {
  ING.api(url=$endpoint) ->
  EVAL.status ->
  SET.ready
}
```

Loops must include an `until=` condition. Open loops are rejected by the validator.
Maximum nesting depth: 3.

### Named plan

```
PLAN:name {
  VERB.a -> VERB.b -> VERB.c
}

// Invoke
CALL.name
```

Plans are reusable named sub-programs declared within a `GOAL` program.

### Control keywords

| Keyword | Effect |
|---------|--------|
| `SKIP` | No-op, returns immediately |
| `BREAK` | Exit the enclosing LOOP |
| `WAIT` | Pause; stub for Sprint 2 async |

---

## The 51 Verbs

### Data (10)

| Verb | Purpose |
|------|---------|
| `ING` | Ingest data from a source |
| `CLN` | Clean / deduplicate |
| `TRN` | Transform / reshape |
| `NORM` | Normalize / scale |
| `MERGE` | Merge multiple inputs |
| `JOIN` | Join on key |
| `SPLIT` | Split into multiple outputs |
| `FILTER` | Filter rows/records |
| `SORT` | Sort by field |
| `SAMPLE` | Sample a subset |

### AI/ML (10)

| Verb | Purpose |
|------|---------|
| `TRAIN` | Train a model |
| `EVAL` | Evaluate / score |
| `PRED` | Predict |
| `RANK` | Rank items |
| `CLUST` | Cluster |
| `EMBED` | Generate embeddings |
| `GEN` | Generate text / content |
| `SUMM` | Summarize |
| `CLASS` | Classify |
| `SCORE` | Score a result |

### I/O (8)

| Verb | Purpose |
|------|---------|
| `READ` | Read file |
| `WRITE` | Write file (GATE required in prod) |
| `FETCH` | HTTP GET |
| `POST` | HTTP POST |
| `OUT` | Output to channel (Telegram, Notion, Slack, etc.) |
| `STORE` | Write to key-value store |
| `RECALL` | Read from key-value store |
| `SEARCH` | Vector search |

### Agents (8)

| Verb | Purpose |
|------|---------|
| `SPAWN` | Spawn a sub-agent |
| `MSG` | Send message to agent |
| `SYNC` | Wait for agent response |
| `CAP` | Declare capability (role=...) |
| `SIGN` | Sign a message |
| `VERIFY` | Verify a signature |
| `CALL` | Invoke a named PLAN |
| `SET` | Capture output into variable |

### Deploy (5)

| Verb | Purpose |
|------|---------|
| `BUILD` | Build an artifact |
| `DEP` | Deploy (GATE required in prod) |
| `TEST` | Run test suite |
| `ROLLBACK` | Rollback deployment |
| `GATE` | Pause for human confirmation |

### Control (8)

| Verb | Purpose |
|------|---------|
| `IF` | Conditional branch |
| `LOOP` | Loop with until condition |
| `PAR` | Parallel execution block |
| `GOAL` | Declare goal name |
| `PLAN` | Declare named sub-plan |
| `SKIP` | No-op |
| `BREAK` | Break out of loop |
| `WAIT` | Wait / pause |

### Error (3)

| Verb | Purpose |
|------|---------|
| `RETRY` | Retry previous step N times |
| `FALLBACK` | Execute on upstream failure |
| `ALERT` | Send alert to channel |

### Audit (3)

| Verb | Purpose |
|------|---------|
| `LOG` | Log a message |
| `AUDIT` | Emit structured audit entry |
| `TRACE` | Attach trace ID to context |

---

## Validator Rules

| Rule | Condition |
|------|-----------|
| Unknown verb | Any verb not in the 51-token vocabulary is rejected |
| SET path | `SET.a.b` (multi-segment) is rejected; only `SET.varname` is valid |
| CALL target | `CALL.name` is rejected if `PLAN:name` is not declared |
| CAP role | `CAP` without `role=` param is rejected |
| LOOP depth | LOOP nested more than 3 levels deep is rejected |
| Prod GATE | `DEP`, `WRITE`, `SPAWN` without a preceding `GATE` in the same chain are rejected in `prod` mode |

---

## Execution Modes

| Mode | Behavior |
|------|---------|
| `dev` (default) | All verbs run; no GATE enforcement |
| `prod` | `DEP`, `WRITE`, `SPAWN` require `GATE` before them in the chain |

Pass mode to the CLI: `praxis run --mode prod myprogram.px`

Pass mode to the API: `validate(program, mode="prod")`

---

## File Extension

Praxis program files use the `.px` extension by convention.
