# Praxis Launch Content

---

## Hacker News — Show HN

**Title:**
Show HN: Praxis – an AI-native intermediate language for agentic workflows

**Body:**
I've been building personal AI agents for a while and kept running into the same wall: the agent does something useful, and then it's gone. No record of the plan. No way to retrieve it next time. No way to replay or improve it. Just prose in a log file.

So I built a language for it.

Praxis is a 51-token symbolic DSL that sits between natural language and execution:

    ING.flights(dest=denver) -> EVAL.price(threshold=200) -> IF.$price < 200 -> OUT.telegram(msg="drop!")

You describe a goal in plain English. Praxis generates a program, validates it against a grammar and semantic rules, executes it step by step, and stores it in SQLite with a vector embedding. Next time you ask something similar, it retrieves and adapts the stored program instead of generating from scratch.

The core insight: LangChain/LangGraph "programs" are Python objects — you can't store them as strings, send them over a message queue, or have an LLM generate and validate them against a grammar. Praxis programs are first-class strings. They're portable, storable, transmittable, and versionable.

It also includes:
- A constitutional rule system (`[verb:ING,TRN] NEVER chain TRN after ING without CLN`) that gets injected into every planning prompt
- A FastAPI bridge so any language (TypeScript, Go, etc.) can POST a goal and get back a program
- Production mode with GATE enforcement before destructive verbs
- 87 tests, all passing, no API key required for core features

The language name comes from the Greek πρᾶξις — the act of turning knowledge into deed. That's the gap: every AI company has models and tool-calling protocols. Nobody has the layer that turns plans into portable, learnable programs.

GitHub: https://github.com/cssmith615/praxis
Install: pip install praxis-lang

Curious what people think about the verb vocabulary and whether the constitutional rules approach resonates with anyone building similar things.

---

## Reddit — r/LocalLLaMA

**Title:**
I built an intermediate language so my AI agents can remember what they did — Praxis (open source, MIT)

**Body:**
Been thinking about this problem for months: every time my agent completes a task, the "plan" disappears. It was just tokens in a context window. There's nothing to retrieve, replay, or learn from.

I built Praxis to fix that. It's a 51-token AI-native DSL:

```
ING.flights(dest=denver) -> EVAL.price(threshold=200) -> IF.$price < 200 -> OUT.telegram(msg="drop!")
```

Every program gets stored in SQLite with a vector embedding of the goal that triggered it. Next time you run a similar goal, it finds the closest match and adapts the existing program instead of generating fresh. The planner (works with Claude, Ollama support coming) uses past programs + constitutional rules as context — so it gets better at *your* specific goals over time.

**What makes it different from LangChain:**
LangGraph programs are Python objects. You can't serialize them to a flat string, send them between agents, or have an LLM generate and validate them. Praxis programs are strings. Store them anywhere, send them over Redis, version them in git.

**The LLVM comparison:**
Everyone's building compilers (model APIs, agent frameworks) but nobody standardized the intermediate representation. That's what this is trying to be — the IR that makes agent plans portable and interoperable.

**The local angle:**
The semantic memory uses sentence-transformers by default but the embedder is injectable — swap in Ollama embeddings, nomic-embed-text, whatever you're running locally. Provider abstraction for the planner (Claude/Ollama/OpenAI) is the next thing I'm building.

**Current state:** v0.1.0, 87 tests passing, MIT license.

`pip install praxis-lang` or `pip install praxis-lang[all]` for everything.

GitHub: https://github.com/cssmith615/praxis

Happy to answer questions about the grammar design, the constitutional rules system, or the program memory approach.

---

## X / Twitter Thread

**Tweet 1:**
I built an intermediate language for AI agents so they can remember what they did, adapt what worked, and coordinate without natural language guessing.

It's called Praxis. It's open source. Here's what it does and why I think the industry is missing this layer. 🧵

**Tweet 2:**
The problem: every AI agent framework gives you models + tool-calling. LangChain, AutoGen, OpenAI Agents SDK — they all have this.

What none of them have: a portable, serializable, storable *language* that the agent uses to express its plan.

Your agent plans something. It executes. The plan is gone.

**Tweet 3:**
Praxis programs are first-class strings:

ING.flights(dest=denver) -> EVAL.price -> IF.$price < 200 -> OUT.telegram

Stored in SQLite. Retrieved by cosine similarity. Adapted by the planner. Transmitted between agents. Replayed. Versioned.

LangGraph programs are Python objects. This is not.

**Tweet 4:**
The LLVM comparison:

Everyone had compilers. Nobody had a standard IR. LLVM became the IR and captured enormous value.

Every AI company has models. Nobody has the portable plan representation. Praxis is trying to be that layer.

**Tweet 5:**
51 tokens. 8 categories:

DATA: ING CLN TRN NORM MERGE JOIN SPLIT FILTER SORT SAMPLE
AI/ML: EVAL PRED RANK EMBED GEN SUMM CLASS SCORE
I/O: READ WRITE FETCH POST OUT STORE RECALL SEARCH
AGENTS: SPAWN MSG SYNC CAP SIGN VERIFY CALL SET
DEPLOY: BUILD DEP TEST ROLLBACK GATE
CONTROL: IF LOOP PAR GOAL PLAN SKIP BREAK WAIT
ERROR: RETRY FALLBACK ALERT
AUDIT: LOG AUDIT TRACE

**Tweet 6:**
Program memory: every run stores the goal + program + outcome + vector embedding.

Next similar goal → retrieve nearest programs → planner adapts instead of generating from scratch.

Your agent gets smarter at *your* goals over time. Not just smarter in general.

**Tweet 7:**
Constitutional rules:

[verb:ING,TRN] NEVER chain TRN after ING without CLN.
[verb:WRITE,DEP] ALWAYS precede WRITE with GATE in production mode.

Tagged by verb. Injected into every planning prompt. Accumulated over time. Your agent's operational wisdom, encoded.

**Tweet 8:**
The REST bridge means any language connects:

python -m praxis.bridge

Then POST /plan with a goal from TypeScript, Go, whatever. Get back a validated Praxis program. POST /execute. Done.

**Tweet 9:**
v0.1.0 is live.

pip install praxis-lang

87 tests passing. MIT license. Works with Claude today, Ollama/OpenAI provider abstraction coming in v0.2.

GitHub: https://github.com/cssmith615/praxis

Built this because I needed it. Putting it out there because I think others do too.

---

## LinkedIn

**Post:**

I've been building personal AI agents for about a year. The agents got good at tasks. What they couldn't do was *remember* how they did them.

Every successful run produced a response. Never a reusable program.

I thought about this for a while and concluded the industry has a missing layer. Every major AI company has:
- A model layer (GPT-4o, Claude, Gemini)
- A tool-calling layer (function calling, MCP)

Nobody has the language layer — the portable, serializable, evolvable representation of what the agent actually *planned* to do.

So I built one. I'm calling it Praxis (from the Greek πρᾶξις — the act of turning knowledge into deed).

A Praxis program looks like this:

ING.flights(dest=denver) → EVAL.price(threshold=200) → IF.$price < 200 → OUT.telegram(msg="drop!")

That's a complete agentic workflow. 51 tokens. Eight categories covering data, AI/ML, I/O, agents, deploy, control, error handling, and audit.

What makes it different from YAML configs or Python agent frameworks:

**Programs are first-class strings.** You can store them in a database, retrieve them by cosine similarity, transmit them between agents, and have a language model generate and validate them against a grammar. LangGraph programs are Python objects — you can't do any of that.

**Program memory.** Every run stores the goal + program + outcome + vector embedding. Next similar goal retrieves and adapts the stored program instead of generating from scratch. The system compounds.

**Constitutional rules.** A markdown file of tagged operational rules that gets injected into every planning prompt. Accumulated lessons about what patterns work and what patterns fail, encoded so they propagate to every future plan.

**A REST bridge.** Start a FastAPI sidecar, POST a goal from any language. Works with TypeScript, Go, Ruby — anything.

v0.1.0 is live on GitHub under MIT license: github.com/cssmith615/praxis

pip install praxis-lang

I built this because I needed it. Sharing it because I think the industry needs it. Curious what others building in the agent space think about the approach.

#AI #OpenSource #Agents #Python #LLM #MachineLearning

---

## dev.to / Medium — Opening Hook

**Title:** The AI industry has a missing layer. I built it.

**Lede:**

Every major AI company is racing to build better models. OpenAI, Anthropic, Google, AWS — they're all competing at the model layer and the tool-calling layer.

There's a layer nobody has built yet: the **language layer**. The portable, serializable, storable representation of what an AI agent *planned* to do — not just what it said, but the structured program that drove its actions.

LangChain programs are Python objects. AutoGen agents coordinate in natural language. OpenAI function calls are ephemeral — the plan exists for one API call, then it's gone. There is no standard intermediate representation.

I spent a year building personal AI agents and kept losing the work. Every successful run produced a response. Never a reusable program. The agent that checked my flight prices and sent me an alert had done the work once, perfectly, and had no memory of how.

So I built the missing layer. I'm calling it Praxis.

[Continue to full article...]
