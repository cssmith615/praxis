# SHIELD — Praxis Security Policy

> **SHIELD** = **S**ecure **H**andling, **I**nfrastructure, **E**xecution, **L**ocking, and **D**isclosure

This document is the authoritative security policy for the Praxis project. It covers vulnerability disclosure, threat model, secure configuration, and design constraints.

---

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.2.x   | ✅ Current |
| < 1.2   | ❌ Not supported — upgrade |

---

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Email: **security@praxis-lang.dev** *(or open a [GitHub private security advisory](https://github.com/cssmith615/praxis/security/advisories/new))*

Please include:
- A description of the vulnerability and its impact
- Steps to reproduce (minimal example)
- Affected version(s)
- Any mitigations you have identified

**Response SLA:**
| Severity | Acknowledgement | Fix target |
|----------|----------------|------------|
| Critical | 24 hours | 7 days |
| High     | 48 hours | 14 days |
| Medium   | 5 days   | 30 days |
| Low      | 10 days  | 60 days |

We follow **responsible disclosure** — reporters who follow this process will receive credit in the fix release notes unless they prefer anonymity.

---

## Threat Model

Praxis runs as a local agent with access to your API keys, file system, and outbound network. The primary attack surfaces are:

| Surface | Risk | Mitigation |
|---------|------|------------|
| Praxis programs (`.px` files) | Malicious programs could exfiltrate data, write files, or hit unexpected endpoints | Run `praxis validate` before `run`; use `CAP.self` capability guards |
| Telegram / Slack / Discord webhooks | Credentials exposed in env or `.env` files | Keep `.env` out of git (enforced via `.gitignore`); restrict `TELEGRAM_CHAT_IDS` |
| FETCH / POST handlers | SSRF — programs could probe internal network endpoints | Coming: GATE on allowed URL prefixes; currently user-controlled programs only |
| LLM-generated programs (planner) | Prompt injection in goals could generate harmful programs | Programs pass through the validator before execution; humans review before scheduling |
| SQLite KV store / memory DB | Path-traversal on `db_path` param | Paths are resolved via `Path.home()` defaults; custom paths are user-supplied only |
| Agent API key | Leaked `ANTHROPIC_API_KEY` enables unauthorized API usage | Key is read from env; never logged; never transmitted except to Anthropic's API |

---

## Secure Configuration Checklist

### Essential (do before running in any shared environment)

- [ ] **Restrict Telegram access** — set `TELEGRAM_CHAT_IDS` to your personal chat ID only. The agent will silently ignore messages from any other chat.
- [ ] **Keep `.env` out of git** — `.gitignore` includes `.env` by default. Verify with `git status`.
- [ ] **Never log API keys** — Praxis never logs credentials; do not add debug prints that expose `os.environ`.
- [ ] **Validate before scheduling** — always run `praxis validate <file>` before `praxis schedule`.

### VPS / server deployments

- [ ] **Firewall** — only expose port 22 (SSH). Port 7821 (Praxis bridge) should be firewalled unless intentionally public.
  ```bash
  ufw allow 22 && ufw enable
  ```
- [ ] **SSH key auth only** — disable password login in `/etc/ssh/sshd_config`.
- [ ] **Minimal permissions** — run the Docker container as a non-root user (coming in 1.3).
- [ ] **Volume backups** — the `praxis_data` volume contains `memory.db`, `schedule.db`, and `execution.log`. Back these up if they matter.

### Developer workstations

- [ ] Do not commit `.env` files.
- [ ] Do not share your bot token publicly — anyone with the token can send messages as your bot.
- [ ] Use `praxis.mode = dev` (default) while iterating; only switch to `prod` for scheduled tasks you trust.

---

## Design-Level Security Constraints

These are standing constraints enforced in code review for all PRs:

1. **No eval / exec on user input** — The Praxis grammar and executor never call `eval()` or `exec()` on user-supplied text. Programs are parsed to an AST; only AST nodes are evaluated.

2. **No shell passthrough** — Praxis verbs do not spawn shell subprocesses for user-supplied strings. Commands like `FETCH` call `httpx` directly; `WRITE` calls `open()` directly.

3. **Capability guards (`CAP` verb)** — Scheduled or agent-spawned programs should use `CAP.self(allow=[...])` to declare the verbs they need. The executor enforces this allowlist.

4. **Credentials from environment only** — Built-in OUT channels (`telegram`, `slack`, `discord`) read credentials from environment variables. Params in programs are allowed for flexibility but documented as lower-security.

5. **Memory DB is local-only** — `~/.praxis/programs.db` and `~/.praxis/kv.db` are SQLite files on the local filesystem. They are not exposed over the network.

6. **No persistent outbound connections** — The agent uses polling (Telegram long-poll) and webhook POSTs, not persistent TCP connections to third parties.

---

## Known Limitations (Non-goals for v1.x)

- **No sandboxing** — Praxis programs run in the same Python process as the agent. A malicious `.px` file with `WRITE` can write files anywhere the process has permission.
- **No rate limiting** — The agent does not throttle incoming messages. Protect via `TELEGRAM_CHAT_IDS`.
- **No authentication on the bridge** — The HTTP bridge (port 7821) has no auth. Do not expose it to the public internet without a reverse proxy with auth.

These are on the roadmap for v1.3.

---

## Dependency Security

Praxis uses a small, auditable dependency tree:

| Dependency | Purpose | Notes |
|------------|---------|-------|
| `anthropic` | Claude API client | Official Anthropic SDK |
| `httpx` | FETCH / POST handlers | Widely used, well-maintained |
| `lark` | Grammar parsing | Pure-Python, no native code |
| `numpy` | Embedding similarity | Optional; only loaded with `[memory]` extra |
| `sentence-transformers` | Goal embeddings | Optional; only loaded with `[memory]` extra |
| `fastapi` | Bridge REST API | Optional; only loaded with `[bridge]` extra |

To audit the installed dependency tree:
```bash
pip install pipdeptree
pipdeptree -p praxis-lang
```

---

## Changelog

| Date | Version | Change |
|------|---------|--------|
| 2026-03-18 | 1.2.0 | Initial SHIELD document |
