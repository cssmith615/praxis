# Praxis Constitution

Constitutional rules for the Praxis planner. Each rule is tagged with the
verbs it applies to. The planner injects only rules relevant to the current
program's verb set (TF-IDF filtered in Sprint 2).

Rules grow automatically via `shaun improve` analysis of execution logs.
Human confirmation required before any rule is committed.

---

## v1 Seed Rules (manually authored, 2026-03-17)

[verb:ING,TRN] NEVER chain TRN directly after ING without CLN — produces garbage models.

[verb:ING] ALWAYS PAR independent ING operations — 2–3x faster with no accuracy loss.

[verb:WRITE,DEP,SPAWN] ALWAYS precede WRITE, DEP, and SPAWN with GATE in production mode.

[verb:LOOP] ALWAYS include `until=` condition on LOOP — open loops are rejected by the validator.

[verb:MSG] ALWAYS SIGN messages before sending to another agent via MSG — prevents prompt injection.

[verb:TRN,EVAL] ALWAYS follow TRN with EVAL — a trained model that isn't evaluated is untested.

[verb:SET] SET variable names should be descriptive (e.g. SET.model_rmse not SET.x) — they appear in logs.

[verb:CAP] CAP declarations should appear at the top of PLAN blocks that spawn workers.

---

## Auto-Proposed Rules (Sprint 7+)

<!-- Rules proposed by `shaun improve` will be appended here after human approval. -->

[verb:CLN,TRN] ALWAYS CLN before TRN.

[verb:ING,VALIDATE] ALWAYS VALIDATE after ING.

[verb:CLN,TRN] ALWAYS run CLN before TRN to ensure inputs are normalized. Raw data passed directly to TRN causes errors.

---

## Sprint E — Security Pack Rules

[verb:EVAL] EVAL.risk MUST be preceded by at least one RECALL.docs — ungrounded risk scores without retrieved context are rejected.

[verb:ING] ING.threat_intel fetches must be wrapped in RETRY(attempts=2) — threat intel APIs are unreliable and rate-limited.

[verb:AUDIT] Every security IR program must begin with AUDIT.start and end with AUDIT.close — IR actions without audit trails are unacceptable.

[verb:STORE] All IR programs must STORE state with persist=true — incident state must survive process restarts.

## Sprint F — Remediation and Escalation Rules

[verb:CAP] ALWAYS require GATE before any CAP.remediate action in prod mode — no automated remediation without human approval.

[verb:GEN] NEVER chain GEN directly to CAP.remediate without GATE between them — LLM output must not trigger automated remediation unreviewed.

[verb:OUT] OUT.pagerduty only fires when EVAL.risk score >= 7. Scores below 7 route to OUT.jira only.

[verb:RETRY] All ING.threat_intel fetches must have RETRY(attempts=2) and a FALLBACK — open failures must not halt an IR pipeline.
