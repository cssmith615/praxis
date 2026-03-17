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
