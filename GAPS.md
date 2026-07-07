# Gaps — what the framework demands vs. what the data supports

Every gap below was **validated directly against the warehouse on 2026-07-06** (RUMI_DB
and tbproddb, via the governed query layer). Each names what blocks it and the smallest
concrete step to unblock it. Steps not listed here run today on real extracts.

## Feasibility check — the numbers this was validated against

| Check | Result |
|---|---|
| Scored sessions (`coaching_sessions_with_fico_scores`, non-test) | **675 rows**, all 30 indicators + domain + overall scores **100% populated** |
| Distinct teachers with scored sessions | **142** (37 with 1, 19 with 2, 46 with 3–5, 27 with 6–10, 13 with >10) |
| Scored-session date range | **2025-11-10 → 2026-05-26** (~6.5 months — just enough for Step 6) |
| Session status split (`coaching_sessions`, non-test) | 567 completed / 104 abandoned / 36 failed / 8 cancelled |
| Transcript usability by status | completed 100%, abandoned 93%, failed 28%, cancelled 25% |
| Human FICO observations (`tbproddb`, ICT) | **8,451** clean, 8,089 with structured scored answers, 2025-08 → 2026-06 |
| **Rumi sessions ↔ human observations sharing an `audio_url`** | **0** (676 Rumi audio URLs, 1,149 ICT observation audio URLs, no intersection) |

---

## G1 — Human-vs-AI IRR — ✅ RESOLVED by the paired study (with a critical caveat)

**Framework wants:** Cohen's kappa >0.70 per indicator between the AI and a trained human
coach on the *same* lesson.

**Original blocker (still true):** The human FICO observations in `tbproddb.coaching_observation`
and the AI sessions in `RUMI_DB.coaching_sessions` share **zero** `audio_url` values — different
programs, different teachers. There is no human↔AI pairing *in the warehouse*.

**Resolved by:** the **paired scoring study** (Railway Postgres, `study_data.py`): 300
recordings, 55 coaches (up to 3 per recording), scored against the same observation rubric
that 6 AI models also scored. `study_compiled` pairs human consensus vs each model.
`step3a_human_irr.py` computes real per-indicator kappa. Run:

```bash
export RUMI_STUDY_PG_URL=postgresql://...   # the study DB
python -m rumi_evals.cli --steps 3a
```

**What the real numbers say (2026-07-06 run):**

| Model | AI-scored recordings | median weighted κ vs human | indicators ≥0.70 | mean AI−human (ordinal) |
|---|---|---|---|---|
| gpt-5.1 | 101 | **0.195** | 0/21 | −0.24 |
| minimax-m2.7 | 100 | 0.087 | 0/21 | −0.54 |
| deepseek-v4-pro | 75 | 0.046 | 0/20 | −0.83 |

*(kimi/mistral/nemotron are mid-run — 15/8/2 of 100 scored — reported but flagged low-coverage.)*

**⚠️ The critical caveat — the 0.70 bar is currently unreachable by construction.** The
**human ceiling** (inter-coach agreement) is **median Fleiss κ = 0.068, pairwise exact = 0.44**
— the coaches barely agree with each other, near chance. No AI can agree with a "human
consensus" better than the humans agree among themselves. So a failing kappa here is
**as much a rater-calibration / rubric-clarity problem as a model problem.** Worst example:
indicator **SI3** — human Fleiss κ ≈ −0.01 (coaches essentially random) *and* AI scores it a
full category harsher (gpt −1.25). Best-aligned: PIA-2, S1, S2.

**Two distinct problems this surfaces:**
1. **Raise the ceiling first** — rater calibration training and/or sharpening the definitions
   of low-agreement indicators (SI3, PIA-3/4, PIC-2). Until the ceiling rises, 0.70 is not a
   meaningful target.
2. **AI harshness bias** — every model scores systematically *below* human coaches
   (deepseek −0.83 overall; L2/SI3 down >1.3 categories). This is a prompt/calibration fix on
   the scoring side, independent of (1).

**Recommendation:** report AI–human agreement *relative to the human ceiling* (e.g. as a % of
achievable agreement), not against a fixed 0.70, until rater reliability is addressed.

## G2 — No gold transcripts (blocks Step 2 WER)

**Framework wants:** WER <5% primary language, <10% code-switched, by language and speaker.

**Reality:** `coaching_sessions.transcript_text` is the *hypothesis*; there is no
human-verified reference transcription anywhere, and no speaker-type labels.

**To unblock:** `step2_stt_wer.py` already writes a stratified annotation sample sheet on
first run. Have bilingual annotators transcribe those recordings into
`data/gold_transcripts/gold.csv`; WER/CER by language and code-switch stratum compute
automatically. (`diarization_data` exists per session but its speaker-ID accuracy is itself
unvalidated — a second, smaller gold-labeling task.)

## G3 — Satisfaction instrumentation is empty (blocks Step 4a survey)

**Framework wants:** teacher survey mean >4.0/5 on specificity, actionability, tone.

**Reality:** `coaching_quality_metrics.user_satisfaction_rating` and `user_feedback` exist as
columns but are **100% NULL** — nothing writes to them. There is no captured teacher
sentiment anywhere in the warehouse.

**Impact:** The framework's actual metric (how feedback is *received*) cannot be measured.
Shipped instead: an LLM coach-judge that scores the feedback text itself on the same three
dimensions — a quality proxy, not a receipt-of-feedback measure.

**To unblock:** Wire a lightweight post-session WhatsApp rating prompt into those columns.
Until then, Step 4a runs in labelled PROXY mode.

## G4 — No endpoint to red-team (blocks Step 4b live testing)

**Framework wants:** zero successful jailbreaks in red-team testing against the system.

**Reality:** Guardrail testing must hit the live Rumi WhatsApp pipeline (its exact
prompt + model + multi-layer content checks). This repo reads the warehouse; it has no
access to that runtime.

**To unblock:** `step4b_guardrails.run_against(respond, cfg)` takes any
`respond(text) -> str` callable. Point it at a Rumi staging webhook or the coaching service
directly. The seed attack set (`prompts/redteam_attacks.yaml`, 12 attacks across scope
escape / harmful content / data exposure / prompt extraction, including code-switched
phrasings) and the LLM grader are ready.

## G5 — No Rumi-linked student outcomes (blocks half of Step 6)

**Framework wants:** where student data exists, correlate teacher improvement with student
learning gains.

**Reality:** Student assessment tables exist but are not linked to Rumi usage, and the
closest teacher-outcome regression table has no coaching-session linkage field. No
treatment/control structure exists anywhere.

**To unblock:** Out of scope for an AI-eval workstream — needs M&E instrumentation linking
`users.emis_code` to student assessment records. The within-teacher trend half of Step 6
runs today without it.

---

## What runs today on real extracts (no unblocking needed)

- **Step 1 (audio capture gate)** — rejection/abandonment rates from session statuses.
- **Step 3a cross-LLM agreement + wobble test** — needs the real FICO rubric pasted into
  `prompts/fico_rubric.md` (one paste), then runs against stored transcripts.
- **Step 3b (hallucination spot-check)** — LLM judge verifies analysis evidence against
  transcripts. Fully runnable.
- **Step 4a (feedback quality PROXY)** — runnable now; upgrade to the real survey per G3.
- **Step 5 (AI-side monthly scoring profile / drift)** — runnable; becomes full coach–AI
  alignment once G1 pairs exist.
- **Step 6 (within-teacher longitudinal trend)** — runnable (142 teachers, 6-month window).

## Decisions needed from you

1. **~~G1 is the crux~~ — now resolved.** The paired study exists and IRR runs. The new crux is
   the **human ceiling**: coaches agree at ~chance (Fleiss κ 0.07). Do you want to (a) invest in
   rater calibration + rubric sharpening before holding models to 0.70, and (b) treat the AI
   harshness bias as a separate scoring-prompt fix? The eval reports both; acting on them is a
   product decision.
2. **Finish the AI scoring runs.** Only gpt-5.1 and minimax are fully scored; deepseek is 75%,
   and kimi/mistral/nemotron are <15% done. Complete those runs (via the DC API / study
   harness) for a fair 6-model comparison.
3. **Only FICO/observation-rubric is scored.** OECD/HOTS/TEACH: scored elsewhere, not yet in
   production, or deprioritized? Extending the harness is mechanical once indicators exist.

## Note on the score source

The study uses the real human observation rubric (yes/partial/no on SI/PIA/PIC/MA + subject
Section C: L*/M*/S*), **not** the warehouse's B/C/D/F FICO columns — these are different
rubrics. `step3a_human_irr` targets the study rubric; the warehouse steps (5, 6) still use
B/C/D/F. Confirm which rubric is canonical for production if they need to be reconciled.
