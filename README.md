# Rumi Evals

An end-to-end evaluation pipeline for Rumi, structured one-to-one on the
**Rumi Evaluation Measurement Framework (v2)**. Each of the framework's pipeline steps maps
to a runnable module that reports the framework's own criterion (kappa >0.70, WER <5%,
zero fabricated citations, etc.) against the existing Taleemabad warehouse
(`RUMI_DB` / `tbproddb`).

It combines two eval styles:
- **data checks** (steps 1, 5, 6) — computed directly from warehouse tables;
- **LLM-judge evals** (steps 3a, 3b, 4a, 4b) — Claude scores/verifies transcripts and
  feedback with schema-validated structured output, modelled on calibrate's
  persona/scenario/metric idea (here: rubric + transcript → judged verdict).

> Read **`GAPS.md` first.** It maps every framework criterion to the data reality (validated
> against the live warehouse + study DB on 2026-07-06). **Headline result: the human-vs-AI
> IRR metric (gap G1) is now unblocked** by the paired scoring study — and it shows the 0.70
> kappa bar is currently unreachable because the *human coaches barely agree with each other*
> (median Fleiss κ ≈ 0.07). See G1 for the full numbers and what to do about it.

## Two data sources

| Source | Backend | Feeds |
|---|---|---|
| Warehouse `RUMI_DB` / `tbproddb` (BigQuery) | `data.py` (bigquery/csv) | Steps 1, 2, 3b, 4a, 5, 6 |
| **Paired scoring study** (Railway Postgres) | `study_data.py` | **Step 3a — real human-vs-AI IRR + cross-LLM** |
| Digital Coach external API | `dc_api.py` | Step 3a wobble test (fresh scoring), Step 4b guardrails |

### Secrets (never committed to the repo)

The study DB URL and the DC API key are read from environment variables only:

```bash
export RUMI_STUDY_PG_URL='postgresql://...'   # study Postgres (from postgres.txt)
export DC_API_KEY='dc_...'                     # Digital Coach X-API-Key (from api key.txt)
```

`postgres.txt` and `api key.txt` contain live credentials — keep them out of git
(add to `.gitignore`) and rotate if they leak.

## Layout

```
rumi-evals/
├── config.yaml               # thresholds (from framework v2), models, sampling
├── requirements.txt
├── sql/                      # governed extract queries, grounded in real schemas
│   ├── fico_scores.sql       #   675 scored sessions, 30 indicators
│   ├── step1_sessions.sql    #   session status / audio gate
│   ├── session_transcripts.sql   # transcript + analysis_data for judge steps
│   └── human_fico_answers.sql    # ICT human observations (reference source)
├── prompts/
│   ├── fico_rubric.md        # PLACEHOLDER — paste the real production rubric
│   └── redteam_attacks.yaml  # seed jailbreak set for step 4b
├── rumi_evals/
│   ├── config.py  data.py  metrics.py  judge.py  report.py  cli.py
│   └── steps/                # one module per framework step
├── scripts/make_sample_extracts.py   # synthetic extracts so steps 1/5/6 run offline
└── results/                  # scorecard + per-step JSON (generated)
```

## Quick start

```bash
pip install -r requirements.txt

# Option A — offline demo (steps 1, 5, 6) on synthetic extracts shaped like the real data:
python scripts/make_sample_extracts.py
python -m rumi_evals.cli --steps 1 5 6

# Option B — real data. Either:
#   (i)  set data.backend: bigquery in config.yaml + `gcloud auth application-default login`
#        (uncomment google-cloud-bigquery in requirements.txt), or
#   (ii) run each sql/*.sql via the taleemabad-data plugin / BigQuery console and save the
#        result to data/extracts/<name>.csv (csv backend, the default).
python -m rumi_evals.cli --steps all
```

LLM-judge steps (3a, 3b, 4a) additionally need `ANTHROPIC_API_KEY` (or `ant auth login`) and
a real `session_transcripts.csv` extract. Step 3a also needs the real rubric in
`prompts/fico_rubric.md`.

## Step → module → status

| Step | Module | Runs today? |
|---|---|---|
| 1. Audio capture | `step1_audio_capture` | ✅ data |
| 2. STT WER | `step2_stt_wer` | ⛔ needs gold transcripts (G2) — writes annotation sheet |
| 3a. Reliability (IRR) | `step3a_human_irr` | ✅ **real human-vs-AI IRR + cross-LLM** (study DB) |
| 3b. Hallucination | `step3b_hallucination` | ✅ LLM judge |
| 4a. Feedback quality | `step4a_feedback_quality` | ✅ LLM-judge proxy; ⛔ survey blocked (G3) |
| 4b. Guardrails | `step4b_guardrails` | ⛔ needs live endpoint (G4) — attack set ready |
| 5. Coach–AI alignment | `step5_alignment` | 🟡 AI-side only until G1 |
| 6. Longitudinal | `step6_longitudinal` | ✅ within-teacher trend; ⛔ student link (G5) |

The judge model, thresholds, and sample sizes all live in `config.yaml`. For runs over ~200
sessions, set `judge.batch: true` to route through the Message Batches API (50% cost).

## Design notes

- **Scorer-agnostic reliability.** Step 3a's wobble test and cross-LLM agreement take any
  scorer; point them at the production GPT-5 scorer inside rumi-platform to wobble-test the
  real system, or at the Claude judge for an independent second opinion.
- **Persistence.** `results/step3b_history.csv` accumulates quarterly runs for the
  framework's drift-variance criterion. Consider promoting the JSON results to warehouse
  tables (`rumi_eval_runs`, `rumi_eval_scores`) once the cadence is regular.
- **FICO only.** Only FICO has structured scores in the warehouse; the domain/indicator map
  in `config.yaml` is the single place to extend to OECD/HOTS/TEACH when those land.
