"""Step 3a — LLM rubric scoring: accuracy and reliability.

Framework metrics and what is runnable today (validated 2026-07-06):

1. Human-vs-AI IRR (Cohen's kappa >0.70 per indicator) — BLOCKED.
   Zero audio_url overlap exists between RUMI_DB.coaching_sessions and
   tbproddb.coaching_observation: the human FICO observations are a different
   program on a different teacher population. See GAPS.md gap G1.

2. Cross-LLM agreement — RUNNABLE. A Claude judge independently re-scores
   stored transcripts against the FICO rubric; agreement with the stored
   production (GPT-5) scores is computed per indicator. Low-agreement
   indicators flag rubric/prompt ambiguity. Requires prompts/fico_rubric.md
   to contain the real production rubric text (placeholder shipped).

3. Wobble test (intra-model consistency, intraclass kappa >0.6) — RUNNABLE
   via the same judge: the same transcript is scored N times and per-indicator
   ICC(2,1)/exact-match rates are computed. To wobble-test the *production*
   scorer (GPT-5 inside rumi-platform), point `score_fn` at that service
   instead — the harness is scorer-agnostic.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PACKAGE_ROOT
from ..judge import FicoIndicatorScores, judge_parse
from ..metrics import cohen_kappa, exact_agreement, icc_2_1, indicator_agreement_table

RUBRIC_PATH = PACKAGE_ROOT / "prompts" / "fico_rubric.md"
RESULTS_DIR = PACKAGE_ROOT / "results"


def _indicators(cfg: dict) -> list[str]:
    return [i for dom in cfg["fico"]["domains"].values() for i in dom]


def _judge_score(transcript: str, cfg: dict, rubric: str) -> dict[str, int]:
    system = (
        "You are an expert classroom-observation rater. Score the following lesson "
        "transcript against the FICO rubric below. Score ONLY from evidence in the "
        "transcript; use the rubric's own scale for every indicator.\n\n" + rubric
    )
    result = judge_parse(system, f"<transcript>\n{transcript}\n</transcript>", FicoIndicatorScores, cfg)
    return result.scores


def cross_llm_agreement(backend, cfg: dict, n_sessions: int = 30) -> dict:
    """Claude re-scores transcripts; compare with stored production scores."""
    rubric = RUBRIC_PATH.read_text(encoding="utf-8")
    if "PLACEHOLDER" in rubric:
        return {
            "status": "blocked",
            "reason": f"{RUBRIC_PATH} still contains the placeholder — paste the real "
            "production FICO rubric (indicator definitions + scale) from rumi-platform.",
        }

    scores = backend.fetch("fico_scores")
    transcripts = backend.fetch("session_transcripts")
    merged = scores.merge(transcripts[["id", "transcript_text"]], on="id")
    sample = merged.sample(n=min(n_sessions, len(merged)), random_state=cfg["sampling"]["random_seed"])

    indicators = _indicators(cfg)
    rows = []
    for _, r in sample.iterrows():
        judged = _judge_score(r["transcript_text"], cfg, rubric)
        row = {"id": r["id"]}
        for ind in indicators:
            row[f"{ind}_prod"] = r[f"{ind}_score"]
            row[f"{ind}_judge"] = judged.get(ind)
        rows.append(row)
    paired = pd.DataFrame(rows)

    table = indicator_agreement_table(paired, "prod", "judge", indicators)
    th = cfg["thresholds"]["step3a_reliability"]
    table["meets_deploy_bar"] = table["kappa"] >= th["min_kappa_deploy"]

    RESULTS_DIR.mkdir(exist_ok=True)
    paired.to_csv(RESULTS_DIR / "step3a_cross_llm_pairs.csv", index=False)
    return {
        "status": "ok",
        "n_sessions": len(paired),
        "indicators_meeting_deploy_bar": int(table["meets_deploy_bar"].sum()),
        "indicators_total": len(table),
        "median_kappa": round(float(table["kappa"].median()), 3),
        "per_indicator": table.to_dict("records"),
    }


def wobble_test(backend, cfg: dict, n_transcripts: int = 5, n_runs: int = 10) -> dict:
    """Same transcript, same judge, repeated: within-model variance."""
    rubric = RUBRIC_PATH.read_text(encoding="utf-8")
    if "PLACEHOLDER" in rubric:
        return {"status": "blocked", "reason": f"Fill in {RUBRIC_PATH} first."}

    transcripts = backend.fetch("session_transcripts").sample(
        n=n_transcripts, random_state=cfg["sampling"]["random_seed"]
    )
    indicators = _indicators(cfg)

    per_transcript = []
    for _, r in transcripts.iterrows():
        runs = [_judge_score(r["transcript_text"], cfg, rubric) for _ in range(n_runs)]
        mat = np.array([[run.get(ind, np.nan) for run in runs] for ind in indicators], dtype=float)
        # subjects = indicators, raters = repeated runs
        clean = mat[~np.isnan(mat).any(axis=1)]
        per_transcript.append(
            {
                "id": r["id"],
                "icc_2_1": round(icc_2_1(clean), 3) if len(clean) >= 2 else None,
                "mean_exact_match_vs_first_run": round(
                    float(np.nanmean([(mat[:, j] == mat[:, 0]).mean() for j in range(1, n_runs)])), 3
                ),
            }
        )
        (RESULTS_DIR / "wobble").mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "wobble" / f"{r['id']}.json", "w") as f:
            json.dump(runs, f, indent=2)

    th = cfg["thresholds"]["step3a_reliability"]["min_intraclass_kappa"]
    iccs = [t["icc_2_1"] for t in per_transcript if t["icc_2_1"] is not None]
    return {
        "status": "ok",
        "n_transcripts": n_transcripts,
        "n_runs_each": n_runs,
        "median_icc": round(float(np.median(iccs)), 3) if iccs else None,
        "passes": bool(iccs and float(np.median(iccs)) >= th),
        "per_transcript": per_transcript,
    }


def run(backend, cfg: dict) -> dict:
    return {
        "step": "3a_reliability",
        "human_vs_ai_irr": {
            "status": "blocked",
            "reason": "No human-scored Rumi sessions exist (zero audio_url overlap with "
            "tbproddb.coaching_observation). See GAPS.md G1 for the paired-scoring protocol.",
        },
        "cross_llm_agreement": cross_llm_agreement(backend, cfg),
        "wobble_test": wobble_test(backend, cfg),
    }
