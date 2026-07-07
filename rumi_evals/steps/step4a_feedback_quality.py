"""Step 4a — Feedback quality.

Framework metric: teacher survey mean >4.0/5 on specificity, actionability,
tone — BLOCKED: coaching_quality_metrics.user_satisfaction_rating is 100% NULL
(instrumentation exists, nothing writes to it). See GAPS.md G3.

Runnable proxy shipped here: an LLM coach-judge scores the AI feedback itself
on the same three dimensions. This answers "is the feedback good?" but not
"is it *received* as constructive?" — the survey is still required.
"""
from __future__ import annotations

import json

import pandas as pd

from ..config import PACKAGE_ROOT
from ..judge import FEEDBACK_QUALITY_SYSTEM, FeedbackQualityScore, judge_parse

RESULTS_DIR = PACKAGE_ROOT / "results"


def _feedback_text(analysis_data: str, prioritized_action) -> str:
    parts = []
    try:
        parsed = json.loads(analysis_data)
        # pull any obviously feedback-like keys; fall back to the whole payload
        for key in ("feedback", "recommendations", "suggestions", "summary", "debrief"):
            if isinstance(parsed, dict) and key in parsed:
                parts.append(json.dumps(parsed[key], ensure_ascii=False))
        if not parts:
            parts.append(json.dumps(parsed, ensure_ascii=False)[:8000])
    except (json.JSONDecodeError, TypeError):
        parts.append(str(analysis_data)[:8000])
    if isinstance(prioritized_action, str) and prioritized_action:
        parts.append(f"Prioritized action: {prioritized_action}")
    return "\n\n".join(parts)


def run(backend, cfg: dict) -> dict:
    df = backend.fetch("session_transcripts")
    df = df[df["analysis_data"].notna()]
    n = cfg["sampling"]["feedback_quality_sample_size"]
    sample = df.sample(n=min(n, len(df)), random_state=cfg["sampling"]["random_seed"])

    rows = []
    for _, r in sample.iterrows():
        try:
            s = judge_parse(
                FEEDBACK_QUALITY_SYSTEM,
                f"<lesson_transcript_excerpt>\n{str(r['transcript_text'])[:6000]}\n"
                f"</lesson_transcript_excerpt>\n\n<feedback>\n"
                f"{_feedback_text(r['analysis_data'], r.get('prioritized_action'))}\n</feedback>",
                FeedbackQualityScore,
                cfg,
            )
            rows.append({"id": r["id"], "specificity": s.specificity,
                         "actionability": s.actionability, "tone": s.tone,
                         "rationale": s.rationale})
        except Exception as e:
            print(f"  judge failed on {r['id']}: {e}")

    res = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    res.to_csv(RESULTS_DIR / "step4a_feedback_scores.csv", index=False)

    th = cfg["thresholds"]["step4a_feedback_quality"]["min_mean_rating"]
    means = res[["specificity", "actionability", "tone"]].mean()
    return {
        "step": "4a_feedback_quality",
        "mode": "LLM-judge PROXY (teacher survey not instrumented — see GAPS.md G3)",
        "n_scored": len(res),
        "mean_specificity": round(float(means["specificity"]), 2),
        "mean_actionability": round(float(means["actionability"]), 2),
        "mean_tone": round(float(means["tone"]), 2),
        "passes_proxy": bool((means >= th).all()),
        "lowest_scoring_sessions": res.nsmallest(5, ["specificity", "actionability", "tone"])
        .to_dict("records"),
    }
