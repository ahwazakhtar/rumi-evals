"""Step 5 — Coach–AI alignment (monthly report).

Framework metric: per-indicator agreement rate between AI and human coach,
direction of disagreement, month-on-month improvement.

STATUS: partially blocked — depends on the same human↔AI pairing as step 3a
(zero pairs exist today; GAPS.md G1). What runs now is the AI-side monthly
scoring profile: per-indicator monthly means and score-distribution shifts,
which (a) becomes the AI half of the alignment report the moment human pairs
exist, and (b) already serves the framework's drift-detection intent by
flagging indicators whose scoring behaviour moves without a rubric change.
"""
from __future__ import annotations

import pandas as pd


def run(backend, cfg: dict) -> dict:
    df = backend.fetch("fico_scores")
    df["month"] = pd.to_datetime(df["session_date"]).dt.to_period("M").astype(str)
    indicators = [i for dom in cfg["fico"]["domains"].values() for i in dom]
    ind_cols = [f"{i}_score" for i in indicators]

    monthly_means = df.groupby("month")[ind_cols + ["overall_score"]].mean().round(3)
    monthly_n = df.groupby("month")["id"].count()

    # month-over-month shift per indicator, flag largest movers
    shifts = monthly_means[ind_cols].diff().abs()
    flagged = []
    if len(monthly_means) >= 2:
        last = shifts.iloc[-1].sort_values(ascending=False)
        flagged = [
            {"indicator": k.replace("_score", ""), "abs_shift_last_month": round(float(v), 3)}
            for k, v in last.head(5).items()
        ]

    return {
        "step": "5_coach_ai_alignment",
        "status": "partial",
        "human_pairs_available": 0,
        "reason_partial": "No human-scored Rumi sessions to align against (GAPS.md G1); "
        "reporting AI-side monthly scoring profile / drift only.",
        "months_covered": monthly_n.to_dict(),
        "monthly_overall_mean": monthly_means["overall_score"].to_dict(),
        "largest_indicator_shifts_last_month": flagged,
    }
