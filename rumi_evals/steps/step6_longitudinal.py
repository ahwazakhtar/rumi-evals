"""Step 6 — Longitudinal instructional change.

Framework criterion: statistically significant improvement on >=2 rubric
indicators after 6 months of regular coaching, effect size with CIs.

RUNNABLE today at the teacher-trend level: 142 non-test teachers with scored
sessions (86 with >=3, 40 with >=6), Nov 2025 – May 2026. Method: per-indicator
OLS of score on months-since-first-session across all teachers' repeated
observations (teacher-demeaned, i.e. within-teacher change), with slope CIs.

Caveats reported in-line: no control group (improvement is not causal),
score source is the AI itself (a scoring drift would masquerade as teacher
change — read alongside step 5), and the student-learning correlation half of
Step 6 is blocked (GAPS.md G5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _within_teacher_slope(df: pd.DataFrame, col: str) -> dict | None:
    """OLS slope of teacher-demeaned score on months since teacher's first session."""
    d = df[["user_id", "months_since_first", col]].dropna()
    if d["user_id"].nunique() < 5 or len(d) < 30:
        return None
    d = d.assign(
        y=d[col] - d.groupby("user_id")[col].transform("mean"),
        x=d["months_since_first"] - d.groupby("user_id")["months_since_first"].transform("mean"),
    )
    if d["x"].std() == 0:
        return None
    res = stats.linregress(d["x"], d["y"])
    # 95% CI on slope
    ci = 1.96 * res.stderr
    sd = df[col].std()
    return {
        "indicator": col.replace("_score", ""),
        "n_obs": len(d),
        "n_teachers": int(d["user_id"].nunique()),
        "slope_per_month": round(float(res.slope), 4),
        "ci_low": round(float(res.slope - ci), 4),
        "ci_high": round(float(res.slope + ci), 4),
        "p_value": round(float(res.pvalue), 4),
        "effect_size_6mo_sd": round(float(res.slope * 6 / sd), 3) if sd else None,
    }


def run(backend, cfg: dict) -> dict:
    df = backend.fetch("fico_scores")
    df["session_date"] = pd.to_datetime(df["session_date"])
    first = df.groupby("user_id")["session_date"].transform("min")
    df["months_since_first"] = (df["session_date"] - first).dt.days / 30.44

    # keep teachers with repeated observations
    counts = df.groupby("user_id")["id"].transform("count")
    rep = df[counts >= 3].copy()

    indicators = [i for dom in cfg["fico"]["domains"].values() for i in dom]
    results = []
    for ind in indicators + ["overall"]:
        col = f"{ind}_score" if ind != "overall" else "overall_score"
        r = _within_teacher_slope(rep, col)
        if r:
            results.append(r)

    alpha = cfg["thresholds"]["step6_longitudinal"]["alpha"]
    improving = [r for r in results if r["indicator"] != "overall"
                 and r["p_value"] < alpha and r["slope_per_month"] > 0]
    need = cfg["thresholds"]["step6_longitudinal"]["min_significant_indicators"]

    return {
        "step": "6_longitudinal",
        "n_teachers_with_3plus_sessions": int(rep["user_id"].nunique()),
        "n_observations_used": len(rep),
        "date_range": [str(df["session_date"].min().date()), str(df["session_date"].max().date())],
        "significant_improving_indicators": len(improving),
        "criterion": f">={need} indicators improving at p<{alpha}",
        "passes": len(improving) >= need,
        "improving": improving,
        "overall": next((r for r in results if r["indicator"] == "overall"), None),
        "all_indicators": results,
        "caveats": [
            "Within-teacher trend, no control group — improvement is descriptive, not causal.",
            "Scores come from the AI scorer itself; scoring drift would mimic teacher change "
            "(cross-check step 5 drift flags before believing a trend).",
            "Student-learning correlation is blocked: no Rumi-linked student outcome data (GAPS.md G5).",
        ],
    }
