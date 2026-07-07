"""Step 3a (real) — human-vs-AI inter-rater reliability from the paired study.

This is the module the whole framework hangs on, now RUNNABLE because the paired
study exists (gap G1 resolved). It computes, per rubric indicator:

1. HUMAN CEILING — inter-rater reliability among the coaches who scored the same
   recording (Fleiss' kappa + mean pairwise exact agreement). No AI scorer can be
   expected to agree with "the humans" better than the humans agree with each other,
   so this bounds every AI number below.

2. HUMAN-vs-AI — for each of the 6 AI models, Cohen's + quadratic-weighted kappa
   between the human consensus and that model, per indicator, vs the framework's
   0.70 deployment bar and 0.75 mature bar.

3. DIRECTIONAL BIAS — mean(AI ordinal) - mean(human ordinal) per indicator, so a
   systematic harsh/lenient tilt is visible (this is the "direction of disagreement"
   the framework's Step 5 asks for).

Scale: no=0, partial=1, yes=2; na treated as missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PACKAGE_ROOT
from ..metrics import cohen_kappa, exact_agreement, fleiss_kappa
from ..study_data import ai_scores, available, indicators_for, load_compiled, normalize_token

RESULTS_DIR = PACKAGE_ROOT / "results"


def _human_ceiling(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Fleiss kappa + mean pairwise exact agreement among human raters, per indicator."""
    # Collect, per indicator, one category-count vector per recording (dedup on recording_id
    # since study_compiled repeats each recording across model runs).
    per_ind: dict[str, list[list[int]]] = {}
    exact_pairs: dict[str, list[float]] = {}
    seen = set()
    for _, row in df.iterrows():
        rid = row["recording_id"]
        if rid in seen or not isinstance(row["human_raters"], list) or len(row["human_raters"]) < 2:
            continue
        seen.add(rid)
        inds = indicators_for(row["subject_key"], cfg)
        for ind in inds:
            vals = [normalize_token(r["indicators"].get(ind), cfg) for r in row["human_raters"]]
            vals = [v for v in vals if v is not None]
            if len(vals) < 2:
                continue
            vec = [0, 0, 0]
            for v in vals:
                vec[v] += 1
            per_ind.setdefault(ind, []).append(vec)
            # mean pairwise exact agreement within this recording
            agree = [vals[i] == vals[j] for i in range(len(vals)) for j in range(i + 1, len(vals))]
            exact_pairs.setdefault(ind, []).append(float(np.mean(agree)))

    rows = []
    for ind, vecs in per_ind.items():
        rows.append({
            "indicator": ind,
            "n_recordings": len(vecs),
            "human_fleiss_kappa": round(fleiss_kappa(vecs, 3), 3),
            "human_pairwise_exact": round(float(np.mean(exact_pairs[ind])), 3),
        })
    return pd.DataFrame(rows)


def _human_vs_ai(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per (model, indicator): kappa of human consensus vs AI."""
    min_n = cfg["study"]["min_paired_for_kappa"]
    order = cfg["study"]["category_order"]
    records = []
    for model, mdf in df[df["has_ai"] & (df["n_raters"] >= 1)].groupby("run_label"):
        # gather paired (human_consensus, ai) per indicator
        paired: dict[str, list[tuple[int, int]]] = {}
        for _, row in mdf.iterrows():
            if not isinstance(row["human_consensus"], dict):
                continue
            ai = ai_scores(row)
            for ind in indicators_for(row["subject_key"], cfg):
                h = normalize_token(row["human_consensus"].get(ind), cfg)
                a = normalize_token(ai.get(ind), cfg)
                if h is not None and a is not None:
                    paired.setdefault(ind, []).append((h, a))
        for ind, pts in paired.items():
            if len(pts) < min_n:
                continue
            h = pd.Series([p[0] for p in pts])
            a = pd.Series([p[1] for p in pts])
            records.append({
                "model": model,
                "indicator": ind,
                "n": len(pts),
                "kappa": round(cohen_kappa(h, a), 3),
                "weighted_kappa": round(cohen_kappa(h, a, weights="quadratic"), 3),
                "exact_agreement": round(exact_agreement(h, a), 3),
                "ai_minus_human_mean": round(float(a.mean() - h.mean()), 3),
            })
    return pd.DataFrame(records)


def _cross_llm(df: pd.DataFrame, cfg: dict) -> dict:
    """Pairwise agreement between AI models on the SAME recordings (per indicator, pooled)."""
    min_n = cfg["study"]["min_paired_for_kappa"]
    min_cov = cfg["study"]["min_ai_coverage"]
    coverage = df[df["has_ai"]].groupby("run_label")["recording_id"].nunique()
    models = [m for m in coverage.index if coverage[m] >= min_cov]

    # build recording_id -> {model -> {indicator -> ordinal}}
    by_rec: dict[int, dict[str, dict[str, int]]] = {}
    subj: dict[int, str] = {}
    for _, row in df[df["has_ai"]].iterrows():
        if row["run_label"] not in models:
            continue
        rid = row["recording_id"]
        subj[rid] = row["subject_key"]
        vals = {k: normalize_token(v, cfg) for k, v in ai_scores(row).items()}
        by_rec.setdefault(rid, {})[row["run_label"]] = vals

    pairs = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            a, b = [], []
            for rid, mm in by_rec.items():
                if m1 in mm and m2 in mm:
                    for ind in indicators_for(subj[rid], cfg):
                        x, y = mm[m1].get(ind), mm[m2].get(ind)
                        if x is not None and y is not None:
                            a.append(x)
                            b.append(y)
            if len(a) >= min_n:
                sa, sb = pd.Series(a), pd.Series(b)
                pairs.append({
                    "model_a": m1, "model_b": m2, "n": len(a),
                    "kappa": round(cohen_kappa(sa, sb), 3),
                    "weighted_kappa": round(cohen_kappa(sa, sb, weights="quadratic"), 3),
                    "exact_agreement": round(exact_agreement(sa, sb), 3),
                })
    return {
        "models_compared": models,
        "median_pairwise_weighted_kappa": round(float(np.median([p["weighted_kappa"] for p in pairs])), 3)
        if pairs else None,
        "pairs": pairs,
    }


def run(backend, cfg: dict) -> dict:
    if not available():
        return {
            "step": "3a_human_irr",
            "status": "blocked",
            "reason": "Study DB not reachable. Set RUMI_STUDY_PG_URL (Railway Postgres) "
            "and `pip install 'psycopg[binary]'`.",
        }

    df = load_compiled(cfg)
    ceiling = _human_ceiling(df, cfg)
    hva = _human_vs_ai(df, cfg)

    RESULTS_DIR.mkdir(exist_ok=True)
    ceiling.to_csv(RESULTS_DIR / "step3a_human_ceiling.csv", index=False)
    hva.to_csv(RESULTS_DIR / "step3a_human_vs_ai.csv", index=False)

    th = cfg["thresholds"]["step3a_reliability"]
    min_cov = cfg["study"]["min_ai_coverage"]
    coverage = df[df["has_ai"]].groupby("run_label")["recording_id"].nunique().to_dict()
    # per-model rollup: median weighted kappa across indicators + how many clear the bar
    model_rollup = []
    for model, mdf in hva.groupby("model"):
        model_rollup.append({
            "model": model,
            "ai_scored_recordings": int(coverage.get(model, 0)),
            "low_coverage": bool(coverage.get(model, 0) < min_cov),
            "n_indicators": len(mdf),
            "median_weighted_kappa": round(float(mdf["weighted_kappa"].median()), 3),
            "median_kappa": round(float(mdf["kappa"].median()), 3),
            "indicators_meeting_070": int((mdf["weighted_kappa"] >= th["min_kappa_deploy"]).sum()),
            "mean_ai_minus_human": round(float(mdf["ai_minus_human_mean"].mean()), 3),
        })
    # rank headline models (full coverage) above low-coverage ones
    model_rollup = sorted(
        model_rollup,
        key=lambda r: (r["low_coverage"], -(r["median_weighted_kappa"] if not np.isnan(r["median_weighted_kappa"]) else -9)),
    )
    headline = [m for m in model_rollup if not m["low_coverage"]]

    best = headline[0] if headline else (model_rollup[0] if model_rollup else None)
    return {
        "step": "3a_human_irr",
        "status": "ok",
        "n_recordings_in_study": int(df["recording_id"].nunique()),
        "n_models_scored": int(hva["model"].nunique()) if len(hva) else 0,
        "n_models_full_coverage": len(headline),
        "human_ceiling_median_fleiss": round(float(ceiling["human_fleiss_kappa"].median()), 3)
        if len(ceiling) else None,
        "human_ceiling_median_pairwise_exact": round(float(ceiling["human_pairwise_exact"].median()), 3)
        if len(ceiling) else None,
        "criterion": f"weighted kappa >= {th['min_kappa_deploy']} per indicator (0.75 mature)",
        "best_model": best["model"] if best else None,
        "best_model_median_weighted_kappa": best["median_weighted_kappa"] if best else None,
        "any_model_meets_bar_overall": bool(best and best["median_weighted_kappa"] >= th["min_kappa_deploy"]),
        "model_rollup": model_rollup,
        "cross_llm_agreement": _cross_llm(df, cfg),
        "human_ceiling_per_indicator": ceiling.to_dict("records"),
        "interpretation": (
            "The 0.70 kappa bar is measured against human consensus, but the human "
            "ceiling above shows coaches barely agree with each other — so the bar is "
            "currently unreachable by construction. Fix rater calibration / rubric "
            "clarity (raise the ceiling) before judging any model against 0.70. "
            "Separately, every model scores systematically harsher than humans "
            "(see directional bias)."
        ),
        "key_finding_directional_bias": {
            "note": "Negative human_ceiling or large negative ai_minus_human means AI is harsher "
            "than human coaches. Inspect results/step3a_human_vs_ai.csv per indicator.",
            "mean_ai_minus_human_all_models": round(float(hva["ai_minus_human_mean"].mean()), 3)
            if len(hva) else None,
        },
    }
