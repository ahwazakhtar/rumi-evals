"""Step 3a (real) — human-vs-AI inter-rater reliability from the paired study.

This is the module the whole framework hangs on, now RUNNABLE because the paired
study exists (gap G1 resolved). It computes, per rubric indicator:

1. HUMAN CEILING — inter-rater reliability among the coaches who scored the same
   recording (Fleiss' kappa + mean pairwise exact agreement). No AI scorer can be
   expected to agree with "the humans" better than the humans agree with each other,
   so this bounds every AI number below.

2. HUMAN-vs-AI — for each of the 6 AI models, Cohen's + LINEAR-weighted kappa
   (runbook convention; quadratic kept as a secondary column) between the human
   consensus and that model, per indicator, vs the framework's 0.70 deployment
   bar and 0.75 mature bar. DC-vs-each-individual-coach and a unit-identical
   pooled coach-coach kappa ceiling make the like-for-like comparison; alpha
   covers the multi-rater reliability CI and the exchangeability test.

3. DIRECTIONAL BIAS — mean(AI ordinal) - mean(human ordinal) per indicator, so a
   systematic harsh/lenient tilt is visible (this is the "direction of disagreement"
   the framework's Step 5 asks for).

Scale: no=0, partial=1, yes=2; na treated as missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PACKAGE_ROOT
from ..metrics import (
    cluster_bootstrap_ci,
    cohen_kappa,
    exact_agreement,
    fleiss_kappa,
    gwet_ac,
    krippendorff_alpha,
)
from ..study_data import (
    ai_scores,
    applicability_token,
    available,
    indicators_for,
    load_compiled,
    normalize_token,
)

RESULTS_DIR = PACKAGE_ROOT / "results"

CATEGORY_NAMES = ["no", "partial", "yes"]
SCALE = [0, 1, 2]  # full ordinal scale, pinned so weight matrices never shrink


def _r3(x) -> float | None:
    """Round to 3dp; NaN becomes None so results stay valid JSON."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)


def _section(ind: str, cfg: dict) -> str:
    return "B" if ind in set(cfg["study"]["section_b"]) else "C"


def _human_units(df: pd.DataFrame, cfg: dict) -> dict[int, dict[str, list[int]]]:
    """Per recording, per indicator, the list of human ordinal ratings (na dropped).

    Deduped on recording_id (study_compiled repeats recordings across model runs).
    """
    out: dict[int, dict[str, list[int]]] = {}
    seen = set()
    for _, row in df.iterrows():
        rid = row["recording_id"]
        if rid in seen or not isinstance(row["human_raters"], list):
            continue
        seen.add(rid)
        by_ind = {}
        for ind in indicators_for(row["subject_key"], cfg):
            vals = [normalize_token(r["indicators"].get(ind), cfg) for r in row["human_raters"]]
            vals = [int(v) for v in vals if v is not None]
            if vals:
                by_ind[ind] = vals
        if by_ind:
            out[rid] = by_ind
    return out


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
        # units for alpha/AC2: expand each count vector back to a rating list
        units = [[c for c in range(3) for _ in range(int(v[c]))] for v in vecs]
        totals = np.array(vecs, dtype=float).sum(axis=0)
        prev = totals / totals.sum() if totals.sum() else totals
        rows.append({
            "indicator": ind,
            "section": _section(ind, cfg),
            "n_recordings": len(vecs),
            "human_fleiss_kappa": _r3(fleiss_kappa(vecs, 3)),
            "human_alpha_ordinal": _r3(krippendorff_alpha(units, level="ordinal")),
            "human_gwet_ac2_ordinal": _r3(gwet_ac(units, 3, weights="ordinal")),
            "human_pairwise_exact": _r3(float(np.mean(exact_pairs[ind]))),
            **{f"prev_{CATEGORY_NAMES[c]}": round(float(prev[c]), 3) for c in range(3)},
        })
    return pd.DataFrame(rows)


def _pooled_ceiling(units_by_rec: dict[int, dict[str, list[int]]], cfg: dict) -> dict:
    """One pooled human-human alpha over all (recording x indicator) units, with a
    bootstrap CI that resamples recordings (ratings within a recording are clustered)."""
    clusters = [
        [vals for vals in by_ind.values() if len(vals) >= 2]
        for by_ind in units_by_rec.values()
    ]
    clusters = [c for c in clusters if c]

    def stat(cl):
        return krippendorff_alpha([u for rec in cl for u in rec], level="ordinal")

    alpha = stat(clusters)
    lo, hi = cluster_bootstrap_ci(
        clusters, stat, n_boot=cfg["study"]["bootstrap_n"], seed=cfg["study"]["bootstrap_seed"]
    )
    all_units = [u for rec in clusters for u in rec]
    return {
        "alpha_ordinal": _r3(alpha),
        "alpha_ci95": [_r3(lo), _r3(hi)],
        "gwet_ac2_ordinal": _r3(gwet_ac(all_units, 3, weights="ordinal")),
        "n_recordings": len(clusters),
        "n_units": len(all_units),
    }


def _dc_vs_each_coach(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """DC vs every INDIVIDUAL coach rating (not the consensus), pooled per
    (model, indicator). Treats DC as 'just another rater' — the consensus
    comparison systematically inflates agreement, so both are reported.

    Returns (per-indicator table, pooled-over-everything linear kappa per model);
    the pooled figure is the runbook's headline 'DC vs individual coach' number.
    """
    min_n = cfg["study"]["min_paired_for_kappa"]
    records = []
    pooled = {}
    for model, mdf in df[df["has_ai"]].groupby("run_label"):
        paired: dict[str, list[tuple[int, int]]] = {}
        for _, row in mdf.iterrows():
            if not isinstance(row["human_raters"], list):
                continue
            ai = ai_scores(row)
            for ind in indicators_for(row["subject_key"], cfg):
                a = normalize_token(ai.get(ind), cfg)
                if a is None:
                    continue
                for r in row["human_raters"]:
                    h = normalize_token(r["indicators"].get(ind), cfg)
                    if h is not None:
                        paired.setdefault(ind, []).append((int(h), int(a)))
        all_pts = [p for pts in paired.values() for p in pts]
        if all_pts:
            h = pd.Series([p[0] for p in all_pts])
            a = pd.Series([p[1] for p in all_pts])
            pooled[model] = {
                "n_pairs": len(all_pts),
                "weighted_kappa": _r3(cohen_kappa(h, a, weights="linear", categories=SCALE)),
            }
        for ind, pts in paired.items():
            if len(pts) < min_n:
                continue
            h = pd.Series([p[0] for p in pts])
            a = pd.Series([p[1] for p in pts])
            records.append({
                "model": model,
                "indicator": ind,
                "section": _section(ind, cfg),
                "n_pairs": len(pts),
                "weighted_kappa": _r3(cohen_kappa(h, a, weights="linear", categories=SCALE)),
                "weighted_kappa_quadratic": _r3(cohen_kappa(h, a, weights="quadratic", categories=SCALE)),
                "exact_agreement": _r3(exact_agreement(h, a)),
            })
    return pd.DataFrame(records), pooled


def _pairwise_kappa_ceiling(
    units_by_rec: dict[int, dict[str, list[int]]], cfg: dict
) -> tuple[dict, pd.DataFrame]:
    """Runbook Step 3 ceiling: all coach-coach rating pairs pooled through the SAME
    linear-weighted kappa used for the DC comparisons, so ceiling and DC numbers
    are unit-identical. Pairs are symmetrized (both orders) because coach slots
    are interchangeable draws from the pool, not consistent people."""
    clusters: list[list[tuple[int, int]]] = []
    per_ind: dict[str, list[tuple[int, int]]] = {}
    for by_ind in units_by_rec.values():
        rec_pairs = []
        for ind, vals in by_ind.items():
            for i in range(len(vals)):
                for j in range(len(vals)):
                    if i != j:
                        rec_pairs.append((vals[i], vals[j]))
                        per_ind.setdefault(ind, []).append((vals[i], vals[j]))
        if rec_pairs:
            clusters.append(rec_pairs)

    def stat(cl):
        pts = [p for rec in cl for p in rec]
        return cohen_kappa(
            pd.Series([p[0] for p in pts]), pd.Series([p[1] for p in pts]),
            weights="linear", categories=SCALE,
        )

    kappa = stat(clusters) if clusters else float("nan")
    lo, hi = cluster_bootstrap_ci(
        clusters, stat, n_boot=cfg["study"]["bootstrap_n"], seed=cfg["study"]["bootstrap_seed"]
    )
    ind_rows = []
    min_n = cfg["study"]["min_paired_for_kappa"]
    for ind, pts in per_ind.items():
        if len(pts) < min_n:
            continue
        h = pd.Series([p[0] for p in pts])
        a = pd.Series([p[1] for p in pts])
        ind_rows.append({
            "indicator": ind,
            "human_pairwise_kappa": _r3(cohen_kappa(h, a, weights="linear", categories=SCALE)),
        })
    pooled = {
        "weighted_kappa": _r3(kappa),
        "kappa_ci95": [_r3(lo), _r3(hi)],
        "n_pairs_symmetrized": sum(len(rec) for rec in clusters),
        "n_recordings": len(clusters),
    }
    return pooled, pd.DataFrame(ind_rows)


def _exchangeability(
    df: pd.DataFrame, units_by_rec: dict[int, dict[str, list[int]]], cfg: dict
) -> list[dict]:
    """Headline test: does pooled alpha drop when DC joins the 3 coaches as a 4th
    rater? delta ~ 0 means DC is statistically exchangeable with a human coach.
    Computed per model, on the recordings that model scored; CI via paired
    cluster bootstrap on recordings."""
    min_rec = cfg["study"]["min_exchangeability_recordings"]
    out = []
    for model, mdf in df[df["has_ai"]].groupby("run_label"):
        # clusters: per recording, list of (human_unit, human_plus_ai_unit) pairs
        clusters = []
        for _, row in mdf.drop_duplicates("recording_id").iterrows():
            rid = row["recording_id"]
            by_ind = units_by_rec.get(rid)
            if not by_ind:
                continue
            ai = ai_scores(row)
            pairs = []
            for ind, vals in by_ind.items():
                if len(vals) < 2:
                    continue
                a = normalize_token(ai.get(ind), cfg)
                pairs.append((vals, vals + [int(a)] if a is not None else vals))
            if pairs:
                clusters.append(pairs)
        if len(clusters) < min_rec:
            continue

        def stat_delta(cl):
            h = [p[0] for rec in cl for p in rec]
            ha = [p[1] for rec in cl for p in rec]
            return krippendorff_alpha(ha, level="ordinal") - krippendorff_alpha(h, level="ordinal")

        alpha_h = krippendorff_alpha([p[0] for rec in clusters for p in rec], level="ordinal")
        alpha_ha = krippendorff_alpha([p[1] for rec in clusters for p in rec], level="ordinal")
        lo, hi = cluster_bootstrap_ci(
            clusters, stat_delta,
            n_boot=cfg["study"]["bootstrap_n"], seed=cfg["study"]["bootstrap_seed"],
        )
        out.append({
            "model": model,
            "n_recordings": len(clusters),
            "alpha_humans_only": _r3(alpha_h),
            "alpha_humans_plus_ai": _r3(alpha_ha),
            "delta_alpha": _r3(alpha_ha - alpha_h),
            "delta_alpha_ci95": [_r3(lo), _r3(hi)],
            "exchangeable": bool(hi >= 0) if not np.isnan(hi) else None,
        })
    return sorted(out, key=lambda r: -(r["delta_alpha"] if r["delta_alpha"] is not None else -9))


def _applicability(df: pd.DataFrame, cfg: dict) -> dict:
    """Stage 1 of the two-stage N/A handling: do raters agree an indicator even
    APPLIES (scored vs marked na)? Score-level stats elsewhere treat na as missing;
    this quantifies what that convention hides."""
    human_units: list[list[int]] = []
    n_na = n_rated = 0
    seen = set()
    dc_pairs: dict[str, list[tuple[int, int]]] = {}
    for _, row in df.iterrows():
        rid = row["recording_id"]
        inds = indicators_for(row["subject_key"], cfg)
        if rid not in seen and isinstance(row["human_raters"], list):
            seen.add(rid)
            for ind in inds:
                vals = [applicability_token(r["indicators"].get(ind), cfg) for r in row["human_raters"]]
                vals = [v for v in vals if v is not None]
                n_na += sum(1 for v in vals if v == 0)
                n_rated += len(vals)
                if len(vals) >= 2:
                    human_units.append(vals)
        if row["has_ai"] and isinstance(row["human_consensus"], dict):
            ai = ai_scores(row)
            for ind in inds:
                h = applicability_token(row["human_consensus"].get(ind), cfg)
                a = applicability_token(ai.get(ind), cfg)
                if h is not None and a is not None:
                    dc_pairs.setdefault(row["run_label"], []).append((h, a))
    per_model = []
    for model, pts in dc_pairs.items():
        h = pd.Series([p[0] for p in pts])
        a = pd.Series([p[1] for p in pts])
        per_model.append({
            "model": model,
            "n": len(pts),
            "human_na_rate": round(float((h == 0).mean()), 4),
            "ai_na_rate": round(float((a == 0).mean()), 4),
            "kappa_binary": _r3(cohen_kappa(h, a, categories=[0, 1])),
            "raw_agreement": _r3(exact_agreement(h, a)),
        })
    return {
        "note": "Agreement on WHETHER an indicator applies (scored vs na). All other "
        "stats treat na as missing — this is the na-decision agreement itself.",
        "human_na_rate": round(n_na / n_rated, 4) if n_rated else None,
        "human_alpha_binary": _r3(krippendorff_alpha(human_units, level="nominal")),
        "dc_vs_consensus_by_model": sorted(per_model, key=lambda r: -(r["n"])),
    }


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
                "section": _section(ind, cfg),
                "n": len(pts),
                "kappa": _r3(cohen_kappa(h, a, categories=SCALE)),
                # Runbook convention: LINEAR weights are the primary weighted kappa;
                # quadratic half-forgives adjacent misses — DC's dominant error mode.
                "weighted_kappa": _r3(cohen_kappa(h, a, weights="linear", categories=SCALE)),
                "weighted_kappa_quadratic": _r3(cohen_kappa(h, a, weights="quadratic", categories=SCALE)),
                "exact_agreement": _r3(exact_agreement(h, a)),
                "ai_minus_human_mean": _r3(float(a.mean() - h.mean())),
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
    units_by_rec = _human_units(df, cfg)
    pooled = _pooled_ceiling(units_by_rec, cfg)
    pairwise, dc_pooled = _dc_vs_each_coach(df, cfg)
    kappa_ceiling, ceiling_ind_kappa = _pairwise_kappa_ceiling(units_by_rec, cfg)
    exchange = _exchangeability(df, units_by_rec, cfg)
    applicability = _applicability(df, cfg)

    if len(ceiling_ind_kappa):
        ceiling = ceiling.merge(ceiling_ind_kappa, on="indicator", how="left")

    RESULTS_DIR.mkdir(exist_ok=True)
    ceiling.to_csv(RESULTS_DIR / "step3a_human_ceiling.csv", index=False)
    hva.to_csv(RESULTS_DIR / "step3a_human_vs_ai.csv", index=False)
    pairwise.to_csv(RESULTS_DIR / "step3a_dc_vs_each_coach.csv", index=False)

    # runbook headline: DC pooled linear kappa as a fraction of the human ceiling,
    # both measured with the identical estimator and weight matrix
    ceil_k = kappa_ceiling["weighted_kappa"]
    ai_coverage = df[df["has_ai"]].groupby("run_label")["recording_id"].nunique().to_dict()
    for m, d in dc_pooled.items():
        d["low_coverage"] = bool(ai_coverage.get(m, 0) < cfg["study"]["min_ai_coverage"])
        d["pct_of_human_ceiling"] = (
            _r3(d["weighted_kappa"] / ceil_k) if d["weighted_kappa"] is not None and ceil_k else None
        )

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
        "human_ceiling_median_alpha_ordinal": round(float(ceiling["human_alpha_ordinal"].median()), 3)
        if len(ceiling) else None,
        "human_ceiling_median_gwet_ac2": round(float(ceiling["human_gwet_ac2_ordinal"].median()), 3)
        if len(ceiling) else None,
        "human_ceiling_median_pairwise_exact": round(float(ceiling["human_pairwise_exact"].median()), 3)
        if len(ceiling) else None,
        "human_ceiling_pooled": pooled,
        "human_ceiling_pairwise_kappa": {
            "note": "Runbook Step 3: all coach-coach pairs pooled through the same "
            "linear-weighted kappa as the DC comparisons — the unit-identical ceiling.",
            **kappa_ceiling,
        },
        "dc_vs_each_coach_pooled_by_model": {
            "note": "Runbook headline: all DC-coach pairs pooled, linear-weighted "
            "kappa, with pct_of_human_ceiling vs human_ceiling_pairwise_kappa.",
            "by_model": dc_pooled,
        },
        "exchangeability_delta_alpha": {
            "note": "alpha(3 coaches + DC as 4th rater) - alpha(3 coaches). delta ~ 0 "
            "=> DC is statistically exchangeable with a human coach. CI: cluster "
            "bootstrap over recordings.",
            "by_model": exchange,
        },
        "na_applicability_agreement": applicability,
        "dc_vs_each_coach_median_by_model": {
            m: _r3(float(g["weighted_kappa"].median()))
            for m, g in pairwise.groupby("model")
        } if len(pairwise) else {},
        "criterion": f"LINEAR-weighted kappa >= {th['min_kappa_deploy']} per indicator "
        "(0.75 mature); weights per runbook — adjacent miss 0.5, extreme miss 1.0, "
        "categories pinned to the full no/partial/yes scale",
        "best_model": best["model"] if best else None,
        "best_model_median_weighted_kappa": best["median_weighted_kappa"] if best else None,
        "any_model_meets_bar_overall": bool(best and best["median_weighted_kappa"] >= th["min_kappa_deploy"]),
        "model_rollup": model_rollup,
        "cross_llm_agreement": _cross_llm(df, cfg),
        "human_ceiling_per_indicator": ceiling.to_dict("records"),
        "interpretation": (
            "All weighted kappas are LINEAR with categories pinned to no/partial/yes "
            "(runbook Step 2); quadratic is kept as an explicit secondary column only. "
            "Judge DC against the human ceiling, not the absolute 0.70 bar — in kappa "
            "units use human_ceiling_pairwise_kappa (same estimator and weights as the "
            "DC numbers; pct_of_human_ceiling in dc_vs_each_coach_pooled_by_model is "
            "the like-for-like headline). Alpha is kept for what kappa cannot do: "
            "human_ceiling_pooled uses the full 3-rater structure with a clustered "
            "bootstrap CI, and exchangeability_delta_alpha is the single cleanest "
            "verdict — if adding DC as a 4th rater does not drop alpha (CI includes "
            "0), DC is statistically exchangeable with a coach. Where Fleiss/alpha "
            "look near-zero but gwet_ac2 is much higher, the indicator has skewed "
            "prevalence (kappa paradox) — read prev_* columns before concluding "
            "raters disagree. Consensus-based kappa (human_vs_ai) is inflated "
            "relative to dc_vs_each_coach; report both."
        ),
        "key_finding_directional_bias": {
            "note": "Negative human_ceiling or large negative ai_minus_human means AI is harsher "
            "than human coaches. Inspect results/step3a_human_vs_ai.csv per indicator.",
            "mean_ai_minus_human_all_models": round(float(hva["ai_minus_human_mean"].mean()), 3)
            if len(hva) else None,
        },
    }
