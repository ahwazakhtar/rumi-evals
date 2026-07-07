"""Agreement and reliability metrics used across steps 3a, 3b and 5.

Implemented directly (numpy/pandas only) so the pipeline has no dependency on
statsmodels for its core numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cohen_kappa(a: pd.Series, b: pd.Series, weights: str | None = None) -> float:
    """Cohen's kappa between two raters. weights: None | 'linear' | 'quadratic'."""
    mask = a.notna() & b.notna()
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return float("nan")
    cats = np.sort(pd.unique(pd.concat([a, b])))
    n_cat = len(cats)
    if n_cat < 2:
        return float("nan")
    idx = {c: i for i, c in enumerate(cats)}
    obs = np.zeros((n_cat, n_cat))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()

    row = obs.sum(axis=1)
    col = obs.sum(axis=0)
    exp = np.outer(row, col)

    if weights is None:
        w = 1 - np.eye(n_cat)
    else:
        diff = np.abs(np.subtract.outer(np.arange(n_cat), np.arange(n_cat)))
        w = diff / (n_cat - 1) if weights == "linear" else (diff / (n_cat - 1)) ** 2

    denom = (w * exp).sum()
    if denom == 0:
        return float("nan")
    return float(1 - (w * obs).sum() / denom)


def exact_agreement(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if mask.sum() == 0:
        return float("nan")
    return float((a[mask] == b[mask]).mean())


def adjacent_agreement(a: pd.Series, b: pd.Series, tolerance: int = 1) -> float:
    mask = a.notna() & b.notna()
    if mask.sum() == 0:
        return float("nan")
    return float((np.abs(a[mask] - b[mask]) <= tolerance).mean())


def icc_2_1(matrix: np.ndarray) -> float:
    """ICC(2,1), two-way random effects, absolute agreement, single measure.

    matrix: subjects x raters, no NaNs.
    """
    n, k = matrix.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)

    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_total = ((matrix - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))

    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    if denom == 0:
        return float("nan")
    return float((ms_rows - ms_err) / denom)


def fleiss_kappa(ratings: list[list[int]], n_categories: int) -> float:
    """Fleiss' kappa for N items each rated by a variable number of raters.

    ratings: list of per-item category-count vectors (length n_categories).
    Items may have different rater counts; items with <2 ratings are skipped.
    """
    rows = [r for r in ratings if sum(r) >= 2]
    if len(rows) < 2:
        return float("nan")
    # require equal rater count per item for the classic estimator; use the
    # modal rater count and drop items that don't match (keeps it well-defined).
    from collections import Counter

    counts = Counter(sum(r) for r in rows)
    n_raters = counts.most_common(1)[0][0]
    rows = [r for r in rows if sum(r) == n_raters]
    if len(rows) < 2 or n_raters < 2:
        return float("nan")
    mat = np.array(rows, dtype=float)
    n_items = mat.shape[0]
    p_j = mat.sum(axis=0) / (n_items * n_raters)
    p_i = (np.square(mat).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_e = np.square(p_j).sum()
    if p_e >= 1:
        return float("nan")
    return float((p_bar - p_e) / (1 - p_e))


def krippendorff_alpha(units: list[list[int]], level: str = "ordinal") -> float:
    """Krippendorff's alpha for any number of raters per unit, missing allowed.

    units: per unit, the list of category indices (0..q-1) actually assigned;
    drop missing values before calling. Units with <2 values are ignored.
    level: 'nominal' | 'ordinal' | 'interval'. Ordinal uses Krippendorff's
    rank-based distance computed from the coincidence-matrix marginals.
    """
    units = [u for u in units if len(u) >= 2]
    if not units:
        return float("nan")
    q = max(max(u) for u in units) + 1
    coin = np.zeros((q, q))
    for u in units:
        m = len(u)
        for i in range(m):
            for j in range(m):
                if i != j:
                    coin[u[i], u[j]] += 1.0 / (m - 1)
    n_c = coin.sum(axis=1)
    n = n_c.sum()
    if n <= 1:
        return float("nan")

    delta = np.zeros((q, q))
    for c in range(q):
        for k in range(c + 1, q):
            if level == "nominal":
                d = 1.0
            elif level == "interval":
                d = float((c - k) ** 2)
            else:  # ordinal
                d = float((n_c[c : k + 1].sum() - (n_c[c] + n_c[k]) / 2) ** 2)
            delta[c, k] = delta[k, c] = d

    d_o = (coin * delta).sum()
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n - 1)
    if d_e == 0:
        return float("nan")
    return float(1 - d_o / d_e)


def _agreement_weights(q: int, weights: str) -> np.ndarray:
    """Agreement weight matrix (1 on diagonal): 'identity'|'linear'|'quadratic'|'ordinal'."""
    if weights == "identity":
        return np.eye(q)
    diff = np.abs(np.subtract.outer(np.arange(q), np.arange(q))).astype(float)
    if weights == "linear":
        return 1 - diff / (q - 1)
    if weights == "quadratic":
        return 1 - (diff / (q - 1)) ** 2
    # Gwet's ordinal weights: M_ck = C(|c-k|+1, 2), normalized by the max.
    m = (diff + 1) * diff / 2
    return 1 - m / m.max()


def gwet_ac(units: list[list[int]], n_categories: int, weights: str = "ordinal") -> float:
    """Gwet's AC1 (weights='identity') / AC2 for multi-rater data, missing allowed.

    units: per unit, the list of category indices assigned (missing dropped).
    Units with 1 rating still inform chance agreement; <2 skip percent agreement.
    """
    units = [u for u in units if len(u) >= 1]
    q = n_categories
    if q < 2 or not units:
        return float("nan")
    w = _agreement_weights(q, weights)

    pi = np.zeros(q)
    pa_terms = []
    for u in units:
        r = np.bincount(u, minlength=q).astype(float)
        r_u = r.sum()
        pi += r / r_u
        if r_u >= 2:
            r_star = w @ r
            pa_terms.append(float((r * (r_star - 1)).sum() / (r_u * (r_u - 1))))
    if not pa_terms:
        return float("nan")
    pi /= len(units)
    p_a = float(np.mean(pa_terms))
    t_w = float(w.sum())
    p_e = t_w / (q * (q - 1)) * float((pi * (1 - pi)).sum())
    if p_e >= 1:
        return float("nan")
    return float((p_a - p_e) / (1 - p_e))


def cluster_bootstrap_ci(
    clusters: list, stat_fn, n_boot: int = 500, seed: int = 0, ci: float = 0.95
) -> tuple[float, float]:
    """Percentile CI for stat_fn(list_of_clusters), resampling clusters with replacement.

    clusters: list of per-cluster payloads (e.g. one recording's units); stat_fn
    receives a resampled list of payloads and returns a float (nan allowed).
    """
    if not clusters:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(clusters)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = stat_fn([clusters[i] for i in idx])
        if not np.isnan(s):
            stats.append(s)
    if len(stats) < max(20, n_boot // 10):
        return (float("nan"), float("nan"))
    lo = (1 - ci) / 2
    return (
        float(np.quantile(stats, lo)),
        float(np.quantile(stats, 1 - lo)),
    )


def indicator_agreement_table(
    df: pd.DataFrame, rater_a: str, rater_b: str, indicators: list[str]
) -> pd.DataFrame:
    """Per-indicator agreement between two rater column-suffixes.

    Expects columns like f"{ind}_{rater_a}" and f"{ind}_{rater_b}".
    """
    rows = []
    for ind in indicators:
        a, b = df[f"{ind}_{rater_a}"], df[f"{ind}_{rater_b}"]
        rows.append(
            {
                "indicator": ind,
                "n": int((a.notna() & b.notna()).sum()),
                "kappa": cohen_kappa(a, b),
                "weighted_kappa": cohen_kappa(a, b, weights="quadratic"),
                "exact_agreement": exact_agreement(a, b),
                "adjacent_agreement": adjacent_agreement(a, b),
            }
        )
    return pd.DataFrame(rows)
