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
