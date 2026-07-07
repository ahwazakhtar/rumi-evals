"""Generate synthetic CSV extracts so the data-only steps (1, 5, 6) run without
BigQuery access. Row counts / distributions are seeded to resemble the real
warehouse figures validated 2026-07-06 (675 scored sessions, 142 teachers,
status split 567 completed / 104 abandoned / 36 failed / 8 cancelled).

For real runs, replace these with true extracts of the sql/*.sql files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "extracts"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20260706)

INDICATORS = (
    [f"B{i}" for i in range(1, 11)]
    + [f"C{i}" for i in range(1, 13)]
    + [f"D{i}" for i in range(1, 8)]
    + [f"F{i}" for i in range(1, 9)]
)


def make_fico_scores(n_teachers=142, n_rows=675) -> pd.DataFrame:
    # teacher session counts roughly matching the observed distribution
    per = np.concatenate([
        np.ones(37), np.full(19, 2), rng.integers(3, 6, 46),
        rng.integers(6, 11, 27), rng.integers(11, 20, 13),
    ]).astype(int)
    user_ids, dates = [], []
    start = pd.Timestamp("2025-11-10")
    for t, k in enumerate(per):
        # each teacher slowly improves; sessions spread over up to 6 months
        offsets = np.sort(rng.integers(0, 197, k))
        for off in offsets:
            user_ids.append(f"user_{t:03d}")
            dates.append(start + pd.Timedelta(days=int(off)))
    df = pd.DataFrame({"user_id": user_ids, "session_date": dates})
    df = df.iloc[:n_rows].reset_index(drop=True)
    df["id"] = [f"sess_{i:04d}" for i in range(len(df))]
    df["session_id"] = df["id"]
    df["audio_duration_seconds"] = rng.integers(300, 1800, len(df))
    df["transcript_length"] = rng.integers(1000, 8000, len(df))
    df["is_coaching_transcript"] = True
    df["evaluation_confidence"] = np.round(rng.uniform(0.6, 0.98, len(df)), 2)

    # teacher baseline + gentle within-teacher upward trend over months
    months = (df["session_date"] - df.groupby("user_id")["session_date"].transform("min")).dt.days / 30.44
    for ind in INDICATORS:
        base = df["user_id"].map({u: rng.uniform(1.5, 3.5) for u in df["user_id"].unique()})
        trend = 0.12 if ind in ("C1", "C2", "D3", "F1") else 0.03
        val = base + trend * months + rng.normal(0, 0.5, len(df))
        df[f"{ind}_score"] = np.clip(val.round(), 1, 4).astype(int)
    for dom, cols in {"B": [c for c in INDICATORS if c.startswith("B")],
                      "C": [c for c in INDICATORS if c.startswith("C")],
                      "D": [c for c in INDICATORS if c.startswith("D")],
                      "F": [c for c in INDICATORS if c.startswith("F")]}.items():
        df[f"{dom}_score"] = df[[f"{c}_score" for c in cols]].mean(axis=1).round(2)
    df["overall_score"] = df[["B_score", "C_score", "D_score", "F_score"]].mean(axis=1).round(2)
    df["session_date"] = df["session_date"].dt.date
    return df


def make_step1_sessions() -> pd.DataFrame:
    counts = {"completed": 567, "abandoned": 104, "failed": 36, "cancelled": 8}
    rows = []
    start = pd.Timestamp("2025-11-01")
    for status, n in counts.items():
        for _ in range(n):
            usable = {"completed": 1.0, "abandoned": 0.93, "failed": 0.28, "cancelled": 0.25}[status]
            rows.append({
                "id": f"cs_{len(rows):04d}",
                "user_id": f"user_{rng.integers(0, 142):03d}",
                "session_date": (start + pd.Timedelta(days=int(rng.integers(0, 210)))).date(),
                "status": status,
                "audio_duration_seconds": int(rng.integers(300, 1800)),
                "transcript_chars": int(rng.integers(300, 8000)) if rng.random() < usable else int(rng.integers(0, 200)),
                "retry_count": int(rng.integers(0, 3)),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    make_fico_scores().to_csv(OUT / "fico_scores.csv", index=False)
    make_step1_sessions().to_csv(OUT / "step1_sessions.csv", index=False)
    print(f"Wrote sample extracts to {OUT}")
    print("Note: session_transcripts.csv is NOT synthesized (LLM-judge steps need real "
          "transcripts + analysis_data). Extract those from the warehouse to run steps 3a/3b/4a.")
