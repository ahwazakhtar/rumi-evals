"""Step 1 — Audio capture gate.

Framework metric: recording usability. Criterion: <5% of submitted recordings
rejected at the gate.

Interpretation against available data (validated 2026-07-06):
- 'failed' and 'cancelled' sessions count as gate rejections.
- 'abandoned' is reported separately: it is teacher drop-off mid-flow, not a
  recording-quality rejection, but it is the closest proxy for the framework's
  "teacher retry rate" until an explicit retry event is instrumented.
"""
from __future__ import annotations

import pandas as pd

REJECTED_STATUSES = {"failed", "cancelled"}


def run(backend, cfg: dict) -> dict:
    df = backend.fetch("step1_sessions")
    total = len(df)
    rejected = df["status"].isin(REJECTED_STATUSES).sum()
    abandoned = (df["status"] == "abandoned").sum()

    by_status = (
        df.groupby("status")
        .agg(
            sessions=("id", "count"),
            avg_audio_seconds=("audio_duration_seconds", "mean"),
            usable_transcripts=("transcript_chars", lambda s: (s.fillna(0) > 200).sum()),
        )
        .reset_index()
    )

    monthly = (
        df.assign(month=pd.to_datetime(df["session_date"]).dt.to_period("M").astype(str))
        .groupby("month")
        .agg(
            sessions=("id", "count"),
            rejected=("status", lambda s: s.isin(REJECTED_STATUSES).sum()),
            abandoned=("status", lambda s: (s == "abandoned").sum()),
        )
        .assign(rejection_rate=lambda d: d["rejected"] / d["sessions"])
        .reset_index()
    )

    threshold = cfg["thresholds"]["step1_audio_capture"]["max_rejection_rate"]
    rejection_rate = rejected / total if total else float("nan")

    return {
        "step": "1_audio_capture",
        "n_sessions": int(total),
        "rejection_rate": round(float(rejection_rate), 4),
        "abandonment_rate": round(float(abandoned / total), 4) if total else None,
        "criterion": f"rejection_rate < {threshold}",
        "passes": bool(rejection_rate < threshold),
        "by_status": by_status.to_dict("records"),
        "monthly": monthly.to_dict("records"),
        "caveats": [
            "'abandoned' sessions are excluded from the rejection rate (teacher drop-off, "
            "not a quality-gate rejection); tracked separately.",
            "No explicit rejected-at-upload event exists in the warehouse — sessions rejected "
            "before a coaching_sessions row is created are invisible to this metric.",
        ],
    }
