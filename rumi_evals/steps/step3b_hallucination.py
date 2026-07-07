"""Step 3b — Hallucination spot-check.

Framework criterion: zero fabricated transcript citations in any spot-check
sample. A Claude judge extracts every evidential claim from analysis_data and
verifies it against the transcript. RUNNABLE today.

Drift half of 3b (kappa variance across quarterly checks <0.05) accumulates in
results/step3b_history.csv — it needs >=2 quarterly runs before it reports.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from ..config import PACKAGE_ROOT
from ..judge import HALLUCINATION_SYSTEM, HallucinationVerdict, judge_parse

RESULTS_DIR = PACKAGE_ROOT / "results"
HISTORY = RESULTS_DIR / "step3b_history.csv"


def _analysis_summary(analysis_data: str, max_chars: int = 12000) -> str:
    """analysis_data is stored as a JSON string; pass it through compactly."""
    try:
        parsed = json.loads(analysis_data)
        return json.dumps(parsed, ensure_ascii=False)[:max_chars]
    except (json.JSONDecodeError, TypeError):
        return str(analysis_data)[:max_chars]


def run(backend, cfg: dict) -> dict:
    df = backend.fetch("session_transcripts")
    df = df[df["analysis_data"].notna()]
    n = cfg["sampling"]["hallucination_sample_size"]
    sample = df.sample(n=min(n, len(df)), random_state=cfg["sampling"]["random_seed"])

    verdicts = []
    for _, r in sample.iterrows():
        user = (
            f"<transcript>\n{str(r['transcript_text'])[:60000]}\n</transcript>\n\n"
            f"<analysis>\n{_analysis_summary(r['analysis_data'])}\n</analysis>\n\n"
            f"session_id: {r['id']}"
        )
        try:
            v = judge_parse(HALLUCINATION_SYSTEM, user, HallucinationVerdict, cfg)
            verdicts.append(v)
        except Exception as e:  # keep going; report failures
            verdicts.append(None)
            print(f"  judge failed on {r['id']}: {e}")

    ok = [v for v in verdicts if v is not None]
    total_citations = sum(len(v.citations) for v in ok)
    fabricated = sum(v.fabricated_count for v in ok)
    flagged = [
        {"session_id": v.session_id, "fabricated": v.fabricated_count,
         "claims": [c.claim for c in v.citations if not c.found_in_transcript]}
        for v in ok if v.fabricated_count > 0
    ]

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "step3b_verdicts.json", "w", encoding="utf-8") as f:
        json.dump([v.model_dump() for v in ok], f, ensure_ascii=False, indent=2)

    # append to drift history
    row = pd.DataFrame(
        [{"run_date": date.today().isoformat(), "n_sessions": len(ok),
          "total_citations": total_citations, "fabricated": fabricated}]
    )
    if HISTORY.exists():
        pd.concat([pd.read_csv(HISTORY), row]).to_csv(HISTORY, index=False)
    else:
        row.to_csv(HISTORY, index=False)

    threshold = cfg["thresholds"]["step3b_hallucination"]["max_fabricated_citations"]
    return {
        "step": "3b_hallucination",
        "n_sessions_checked": len(ok),
        "n_judge_failures": len(verdicts) - len(ok),
        "total_citations_checked": total_citations,
        "fabricated_citations": fabricated,
        "passes": bool(fabricated <= threshold),
        "flagged_sessions": flagged,
        "caveats": [
            "LLM-judge verification, not the framework's field-coordinator manual review — "
            "have a human spot-check flagged_sessions before treating a failure as real.",
            "Drift tracking needs >=2 quarterly runs in results/step3b_history.csv.",
        ],
    }
