"""Step 2 — Speech-to-text WER.

Framework criterion: WER <5% primary instructional language, <10% code-switched.

STATUS: harness only — BLOCKED on gold transcripts. The warehouse has hypothesis
transcripts (coaching_sessions.transcript_text) but no human reference
transcriptions. To activate:

1. Sample sessions (this module writes a stratified sample sheet).
2. Have bilingual annotators transcribe the audio (audio_url) into
   data/gold_transcripts/gold.csv with columns:
       session_id, gold_transcript, language, is_code_switched (0/1)
3. Re-run; WER/CER are computed per language and code-switch stratum.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import PACKAGE_ROOT

GOLD_PATH = PACKAGE_ROOT / "data" / "gold_transcripts" / "gold.csv"
SAMPLE_SHEET = PACKAGE_ROOT / "data" / "gold_transcripts" / "annotation_sample.csv"


def _normalize(text: str) -> str:
    import re

    text = text.lower()
    text = re.sub(r"[^\w\s؀-ۿ]", " ", text)  # keep Arabic-script chars
    return re.sub(r"\s+", " ", text).strip()


def run(backend, cfg: dict) -> dict:
    sessions = backend.fetch("session_transcripts")

    if not GOLD_PATH.exists():
        SAMPLE_SHEET.parent.mkdir(parents=True, exist_ok=True)
        sample = sessions.sample(
            n=min(40, len(sessions)), random_state=cfg["sampling"]["random_seed"]
        )[["id", "user_id", "session_date", "transcript_language"]]
        sample.to_csv(SAMPLE_SHEET, index=False)
        return {
            "step": "2_stt_wer",
            "status": "blocked",
            "reason": f"No gold transcripts at {GOLD_PATH}",
            "action": f"Annotation sample sheet written to {SAMPLE_SHEET} "
            f"({len(sample)} sessions). Have bilingual annotators transcribe these.",
        }

    import jiwer

    gold = pd.read_csv(GOLD_PATH)
    merged = gold.merge(sessions[["id", "transcript_text"]], left_on="session_id", right_on="id")

    rows = []
    for _, r in merged.iterrows():
        ref, hyp = _normalize(str(r["gold_transcript"])), _normalize(str(r["transcript_text"]))
        rows.append(
            {
                "session_id": r["session_id"],
                "language": r.get("language"),
                "is_code_switched": bool(r.get("is_code_switched", 0)),
                "wer": jiwer.wer(ref, hyp),
                "cer": jiwer.cer(ref, hyp),
            }
        )
    res = pd.DataFrame(rows)

    th = cfg["thresholds"]["step2_stt"]
    primary = res[~res["is_code_switched"]]["wer"].mean()
    cs = res[res["is_code_switched"]]["wer"].mean()

    return {
        "step": "2_stt_wer",
        "status": "ok",
        "n_gold": len(res),
        "wer_primary": round(float(primary), 4),
        "wer_codeswitched": round(float(cs), 4) if not pd.isna(cs) else None,
        "by_language": res.groupby("language")["wer"].agg(["mean", "count"]).reset_index().to_dict("records"),
        "passes": bool(primary < th["max_wer_primary"] and (pd.isna(cs) or cs < th["max_wer_codeswitched"])),
        "per_session": res.to_dict("records"),
    }
