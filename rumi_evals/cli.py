"""CLI entry point.

    python -m rumi_evals.cli --steps 1 6          # data-only steps
    python -m rumi_evals.cli --steps 3b 4a        # LLM-judge steps (needs API key)
    python -m rumi_evals.cli --steps all
"""
from __future__ import annotations

import argparse
import json

from .config import load_config
from .data import get_backend
from .report import write_scorecard
from .steps import (
    step1_audio_capture,
    step2_stt_wer,
    step3a_human_irr,
    step3a_reliability,
    step3b_hallucination,
    step4a_feedback_quality,
    step4b_guardrails,
    step5_alignment,
    step6_longitudinal,
)

STEPS = {
    "1": ("step1", step1_audio_capture.run),
    "2": ("step2", step2_stt_wer.run),
    # 3a is now the real human-vs-AI IRR + cross-LLM from the paired study (gap G1).
    "3a": ("step3a", step3a_human_irr.run),
    # The warehouse wobble/re-scoring harness remains available under 3a-warehouse.
    "3a-warehouse": ("step3a_warehouse", step3a_reliability.run),
    "3b": ("step3b", step3b_hallucination.run),
    "4a": ("step4a", step4a_feedback_quality.run),
    "4b": ("step4b", step4b_guardrails.run),
    "5": ("step5", step5_alignment.run),
    "6": ("step6", step6_longitudinal.run),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rumi end-to-end evals")
    parser.add_argument("--steps", nargs="+", default=["all"], choices=[*STEPS, "all"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    backend = get_backend(cfg)

    all_steps = [s for s in STEPS if s != "3a-warehouse"]  # warehouse variant is opt-in
    selected = all_steps if "all" in args.steps else args.steps
    results: dict = {}
    for key in selected:
        name, fn = STEPS[key]
        print(f"\n=== Running step {key} ===")
        try:
            results[name] = fn(backend, cfg)
        except FileNotFoundError as e:
            results[name] = {"status": "blocked", "reason": str(e)}
        print(json.dumps(results[name], indent=2, default=str)[:2000])

    out = write_scorecard(results)
    print(f"\nScorecard written to {out}")


if __name__ == "__main__":
    main()
