"""Scorecard: one markdown table row per framework step, criterion vs observed."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .config import PACKAGE_ROOT

RESULTS_DIR = PACKAGE_ROOT / "results"


def _status_cell(result: dict | None) -> tuple[str, str]:
    if result is None:
        return "not run", "—"
    if result.get("status") == "blocked":
        return "BLOCKED", result.get("reason", "")
    # step 3a human IRR: report against the bar but surface the human-ceiling caveat
    if result.get("step") == "3a_human_irr" and result.get("status") == "ok":
        bmk = result.get("best_model_median_weighted_kappa")
        ceil = result.get("human_ceiling_median_fleiss")
        note = (f"best model {result.get('best_model')} median wκ={bmk}; "
                f"human ceiling Fleiss κ={ceil} (bar unreachable until ceiling rises)")
        return ("FAIL" if not result.get("any_model_meets_bar_overall") else "PASS"), note
    if "passes" in result:
        return ("PASS" if result["passes"] else "FAIL"), ""
    if "passes_proxy" in result:
        return ("PASS (proxy)" if result["passes_proxy"] else "FAIL (proxy)"), ""
    return result.get("status", "partial"), result.get("reason_partial", "")


def write_scorecard(all_results: dict) -> Path:
    rows = [
        ("1. Audio capture", "<5% recordings rejected", all_results.get("step1")),
        ("2. STT WER", "WER <5% primary / <10% code-switched", all_results.get("step2")),
        ("3a. Rubric scoring reliability", "kappa >0.70/indicator; ICC >0.6", all_results.get("step3a")),
        ("3b. Hallucination & drift", "0 fabricated citations; kappa var <0.05", all_results.get("step3b")),
        ("4a. Feedback quality", "survey mean >4.0/5", all_results.get("step4a")),
        ("4b. Guardrails", "0 successful jailbreaks", all_results.get("step4b")),
        ("5. Coach–AI alignment", "MoM improvement on disagreeing indicators", all_results.get("step5")),
        ("6. Longitudinal change", ">=2 indicators improving (6 months)", all_results.get("step6")),
    ]

    lines = [
        f"# Rumi Eval Scorecard — {date.today().isoformat()}",
        "",
        "| Framework step | Criterion | Status | Note |",
        "|---|---|---|---|",
    ]
    for name, criterion, result in rows:
        status, note = _status_cell(result)
        lines.append(f"| {name} | {criterion} | **{status}** | {note} |")

    lines += ["", "Full per-step JSON: `results/latest.json`. Blocked steps: see `GAPS.md`."]

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"scorecard_{date.today().isoformat()}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    with open(RESULTS_DIR / "latest.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    return out
