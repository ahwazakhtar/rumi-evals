"""Presentation-layer service over the ``rumi_evals`` package.

This module does NOT re-implement any eval logic. It:
  * loads the cached full-run results (``results/latest.json`` if present, else the
    committed fallback in ``app/cached/latest.json``);
  * calls the real ``rumi_evals.steps.step3a_human_irr.run`` for a live refresh of
    Step 3a against the study Postgres, falling back to the cache on any error;
  * carries the human-authored step metadata (title / criterion / method / gap
    links) sourced from the framework doc, ``config.yaml`` thresholds and GAPS.md;
  * assembles per-step and roadmap view-models (including pre-rendered SVG charts).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from .charts import BarChart, Ref, Row, model_color

# ---------------------------------------------------------------------------
# Paths — import rumi_evals lazily / defensively so the web app boots even if a
# heavy optional dep is missing.
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
RESULTS_JSON = REPO_ROOT / "results" / "latest.json"
CACHED_JSON = APP_DIR / "cached" / "latest.json"

RESULT_KEYS = {
    "1": "step1", "2": "step2", "3a": "step3a", "3b": "step3b",
    "4a": "step4a", "4b": "step4b", "5": "step5", "6": "step6",
}

# ---------------------------------------------------------------------------
# Step metadata (framework v2 + config.yaml thresholds + GAPS.md). Ordered.
# ---------------------------------------------------------------------------
GROUPS = [
    ("Capture & Transcription", "Turning classroom audio into usable, accurate text.", ["1", "2"]),
    ("Rubric Scoring Reliability", "Can the AI score a lesson like a trained coach — reproducibly and without fabrication?", ["3a", "3b"]),
    ("Feedback & Safety", "Is the feedback good, and does the system stay in scope?", ["4a", "4b"]),
    ("Alignment & Longitudinal Change", "Do AI and coaches converge, and do teachers actually improve?", ["5", "6"]),
]

STEP_META: dict[str, dict] = {
    "1": {
        "num": "1", "title": "Audio capture",
        "criterion": "Fewer than 5% of submitted recordings rejected at the gate; teacher retry rate below 5% after onboarding.",
        "measure": "Automated pre-processing checks — file duration, signal-to-noise estimate, language detection. Manual review flag when checks fail.",
        "reads": "Warehouse RUMI_DB / tbproddb — session statuses & audio durations.",
        "gap": None,
        "next_action": "Investigate the Jan-2026 (9.3%) and Mar-2026 (14.8%) rejection spikes. No explicit 'rejected-at-upload' event exists in the warehouse, so recordings rejected before a coaching_sessions row is created are invisible — instrument the upstream gate to close that blind spot.",
    },
    "2": {
        "num": "2", "title": "Speech-to-text (WER)",
        "criterion": "WER below 5% for the primary instructional language; below 10% for code-switched segments — reported by language and speaker type.",
        "measure": "A stratified sample of recordings manually transcribed by bilingual annotators; WER/CER computed against the gold standard, split by language and speaker.",
        "reads": "Warehouse transcripts (hypothesis) + a not-yet-existing bilingual gold set.",
        "gap": "G2",
        "next_action": "step2_stt_wer.py already writes a stratified annotation sheet on first run. Have bilingual annotators transcribe those recordings into data/gold_transcripts/gold.csv; WER/CER by language and code-switch stratum then compute automatically.",
    },
    "3a": {
        "num": "3a", "title": "Rubric scoring reliability (IRR)",
        "criterion": "Cohen's / weighted kappa above 0.70 on each indicator before deployment (0.75 mature); intra-class kappa above 0.6 across models.",
        "measure": "AI and human coaches score the same recordings independently. Per-indicator weighted kappa of the human consensus vs each model; Fleiss' kappa among the coaches themselves (the 'human ceiling'); cross-model agreement; and the directional AI-minus-human bias.",
        "reads": "LIVE from the paired-scoring study Postgres (300 recordings, 55 coaches, 6 models) via step3a_human_irr.run — cached fallback on any DB error.",
        "gap": "G1",
        "next_action": "Two distinct fixes: (1) Raise the human ceiling first — rater calibration + sharper definitions for the lowest-agreement indicators (SI3, PIA-3/4, PIC-2); 0.70 is not a meaningful target until coaches agree with each other. (2) Correct the AI harshness bias — every model scores systematically below humans; a scoring-prompt/calibration fix independent of (1). Report agreement relative to the human ceiling, not a fixed 0.70. Also finish the mid-run model scorings (deepseek 75%, kimi/mistral/nemotron <15%).",
    },
    "3b": {
        "num": "3b", "title": "Hallucination & drift",
        "criterion": "Zero fabricated transcript citations in any spot-check sample; kappa variance across quarterly checks within 0.05 points.",
        "measure": "A random sample of AI rationales verified against the source transcript (LLM judge / field coordinator). Longitudinal kappa tracking to detect scoring drift.",
        "reads": "Warehouse session transcripts + analysis_data (LLM-judge step).",
        "gap": None,
        "next_action": "Runnable today once the extract exists: run sql/session_transcripts.sql against BigQuery and save it to data/extracts/session_transcripts.csv (or switch data.backend to bigquery). The LLM judge then verifies each cited piece of evidence against the transcript.",
    },
    "4a": {
        "num": "4a", "title": "Feedback quality",
        "criterion": "Mean score above 4.0 / 5.0 on specificity, actionability and tone; senior-coach review flags no systemic tone failures.",
        "measure": "A structured teacher survey after each feedback cycle, plus qualitative review of a subsample by a senior coach.",
        "reads": "Warehouse coaching_quality_metrics satisfaction columns (currently empty) — LLM-judge proxy available.",
        "gap": "G3",
        "next_action": "coaching_quality_metrics.user_satisfaction_rating / user_feedback exist but are 100% NULL — nothing writes to them. Wire a lightweight post-session WhatsApp rating prompt into those columns. Until then Step 4a runs in a labelled PROXY mode (an LLM coach-judge scores the feedback text on the same three dimensions — a quality proxy, not a receipt-of-feedback measure).",
    },
    "4b": {
        "num": "4b", "title": "Guardrails & jailbreak resistance",
        "criterion": "Zero successful jailbreaks producing out-of-scope output in red-team testing; no confirmed data cross-contamination in production.",
        "measure": "Red-team testing across attack categories (scope escape, harmful content, data exposure, prompt extraction) plus monthly review of multi-layer content-check logs.",
        "reads": "The live Rumi WhatsApp pipeline (its exact prompt + model + content checks) — not the warehouse.",
        "gap": "G4",
        "next_action": "step4b_guardrails.run_against(respond, cfg) takes any respond(text)->str callable. Point it at a Rumi staging webhook or the coaching service directly. The seed attack set (prompts/redteam_attacks.yaml, 12 attacks incl. code-switched phrasings) and the LLM grader are ready.",
    },
    "5": {
        "num": "5", "title": "Coach–AI alignment",
        "criterion": "Month-on-month improvement on the indicators that showed disagreement in the prior cycle.",
        "measure": "A monthly per-indicator agreement rate, the direction of disagreement, and qualitative notes on disputed cases.",
        "reads": "Warehouse AI scores by month (AI-side drift) — human pairs blocked in the warehouse (see G1).",
        "gap": "G1",
        "next_action": "No human-scored Rumi sessions exist in the warehouse to align against (0 shared audio_url between AI sessions and human observations). The paired study (G1) provides pairs off-warehouse; a full month-on-month alignment report needs those pairs promoted into the warehouse. Until then this reports the AI-side monthly scoring profile / drift only.",
    },
    "6": {
        "num": "6", "title": "Longitudinal instructional change",
        "criterion": "Statistically significant improvement on at least two rubric indicators after six months; effect size documented with confidence intervals.",
        "measure": "Individual teacher trend lines across repeated observations (within-teacher slope, mixed over 142 teachers), with p-values and 6-month effect sizes.",
        "reads": "Warehouse coaching_sessions_with_fico_scores over a ~6.5-month window.",
        "gap": "G5",
        "next_action": "The within-teacher trend half runs today and passes. The student-outcome-correlation half is blocked: no Rumi-linked student assessment data and no treatment/control structure exist. Unblocking needs M&E instrumentation linking users.emis_code to student assessment records — out of scope for an AI-eval workstream.",
    },
}

# ---------------------------------------------------------------------------
# Gap register (GAPS.md, restated as structured data).
# ---------------------------------------------------------------------------
GAPS = [
    {
        "id": "G1", "title": "Human-vs-AI IRR", "status": "resolved",
        "steps": ["3a", "5"],
        "wants": "Cohen's kappa >0.70 per indicator between the AI and a trained human coach on the same lesson.",
        "reality": "The human FICO observations and the AI sessions share ZERO audio_url values in the warehouse — different programs, different teachers. No human-AI pairing exists in the warehouse.",
        "resolution": "Resolved by the paired scoring study (Railway Postgres): 300 recordings, 55 coaches (up to 3 per recording), scored against the same rubric that 6 AI models also scored. step3a_human_irr computes real per-indicator kappa. CRITICAL caveat: the human ceiling is only median Fleiss kappa 0.068 — coaches barely agree with each other, so the 0.70 bar is currently unreachable by construction. Two problems surface: raise the ceiling (rater calibration) AND fix AI harshness bias (every model scores below humans).",
    },
    {
        "id": "G2", "title": "No gold transcripts", "status": "blocked",
        "steps": ["2"],
        "wants": "WER <5% primary language, <10% code-switched, by language and speaker.",
        "reality": "coaching_sessions.transcript_text is the hypothesis; there is no human-verified reference transcription anywhere, and no speaker-type labels.",
        "resolution": "step2_stt_wer.py writes a stratified annotation sample sheet on first run. Bilingual annotators transcribe those recordings into data/gold_transcripts/gold.csv; WER/CER by language and code-switch stratum compute automatically. (diarization_data exists but its speaker-ID accuracy is itself unvalidated — a second gold-labeling task.)",
    },
    {
        "id": "G3", "title": "Satisfaction instrumentation empty", "status": "blocked",
        "steps": ["4a"],
        "wants": "Teacher survey mean >4.0/5 on specificity, actionability, tone.",
        "reality": "coaching_quality_metrics.user_satisfaction_rating and user_feedback exist as columns but are 100% NULL — nothing writes to them. There is no captured teacher sentiment anywhere in the warehouse.",
        "resolution": "Wire a lightweight post-session WhatsApp rating prompt into those columns. Until then Step 4a runs in labelled PROXY mode (an LLM coach-judge scores the feedback text itself — a quality proxy, not a receipt-of-feedback measure).",
    },
    {
        "id": "G4", "title": "No endpoint to red-team", "status": "blocked",
        "steps": ["4b"],
        "wants": "Zero successful jailbreaks in red-team testing against the system.",
        "reality": "Guardrail testing must hit the live Rumi WhatsApp pipeline (its exact prompt + model + multi-layer content checks). This repo reads the warehouse; it has no access to that runtime.",
        "resolution": "step4b_guardrails.run_against(respond, cfg) takes any respond(text)->str callable. Point it at a Rumi staging webhook or the coaching service directly. The seed attack set (12 attacks across scope escape / harmful content / data exposure / prompt extraction, incl. code-switched phrasings) and the LLM grader are ready.",
    },
    {
        "id": "G5", "title": "No Rumi-linked student outcomes", "status": "blocked",
        "steps": ["6"],
        "wants": "Where student data exists, correlate teacher improvement with student learning gains.",
        "reality": "Student assessment tables exist but are not linked to Rumi usage, and the closest teacher-outcome regression table has no coaching-session linkage. No treatment/control structure exists anywhere.",
        "resolution": "Out of scope for an AI-eval workstream — needs M&E instrumentation linking users.emis_code to student assessment records. The within-teacher trend half of Step 6 runs today without it.",
    },
]

DATA_SOURCES = [
    {
        "name": "Warehouse — RUMI_DB / tbproddb",
        "backend": "BigQuery (via data.py; csv extracts by default)",
        "role": "The production data warehouse. AI coaching sessions, LLM-scored FICO rubric output, human ICT observations, quality/cost metrics.",
        "feeds": ["1", "2", "3b", "4a", "5", "6"],
        "notes": [
            "coaching_sessions_with_fico_scores: 675 scored sessions, 30 indicators, 100% populated (B/C/D/F rubric).",
            "142 distinct teachers with scored sessions; 2025-11-10 → 2026-05-26 (~6.5 months).",
            "tbproddb.coaching_observation: 8,451 clean human FICO observations — but 0 shared audio_url with Rumi AI sessions (the reason G1 needed a separate study).",
        ],
    },
    {
        "name": "Paired scoring study — Railway Postgres",
        "backend": "Postgres (via study_data.py; RUMI_STUDY_PG_URL)",
        "role": "The purpose-built study that unblocks Step 3a. 300 recordings, up to 3 coaches each, scored against the same observation rubric that 6 AI models also scored. study_compiled pairs human consensus vs each model.",
        "feeds": ["3a"],
        "notes": [
            "Uses the real human observation rubric (yes/partial/no on SI/PIA/PIC/MA + subject Section C: L*/M*/S*), NOT the warehouse B/C/D/F FICO columns — different rubrics.",
            "This is the ONLY live-read source in the dashboard; Step 3a refreshes from it on demand.",
            "gpt-5.1 (101) and minimax (100) fully scored; deepseek 75; kimi/mistral/nemotron mid-run.",
        ],
    },
    {
        "name": "Digital Coach external API",
        "backend": "HTTPS (dc_api.py; DC_API_KEY)",
        "role": "Fresh re-scoring for the Step 3a wobble/self-consistency test and the Step 4b guardrail red-team harness.",
        "feeds": ["3a", "4b"],
        "notes": [
            "base_url https://digitalcoach.taleemabad.com; reproduces the gpt-5.1 study run for wobble testing.",
        ],
    },
]

STATUS_ORDER = {"pass": 0, "partial": 1, "fail": 2, "blocked": 3}
STATUS_LABEL = {"pass": "PASS", "partial": "PARTIAL", "fail": "FAIL", "blocked": "BLOCKED"}


# ---------------------------------------------------------------------------
# Status + headline derivation (per step, from the result dict).
# ---------------------------------------------------------------------------
def _derive_status(step_id: str, res: dict | None) -> str:
    if not res:
        return "blocked"
    st = res.get("status")
    if st == "blocked":
        return "blocked"
    if st == "partial":
        return "partial"
    if step_id == "3a":
        return "pass" if res.get("any_model_meets_bar_overall") else "fail"
    if "passes" in res:
        return "pass" if res["passes"] else "fail"
    if "passes_proxy" in res:
        return "pass" if res["passes_proxy"] else "fail"
    return "partial"


def _headline(step_id: str, res: dict | None, status: str) -> str:
    if not res:
        return "No cached result available."
    if step_id == "1":
        return (f"Rejection rate {res['rejection_rate'] * 100:.1f}% vs <5% bar "
                f"({res['n_sessions']} sessions; abandonment {res['abandonment_rate'] * 100:.1f}%).")
    if step_id == "2":
        return "Blocked — no human-verified gold transcripts exist to compute WER against (G2)."
    if step_id == "3a":
        return (f"Best model {res['best_model']} median weighted κ = {res['best_model_median_weighted_kappa']} "
                f"vs a 0.70 bar — but the human ceiling is only Fleiss κ = {res['human_ceiling_median_fleiss']}, "
                f"so the bar is unreachable by construction.")
    if step_id == "3b":
        return "Blocked — needs a session_transcripts.csv extract to run the hallucination judge."
    if step_id == "4a":
        return "Survey blocked — satisfaction columns are 100% NULL (G3); LLM-judge proxy available."
    if step_id == "4b":
        return "Blocked — needs a live Rumi endpoint to red-team (G4); 12-attack seed set is ready."
    if step_id == "5":
        return (f"Partial — {res.get('human_pairs_available', 0)} human pairs in the warehouse; "
                f"reporting AI-side monthly drift only.")
    if step_id == "6":
        return (f"PASS — {res['significant_improving_indicators']} indicators improving at p<0.05 "
                f"over 6 months ({res['n_teachers_with_3plus_sessions']} teachers).")
    return STATUS_LABEL.get(status, status)


# ---------------------------------------------------------------------------
# The service.
# ---------------------------------------------------------------------------
class EvalService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._results: dict = {}
        self._source_label = "unknown"
        self._live_3a = False        # True once a successful live refresh happened
        self._live_3a_error: str | None = None
        self.reload_cache()

    # ---- cache loading ----------------------------------------------------
    def reload_cache(self) -> None:
        path, label = (RESULTS_JSON, "results/latest.json") if RESULTS_JSON.exists() else (CACHED_JSON, "app/cached/latest.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._results = json.load(f)
            self._source_label = f"cached full-run ({label})"
        except Exception as exc:  # pragma: no cover - defensive
            self._results = {}
            self._source_label = f"cache unavailable ({exc})"

    # ---- live Step-3a refresh (never blocks startup) ----------------------
    def refresh_step3a(self, timeout_ok: bool = True) -> dict:
        """Run the real study-backed Step 3a. On any failure keep the cache."""
        with self._lock:
            try:
                from rumi_evals.config import load_config
                from rumi_evals.study_data import available
                from rumi_evals.steps import step3a_human_irr

                if not available():
                    self._live_3a_error = ("Study DB unreachable — RUMI_STUDY_PG_URL is unset "
                                           "or psycopg is not installed. Showing cached results.")
                    return {"ok": False, "error": self._live_3a_error}

                cfg = load_config()
                res = step3a_human_irr.run(None, cfg)
                if res.get("status") != "ok":
                    self._live_3a_error = f"Study step returned status={res.get('status')}: {res.get('reason', '')}"
                    return {"ok": False, "error": self._live_3a_error}
                self._results["step3a"] = res
                self._live_3a = True
                self._live_3a_error = None
                return {"ok": True}
            except Exception as exc:
                self._live_3a_error = f"Study DB unreachable — showing cached results/latest.json ({exc})."
                return {"ok": False, "error": self._live_3a_error}

    def try_live_refresh_on_start(self) -> None:
        """Best-effort: attempt one live 3a refresh; swallow all errors."""
        try:
            self.refresh_step3a()
        except Exception:
            pass

    # ---- accessors --------------------------------------------------------
    @property
    def source_label(self) -> str:
        return self._source_label

    def result(self, step_id: str) -> dict | None:
        return self._results.get(RESULT_KEYS[step_id])

    def step_card(self, step_id: str) -> dict:
        res = self.result(step_id)
        status = _derive_status(step_id, res)
        meta = STEP_META[step_id]
        return {
            "id": step_id,
            "num": meta["num"],
            "title": meta["title"],
            "status": status,
            "status_label": STATUS_LABEL[status],
            "headline": _headline(step_id, res, status),
            "gap": meta["gap"],
        }

    def roadmap(self) -> dict:
        groups = []
        counts = {"pass": 0, "fail": 0, "blocked": 0, "partial": 0}
        for title, subtitle, ids in GROUPS:
            cards = [self.step_card(i) for i in ids]
            for c in cards:
                counts[c["status"]] += 1
            groups.append({"title": title, "subtitle": subtitle, "cards": cards})
        return {
            "groups": groups,
            "counts": counts,
            "total": sum(counts.values()),
            "source_label": self._source_label,
            "live_3a": self._live_3a,
            "live_3a_error": self._live_3a_error,
        }

    # ---- per-step view models --------------------------------------------
    def step_view(self, step_id: str) -> dict | None:
        if step_id not in STEP_META:
            return None
        res = self.result(step_id)
        meta = STEP_META[step_id]
        status = _derive_status(step_id, res)
        gap = next((g for g in GAPS if g["id"] == meta["gap"]), None) if meta["gap"] else None

        view = {
            "id": step_id,
            "meta": meta,
            "status": status,
            "status_label": STATUS_LABEL[status],
            "headline": _headline(step_id, res, status),
            "gap": gap,
            "stats": [],
            "tables": [],
            "charts": [],
            "caveats": (res or {}).get("caveats", []),
            "interpretation": None,
            "source_label": self._source_label,
            "is_3a": step_id == "3a",
            "live_3a": self._live_3a,
            "live_3a_error": self._live_3a_error,
        }
        dispatch = {
            "1": self._build_1, "2": self._build_blocked, "3a": self._build_3a,
            "3b": self._build_blocked, "4a": self._build_blocked, "4b": self._build_blocked,
            "5": self._build_5, "6": self._build_6,
        }
        dispatch[step_id](view, res)
        return view

    # -- step 1 -------------------------------------------------------------
    def _build_1(self, view: dict, res: dict | None) -> None:
        if not res:
            return
        view["stats"] = [
            {"label": "Rejection rate", "value": f"{res['rejection_rate'] * 100:.1f}%", "sub": "bar: <5%", "tone": "critical"},
            {"label": "Abandonment rate", "value": f"{res['abandonment_rate'] * 100:.1f}%", "sub": "tracked separately", "tone": "muted"},
            {"label": "Sessions", "value": f"{res['n_sessions']:,}", "sub": "in window", "tone": "muted"},
        ]
        monthly = res.get("monthly", [])
        rows = [
            Row(m["month"], m["rejection_rate"] * 100,
                color="var(--critical)" if m["rejection_rate"] > 0.05 else "var(--good)",
                display=f"{m['rejection_rate'] * 100:.1f}%")
            for m in monthly
        ]
        chart = BarChart(rows, vmin=0, vmax=16, decimals=0, unit="%",
                         refs=[Ref(5, "5% bar", "var(--critical)")],
                         label_w=70, chart_w=520)
        view["charts"].append({
            "title": "Monthly rejection rate vs the 5% gate",
            "svg": chart.svg(),
            "caption": "Red bars breach the 5% bar. Spikes in Jan (9.3%) and Mar (14.8%) drive the overall miss.",
        })
        view["tables"].append({
            "title": "Sessions by status",
            "columns": ["Status", "Sessions", "Avg audio (s)", "Usable transcripts"],
            "rows": [[b["status"], f"{b['sessions']:,}", f"{b['avg_audio_seconds']:.0f}", f"{b['usable_transcripts']:,}"]
                     for b in res.get("by_status", [])],
            "note": None,
        })

    # -- generic blocked steps (2, 3b, 4a, 4b) ------------------------------
    def _build_blocked(self, view: dict, res: dict | None) -> None:
        reason = (res or {}).get("reason")
        if reason:
            view["stats"] = [{"label": "Status", "value": view["status_label"],
                              "sub": "not yet measurable", "tone": "muted"}]
            view["tables"].append({
                "title": "Why this can't run yet",
                "columns": ["Blocker"],
                "rows": [[reason]],
                "note": None,
            })

    # -- step 3a (the headline) --------------------------------------------
    def _build_3a(self, view: dict, res: dict | None) -> None:
        if not res or res.get("status") != "ok":
            self._build_blocked(view, res)
            return
        view["interpretation"] = res.get("interpretation")
        bias = res.get("key_finding_directional_bias", {}).get("mean_ai_minus_human_all_models")
        rollup = res.get("model_rollup", [])
        cross = res.get("cross_llm_agreement", {})
        ceiling = res.get("human_ceiling_per_indicator", [])

        view["stats"] = [
            {"label": "Human ceiling (Fleiss κ)", "value": f"{res['human_ceiling_median_fleiss']:.3f}",
             "sub": "coaches vs each other — near chance", "tone": "critical"},
            {"label": "Human pairwise exact", "value": f"{res['human_ceiling_median_pairwise_exact']:.2f}",
             "sub": "median across indicators", "tone": "muted"},
            {"label": f"Best model ({res['best_model']}) wκ", "value": f"{res['best_model_median_weighted_kappa']:.3f}",
             "sub": "vs 0.70 deploy bar", "tone": "warning"},
            {"label": "Indicators clearing 0.70", "value": "0 / 21",
             "sub": "across all models", "tone": "critical"},
            {"label": "Cross-LLM median wκ", "value": f"{cross.get('median_pairwise_weighted_kappa')}",
             "sub": "models agree more with each other", "tone": "muted"},
            {"label": "Mean AI − human", "value": f"{bias}",
             "sub": "AI scores systematically harsher", "tone": "critical"},
        ]

        # Chart A — median weighted kappa per model vs ceiling & deploy bar
        rows_a = [Row(m["model"].split("/")[-1], m["median_weighted_kappa"], color=model_color(m["model"]),
                      display=f"{m['median_weighted_kappa']:.3f}",
                      sub=f"{m['ai_scored_recordings']} recs" + (" · low cov" if m["low_coverage"] else ""))
                  for m in rollup]
        chart_a = BarChart(rows_a, vmin=0, vmax=0.75, decimals=2,
                           refs=[Ref(res["human_ceiling_median_fleiss"], "human ceiling", "var(--warning)"),
                                 Ref(0.70, "0.70 deploy bar", "var(--critical)")],
                           label_w=140, chart_w=520)
        view["charts"].append({
            "title": "Median weighted κ per model — against the human ceiling and the 0.70 bar",
            "svg": chart_a.svg(),
            "caption": "Every model sits far below 0.70 — but also barely above the human ceiling (0.068). No scorer can agree with a human consensus better than the humans agree among themselves.",
        })

        # Chart B — directional bias (AI - human), diverging
        rows_b = [Row(m["model"].split("/")[-1], m["mean_ai_minus_human"],
                      color="var(--critical)" if m["mean_ai_minus_human"] < 0 else "var(--good)",
                      display=f"{m['mean_ai_minus_human']:+.2f}")
                  for m in rollup]
        chart_b = BarChart(rows_b, vmin=-1.0, vmax=0.2, decimals=1, diverging=True,
                           label_w=140, chart_w=520)
        view["charts"].append({
            "title": "Directional bias — mean (AI − human) on the ordinal scale",
            "svg": chart_b.svg(),
            "caption": "All negative: every model scores HARSHER than human coaches (0 = no bias; −1 ≈ a full category harsher). A scoring-prompt calibration issue, separate from the ceiling problem.",
        })

        # Chart C — per-indicator human ceiling
        rows_c = [Row(c["indicator"], c["human_fleiss_kappa"],
                      color="var(--critical)" if c["human_fleiss_kappa"] < 0.05 else "var(--warning)",
                      display=f"{c['human_fleiss_kappa']:.2f}")
                  for c in ceiling]
        chart_c = BarChart(rows_c, vmin=-0.1, vmax=0.4, decimals=2, diverging=True,
                           refs=[Ref(0.0, "chance", "var(--muted)")],
                           label_w=70, chart_w=520, bar_h=15, gap=6)
        view["charts"].append({
            "title": "Human ceiling by indicator (inter-coach Fleiss κ)",
            "svg": chart_c.svg(),
            "caption": "The reliability ceiling for each indicator. SI3, PIA-3/4 and L1 sit at or below chance — those definitions need sharpening most. Best-aligned: S2 (0.33), SI2, PIA-2.",
        })

        # Table — model rollup
        view["tables"].append({
            "title": "Per-model rollup",
            "columns": ["Model", "AI recs", "Coverage", "Indicators", "Median wκ", "Median κ", "≥0.70", "Mean AI−human"],
            "rows": [[m["model"], f"{m['ai_scored_recordings']}",
                      "low" if m["low_coverage"] else "full", f"{m['n_indicators']}",
                      f"{m['median_weighted_kappa']:.3f}", f"{m['median_kappa']:.3f}",
                      f"{m['indicators_meeting_070']}", f"{m['mean_ai_minus_human']:+.3f}"]
                     for m in rollup],
            "note": "Weighted κ = quadratic-weighted Cohen's kappa, human consensus vs model, median across indicators.",
        })

        # Table — per-indicator ceiling + each model's weighted kappa
        by_model = self._load_human_vs_ai()
        models = [m["model"] for m in rollup]
        cols = ["Indicator", "Human Fleiss κ", "Human exact"] + [m.split("/")[-1] + " wκ" for m in models]
        prows = []
        for c in ceiling:
            ind = c["indicator"]
            row = [ind, f"{c['human_fleiss_kappa']:.3f}", f"{c['human_pairwise_exact']:.2f}"]
            for m in models:
                wk = by_model.get((m, ind))
                row.append(f"{wk:.3f}" if wk is not None else "—")
            prows.append(row)
        view["tables"].append({
            "title": "Per-indicator: human ceiling vs each model (weighted κ)",
            "columns": cols,
            "rows": prows,
            "note": "Read a model's number against the Human Fleiss κ in the same row — not against 0.70. Source: results/step3a_human_vs_ai.csv.",
        })

        # Table — cross-LLM pairs
        view["tables"].append({
            "title": "Cross-LLM agreement (models scoring the same recordings)",
            "columns": ["Model A", "Model B", "n", "κ", "Weighted κ", "Exact"],
            "rows": [[p["model_a"], p["model_b"], f"{p['n']:,}", f"{p['kappa']:.3f}",
                      f"{p['weighted_kappa']:.3f}", f"{p['exact_agreement']:.3f}"]
                     for p in cross.get("pairs", [])],
            "note": f"Median pairwise weighted κ = {cross.get('median_pairwise_weighted_kappa')} — models agree with each other more than with humans, consistent with a shared LLM harshness bias.",
        })

    def _load_human_vs_ai(self) -> dict:
        """(model, indicator) -> weighted_kappa, from the study CSV (results or cached)."""
        import csv
        path = REPO_ROOT / "results" / "step3a_human_vs_ai.csv"
        if not path.exists():
            path = APP_DIR / "cached" / "step3a_human_vs_ai.csv"
        out: dict = {}
        try:
            with open(path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    try:
                        out[(r["model"], r["indicator"])] = float(r["weighted_kappa"])
                    except (ValueError, KeyError):
                        continue
        except OSError:
            pass
        return out

    # -- step 5 -------------------------------------------------------------
    def _build_5(self, view: dict, res: dict | None) -> None:
        if not res:
            return
        view["interpretation"] = res.get("reason_partial")
        monthly = res.get("monthly_overall_mean", {})
        view["stats"] = [
            {"label": "Human pairs in warehouse", "value": f"{res.get('human_pairs_available', 0)}",
             "sub": "blocks full alignment (G1)", "tone": "critical"},
            {"label": "Months covered", "value": f"{len(monthly)}", "sub": "AI-side drift", "tone": "muted"},
        ]
        rows = [Row(mon, val, color="var(--series-1)", display=f"{val:.3f}")
                for mon, val in monthly.items()]
        if rows:
            vals = [r.value for r in rows]
            chart = BarChart(rows, vmin=min(vals) - 0.05, vmax=max(vals) + 0.05, decimals=2,
                             label_w=70, chart_w=520)
            view["charts"].append({
                "title": "AI-side monthly overall mean score (drift view)",
                "svg": chart.svg(),
                "caption": "A gentle upward drift in the AI's own scoring over the window (2.52 → 2.67). Cross-check before reading this as teacher change.",
            })
        view["tables"].append({
            "title": "Largest indicator shifts (last month)",
            "columns": ["Indicator", "Abs shift"],
            "rows": [[s["indicator"], f"{s['abs_shift_last_month']:.3f}"]
                     for s in res.get("largest_indicator_shifts_last_month", [])],
            "note": None,
        })

    # -- step 6 -------------------------------------------------------------
    def _build_6(self, view: dict, res: dict | None) -> None:
        if not res:
            return
        overall = res.get("overall", {})
        view["stats"] = [
            {"label": "Improving indicators", "value": f"{res['significant_improving_indicators']}",
             "sub": "bar: ≥2 at p<0.05", "tone": "good"},
            {"label": "Teachers", "value": f"{res['n_teachers_with_3plus_sessions']}",
             "sub": f"{res['n_observations_used']} observations", "tone": "muted"},
            {"label": "Overall effect (6mo SD)", "value": f"{overall.get('effect_size_6mo_sd', 0):.2f}",
             "sub": f"slope {overall.get('slope_per_month', 0):.3f}/mo", "tone": "good"},
        ]
        improving = sorted(res.get("improving", []), key=lambda x: -x["effect_size_6mo_sd"])[:12]
        rows = [Row(i["indicator"], i["effect_size_6mo_sd"], color="var(--series-2)",
                    display=f"{i['effect_size_6mo_sd']:.2f}")
                for i in improving]
        chart = BarChart(rows, vmin=0, vmax=1.0, decimals=1, label_w=70, chart_w=520, bar_h=17, gap=7)
        view["charts"].append({
            "title": "Top improving indicators — 6-month effect size (within-teacher SD)",
            "svg": chart.svg(),
            "caption": "F1, C1, C2, D3 show the largest within-teacher gains (~0.8–0.95 SD over six months). All 24 significant indicators are in the table below.",
        })
        view["tables"].append({
            "title": "Significant improving indicators (p<0.05)",
            "columns": ["Indicator", "Slope/mo", "95% CI", "p", "Effect (6mo SD)"],
            "rows": [[i["indicator"], f"{i['slope_per_month']:.4f}",
                      f"[{i['ci_low']:.4f}, {i['ci_high']:.4f}]", f"{i['p_value']:.4f}",
                      f"{i['effect_size_6mo_sd']:.3f}"]
                     for i in sorted(res.get("improving", []), key=lambda x: -x["effect_size_6mo_sd"])],
            "note": "Within-teacher OLS slope per indicator; no control group — descriptive, not causal.",
        })

    # ---- gaps / data page models -----------------------------------------
    def gaps(self) -> list[dict]:
        out = []
        for g in GAPS:
            out.append({**g, "step_titles": [(s, STEP_META[s]["title"]) for s in g["steps"]]})
        return out

    def data_sources(self) -> list[dict]:
        out = []
        for s in DATA_SOURCES:
            out.append({**s, "feeds_titles": [(i, STEP_META[i]["title"]) for i in s["feeds"]]})
        return out


service = EvalService()
