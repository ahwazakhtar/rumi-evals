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
        "measure": "Automated pre-processing checks — recording length, audio clarity, and language detection — with a manual review flag when checks fail.",
        "reads": "Programme records of submitted coaching sessions and their audio.",
        "gap": None,
        "caveats": [
            "Abandoned sessions (teachers who drop off mid-flow) are excluded from the rejection rate and tracked separately.",
            "Recordings rejected before a session record is created aren't captured by this metric — the true rejection rate could be somewhat higher.",
        ],
        "next_action": "Investigate the two rejection spikes — January (about 9%) and March (about 15%). Recordings rejected before a session record is created aren't yet captured, so add tracking at the upload step to close that blind spot.",
    },
    "2": {
        "num": "2", "title": "Speech-to-text accuracy",
        "criterion": "Word-error rate below 5% for the primary instructional language; below 10% for mixed-language (code-switched) speech — reported by language and speaker.",
        "measure": "A representative sample of recordings transcribed by hand by bilingual annotators to serve as a reference, then compared against the automatic transcripts by language and speaker.",
        "reads": "Automatic lesson transcripts, plus a planned set of hand-verified reference transcripts.",
        "gap": "G2",
        "blocked_reason": "This requires hand-verified reference transcripts to measure the automatic transcripts against. Those haven't been collected yet.",
        "next_action": "Commission a bilingual annotation pass: a representative sample of lessons transcribed by hand as the reference. Accuracy by language and by mixed-language speech then reports automatically.",
    },
    "3a": {
        "num": "3a", "title": "Rubric scoring reliability",
        "criterion": "Agreement (weighted kappa) above 0.70 on each indicator before deployment (0.75 at maturity); consistency above 0.6 across AI models.",
        "measure": "AI and human coaches score the same lessons independently. We report per-indicator agreement between the coaches' consensus and each AI model; how well the coaches agree with one another (the 'human ceiling'); agreement between AI models; and whether the AI scores systematically higher or lower than coaches.",
        "reads": "The paired scoring study: 300 lessons independently scored by 55 trained coaches and by 6 AI models. Read live, refreshed on demand.",
        "gap": "G1",
        "next_action": "Two distinct fixes. (1) Raise the human ceiling first — coach calibration and sharper definitions for the lowest-agreement indicators; a 0.70 target isn't meaningful until coaches agree with one another. (2) Correct the AI's harshness bias — every model scores systematically below coaches, a scoring-calibration fix independent of (1). Report agreement relative to the human ceiling rather than a fixed 0.70. Also complete the partially-scored AI models.",
    },
    "3b": {
        "num": "3b", "title": "Fabrication & drift",
        "criterion": "Zero fabricated evidence in any review sample; scoring behaviour stable over time (agreement varying by less than 0.05 between quarterly checks).",
        "measure": "A random sample of the AI's written rationales checked against the source transcript, so we can confirm every quoted piece of evidence really appears in the lesson. Agreement tracked over time to detect any drift.",
        "reads": "Completed lessons — the transcript plus the AI's written analysis of each.",
        "gap": None,
        "blocked_reason": "This needs a batch of completed lessons (each with its transcript and the AI's analysis) to be made available for review.",
        "next_action": "Runs as soon as a batch of completed lessons is made available. An automated check then verifies every piece of cited evidence against the transcript, flagging anything unsupported for a human to spot-check.",
    },
    "4a": {
        "num": "4a", "title": "Feedback quality",
        "criterion": "Teacher satisfaction above 4.0 out of 5.0 on specificity, actionability and tone; senior-coach review flags no systemic tone problems.",
        "measure": "A short teacher survey after each feedback cycle, plus a senior coach reviewing a subsample.",
        "reads": "Post-session feedback records; teacher satisfaction ratings once they are collected.",
        "gap": "G3",
        "blocked_reason": "This needs teacher satisfaction ratings, which aren't being collected yet.",
        "next_action": "Add a short post-session rating prompt in the teacher's chat. Until then, an automated coach-style review scores the feedback's specificity, actionability and tone as a proxy — a measure of feedback quality, not of how teachers actually received it.",
    },
    "4b": {
        "num": "4b", "title": "Safety & guardrails",
        "criterion": "Zero successful attempts to push the assistant out of its intended scope in red-team testing; no confirmed cases of one teacher's data reaching another.",
        "measure": "Structured red-team testing across attack types (pushing out of scope, eliciting harmful content, exposing other users' data, extracting internal instructions), plus monthly review of the content-safety logs.",
        "reads": "A test version of the live coaching assistant.",
        "gap": "G4",
        "blocked_reason": "This requires access to a test version of the live coaching assistant to safely run the attacks against.",
        "next_action": "Connect the ready-made test harness to a test version of the assistant. The attack set (12 scenarios spanning out-of-scope requests, harmful content, data exposure and instruction extraction, including mixed-language phrasing) and the automated grader are prepared.",
    },
    "5": {
        "num": "5", "title": "Coach–AI alignment over time",
        "criterion": "Month-on-month improvement on the indicators where AI and coaches disagreed in the prior cycle.",
        "measure": "A monthly per-indicator agreement rate, the direction of any disagreement, and notes on disputed cases.",
        "reads": "The AI's monthly scores; paired coach scores once available across the programme.",
        "gap": "G1",
        "next_action": "Within the programme's own records no lesson is yet scored by both a coach and the AI, so month-over-month alignment can't be produced there. The paired study provides that pairing separately; a full alignment report needs those paired scores brought into the programme's records. Until then this shows the AI's own month-to-month scoring drift.",
    },
    "6": {
        "num": "6", "title": "Teacher improvement over time",
        "criterion": "Statistically significant improvement on at least two rubric indicators after six months, with effect sizes and confidence intervals.",
        "measure": "Each teacher's trend across their repeated lessons (within-teacher change, pooled across 142 teachers), with significance tests and six-month effect sizes.",
        "reads": "Repeated AI scores per teacher over roughly six months.",
        "gap": "G5",
        "caveats": [
            "This is a within-teacher trend with no comparison group — it describes improvement, it does not prove the coaching caused it.",
            "The scores come from the AI itself, so a drift in the AI's scoring could look like teacher change (cross-checked against Step 5).",
            "Linking teacher improvement to student learning gains is not yet possible (see the gap register).",
        ],
        "next_action": "The teacher-improvement trend runs today and passes. Linking it to student learning gains isn't yet possible — student assessment records aren't connected to coaching activity, and there is no comparison group. That linkage is a monitoring-and-evaluation task beyond the AI-evaluation scope.",
    },
}

# ---------------------------------------------------------------------------
# Gap register (GAPS.md, restated as structured data).
# ---------------------------------------------------------------------------
GAPS = [
    {
        "id": "G1", "title": "Human-vs-AI agreement", "status": "resolved",
        "steps": ["3a", "5"],
        "wants": "Agreement (kappa above 0.70 per indicator) between the AI's scores and a trained coach's scores on the same lesson.",
        "reality": "Within the programme's own records, no lesson had been scored by both a human coach and the AI — the existing human observations and the AI sessions came from different programmes and different teachers, with no way to match them lesson-for-lesson.",
        "resolution": "Resolved by a purpose-built paired scoring study: 300 lessons, each scored by up to 3 trained coaches (55 in total) and by 6 AI models against the same rubric. This gives real per-indicator agreement. The critical finding: the coaches barely agree with each other (median inter-coach reliability near chance), so the 0.70 target is currently unreachable by construction. Two issues follow — raise the human agreement ceiling through coach calibration, and correct the AI's tendency to score more harshly than coaches.",
    },
    {
        "id": "G2", "title": "No reference transcripts", "status": "blocked",
        "steps": ["2"],
        "wants": "Transcription accuracy below 5% error for the main language and below 10% for mixed-language speech, by language and speaker.",
        "reality": "Only the automatic transcripts exist; there is no hand-verified reference to measure them against, and no speaker labelling.",
        "resolution": "Commission a bilingual annotation pass on a representative sample of lessons; accuracy by language and mixed-language speech then reports automatically. (Confirming who is speaking would be a second, smaller labelling task.)",
    },
    {
        "id": "G3", "title": "Teacher satisfaction not collected", "status": "blocked",
        "steps": ["4a"],
        "wants": "Teacher satisfaction averaging above 4 out of 5 on specificity, actionability and tone.",
        "reality": "No teacher satisfaction is being captured — the place to record it exists, but nothing is written to it.",
        "resolution": "Add a short post-session rating prompt in the teacher's chat. Until then, an automated coach-style review scores the feedback text itself as a proxy — a quality measure, not a measure of how teachers received it.",
    },
    {
        "id": "G4", "title": "No test system to red-team", "status": "blocked",
        "steps": ["4b"],
        "wants": "Zero successful attempts to push the assistant out of its intended scope during red-team testing.",
        "reality": "Guardrail testing has to run against the live coaching assistant itself; this evaluation only reads recorded data and has no access to that live system.",
        "resolution": "Connect the ready-made test harness to a test version of the assistant. The attack set (12 scenarios, including mixed-language phrasing) and the automated grader are prepared.",
    },
    {
        "id": "G5", "title": "No link to student outcomes", "status": "blocked",
        "steps": ["6"],
        "wants": "Where student data exists, link teacher improvement to student learning gains.",
        "reality": "Student assessment records aren't connected to coaching activity, and there is no comparison (treatment/control) structure.",
        "resolution": "Beyond the scope of the AI evaluation — needs monitoring-and-evaluation instrumentation to connect teachers to their students' assessment records. The teacher-improvement trend (Step 6) runs today without it.",
    },
]

DATA_SOURCES = [
    {
        "name": "Programme data warehouse",
        "backend": "Programme records",
        "role": "The programme's central store of coaching activity: submitted sessions, the AI's rubric scores, human classroom observations, and quality and cost metrics.",
        "feeds": ["1", "2", "3b", "4a", "5", "6"],
        "notes": [
            "675 AI-scored coaching sessions across 30 rubric indicators, fully populated.",
            "142 teachers with scored sessions over roughly six and a half months (November 2025 to May 2026).",
            "It also holds 8,451 human classroom observations — but from a different programme and different teachers, with no lesson-level overlap with the AI sessions (the reason Step 3a needed a separate study).",
        ],
    },
    {
        "name": "Paired scoring study",
        "backend": "Study database",
        "role": "A purpose-built study created to answer the core reliability question. 300 lessons, each scored by up to 3 trained coaches and by 6 AI models against the same rubric.",
        "feeds": ["3a"],
        "notes": [
            "Uses the coaches' own classroom-observation rubric.",
            "This is the only source read live in this dashboard; Step 3a refreshes from it on demand.",
            "Two models are fully scored (about 100 lessons each), one is roughly three-quarters done, and three are still in progress.",
        ],
    },
    {
        "name": "Live coaching assistant's scoring service",
        "backend": "Live service",
        "role": "The coaching assistant's own scoring service, used to re-score lessons on demand for the reliability self-consistency check and the guardrail testing.",
        "feeds": ["3a", "4b"],
        "notes": [
            "Reproduces one of the study's model runs for consistency testing.",
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
        return "Pending — needs hand-verified reference transcripts to measure accuracy against (see gap G2)."
    if step_id == "3a":
        return (f"Best model {res['best_model']} median weighted κ = {res['best_model_median_weighted_kappa']} "
                f"vs a 0.70 bar — but the human ceiling is only Fleiss κ = {res['human_ceiling_median_fleiss']}, "
                f"so the bar is unreachable by construction.")
    if step_id == "3b":
        return "Pending — needs a batch of completed lessons (transcript plus AI analysis) to review."
    if step_id == "4a":
        return "Survey pending — teacher satisfaction isn't being collected yet (G3); a quality proxy is available."
    if step_id == "4b":
        return "Pending — needs a test version of the live assistant to run the guardrail attacks against (G4); the 12-scenario attack set is ready."
    if step_id == "5":
        return ("Partial — no lessons are yet scored by both a coach and the AI within the programme's "
                "records; showing the AI's own month-to-month scoring drift.")
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
        path = RESULTS_JSON if RESULTS_JSON.exists() else CACHED_JSON
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._results = json.load(f)
            self._source_label = "the most recent full evaluation run"
        except Exception:  # pragma: no cover - defensive
            self._results = {}
            self._source_label = "no saved results available"

    # ---- live Step-3a refresh (never blocks startup) ----------------------
    def refresh_step3a(self, timeout_ok: bool = True) -> dict:
        """Run the real study-backed Step 3a. On any failure keep the cache."""
        with self._lock:
            try:
                from rumi_evals.config import load_config
                from rumi_evals.study_data import available
                from rumi_evals.steps import step3a_human_irr

                if not available():
                    self._live_3a_error = ("Live refresh from the scoring study is unavailable right "
                                           "now — showing the most recent saved results.")
                    return {"ok": False, "error": self._live_3a_error}

                cfg = load_config()
                res = step3a_human_irr.run(None, cfg)
                if res.get("status") != "ok":
                    self._live_3a_error = ("Live refresh from the scoring study is unavailable right "
                                           "now — showing the most recent saved results.")
                    return {"ok": False, "error": self._live_3a_error}
                self._results["step3a"] = res
                self._live_3a = True
                self._live_3a_error = None
                return {"ok": True}
            except Exception:
                self._live_3a_error = ("Live refresh from the scoring study is unavailable right "
                                       "now — showing the most recent saved results.")
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
            # Prefer the curated, plain-language caveats; the raw pipeline caveats
            # reference internal record/table names and aren't for an external reader.
            "caveats": meta.get("caveats", []),
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
        # Use the curated, plain-language explanation (never the raw technical
        # reason from the pipeline, which contains file paths and internal names).
        reason = STEP_META.get(view["id"], {}).get("blocked_reason")
        if reason:
            view["stats"] = [{"label": "Status", "value": view["status_label"],
                              "sub": "not yet measurable", "tone": "muted"}]
            view["tables"].append({
                "title": "Why this can't run yet",
                "columns": ["What's needed first"],
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
            "note": "Read a model's number against the Human Fleiss κ in the same row — not against 0.70.",
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
        view["interpretation"] = (
            "No lesson is yet scored by both a coach and the AI within the programme's own "
            "records, so a true month-over-month coach–AI alignment can't be produced here yet. "
            "What's shown below is the AI's own scoring drift over time — useful as an early "
            "warning, but read it alongside the reliability findings in Step 3a, not as evidence "
            "of teacher change."
        )
        monthly = res.get("monthly_overall_mean", {})
        view["stats"] = [
            {"label": "Coach + AI scored lessons", "value": f"{res.get('human_pairs_available', 0)}",
             "sub": "needed for month-over-month alignment (G1)", "tone": "critical"},
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
