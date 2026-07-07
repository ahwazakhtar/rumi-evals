"""Paired human-vs-AI scoring study — data layer (Railway Postgres).

This is the study that unblocks the framework's core reliability metric (gap G1):
300 classroom recordings, up to 3 human coaches each, scored against the same
observation rubric that 6 AI models also scored. The `study_compiled` table pairs
human consensus against each model.

Connection string comes from env var RUMI_STUDY_PG_URL — never hardcode it. If
psycopg is not installed or the var is unset, callers get a clear blocked message.

Rubric scale: yes / partial / no (ordinal), plus na (not applicable = missing).
Humans store lowercase, AI stores UPPER / "N/A"; normalize() handles both.
"""
from __future__ import annotations

import json
import os

import pandas as pd

ENV_VAR = "RUMI_STUDY_PG_URL"


def _order_map(cfg: dict) -> dict[str, int]:
    # Coerce keys to lowercase strings — guards against YAML parsing bare
    # no/yes as booleans (the "Norway problem").
    return {str(k).lower(): int(v) for k, v in cfg["study"]["category_order"].items()}


def normalize_token(v, cfg: dict) -> float | None:
    """Map a raw rubric token to its ordinal int, or None for na/missing."""
    if v is None:
        return None
    t = str(v).strip().lower()
    if t in {str(s).lower() for s in cfg["study"]["na_tokens"]} or t in ("", "none"):
        return None
    return _order_map(cfg).get(t)


def _connect(url: str):
    import psycopg  # lazy

    return psycopg.connect(url, connect_timeout=20)


def available() -> bool:
    if os.environ.get(ENV_VAR) is None:
        return False
    try:
        import psycopg  # noqa: F401
    except ImportError:
        return False
    return True


def load_compiled(cfg: dict) -> pd.DataFrame:
    """One row per (recording x model run). JSON columns parsed to dicts."""
    url = os.environ[ENV_VAR]
    sql = """
        SELECT audio_id, run_label, recording_id, subject, grade, subject_key,
               in_study, n_raters, has_ai, transcript_source, transcript_chars,
               human_raters, human_consensus, ai_section_b, ai_section_c, agreement
        FROM study_compiled
        WHERE in_study
    """
    with _connect(url) as conn, conn.cursor() as cur:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    df = pd.DataFrame(rows)
    # psycopg returns jsonb already as python objects; guard against text
    for col in ("human_raters", "human_consensus", "ai_section_b", "ai_section_c", "agreement"):
        df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    return df


def indicators_for(subject_key: str, cfg: dict) -> list[str]:
    sc = cfg["study"]["section_c_by_subject"].get(subject_key, [])
    return list(cfg["study"]["section_b"]) + list(sc)


def ai_scores(row: pd.Series) -> dict:
    """Merge a row's AI section B + C into one indicator->token dict."""
    out = {}
    for sec in ("ai_section_b", "ai_section_c"):
        if isinstance(row.get(sec), dict):
            out.update(row[sec])
    return out
