"""Digital Coach external API client.

The paired-study AI scores were produced by this service. It is used here to
score fresh transcripts on demand — mainly for the Step 3a WOBBLE TEST (same
transcript scored N times, measuring within-model non-determinism), which the
study DB's single stored score per recording can't provide.

Auth: X-API-Key header. Key comes from env var DC_API_KEY — never hardcode it.
Endpoints (see API_AUTHENTICATION.md):
  POST /api/external/process-transcription  -> {task_id}
  GET  /api/external/status/{task_id}       -> {status, result|scores, ...}
"""
from __future__ import annotations

import os
import time

ENV_KEY = "DC_API_KEY"


class DCApiError(RuntimeError):
    pass


def _headers() -> dict:
    key = os.environ.get(ENV_KEY)
    if not key:
        raise DCApiError(f"{ENV_KEY} not set. Export the X-API-Key for the Digital Coach API.")
    return {"Content-Type": "application/json", "X-API-Key": key}


def score_transcription(transcription: str, cfg: dict, *, teacher_id: str = "eval-harness",
                        extra: dict | None = None) -> dict:
    """Submit a transcript, poll to completion, return the parsed result payload."""
    import requests  # lazy

    dc = cfg["dc_api"]
    base = dc["base_url"]
    payload = {"transcription": transcription, "teacher_id": teacher_id, **(extra or {})}

    r = requests.post(f"{base}/api/external/process-transcription",
                      json=payload, headers=_headers(), timeout=dc["timeout_seconds"])
    if r.status_code == 401:
        raise DCApiError("401 Unauthorized — check DC_API_KEY (do not retry unchanged).")
    r.raise_for_status()
    task_id = r.json().get("task_id")
    if not task_id:
        # some deployments return the result inline
        return r.json()

    for _ in range(dc["poll_max_attempts"]):
        s = requests.get(f"{base}/api/external/status/{task_id}",
                         headers=_headers(), timeout=dc["timeout_seconds"])
        s.raise_for_status()
        body = s.json()
        status = str(body.get("status", "")).lower()
        if status in ("completed", "success", "done"):
            return body
        if status in ("failed", "error"):
            raise DCApiError(f"DC task {task_id} failed: {body}")
        time.sleep(dc["poll_interval_seconds"])
    raise DCApiError(f"DC task {task_id} did not complete within the poll window.")


def make_score_fn(cfg: dict):
    """Return score_fn(transcript) -> {indicator: token} for the wobble test.

    Adapts the DC response into the flat indicator dict the metrics expect. The
    exact response shape depends on the DC version; adjust the extraction below
    against a live sample if keys differ.
    """
    def score_fn(transcript: str) -> dict:
        body = score_transcription(transcript, cfg)
        result = body.get("result", body)
        scores = {}
        for sec in ("section_b", "section_c"):
            if isinstance(result.get(sec), dict):
                scores.update(result[sec])
        if not scores and isinstance(result.get("scores"), dict):
            scores = result["scores"]
        return scores

    return score_fn
