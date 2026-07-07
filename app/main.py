"""Rumi Evals dashboard — FastAPI presentation layer.

Run from the repo root (so both ``app`` and ``rumi_evals`` import):

    uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}

Env vars:
  RUMI_STUDY_PG_URL  (optional) study Postgres — enables the live Step-3a refresh.
  APP_PASSWORD       (optional) if set, gates the whole app behind HTTP Basic auth.
  DC_API_KEY         (optional) only used by wobble/guardrail harnesses, not the web app.
"""
from __future__ import annotations

import base64
import hmac
import html
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .eval_service import STEP_META, service

APP_DIR = Path(__file__).resolve().parent
# Build the Jinja env explicitly with the template cache disabled (cache_size=0):
# jinja2's LRUCache trips over Python 3.14's dict internals; recompiling per render
# is negligible for an internal dashboard and sidesteps the issue on every runtime.
_jinja_env = Environment(
    loader=FileSystemLoader(str(APP_DIR / "templates")),
    autoescape=select_autoescape(["html", "xml"]),
    cache_size=0,
    auto_reload=False,
)
templates = Jinja2Templates(env=_jinja_env)

app = FastAPI(title="Rumi Evals Dashboard", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


# ---------------------------------------------------------------------------
# Optional HTTP Basic auth gate (only active when APP_PASSWORD is set).
# ---------------------------------------------------------------------------
def _auth_ok(header: str | None) -> bool:
    password = os.environ.get("APP_PASSWORD")
    if not password:
        return True
    if not header or not header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(header[6:]).decode("utf-8")
        _, _, supplied = decoded.partition(":")
    except Exception:
        return False
    return hmac.compare_digest(supplied, password)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path.startswith("/static") or request.url.path == "/healthz":
        return await call_next(request)
    if not _auth_ok(request.headers.get("authorization")):
        return Response(
            "Authentication required.", status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Rumi Evals"'},
        )
    return await call_next(request)


@app.on_event("startup")
def _startup() -> None:
    # Best-effort live Step-3a refresh; never blocks or crashes startup.
    service.try_live_refresh_on_start()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "source": service.source_label}


@app.get("/", response_class=HTMLResponse)
def roadmap(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "roadmap.html", {
        "roadmap": service.roadmap(), "active": "roadmap",
    })


@app.get("/step/{step_id}", response_class=HTMLResponse)
def step(request: Request, step_id: str) -> HTMLResponse:
    view = service.step_view(step_id)
    if view is None:
        return HTMLResponse(_not_found(step_id), status_code=404)
    ids = list(STEP_META.keys())
    i = ids.index(step_id)
    nav = {"prev": ids[i - 1] if i > 0 else None, "next": ids[i + 1] if i < len(ids) - 1 else None}
    return templates.TemplateResponse(request, "step.html", {
        "v": view, "nav": nav, "active": "roadmap",
    })


@app.post("/step/3a/refresh")
def refresh_3a() -> RedirectResponse:
    service.refresh_step3a()
    return RedirectResponse(url="/step/3a", status_code=303)


@app.get("/gaps", response_class=HTMLResponse)
def gaps(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "gaps.html", {
        "gaps": service.gaps(), "active": "gaps",
    })


@app.get("/data", response_class=HTMLResponse)
def data(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "data.html", {
        "sources": service.data_sources(), "active": "data",
    })


def _not_found(step_id: str) -> str:
    return (f"<h1>Unknown step '{html.escape(step_id)}'</h1>"
            "<p><a href='/'>Back to roadmap</a></p>")
