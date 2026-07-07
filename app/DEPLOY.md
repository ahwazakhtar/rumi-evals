# Rumi Evals Dashboard — deploy & run

A FastAPI presentation layer over the `rumi_evals` package. It shows the 8-step
evaluation roadmap and drills into each step. It re-uses the eval code — it does
**not** re-implement any metric. Step 3a reads live from the study Postgres (with a
cached fallback); every other step is served from a cached full-run
(`results/latest.json`, falling back to the committed `app/cached/latest.json`).

## Environment variables

| Var | Required? | Purpose |
|---|---|---|
| `RUMI_STUDY_PG_URL` | **Recommended** | Study Postgres URL. Enables the **live Step-3a refresh**. Without it, Step 3a shows the cached study numbers (app still boots fine). |
| `APP_PASSWORD` | Optional but advised for shared deploys | If set, the whole dashboard is gated behind HTTP Basic auth (any username, this password). The app surfaces scores and connects to a DB, so set this on any team/leadership deployment. |
| `DC_API_KEY` | Optional | Digital Coach `X-API-Key`. Only used by the wobble / guardrail harnesses in the CLI, **not** by the web app. |
| `PORT` | Set by Railway | Bound automatically by the Procfile / start command. Defaults to 8000 locally. |

Never commit `postgres.txt` or `api key.txt` — they are gitignored. Read the values
from them once and set the env vars.

## Run locally

From the repo root (`rumi-evals/`), so both `app` and `rumi_evals` import:

```bash
pip install -r app/requirements.txt

# with the live study DB (value from postgres.txt):
export RUMI_STUDY_PG_URL='postgresql://USER:PASS@HOST:PORT/DB'
uvicorn app.main:app --host 0.0.0.0 --port 8000

# optional auth gate:
export APP_PASSWORD='choose-a-strong-password'
```

Then open http://127.0.0.1:8000/ . The app also boots **without** `RUMI_STUDY_PG_URL`
— Step 3a simply falls back to the cached study results.

Pages: `/` (roadmap), `/step/{1,2,3a,3b,4a,4b,5,6}`, `/gaps`, `/data`, `/healthz`.

> On Windows PowerShell use `$env:RUMI_STUDY_PG_URL='...'` instead of `export`.

## Deploy on Railway

The study DB is already a Railway Postgres, so the simplest topology is to add this
web service to the **same Railway project** as that database.

### Option A — deploy from a connected GitHub repo (recommended)

1. Push this repo to GitHub.
2. In the Railway project that hosts the study Postgres: **New → GitHub Repo →** pick
   this repo. Railway auto-detects the build via `nixpacks.toml` (Python 3.12, installs
   `app/requirements.txt`) and starts it via the `Procfile` / `railway.json`
   `startCommand`. `$PORT` is injected automatically.
3. **Variables** tab on the new service:
   - `RUMI_STUDY_PG_URL` → reference the Postgres service's connection string. Type
     `${{` and pick the Postgres service, e.g.
     `${{Postgres.DATABASE_URL}}` (or the `DATABASE_PUBLIC_URL` if the app runs
     outside the project's private network). This wires the two services without
     hardcoding the credential.
   - `APP_PASSWORD` → a strong password (set this for any shared/leadership deploy).
   - `DC_API_KEY` → only if you also run the DC-API harnesses; the web app doesn't need it.
4. Railway builds, deploys, and health-checks `/healthz`. Open the generated URL.

### Option B — deploy from your machine with the CLI

```bash
npm i -g @railway/cli      # or: brew install railway
railway login
cd rumi-evals
railway link               # select the project that has the study Postgres
railway up                 # builds & deploys the current directory

railway variables --set "RUMI_STUDY_PG_URL=${{Postgres.DATABASE_URL}}"
railway variables --set "APP_PASSWORD=choose-a-strong-password"
```

(You can also set variables in the dashboard instead of the CLI.)

### Build details

- `nixpacks.toml` pins **Python 3.12** and installs the light `app/requirements.txt`
  (fastapi, uvicorn, jinja2, pandas, numpy, scipy, pyyaml, psycopg[binary]) — it does
  **not** install the repo-root `requirements.txt` (which pulls `anthropic`/`jiwer`
  that the dashboard never imports), keeping the image small and the boot fast.
- `railway.json` sets the start command (honours `$PORT`), a `/healthz` health check,
  and an on-failure restart policy.
- `Procfile` mirrors the start command for any Procfile-based platform (Heroku etc.).

## Notes

- The app never blocks startup on the DB: it attempts one best-effort live Step-3a
  refresh on boot and swallows any error, then serves the cached snapshot. The
  "↻ Refresh from study DB" button on `/step/3a` re-runs the live study read on demand.
- To update the cached fallback after a fresh full run, copy
  `results/latest.json` and `results/step3a_human_vs_ai.csv` into `app/cached/` and commit.
