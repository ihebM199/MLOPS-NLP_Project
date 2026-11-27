# AGENTS.md

## Layout gotcha
- The real project is **`exigence_pdf/`**, not the repo root. Run every command below from
  `/workspaces/MLOPS-NLP_Project/exigence_pdf` (or use `workdir`). The root only holds the
  original notebook, the assignment prompt, and a 4-line README.
- `models/` and `reports/` are git-ignored (`reports/metrics.json` is now an exception). In this
  workspace `models/` holds the **real** `deepfake_fusion.onnx` + `AdaptiveFusion_best_model.pth`
  and `reports/metrics.json` the trained metrics; on a fresh clone use `dvc pull` or
  `scripts/make_dummy_model.py` (dummy, not a real detector).

## DeepGuard stack (colleague's UI, integrated)
- `deepguard/` holds the colleague's full React app + Django backend (same API contract as the
  FastAPI service). It reuses the project's `src/`, `models/` and `reports/metrics.json`.
- Run it with `docker compose -f deepguard/docker-compose.yml up --build` (UI :8010, Prometheus
  :9091, Grafana :3002, MLflow :5001) or see `deepguard/README.md` for the two-terminal dev mode.
- Django exposes `/metrics`, `/monitoring/stats`, `/monitoring/drift` via `src/monitoring`, and
  records every `/predict`. The original FastAPI stack (`backend/app`, `frontend/`) stays intact.

## Setup
- Target **Python 3.11** (CI and Docker use it); `pyproject.toml` targets 3.10/3.11.
- `pip install -r requirements-ci.txt` is enough for lint + API + core tests (it already pins
  torch/timm/onnx/fastapi/prometheus). `requirements.txt` adds MLflow/DVC/Prefect/training;
  `requirements-colab.txt` avoids reinstalling torch.
- Backend loads `.env` from the project root (`PROJECT_ROOT/.env`). Paths resolve against
  `DEEPFAKE_ROOT` (default: repo root); `DATA_DIR`/`MODEL_DIR`/`REPORTS_DIR` are env-overridable.
- Frontend needs Node >= 20: `cd frontend && npm ci` (lockfile committed; `node_modules` not).

## Commands (from `exigence_pdf/`)
- CI order: `black --check src backend tests` -> `flake8 src backend tests` (line length 110) ->
  `pytest backend -q --cov=backend/app --cov-report=term-missing --cov-fail-under=80` ->
  `pytest tests -q -m "not slow"`.
- Single test: `pytest backend/tests/test_api.py::test_health_ok -q`. Markers: `slow` (real
  weights/dataset), `gpu`, `data`; `pyproject.toml` sets both testpaths, so bare `pytest` runs both.
- Frontend: `cd frontend && npm run lint && npm test && npm run build` (`npm test` = `vitest run`).
- Pipeline: `dvc repro` = prepare -> train -> evaluate -> export_onnx (needs Kaggle credentials + GPU).
- CPU smoke train/eval (500 img/class, 2 epochs): `python -m src.models.train --config params.yaml --quick-demo`,
  then `python -m src.models.evaluate --config params.yaml --quick-demo`.
- Export: `python -m src.models.evaluate --config params.yaml --export-onnx --quantize` or
  `python -m src.models.onnx_export --config params.yaml`. `scripts/setup_demo.sh` still prints a
  stale `--export` hint; the real flag is `--export-onnx`.
- Demo without weights: `python scripts/make_dummy_model.py --out models` then
  `docker compose up --build` (5 services: backend 8000, frontend 3000, MLflow 5000, Prometheus
  9090, Grafana 3001). CI does exactly this before building images.
- Backend dev: `uvicorn backend.app.main:app --reload --port 8000` (imports `src.*`, so CWD must be
  `exigence_pdf`). Retraining orchestrator: `python -m src.orchestration.flows --retrain --trigger manual`.

## Architecture contracts (not obvious from filenames)
- ONNX takes **2 inputs** (`image`, `freq_input`) and returns **3 outputs** (`logit`, `w_spatial`,
  `w_frequency`). FFT `log1p(|fftshift(fft2(x))|)` is computed outside the graph in
  `src/models/onnx_export.py`; preserve this contract in inference/export.
- `/predict` uses ONNX Runtime; `/predict/visualize` lazily loads PyTorch for Grad-CAM. Backend tests
  build a dummy ONNX graph with the same signature, so **no real checkpoint/dataset is needed**.
- **The backend does NOT query the MLflow registry.** `InferenceRegistry.load()` reads only
  `settings.model_path` (default `models/deepfake_fusion.onnx`). `MLFLOW_TRACKING_URI` is set in
  docker-compose but unused for serving; `docs/DECISIONS.md` §6 is stale on this — trust the code.
- `params.yaml` is the source of truth for all stages. Dataclass defaults in `src/config.py` differ:
  backbone `efficientnetv2_rw_s` (dataclass/architecture default) vs `convnext_tiny` (params.yaml,
  matching the provided checkpoint). Pass `--config params.yaml` explicitly.
- `models/`, `reports/` and data are git-ignored; DVC tracks pipeline outputs (the checkpoint is a
  `train` stage output, not `dvc add`). Default MLflow store: `<repo>/mlruns/mlflow.db` (SQLite).
- Backend reads metadata/metrics from `models/model_config.json` (`MODEL_CONFIG_PATH`) and
  `reports/metrics.json`, never hard-coded values.

## Conventions that differ from defaults
- Labels: `real = 0`, `fake = 1`; decision is `P(fake) >= DECISION_THRESHOLD` (default 0.5).
- Frontend calls are relative by default; Docker builds with `VITE_API_URL=/api` and nginx strips the
  prefix. Vite dev proxy rewrites `/api` -> `VITE_DEV_API_TARGET` (default `http://localhost:8000`).
- Pre-commit runs black/flake8, nbstripout, `detect-secrets` (baseline; notebooks excluded) and
  forbids files >2 MB. Do not add secrets; history had an exposed Kaggle token (see `docs/DECISIONS.md`).
- Deeper rationale lives in `docs/DECISIONS.md`, `docs/MODEL_CARD.md`, `EXPLICATION_PROJET.md`; do not
  duplicate it here. When docs and code disagree (e.g. MLflow serving), trust the code/scripts.
