# AGENTS.md

## Layout gotcha
The real project is **`exigence_pdf/`**, not the repo root. Run every command below from
`/workspaces/MLOPS-NLP_Project/exigence_pdf` (or use `workdir`). The root only holds the
original notebook, the assignment prompt, and a 4-line README.

## Setup
- Target **Python 3.11** (CI and Docker use it); `pyproject.toml` targets 3.10/3.11.
- `pip install -r requirements-ci.txt` is enough for lint + API + core tests.
  `requirements.txt` adds MLflow/DVC/training; `requirements-colab.txt` avoids reinstalling torch.
- Copy `.env.example` to `.env`; backend settings load `.env` from the project root.
- Frontend needs Node >= 20: `cd frontend && npm ci` (lockfile is committed; `node_modules` is not).

## Commands (from `exigence_pdf/`)
- Lint/format: `black --check src backend tests` then `flake8 src backend tests` (line length 110).
- Backend tests + coverage gate (80%): `pytest backend -q --cov=backend/app --cov-report=term-missing --cov-fail-under=80`
- Core tests: `pytest tests -q -m "not slow"` (`slow` needs real weights/dataset; `gpu`/`data` markers exist).
- Frontend: `cd frontend && npm run lint && npm test && npm run build`.
- Pipeline: `dvc repro` (prepare -> train -> evaluate -> export_onnx; needs Kaggle credentials).
- CPU smoke train/eval (500 img/class, 2 epochs): `python -m src.models.train --config params.yaml --quick-demo`,
  then `python -m src.models.evaluate --config params.yaml --quick-demo`.
- Export is `python -m src.models.evaluate --config params.yaml --export-onnx --quantize` or the
  dedicated `python -m src.models.onnx_export`; the `--export` hint printed by `scripts/setup_demo.sh` is stale.
- Local demo without real weights: `python scripts/make_dummy_model.py --out models` (also writes
  `sample.jpg` + `model_config.json`) then `docker compose up --build`. The dummy is not a real detector.
- Backend dev: `uvicorn backend.app.main:app --reload --port 8000` (imports `src.*`, so CWD must be `exigence_pdf`).

## Architecture contracts (not obvious from filenames)
- ONNX model takes **two inputs** (`image`, `freq_input`) and returns **three outputs**
  (`logit`, `w_spatial`, `w_frequency`). FFT (`log1p(|fftshift(fft2(x))|)`) is computed
  outside the graph in `src/models/onnx_export.py`; keep this contract when editing inference/export.
- `/predict` uses ONNX Runtime; `/predict/visualize` lazily loads PyTorch for Grad-CAM.
  Backend tests use a dummy ONNX graph of the same signature, so **no real checkpoint is needed**.
- `params.yaml` is the source of truth loaded by all stages. Some dataclass defaults in
  `src/config.py` differ from YAML (e.g. backbone); if the YAML is missing, config silently
  falls back to defaults. Pass `--config params.yaml` when running modules.
- `models/` and `reports/` are git-ignored and currently empty. Never commit weights, data,
  `.env`, or generated reports; DVC tracks pipeline outputs (checkpoint is a DVC stage output,
  not `dvc add`). Default MLflow store is `<repo>/mlruns/mlflow.db` (SQLite).
- The backend prefers a registered MLflow "Production" model and falls back to
  `models/deepfake_fusion.onnx`; metrics/metadata come from `reports/*.json`, not hard-coded values.

## Conventions that differ from defaults
- Labels: `real = 0`, `fake = 1`; decision is `P(fake) >= DECISION_THRESHOLD`.
- Frontend calls are relative by default; Docker builds with `VITE_API_URL=/api` and nginx strips
  the prefix. Vite dev proxy rewrites `/api` to `http://localhost:8000`.
- Pre-commit runs black/flake8, nbstripout, and `detect-secrets` (notebooks excluded via baseline).
  Do not add secrets; the repo history has had an exposed Kaggle token (see `docs/DECISIONS.md`).
- Existing deeper rationale lives in `docs/DECISIONS.md` and `docs/MODEL_CARD.md`; do not duplicate it here.
