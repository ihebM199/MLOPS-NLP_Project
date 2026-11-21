#!/usr/bin/env bash
# One-command demo: pull data/model artefacts (if DVC is set up) and start the stack.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required: https://docs.docker.com/engine/install/" >&2
  exit 1
fi

if command -v dvc >/dev/null 2>&1 && [ -f data/raw.dvc ]; then
  echo "[demo] pulling DVC artefacts (data + model)..."
  dvc pull || echo "[demo] WARNING: dvc pull failed (remote not configured?)"
fi

if [ ! -f models/deepfake_fusion.onnx ]; then
  cat >&2 <<'EOF'
[demo] ERROR: models/deepfake_fusion.onnx is missing.
       Train + export first:
         python -m src.models.train --config params.yaml
         python -m src.models.evaluate --config params.yaml --export
       or create a CI-only dummy model:
         python scripts/make_dummy_model.py --out models
EOF
  exit 1
fi

echo "[demo] starting backend + frontend + MLflow (Ctrl+C to stop)..."
docker compose up --build "$@"
