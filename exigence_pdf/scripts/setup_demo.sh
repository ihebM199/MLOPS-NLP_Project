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

if [ ! -f models/deepfake_fusion.onnx ] || [ ! -f models/AdaptiveFusion_best_model.pth ]; then
  echo "[demo] generating CI-only dummy ONNX and PyTorch checkpoints..."
  if docker run --rm \
    --user root \
    --volume "$PWD:/workspace" \
    --workdir /workspace \
    --env PYTHONPATH=/workspace \
    deepfake-adaptive-fusion-backend:latest \
    python scripts/make_dummy_model.py --out models; then
    echo "[demo] dummy artefacts created"
  else
    cat >&2 <<'EOF'
[demo] ERROR: demo artefacts could not be created.
       Train + export first:
         python -m src.models.train --config params.yaml
         python -m src.models.evaluate --config params.yaml --export
EOF
    exit 1
  fi
fi

echo "[demo] starting backend + frontend + MLflow (Ctrl+C to stop)..."
docker compose up --build "$@"
