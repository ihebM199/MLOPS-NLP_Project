#!/usr/bin/env bash
# HTTP smoke test against a running stack (docker compose up).
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_URL="${BASE_URL:-http://localhost:8000}"
IMAGE="${1:-models/sample.jpg}"

if [ ! -f "$IMAGE" ]; then
  echo "Image not found: $IMAGE" >&2
  echo "Pass one explicitly: scripts/smoke_test.sh path/to/face.jpg" >&2
  exit 1
fi

echo "== GET /health =="
curl -fsS "$BASE_URL/health" | tee /dev/stderr | grep -q '"status"'

printf "\n== GET /model/metadata ==\n"
curl -fsS "$BASE_URL/model/metadata" | tee /dev/stderr | grep -q '"architecture"'

printf "\n== POST /predict ==\n"
curl -fsS -F "file=@${IMAGE};type=image/jpeg" "$BASE_URL/predict" | tee /tmp/deepfake_prediction.json
grep -q '"label"' /tmp/deepfake_prediction.json
grep -q '"fusion_weights"' /tmp/deepfake_prediction.json

printf "\n== POST /predict/visualize ==\n"
curl -fsS -F "file=@${IMAGE};type=image/jpeg" "$BASE_URL/predict/visualize" | tee /tmp/deepfake_visualize.json
grep -q '"gradcam_overlay"' /tmp/deepfake_visualize.json

printf "\nAll smoke tests passed.\n"
