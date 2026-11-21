#!/usr/bin/env bash
# Initialise DVC and track the raw + processed datasets.
# Run once, from the project root, after `pip install -r requirements.txt`.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v dvc >/dev/null 2>&1; then
  echo "DVC is not installed: pip install 'dvc>=3.50,<4'" >&2
  exit 1
fi

if [ ! -d .dvc ]; then
  dvc init -q
  echo "[dvc] initialised"
fi

# Local remote by default; replace with S3/GCS/Azure for a shared remote.
dvc remote add -d local ../dvc-remote 2>/dev/null || true

for path in data/raw data/processed; do
  if [ -e "$path" ]; then
    dvc add "$path"
    echo "[dvc] tracking $path"
  else
    echo "[dvc] skipping missing $path"
  fi
done

git add .dvc .dvcignore dvc.yaml params.yaml 2>/dev/null || true
git add data/raw.dvc data/processed.dvc 2>/dev/null || true
cat <<'EOF'
[dvc] done. Next:
  dvc push
  git commit -m "data: track raw/processed datasets with DVC"
EOF
