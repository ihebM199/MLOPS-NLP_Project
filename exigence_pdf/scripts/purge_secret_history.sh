#!/usr/bin/env bash
# DANGER: rewrite git history to purge the leaked Kaggle token.
# Only run after the token has been revoked on kaggle.com (Account > API).
# Requires git-filter-repo: pip install git-filter-repo
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v git-filter-repo >/dev/null 2>&1; then
  echo "git-filter-repo is required: pip install git-filter-repo" >&2
  exit 1
fi

if [ -n "${ALLOW_HISTORY_REWRITE:-}" ] && [ "$ALLOW_HISTORY_REWRITE" != "1" ]; then
  echo "Set ALLOW_HISTORY_REWRITE=1 to confirm." >&2
  exit 1
fi

expressions="$(mktemp)"
cat > "$expressions" <<'EOF'
regex:KGAT_[A-Za-z0-9]+==>REDACTED_KAGGLE_TOKEN
EOF

echo "[purge] rewriting history (local clone only)..."
git filter-repo --replace-text "$expressions" --force
rm -f "$expressions"

cat <<'EOF'
[purge] local history rewritten.
  1. Verify:  git log --all -p | grep -c KGAT_   # must print 0
  2. Revoke the old token on Kaggle.
  3. Force-push explicitly:  git push --force-with-lease --all
  4. Force-push tags too:    git push --force-with-lease --tags
NOTE: collaborators must re-clone; never do this without a backup.
EOF
