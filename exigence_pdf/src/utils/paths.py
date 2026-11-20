"""Single source of truth for filesystem locations.

Everything is relative to the repository root so the *same* code runs:
  * locally            (D:\\...\\exigences_version)
  * inside Docker      (/app)
  * on Google Colab    (/content/deepfake-adaptive-fusion)

Every path can be overridden with an environment variable, which is what the
notebooks and docker-compose do.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# ``src/utils/paths.py`` -> parents[2] is the repository root.
_DEFAULT_ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """Repository root (override with ``DEEPFAKE_ROOT``)."""
    return Path(os.environ.get("DEEPFAKE_ROOT") or _DEFAULT_ROOT).expanduser().resolve()


PROJECT_ROOT: Path = project_root()
DATA_DIR: Path = Path(os.environ.get("DATA_DIR") or PROJECT_ROOT / "data")
MODEL_DIR: Path = Path(os.environ.get("MODEL_DIR") or PROJECT_ROOT / "models")
REPORTS_DIR: Path = Path(os.environ.get("REPORTS_DIR") or PROJECT_ROOT / "reports")
FRONTEND_DIST: Path = Path(os.environ.get("FRONTEND_DIST") or PROJECT_ROOT / "frontend" / "dist")

RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DIR: Path = DATA_DIR / "external"

#: Layout produced by the Kaggle archive of ``xhlulu/140k-real-and-fake-faces``.
DATASET_SUBDIR = Path("real_vs_fake") / "real-vs-fake"


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (parents included) and return it as a ``Path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve(path: Optional[str | Path], *, root: Optional[Path] = None) -> Optional[Path]:
    """Resolve ``path`` against the repository root (absolute paths are kept)."""
    if path in (None, ""):
        return None
    p = Path(str(path)).expanduser()
    return p if p.is_absolute() else (Path(root or PROJECT_ROOT) / p)


def dataset_root() -> Path:
    """Directory holding ``train/valid/test`` (env ``DATASET_ROOT`` wins)."""
    env = os.environ.get("DATASET_ROOT")
    if env:
        return resolve(env)  # type: ignore[return-value]
    processed = PROCESSED_DIR / DATASET_SUBDIR
    raw = RAW_DIR / DATASET_SUBDIR
    if processed.exists():
        return processed
    if raw.exists():
        return raw
    return processed
