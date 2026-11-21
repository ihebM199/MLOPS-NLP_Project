"""Download the 140k dataset (Kaggle) and extract a deterministic subset.

    python scripts/download_data.py --config params.yaml
    python scripts/download_data.py --quick-demo
    python scripts/download_data.py --zip /content/140k.zip --train-per-class 500

Credentials are read from the environment (``KAGGLE_API_TOKEN`` or
``KAGGLE_USERNAME``/``KAGGLE_KEY``) or from ``~/.kaggle``; they are never
written to the repository. External cross-dataset sources are *not* downloaded
here (see ``data/external/README.md``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.data.prepare import prepare_dataset  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download + extract the 140k working subset.")
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--zip", default=None, help="existing Kaggle ZIP (skips the download)")
    parser.add_argument("--quick-demo", action="store_true", help="500 img/class")
    parser.add_argument("--force", action="store_true", help="re-extract even if already present")
    parser.add_argument("--train-per-class", type=int, default=None)
    parser.add_argument("--valid-per-class", type=int, default=None)
    parser.add_argument("--test-per-class", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, quick_demo=args.quick_demo).prepare_dirs()
    per_class = {
        "train": args.train_per_class if args.train_per_class is not None else cfg.data.train_per_class,
        "valid": args.valid_per_class if args.valid_per_class is not None else cfg.data.valid_per_class,
        "test": args.test_per_class if args.test_per_class is not None else cfg.data.test_per_class,
    }
    root = prepare_dataset(cfg, zip_path=Path(args.zip) if args.zip else None, force=args.force, per_class=per_class)
    print(f"Dataset ready at {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
