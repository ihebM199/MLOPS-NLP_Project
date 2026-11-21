"""Dataset preparation: Kaggle download + deterministic subset extraction.

The 140k archive is 3.7 GB zipped / ~8 GB extracted. Instead of extracting
everything, the ZIP is indexed and only the requested number of images per
split/class is extracted — the exact strategy used in the original notebook, but
now idempotent, configurable and reusable from the CLI:

    python -m src.data.prepare --config params.yaml
    python -m src.data.prepare --quick-demo
    python -m src.data.prepare --zip /content/140k.zip --per-class 500

Credentials are **never** hard-coded: environment variables, an existing
``~/.kaggle`` file, Google Colab *Secrets* (``google.colab.userdata``) or an
interactive masked prompt (local only).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Mapping, Optional, Sequence

import pandas as pd

from src.config import load_config
from src.data.dataset import IMG_EXT, LABEL2ID, SPLITS, build_dataframe
from src.utils.io import write_json
from src.utils.logging import get_logger
from src.utils.paths import PROCESSED_DIR, RAW_DIR, ensure_dir

LOGGER = get_logger("data.prepare")

DEFAULT_ZIP_ROOT = "real_vs_fake/real-vs-fake"


# --------------------------------------------------------------------------- #
# Kaggle credentials
# --------------------------------------------------------------------------- #
def _colab_secret(name: str) -> Optional[str]:
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        return userdata.get(name)
    except Exception:
        return None


def setup_kaggle_credentials(prompt: bool = False) -> bool:
    """Make the Kaggle CLI usable. Returns ``True`` when credentials are ready."""
    kaggle_dir = Path.home() / ".kaggle"
    if (kaggle_dir / "kaggle.json").exists() or (kaggle_dir / "access_token").exists():
        LOGGER.info("Kaggle credentials found in %s", kaggle_dir)
        return True

    token = os.environ.get("KAGGLE_API_TOKEN") or _colab_secret("KAGGLE_API_TOKEN")
    username = os.environ.get("KAGGLE_USERNAME") or _colab_secret("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY") or _colab_secret("KAGGLE_KEY")

    kaggle_dir.mkdir(parents=True, exist_ok=True)
    if token:
        path = kaggle_dir / "access_token"
        path.write_text(token.strip())
        path.chmod(0o600)
        os.environ["KAGGLE_API_TOKEN"] = token.strip()
        LOGGER.info("Kaggle API token configured for this session (not written to the repo).")
        return True
    if username and key:
        import json

        path = kaggle_dir / "kaggle.json"
        path.write_text(json.dumps({"username": username, "key": key}))
        path.chmod(0o600)
        LOGGER.info("Kaggle username/key configured for this session.")
        return True

    if prompt:
        import getpass

        entered = getpass.getpass("Paste your Kaggle API token (hidden, empty to skip): ").strip()
        if entered:
            path = kaggle_dir / "access_token"
            path.write_text(entered)
            path.chmod(0o600)
            os.environ["KAGGLE_API_TOKEN"] = entered
            return True
    LOGGER.error(
        "No Kaggle credentials. Set KAGGLE_API_TOKEN (Colab: Tools > Secrets) or place "
        "~/.kaggle/kaggle.json. Never commit these files."
    )
    return False


def kaggle_download(dataset: str, dest: Path) -> Path:
    """``kaggle datasets download`` -> path of the resulting ZIP."""
    import subprocess

    dest = ensure_dir(dest)
    target = dest / f"{dataset.split('/')[-1]}.zip"
    if target.exists() and target.stat().st_size > 0:
        LOGGER.info("Archive already downloaded: %s", target)
        return target
    if not setup_kaggle_credentials():
        raise RuntimeError("Kaggle credentials are missing — see the message above.")
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", dataset, "-p", str(dest), "--force"]
    LOGGER.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    # the CLI writes <slug>.zip
    produced = dest / f"{dataset.split('/')[-1]}.zip"
    if produced.exists():
        return produced
    candidates = sorted(dest.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"Kaggle download produced no ZIP in {dest}")
    shutil.move(str(candidates[0]), produced)
    return produced


# --------------------------------------------------------------------------- #
# ZIP indexing / extraction
# --------------------------------------------------------------------------- #
def index_zip(zip_path: Path, zip_root: str = DEFAULT_ZIP_ROOT) -> dict[tuple[str, str], list[str]]:
    """``{(split, class): [member names]}`` without extracting anything."""
    with zipfile.ZipFile(zip_path) as archive:
        names = [
            n
            for n in archive.namelist()
            if n.startswith(zip_root + "/") and n.lower().endswith(IMG_EXT)
        ]
    index: dict[tuple[str, str], list[str]] = {}
    for name in names:
        parts = name.split("/")
        if len(parts) < 5:
            continue
        _, _, split, cls, _ = parts[0], parts[1], parts[2], parts[3], parts[4]
        if split in SPLITS and cls in LABEL2ID:
            index.setdefault((split, cls), []).append(name)
    return index


def subset_counts(index: Mapping[tuple[str, str], list[str]]) -> pd.DataFrame:
    """Images per (split, class) as a ``split x class`` table."""
    rows = [{"split": split, "class": cls, "n": len(members)} for (split, cls), members in index.items()]
    if not rows:
        return pd.DataFrame(columns=list(LABEL2ID))
    return (
        pd.DataFrame(rows)
        .pivot(index="split", columns="class", values="n")
        .reindex(list(SPLITS))
        .fillna(0)
        .astype(int)
    )


def extract_subset(
    zip_path: Path,
    dest: Path,
    index: Mapping[tuple[str, str], list[str]],
    limits: Mapping[str, Optional[int]],
    seed: int = 42,
) -> int:
    """Extract at most ``limits[split]`` images per class (deterministic, idempotent)."""
    import random

    rnd = random.Random(seed)
    selected: list[str] = []
    for split in SPLITS:
        for cls in LABEL2ID:
            members = sorted(index.get((split, cls), []))
            limit = limits.get(split)
            if limit is not None and len(members) > limit:
                members = rnd.sample(members, limit)
            selected += members
    todo = [name for name in selected if not (dest / name).exists()]
    LOGGER.info(
        "Selected %s images | %s still to extract into %s",
        f"{len(selected):,}",
        f"{len(todo):,}",
        dest,
    )
    if not todo:
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for name in todo:
            archive.extract(name, dest)
    return len(todo)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def prepare_dataset(
    cfg,
    *,
    zip_path: Optional[Path] = None,
    force: bool = False,
    per_class: Optional[dict[str, Optional[int]]] = None,
    zip_root: str = DEFAULT_ZIP_ROOT,
) -> Path:
    """Download (once) then extract the working subset. Returns the dataset root."""
    if per_class is None:
        per_class = {
            "train": cfg.data.train_per_class,
            "valid": cfg.data.valid_per_class,
            "test": cfg.data.test_per_class,
        }
    dest = ensure_dir(PROCESSED_DIR)
    root = dest / zip_root
    existing = build_dataframe(root)
    expected = sum(v for v in per_class.values() if v) * len(LABEL2ID)
    if not force and len(existing) >= 0.9 * expected and expected > 0:
        LOGGER.info("Dataset already prepared (%d images in %s)", len(existing), root)
        return root

    zip_path = Path(zip_path) if zip_path else RAW_DIR / f"{cfg.data.dataset.split('/')[-1]}.zip"
    if not zip_path.exists():
        zip_path = kaggle_download(cfg.data.dataset, zip_path.parent)

    index = index_zip(zip_path, zip_root)
    if not index:
        raise ValueError(
            f"{zip_path} does not contain '{zip_root}/<split>/<class>/<image>' — wrong archive?"
        )
    LOGGER.info("Official composition:\n%s", subset_counts(index).to_string())
    extract_subset(zip_path, dest, index, per_class, seed=cfg.data.seed)

    df = build_dataframe(root)
    report = {
        "dataset": cfg.data.dataset,
        "root": str(root),
        "counts": df.groupby(["split", "class"]).size().unstack(fill_value=0).to_dict(),
        "n_images": int(len(df)),
        "per_class": per_class,
        "seed": cfg.data.seed,
    }
    write_json(Path(cfg.reports_dir) / "dataset_report.json", report)
    LOGGER.info("Working dataset ready: %d images at %s", len(df), root)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and extract a 140k working subset.")
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--zip", default=None, help="existing Kaggle ZIP (skips the download)")
    parser.add_argument("--quick-demo", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--train-per-class", type=int, default=None)
    parser.add_argument("--valid-per-class", type=int, default=None)
    parser.add_argument("--test-per-class", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, quick_demo=args.quick_demo).prepare_dirs()
    per_class = {
        "train": args.train_per_class if args.train_per_class is not None else cfg.data.train_per_class,
        "valid": args.valid_per_class if args.valid_per_class is not None else cfg.data.valid_per_class,
        "test": args.test_per_class if args.test_per_class is not None else cfg.data.test_per_class,
    }
    root = prepare_dataset(cfg, zip_path=args.zip, force=args.force, per_class=per_class)
    print(f"Dataset ready at {root}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
