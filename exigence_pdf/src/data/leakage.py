"""Anti data-leakage toolkit.

Two independent risks are covered:

1. **Identical files** in two splits (very common after re-downloading or manual
   copying) -> detected with an MD5 digest of the *content*, not the path.
2. **Identity / video leakage** — frames of the same video (or the same person)
   must all live in the *same* split. ``group_split`` implements a group-aware
   split for external video datasets (Celeb-DF v2, FaceForensics++ c23).

``python -m src.data.leakage --dataset-root data/processed/...`` prints a JSON
report and exits with code 2 when leakage is detected (usable in CI).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from src.utils.logging import get_logger

LOGGER = get_logger("data.leakage")


class DataLeakageError(RuntimeError):
    """Raised when the same image content appears in two different splits."""


def md5_of_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_paths(paths: Iterable[str | Path], *, limit: int | None = None) -> dict[str, str]:
    """``{path: md5}`` (``limit`` keeps the cost bounded for huge splits)."""
    result: dict[str, str] = {}
    selected = list(paths)
    if limit is not None:
        selected = selected[:limit]
    for path in selected:
        try:
            result[str(path)] = md5_of_file(path)
        except OSError as exc:
            LOGGER.warning("Cannot hash %s (%s) — skipped", path, exc)
    return result


def find_cross_split_duplicates(
    split_paths: Mapping[str, Sequence[str]], *, limit: int | None = None
) -> dict:
    """Return ``{"hashes": [...], "pairs": [...], "n_duplicates": int}``."""
    hashes: dict[str, dict[str, list[str]]] = {}
    for split, paths in split_paths.items():
        for path, digest in hash_paths(paths, limit=limit).items():
            hashes.setdefault(digest, {}).setdefault(split, []).append(path)

    duplicate_pairs = []
    for digest, per_split in hashes.items():
        if len(per_split) > 1:
            duplicate_pairs.append(
                {"md5": digest, "splits": {s: sorted(p)[:3] for s, p in per_split.items()}}
            )
    return {
        "n_files": sum(len(v) for v in split_paths.values()),
        "n_hashes": len(hashes),
        "n_duplicates": len(duplicate_pairs),
        "duplicates": duplicate_pairs[:50],
    }


def assert_no_leakage(
    split_paths: Mapping[str, Sequence[str]], *, limit: int | None = None, strict: bool = True
) -> dict:
    """Raise :class:`DataLeakageError` if a file appears in two splits."""
    report = find_cross_split_duplicates(split_paths, limit=limit)
    if report["n_duplicates"] and strict:
        sample = json.dumps(report["duplicates"][:3], indent=2, ensure_ascii=False)
        raise DataLeakageError(
            f"{report['n_duplicates']} file(s) shared between splits — data leakage!\n{sample}"
        )
    return report


def dedupe_by_hash(paths: Sequence[str], *, keep: str = "first") -> list[str]:
    """Drop duplicate contents inside one split (keeps the lexicographically first path)."""
    seen: set[str] = set()
    kept: list[str] = []
    for path in sorted(paths) if keep == "first" else paths:
        try:
            digest = md5_of_file(path)
        except OSError:
            continue
        if digest in seen:
            continue
        seen.add(digest)
        kept.append(path)
    return kept


def group_key(path: str | Path) -> str:
    """Group id of an image.

    Video frames are stored as ``<dataset>/<subset>/<video_id>/frame.jpg``: the
    parent directory is the video. A flat image file falls back to its own stem.
    """
    path = Path(path)
    return path.parent.name or path.stem


def group_split(
    frame: pd.DataFrame,
    *,
    valid_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    group_column: str = "group",
) -> pd.DataFrame:
    """Assign whole groups (videos/identities) to train/valid/test.

    Prevents the classic video-deepfake leak where two frames of the same video
    end up on both sides of the split.
    """
    import random

    if frame.empty:
        return frame.assign(split=[])
    groups = sorted(frame[group_column].astype(str).unique())
    rnd = random.Random(seed)
    rnd.shuffle(groups)
    n = len(groups)
    n_test = max(1, int(round(n * test_fraction))) if n >= 3 else 0
    n_valid = max(1, int(round(n * valid_fraction))) if n >= 3 else 0
    assignment: dict[str, str] = {}
    for i, group in enumerate(groups):
        if i < n_test:
            assignment[group] = "test"
        elif i < n_test + n_valid:
            assignment[group] = "valid"
        else:
            assignment[group] = "train"
    out = frame.copy()
    out["split"] = out[group_column].astype(str).map(assignment)
    return out


def leakage_report_for_dataframe(df: pd.DataFrame, *, limit: int | None = 4000) -> dict:
    """Split-aware report (``limit`` hashes per split keeps it fast)."""
    paths = {split: part["path"].tolist()[:limit] for split, part in df.groupby("split")}
    return assert_no_leakage(paths, strict=False) if paths else {"n_duplicates": 0, "n_files": 0}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect cross-split duplicates (MD5).")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--limit", type=int, default=4000, help="max images hashed per split")
    parser.add_argument("--json-out", default="reports/leakage_report.json")
    args = parser.parse_args(argv)

    from src.data.dataset import build_dataframe

    df = build_dataframe(args.dataset_root)
    if df.empty:
        LOGGER.error("No images found under %s", args.dataset_root)
        return 1
    report = leakage_report_for_dataframe(df, limit=args.limit)
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(report, indent=2), "utf-8")
    LOGGER.info(
        "Hashed %s files | %s cross-split duplicate(s) -> %s",
        report.get("n_files"),
        report.get("n_duplicates"),
        args.json_out,
    )
    return 2 if report.get("n_duplicates") else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
