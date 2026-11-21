"""Anti-leakage guarantees (no dataset required: tiny synthetic files)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.dataset import build_dataframe, split_paths, subset_per_class
from src.data.leakage import (
    DataLeakageError,
    assert_no_leakage,
    dedupe_by_hash,
    find_cross_split_duplicates,
    group_key,
    group_split,
    md5_of_file,
)


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_identical_content_across_splits_is_detected(tmp_path: Path) -> None:
    train = _write(tmp_path / "train" / "real" / "a.jpg", b"same-bytes")
    test = _write(tmp_path / "test" / "fake" / "b.jpg", b"same-bytes")
    other = _write(tmp_path / "test" / "real" / "c.jpg", b"other-bytes")

    report = find_cross_split_duplicates({"train": [train], "test": [test, other]})
    assert report["n_duplicates"] == 1
    assert report["n_files"] == 3

    with pytest.raises(DataLeakageError):
        assert_no_leakage({"train": [train], "test": [test]}, strict=True)


def test_distinct_content_is_clean(tmp_path: Path) -> None:
    train = _write(tmp_path / "train" / "real" / "a.jpg", b"aaa")
    test = _write(tmp_path / "test" / "real" / "b.jpg", b"bbb")
    report = assert_no_leakage({"train": [train], "test": [test]})
    assert report["n_duplicates"] == 0


def test_dedupe_keeps_one_copy_per_content(tmp_path: Path) -> None:
    first = _write(tmp_path / "a.jpg", b"dup")
    second = _write(tmp_path / "b.jpg", b"dup")
    third = _write(tmp_path / "c.jpg", b"unique")
    kept = dedupe_by_hash([first, second, third])
    assert sorted(Path(p).name for p in kept) == ["a.jpg", "c.jpg"]


def test_subset_per_class_is_deterministic_and_stratified(tmp_path: Path) -> None:
    for split in ("train", "test"):
        for cls in ("real", "fake"):
            for index in range(6):
                _write(tmp_path / split / cls / f"{index}.jpg", f"{split}-{cls}-{index}".encode())

    frame = build_dataframe(tmp_path)
    first = subset_per_class(frame, "train", 3, seed=7)
    second = subset_per_class(frame, "train", 3, seed=7)
    assert first["path"].tolist() == second["path"].tolist()
    assert first.groupby("label").size().tolist() == [3, 3]

    paths = split_paths(frame)
    assert set(paths) == {"train", "test"}
    assert all(len(items) == 12 for items in paths.values())


def test_group_split_keeps_frames_of_one_video_together() -> None:
    rows = []
    for video in range(9):
        for frame in range(4):
            rows.append(
                {
                    "path": f"/data/video_{video}/frame_{frame}.jpg",
                    "label": video % 2,
                }
            )
    frame = pd.DataFrame(rows)
    frame["group"] = frame["path"].map(group_key)
    assigned = group_split(frame, seed=3)

    per_group = assigned.groupby("group")["split"].nunique()
    assert (per_group == 1).all(), "a video was split across several subsets"


def test_md5_is_content_based(tmp_path: Path) -> None:
    one = _write(tmp_path / "one.bin", b"payload")
    two = _write(tmp_path / "two.bin", b"payload")
    assert md5_of_file(one) == md5_of_file(two)
