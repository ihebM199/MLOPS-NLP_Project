"""Cross-dataset (external) evaluation sets.

Design rule from the audit: extra datasets are **never mixed into training**.
They are only used to measure generalisation to unseen generators/sources, with
three safeguards:

* FFHQ is excluded: the 70k "real" images of *140k real-and-fake-faces* already
  come from FFHQ, so adding it again would leak train images into the test.
* Video datasets are grouped by **video** (``leakage.group_split``), never by
  frame, when a train/valid/test split is required.
* Every image is de-duplicated against the training hashes (MD5).

Expected layouts under ``data/external/<dataset>/`` (script-driven, see
``scripts/download_data.py``)::

    celeb-df-v2/Celeb-real/*.jpg        -> real
    celeb-df-v2/Celeb-synthesis/*.jpg   -> fake
    celeb-df-v2/YouTube-real/*.jpg      -> real
    faceforensics-c23/original/*.mp4    -> real   (frames extracted by the script)
    faceforensics-c23/Face2Face/*.mp4   -> fake
    celeba/img_align_celeba/*.jpg       -> real   (identity = filename prefix)
    lfw/*/*.jpg                         -> real
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from src.data.leakage import group_key, md5_of_file
from src.utils.logging import get_logger

LOGGER = get_logger("data.external")

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class ExternalSource:
    name: str
    root: Path
    real_dirs: tuple[str, ...] = ()
    fake_dirs: tuple[str, ...] = ()
    enabled: bool = True
    note: str = ""
    #: extra columns for reporting
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def available(self) -> bool:
        return self.requested_dirs and all((self.root / d).is_dir() for d in self.requested_dirs)

    @property
    def requested_dirs(self) -> tuple[str, ...]:
        return tuple(d for d in (*self.real_dirs, *self.fake_dirs) if (self.root / d).is_dir())


#: Default catalogue. FFHQ is deliberately absent (see module docstring).
DEFAULT_SOURCES: tuple[ExternalSource, ...] = (
    ExternalSource(
        "celeb-df-v2",
        root=Path("data/external/celeb-df-v2"),
        real_dirs=("Celeb-real", "YouTube-real"),
        fake_dirs=("Celeb-synthesis",),
        tags=("video", "identity", "generator:FaceSwap"),
    ),
    ExternalSource(
        "faceforensics-c23",
        root=Path("data/external/faceforensics-c23"),
        real_dirs=("original",),
        fake_dirs=("DeepFakeDetection", "DeepFakeDetection_original", "Face2Face", "FaceSwap", "NeuralTextures"),
        tags=("video", "identity"),
    ),
    ExternalSource(
        "celeba",
        root=Path("data/external/celeba"),
        real_dirs=("img_align_celeba",),
        tags=("identity", "excluded_from_training"),
    ),
    ExternalSource(
        "lfw",
        root=Path("data/external/lfw"),
        real_dirs=("lfw-deepfunneled", "lfw", "."),
        tags=("identity",),
    ),
)


def identity_from_filename(path: Path) -> str:
    """Identity id of an aligned face crop.

    * CelebA: ``000123.jpg`` -> the image index (one identity per crop is
      approximated by the numerical prefix — documented approximation).
    * LFW: ``Aaron_Peirsol_0001.jpg`` -> ``Aaron_Peirsol``.
    """
    stem = path.stem
    match = re.match(r"^([A-Za-z][A-Za-z\-']+)_\d+", stem)
    if match:
        return match.group(1)
    match = re.match(r"^(\d{3,})", stem)
    if match:
        return match.group(1)[:3]  # CelebA ids are 6 digits: first 3 ~ identity block
    return stem


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)


def scan_source(
    source: ExternalSource,
    *,
    max_per_class: int | None = 500,
    seed: int = 42,
    exclude_hashes: Iterable[str] = (),
) -> pd.DataFrame:
    """List (path, label, source, group) rows for one external dataset."""
    excluded = set(exclude_hashes)
    rnd = random.Random(seed)
    rows: list[dict] = []
    duplicates = 0
    for label, folders in ((0, source.real_dirs), (1, source.fake_dirs)):
        for folder_name in folders:
            folder = source.root / folder_name
            if not folder.is_dir():
                continue
            files = _list_images(folder)
            if max_per_class is not None and len(files) > max_per_class:
                files = rnd.sample(files, max_per_class)
            for path in files:
                try:
                    digest = md5_of_file(path)
                except OSError:
                    continue
                if digest in excluded:
                    duplicates += 1
                    continue
                excluded.add(digest)
                rows.append(
                    {
                        "path": str(path),
                        "label": label,
                        "source": source.name,
                        "group": group_key(path) if source.tags and "video" in source.tags else identity_from_filename(path),
                        "md5": digest,
                    }
                )
    frame = pd.DataFrame(rows, columns=["path", "label", "source", "group", "md5"])
    if duplicates:
        LOGGER.info("%s: %d image(s) dropped (already present in the training set)", source.name, duplicates)
    return frame


def build_external_manifest(
    sources: Sequence[ExternalSource] = DEFAULT_SOURCES,
    *,
    max_per_class: int | None = 500,
    seed: int = 42,
    exclude_hashes: Iterable[str] = (),
) -> pd.DataFrame:
    """Concatenate every available external source into one evaluation manifest."""
    frames = []
    for source in sources:
        if not source.enabled or not source.available:
            LOGGER.info("external source %-20s : not found at %s (skipped)", source.name, source.root)
            continue
        frame = scan_source(
            source, max_per_class=max_per_class, seed=seed, exclude_hashes=exclude_hashes
        )
        if not frame.empty:
            frame["tags"] = ",".join(source.tags)
            frames.append(frame)
            LOGGER.info(
                "external source %-20s : %5d images (real=%d, fake=%d)",
                source.name,
                len(frame),
                int((frame["label"] == 0).sum()),
                int((frame["label"] == 1).sum()),
            )
    if not frames:
        return pd.DataFrame(columns=["path", "label", "source", "group", "md5", "tags"])
    return pd.concat(frames, ignore_index=True)


def training_hashes(dataset_root: str | Path, *, limit: int | None = 5000, seed: int = 42) -> set[str]:
    """MD5 set of (a sample of) the training images, used to exclude duplicates."""
    from src.data.dataset import build_dataframe, subset_per_class

    df = build_dataframe(dataset_root)
    train = subset_per_class(df, "train", limit, seed=seed)
    hashes = set()
    for path in train["path"]:
        try:
            hashes.add(md5_of_file(path))
        except OSError:
            continue
    LOGGER.info("Computed %d training hashes for external de-duplication", len(hashes))
    return hashes


def manifest_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(["source", "label"]).size().unstack(fill_value=0)
        .rename(columns={0: "real", 1: "fake"})
        .assign(total=lambda d: d.sum(axis=1))
    )
