"""``FaceDataset`` + DataFrame indexing + DataLoader factory.

Label convention (identical to the pre-trained weights): **real = 0, fake = 1**.
``torchvision.datasets.ImageFolder`` sorts folders alphabetically and would map
``fake = 0``; that is why the mapping is explicit here and never inferred.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.utils.logging import get_logger

LOGGER = get_logger("data.dataset")

LABEL2ID: Dict[str, int] = {"real": 0, "fake": 1}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}
SPLITS: tuple[str, ...] = ("train", "valid", "test")
IMG_EXT: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def build_dataframe(root: str | Path, splits: Sequence[str] = SPLITS) -> pd.DataFrame:
    """Index ``root/<split>/<class>/*.jpg`` into a tidy DataFrame."""
    root = Path(root)
    rows: list[dict[str, Any]] = []
    for split in splits:
        for cls, label in LABEL2ID.items():
            folder = root / split / cls
            if not folder.is_dir():
                continue
            rows += [
                {"split": split, "class": cls, "label": label, "path": str(p)}
                for p in sorted(folder.iterdir())
                if p.suffix.lower() in IMG_EXT
            ]
    return pd.DataFrame(rows, columns=["split", "class", "label", "path"])


def subset_per_class(df: pd.DataFrame, split: str, limit: Optional[int], *, seed: int = 42) -> pd.DataFrame:
    """Deterministically keep at most ``limit`` images per class for ``split``."""
    part = df[df["split"] == split]
    if limit is None or part.empty:
        return part
    kept = [
        group.sample(n=min(len(group), limit), random_state=seed)
        for _, group in part.groupby("label", sort=True)
    ]
    return pd.concat(kept, ignore_index=True) if kept else part


def split_paths(df: pd.DataFrame) -> dict[str, list[str]]:
    """``{split: [paths]}`` — used by the anti-leakage test."""
    return {split: part["path"].tolist() for split, part in df.groupby("split")}


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class FaceDataset(Dataset):
    """Binary dataset (real=0 / fake=1) built from a DataFrame.

    An unreadable image is **not** silently swallowed: it is replaced by a black
    image (a long training run must never crash) and counted, so the caller can
    report how much data was lost.
    """

    def __init__(self, df: pd.DataFrame, transform: Any, *, image_size: int = 224):
        self.paths: list[str] = df["path"].tolist()
        self.labels: list[int] = df["label"].astype(int).tolist()
        self.transform = transform
        self.image_size = image_size
        self.corrupted = 0

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        try:
            with Image.open(self.paths[idx]) as handle:
                img = handle.convert("RGB")
        except Exception as exc:  # noqa: BLE001 - any decode error is handled the same way
            self.corrupted += 1
            if self.corrupted == 1 or self.corrupted % 100 == 0:
                LOGGER.warning(
                    "Corrupted image #%d (%s): %s -> replaced by a black image",
                    self.corrupted,
                    Path(self.paths[idx]).name,
                    exc.__class__.__name__,
                )
            img = Image.new("RGB", (self.image_size, self.image_size))
        return self.transform(img), self.labels[idx]

    def report(self) -> dict[str, int]:
        return {"n_images": len(self), "corrupted": self.corrupted}


def make_loader(
    df: pd.DataFrame,
    split: str,
    transform: Any,
    *,
    batch_size: int = 32,
    num_workers: int = 2,
    shuffle: bool = False,
    image_size: int = 224,
    drop_last: bool = False,
    pin_memory: bool = True,
    seed: int = 42,
) -> Optional[DataLoader]:
    """Build a DataLoader for ``split`` (``None`` when the split is empty)."""
    subset = df[df["split"] == split]
    if subset.empty:
        return None
    dataset = FaceDataset(subset, transform, image_size=image_size)
    generator = None
    if shuffle:
        import torch

        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Images per (split, class) — used by the EDA and the leakage report."""
    if df.empty:
        return pd.DataFrame(columns=["real", "fake"])
    return df.groupby(["split", "class"]).size().unstack(fill_value=0).reindex(list(SPLITS))


def iter_paths(df: pd.DataFrame, split: str | None = None) -> Iterable[str]:
    part = df if split is None else df[df["split"] == split]
    return part["path"].tolist()
