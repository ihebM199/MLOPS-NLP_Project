"""Matplotlib figures (headless-safe, used by the notebook and by reports)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # Colab / Docker: never try to open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.models.metrics import confusion  # noqa: E402

REAL_COLOR = "#2b5c8f"
FAKE_COLOR = "#d95f02"
PALETTE = {"real": REAL_COLOR, "fake": FAKE_COLOR, "REAL": REAL_COLOR, "FAKE": FAKE_COLOR}

plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.25})


def _save(fig, path: Optional[str | Path]) -> Optional[str]:
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return str(path)


def plot_training_history(history: Mapping[str, Sequence[float]], path: Optional[str | Path] = None):
    """2x2 grid: loss / accuracy / F1 / balanced accuracy (train vs validation)."""
    epochs = list(history.get("epoch", range(1, len(history.get("loss", [])) + 1)))
    panels = [
        ("loss", "val_loss", "Loss"),
        ("acc", "val_accuracy", "Accuracy"),
        ("f1", "val_f1", "F1 score"),
        ("bal_acc", "val_balanced_accuracy", "Balanced accuracy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    val_f1 = list(history.get("val_f1", []))
    best_epoch = int(np.argmax(val_f1)) + 1 if val_f1 else None
    for ax, (train_key, val_key, title) in zip(axes.flat, panels):
        if train_key in history:
            ax.plot(epochs, history[train_key], "o-", label="Train")
        if val_key in history:
            ax.plot(epochs, history[val_key], "s-", label="Validation")
        if best_epoch:
            ax.axvline(best_epoch, color="gray", ls=":", label=f"Best epoch ({best_epoch})")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.suptitle("Training history", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_evaluation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
    title: str = "Evaluation",
    path: Optional[str | Path] = None,
    eer: Optional[float] = None,
):
    """Confusion matrix (counts + per-class), ROC and score distribution."""
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    matrix = confusion(y_true, y_pred)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    for ax, data, fmt, subtitle in (
        (axes[0], matrix, "d", "Confusion matrix (counts)"),
        (axes[1], matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1), ".2%", "Per actual class"),
    ):
        image = ax.imshow(data, cmap="Blues")
        ax.set_xticks([0, 1], labels=["REAL", "FAKE"])
        ax.set_yticks([0, 1], labels=["REAL", "FAKE"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(subtitle)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                value = matrix[i, j] if fmt == "d" else data[i, j]
                text = f"{value:d}" if fmt == "d" else f"{value:.2%}"
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=11)
        fig.colorbar(image, ax=ax, fraction=0.046)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[2].plot(fpr, tpr, lw=2.4, color=FAKE_COLOR, label="ROC")
    axes[2].plot([0, 1], [0, 1], "--", color=REAL_COLOR, lw=1.1)
    if eer is not None:
        axes[2].plot(eer, 1 - eer, "ro", label=f"EER = {eer:.4f}")
    axes[2].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
    axes[2].legend(loc="lower right")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_score_distribution(
    y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float = 0.5, path: Optional[str | Path] = None
):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for label, name in ((0, "REAL"), (1, "FAKE")):
        ax.hist(y_prob[y_true == label], bins=40, range=(0, 1), alpha=0.65,
                color=PALETTE[name], label=name)
    ax.axvline(threshold, color="black", ls="--", lw=1, label=f"threshold={threshold:.2f}")
    ax.set(title="Score distribution P(fake)", xlabel="P(fake)", ylabel="Images", yscale="log")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_class_balance(df, path: Optional[str | Path] = None):
    counts = df.groupby(["split", "class"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    counts.plot(kind="bar", ax=ax, color=[REAL_COLOR, FAKE_COLOR])
    ax.set_title("Class balance per split")
    ax.set_xlabel("")
    ax.set_ylabel("Images")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_samples(paths_by_class: Mapping[str, Sequence[str]], path: Optional[str | Path] = None):
    from PIL import Image

    classes = list(paths_by_class)
    n = max((len(v) for v in paths_by_class.values()), default=0)
    fig, axes = plt.subplots(len(classes), n, figsize=(2.2 * n, 2.4 * len(classes)), squeeze=False)
    for row, cls in enumerate(classes):
        for col in range(n):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(paths_by_class[cls]):
                with Image.open(paths_by_class[cls][col]) as img:
                    ax.imshow(img.convert("RGB"))
                ax.set_title(cls.upper(), color=PALETTE.get(cls, "black"), fontsize=9)
    fig.suptitle("Random training samples", fontweight="bold")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_mean_spectra(spectra: Mapping[str, np.ndarray], path: Optional[str | Path] = None):
    names = list(spectra)
    columns = len(names) + (1 if len(names) == 2 else 0)
    fig, axes = plt.subplots(1, columns, figsize=(4.6 * columns, 4.2), squeeze=False)
    for ax, name in zip(axes[0], names):
        image = ax.imshow(spectra[name], cmap="magma")
        ax.set_title(f"Mean spectrum — {name.upper()}")
        ax.axis("off")
        fig.colorbar(image, ax=ax, fraction=0.046)
    if len(names) == 2:
        diff = spectra[names[1]] - spectra[names[0]]
        image = axes[0][-1].imshow(diff, cmap="coolwarm")
        axes[0][-1].set_title("Difference (fake − real)")
        axes[0][-1].axis("off")
        fig.colorbar(image, ax=axes[0][-1], fraction=0.046)
    fig.suptitle("log(1+|FFT|) averaged over images", fontsize=12)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_fusion_weights(
    y_true: np.ndarray, w_spatial: np.ndarray, w_frequency: np.ndarray, path: Optional[str | Path] = None
):
    y_true = np.asarray(y_true).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    for ax, values, title in (
        (axes[0], w_spatial, "Spatial weight"),
        (axes[1], w_frequency, "Frequency weight"),
    ):
        for label, name in ((0, "REAL"), (1, "FAKE")):
            ax.hist(np.asarray(values)[y_true == label], bins=30, alpha=0.6,
                    color=PALETTE[name], label=name)
        ax.set_title(title)
        ax.legend()
    for label, name in ((0, "REAL"), (1, "FAKE")):
        mask = y_true == label
        axes[2].scatter(np.asarray(w_spatial)[mask], np.asarray(w_frequency)[mask],
                        s=12, alpha=0.45, color=PALETTE[name], label=name)
    axes[2].set(title="Dynamic fusion weights", xlabel="w_spatial", ylabel="w_frequency")
    axes[2].legend()
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_gradcam_grid(rows: Sequence[Mapping[str, Any]], path: Optional[str | Path] = None):
    """``rows``: list of ``{"image": HxWx3, "heatmap": HxW, "overlay": HxWx3, "title": str}``."""
    n_cols = 3
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(10, 3.2 * len(rows)), squeeze=False)
    for row, item in zip(axes, rows):
        for ax, key, subtitle in zip(row, ("image", "heatmap", "overlay"),
                                     ("Input", "Grad-CAM", "Overlay")):
            ax.imshow(item[key], cmap="jet" if key == "heatmap" else None)
            ax.set_title(item.get("title", "") if key == "image" else subtitle, fontsize=9)
            ax.axis("off")
    fig.suptitle("Grad-CAM — red = most influential regions", fontweight="bold")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_benchmark(table, path: Optional[str | Path] = None, x: str = "variant", y: str = "latency_ms_p50"):
    fig, ax = plt.subplots(figsize=(9, 4))
    values = getattr(table, "columns", [])
    if x in values and y in values:
        ax.bar(table[x].astype(str), table[y], color=REAL_COLOR)
        ax.set_ylabel(y)
        ax.set_title("Latency comparison (CPU, batch 1)")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, path)
    return fig
