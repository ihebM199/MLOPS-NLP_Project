"""Metrics shared by training, evaluation and the robustness study.

All functions take 0/1 labels (**real=0, fake=1**) and P(fake) scores.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Equal Error Rate: the point where FPR == FNR (lower is better)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        y_prob = np.asarray(y_prob, dtype=float)
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["eer"] = calculate_eer(y_true, y_prob)
    return metrics


def threshold_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Metrics obtained by thresholding P(fake) at ``threshold``."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    result = compute_metrics(y_true, y_pred, y_prob)
    result["threshold"] = float(threshold)
    return result


def calibrate_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, *, min_recall: Optional[float] = None
) -> Dict[str, Any]:
    """Pick the decision threshold that maximises F1 on validation data.

    Deepfake detectors are usually tuned to maximise recall (detect as many
    fakes as possible) while keeping a usable false-positive rate; pass
    ``min_recall`` to enforce that constraint.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    if min_recall is not None:
        allowed = recall >= min_recall
        if allowed.any():
            f1 = np.where(allowed, f1, -1.0)
    best = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    threshold = float(thresholds[best]) if len(thresholds) else 0.5
    at_threshold = threshold_metrics(y_true, y_prob, threshold)
    return {
        "threshold": threshold,
        "f1": at_threshold["f1"],
        "precision": at_threshold["precision"],
        "recall": at_threshold["recall"],
        "curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
        },
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """2x2 confusion matrix with the fixed class order ``[real, fake]``."""
    return confusion_matrix(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int), labels=[0, 1])


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = {}
    for label, name in ((0, "real"), (1, "fake")):
        mask = y_true == label
        out[f"{name}_accuracy"] = float((y_pred[mask] == label).mean()) if mask.any() else float("nan")
    return out


def bootstrap_ci(
    y_true: np.ndarray, y_prob: np.ndarray, *, metric: str = "accuracy", n_boot: int = 500, seed: int = 0
) -> Dict[str, float]:
    """95 % confidence interval of a metric (useful for small external sets)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample_true, sample_prob = y_true[idx], y_prob[idx]
        if len(np.unique(sample_true)) < 2:
            continue
        pred = (sample_prob >= 0.5).astype(int)
        if metric == "accuracy":
            scores.append(accuracy_score(sample_true, pred))
        elif metric == "f1":
            scores.append(f1_score(sample_true, pred, zero_division=0))
        elif metric == "auc":
            scores.append(roc_auc_score(sample_true, sample_prob))
    if not scores:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan")}
    return {
        "mean": float(np.mean(scores)),
        "low": float(np.percentile(scores, 2.5)),
        "high": float(np.percentile(scores, 97.5)),
    }
