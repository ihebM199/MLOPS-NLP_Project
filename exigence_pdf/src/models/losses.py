"""Losses: BCE with label smoothing (reference) and Focal Loss (ablation).

Both take **logits** (not probabilities) to stay numerically stable with AMP.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_targets(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    """``0 -> smoothing``, ``1 -> 1 - smoothing`` (targets must be float)."""
    if smoothing <= 0:
        return targets
    return targets * (1 - 2 * smoothing) + smoothing


class BinaryFocalLoss(nn.Module):
    """Focal loss for binary classification (Lin et al., 2017).

    ``alpha`` down-weights the majority class, ``gamma`` focuses on hard
    examples. With balanced classes the expected gain is small — this is exactly
    what the ablation run measures and reports.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = smooth_targets(targets.float(), self.smoothing)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce
        return loss.mean()


def build_criterion(
    *, focal: bool = False, gamma: float = 2.0, alpha: float = 0.25, smoothing: float = 0.0
) -> nn.Module:
    """Criterion factory shared by train.py and evaluate.py."""
    if focal:
        return BinaryFocalLoss(gamma=gamma, alpha=alpha, smoothing=smoothing)
    if smoothing > 0:
        return LabelSmoothingBCE(smoothing=smoothing)
    return nn.BCEWithLogitsLoss()


class LabelSmoothingBCE(nn.Module):
    """``BCEWithLogitsLoss`` on smoothed targets."""

    def __init__(self, smoothing: float = 0.05) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, smooth_targets(targets.float(), self.smoothing))


def criterion_for(run_cfg: Optional[object]) -> nn.Module:
    """Build the criterion straight from ``TrainParams``.

    Label smoothing is **not** baked into the criterion here: the training loop
    smooths ``target_a``/``target_b`` itself so that MixUp/CutMix and smoothing
    compose correctly (and so validation loss stays on the true labels).
    """
    if run_cfg is None:
        return nn.BCEWithLogitsLoss()
    return build_criterion(
        focal=getattr(run_cfg, "focal_loss", False),
        gamma=getattr(run_cfg, "focal_gamma", 2.0),
        alpha=getattr(run_cfg, "focal_alpha", 0.25),
        smoothing=0.0,
    )
