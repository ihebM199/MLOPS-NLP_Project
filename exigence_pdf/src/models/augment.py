"""Batch-level augmentation: MixUp and CutMix (train only).

Applied on the GPU batch right before the forward pass, which keeps the
DataLoader transform pipeline untouched (valid/test stay deterministic).
"""

from __future__ import annotations

import random
from typing import Tuple

import torch


def rand_bbox(
    width: int, height: int, lam: float, generator: torch.Generator | None = None
) -> Tuple[int, int, int, int]:
    """Random box for CutMix."""
    cut_ratio = (1.0 - lam) ** 0.5
    cut_w, cut_h = int(width * cut_ratio), int(height * cut_ratio)
    cx = int(torch.randint(width, (1,), generator=generator).item())
    cy = int(torch.randint(height, (1,), generator=generator).item())
    x1, y1 = max(cx - cut_w // 2, 0), max(cy - cut_h // 2, 0)
    x2, y2 = min(cx + cut_w // 2, width), min(cy + cut_h // 2, height)
    return x1, y1, x2, y2


def mixup_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.2,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """MixUp: convex combination of two images and their labels."""
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item()) if alpha > 0 else 1.0
    lam = max(lam, 1.0 - lam)  # keep the dominant label first
    index = torch.randperm(images.size(0), device=images.device, generator=generator)
    mixed = lam * images + (1.0 - lam) * images[index]
    return mixed, targets, targets[index], lam


def cutmix_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """CutMix: paste a random rectangle of another image."""
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item()) if alpha > 0 else 1.0
    index = torch.randperm(images.size(0), device=images.device, generator=generator)
    x1, y1, x2, y2 = rand_bbox(images.size(3), images.size(2), lam, generator)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (images.size(2) * images.size(3)))
    return mixed, targets, targets[index], lam


def apply_mix(
    images: torch.Tensor,
    targets: torch.Tensor,
    *,
    probability: float = 0.5,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Randomly apply MixUp or CutMix to a batch.

    Returns ``(images, target_a, target_b, lam)``; when no mixing happens
    ``target_a == target_b`` and ``lam == 1.0`` so the loss stays unchanged.
    """
    if probability <= 0 or random.random() > probability:
        return images, targets, targets, 1.0
    if random.random() < 0.5:
        return mixup_batch(images, targets, mixup_alpha, generator)
    return cutmix_batch(images, targets, cutmix_alpha, generator)


def mix_criterion(criterion, logits, target_a, target_b, lam: float):
    """``lam * L(a) + (1 - lam) * L(b)``."""
    if lam >= 1.0:
        return criterion(logits, target_a)
    return lam * criterion(logits, target_a) + (1.0 - lam) * criterion(logits, target_b)
