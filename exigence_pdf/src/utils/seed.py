"""Reproducibility helpers."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def seed_everything(seed: int = 42, *, deterministic: bool = False) -> int:
    """Seed ``random``, NumPy and PyTorch (CUDA included).

    ``deterministic=True`` also forces cuDNN into deterministic mode: slower,
    but required for strictly reproducible runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    except Exception:  # torch absent (backend-only environments) -> nothing to seed
        pass
    return seed


def seed_worker(worker_id: int) -> None:  # pragma: no cover - called inside workers
    """``DataLoader(worker_init_fn=seed_worker)`` hook (keeps workers reproducible)."""
    worker_seed = (np.uint32(os.getpid()) + np.uint32(worker_id)) % 2**32
    np.random.seed(int(worker_seed))
    random.seed(int(worker_seed))


def rng(seed: int) -> random.Random:
    """Deterministic Python RNG (used for sampling without touching globals)."""
    return random.Random(seed)


def describe_environment() -> dict[str, Any]:
    """Versions + hardware, logged to MLflow and written to the model manifest."""
    import platform

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        info["torch"] = None
    try:
        import timm

        info["timm"] = timm.__version__
    except Exception:
        info["timm"] = None
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["onnxruntime_providers"] = ort.get_available_providers()
    except Exception:
        info["onnxruntime"] = None
    return info
