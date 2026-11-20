"""Device selection shared by training, evaluation and the API."""

from __future__ import annotations

import os
from typing import Any

from src.utils.logging import get_logger

LOGGER = get_logger("utils.device")


def resolve_device(spec: str | None = None) -> Any:
    """``auto`` (default) / ``cuda`` / ``cpu`` / ``cuda:1`` -> ``torch.device``."""
    import torch

    spec = (spec or os.environ.get("TORCH_DEVICE") or "auto").lower()
    if spec == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(spec)
        if device.type == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but unavailable — falling back to CPU.")
            device = torch.device("cpu")
    return device


def describe_device(device: Any) -> str:
    import torch

    if getattr(device, "type", "cpu") == "cuda" and torch.cuda.is_available():
        return f"{device} ({torch.cuda.get_device_name(0)})"
    return f"{device}"


def amp_enabled(device: Any, requested: bool) -> bool:
    """AMP is only meaningful on CUDA (bf16 CPU support is too slow/variable)."""
    return bool(requested and getattr(device, "type", "cpu") == "cuda")
