"""JSON / text helpers that never crash on NumPy or Path objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def json_default(obj: Any) -> Any:
    """``json.dumps(default=...)`` for Path, NumPy scalars/arrays, dataclasses."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


def sanitize(obj: Any) -> Any:
    """Replace NaN/±inf by ``None`` so the output is valid JSON (NaN is not)."""
    import math

    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return sanitize(float(obj))
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    return obj


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            sanitize(payload), indent=indent, default=json_default, ensure_ascii=False, allow_nan=False
        ),
        "utf-8",
    )
    return path


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, "utf-8")
    return path


def flatten(mapping: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts (MLflow params must be flat scalars)."""
    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, prefix=f"{name}."))
        elif isinstance(value, (list, tuple)):
            flat[name] = ",".join(map(str, value))
        elif value is None:
            flat[name] = "None"
        else:
            flat[name] = value
    return flat


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def first_existing(paths: Iterable[str | Path]) -> Path | None:
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    return None
