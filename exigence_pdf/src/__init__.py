"""Deepfake Adaptive Fusion — industrial MLOps package.

The package is intentionally import-light: heavy dependencies (torch, timm,
mlflow, onnxruntime) are imported lazily inside the modules that need them so
that the FastAPI backend can start (and be unit-tested) without them.
"""

from __future__ import annotations

import sys

__version__ = "1.0.0"
__all__ = ["__version__", "IN_COLAB"]

IN_COLAB: bool = "google.colab" in sys.modules
