"""MLflow tracking + Model Registry helpers.

Works in the three supported environments:

* **Colab** — SQLite backend store (``sqlite:///mlruns/mlflow.db``). The model
  registry is fully functional, no server required.
* **Docker** — ``MLFLOW_TRACKING_URI=http://mlflow:5000`` (compose service).
* **Local** — same as Colab by default.

``mlflow`` is imported lazily so the backend and the tests never depend on it.
"""

from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from src.utils.io import flatten
from src.utils.logging import get_logger

LOGGER = get_logger("tracking")


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def git_sha() -> str:
    """Short commit SHA (``unknown`` outside a git checkout)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def is_available() -> bool:
    try:
        import mlflow  # noqa: F401

        return True
    except Exception:
        return False


def _configured(cfg) -> bool:
    """Return True when tracking is enabled *and* usable."""
    if cfg is not None and not getattr(cfg.tracking, "enabled", True):
        return False
    if not is_available():
        if cfg is not None:
            LOGGER.warning("mlflow is not installed — tracking disabled (pip install mlflow).")
        return False
    return True


def configure(cfg) -> bool:
    """Point MLflow to the right store/experiment. Safe to call repeatedly."""
    if not _configured(cfg):
        return False
    import mlflow

    uri = cfg.tracking_uri()
    mlflow.set_tracking_uri(uri)
    if uri.startswith("sqlite"):
        Path(uri.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME") or cfg.tracking.experiment)
    LOGGER.info("MLflow tracking uri = %s", uri)
    return True


@contextmanager
def run(
    cfg,
    run_name: str,
    *,
    tags: Optional[Dict[str, Any]] = None,
    nested: bool = False,
) -> Iterator[Any]:
    """Context manager yielding the active MLflow run (or ``None`` when disabled)."""
    if not configure(cfg):
        yield None
        return
    import mlflow

    base_tags = {
        "git_sha": git_sha(),
        "git_branch": git_branch(),
        "author": os.environ.get("GIT_AUTHOR_NAME", os.environ.get("USERNAME", "unknown")),
        "axis": "axe1-cnn-transfer-learning-gradcam",
        **{k: str(v) for k, v in (tags or {}).items()},
    }
    with mlflow.start_run(run_name=run_name, tags=base_tags, nested=nested):
        LOGGER.info("MLflow run '%s' started (id=%s)", run_name, mlflow.active_run().info.run_id)
        yield mlflow


# --------------------------------------------------------------------------- #
# Logging primitives
# --------------------------------------------------------------------------- #
def log_params(cfg, params: Dict[str, Any]) -> None:
    if not _configured(cfg):
        return
    import mlflow

    for key, value in flatten(params).items():
        try:
            mlflow.log_param(key, value)
        except Exception:  # param already logged / unsupported type
            continue


def log_metrics(cfg, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    if not _configured(cfg):
        return
    import mlflow

    flat = flatten(metrics)
    numeric = {
        k: float(v)
        for k, v in flat.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) == float(v)
    }
    if numeric:
        mlflow.log_metrics(numeric, step=step)


def log_artifact(cfg, path: str | Path, artifact_path: Optional[str] = None) -> None:
    if not _configured(cfg) or not Path(path).exists():
        return
    import mlflow

    try:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    except Exception as exc:  # noqa: BLE001 - tracking must never break a run
        LOGGER.warning("Could not log artifact %s (%s)", path, exc)


def log_artifacts(cfg, path: str | Path, artifact_path: Optional[str] = None) -> None:
    if not _configured(cfg) or not Path(path).exists():
        return
    import mlflow

    try:
        mlflow.log_artifacts(str(path), artifact_path=artifact_path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not log artifacts %s (%s)", path, exc)


def log_dict(cfg, payload: Dict[str, Any], filename: str) -> None:
    if not _configured(cfg):
        return
    import mlflow

    try:
        mlflow.log_dict(payload, filename)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not log dict %s (%s)", filename, exc)


def log_text(cfg, text: str, filename: str) -> None:
    if not _configured(cfg):
        return
    import mlflow

    try:
        mlflow.log_text(text, filename)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not log text %s (%s)", filename, exc)


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
def register_onnx_model(
    cfg,
    onnx_path: str | Path,
    *,
    alias: str = "staging",
    description: str = "Adaptive Fusion Detector (EfficientNetV2 + FFT gate)",
) -> Optional[str]:
    """Log the ONNX model and register it, then move the alias.

    Uses the *alias* API (MLflow >= 2.9) and falls back to the legacy stages
    (``Staging`` / ``Production``) on older servers.
    """
    if not _configured(cfg) or not Path(onnx_path).exists():
        return None
    import mlflow
    from mlflow.tracking import MlflowClient

    name = cfg.tracking.registered_model
    try:
        mlflow.onnx.log_model(
            onnx_model=str(onnx_path),
            artifact_path="onnx_model",
            registered_model_name=name,
        )
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{name}'")
        if not versions:
            return None
        version = max(versions, key=lambda v: int(v.version)).version
        try:
            client.set_registered_model_alias(name, alias, version)
        except Exception:  # older server without aliases
            stage = {"staging": "Staging", "production": "Production"}.get(alias, "Staging")
            client.transition_model_version_stage(name=name, version=version, stage=stage)
        client.update_model_version(
            name=name,
            version=version,
            description=description,
        )
        LOGGER.info("Registered %s v%s with alias '%s'", name, version, alias)
        return str(version)
    except Exception as exc:  # noqa: BLE001 - registry is best-effort
        LOGGER.warning("Model registry unavailable (%s). Artifacts were still logged.", exc)
        return None


def promote(cfg, version: str, alias: str = "production") -> None:
    """Move an existing version to the ``production`` alias."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(cfg.tracking_uri())
    client = MlflowClient()
    name = cfg.tracking.registered_model
    try:
        client.set_registered_model_alias(name, alias, version)
    except Exception:
        client.transition_model_version_stage(
            name=name, version=version, stage="Production" if alias == "production" else "Staging"
        )
    LOGGER.info("Model %s v%s -> alias '%s'", name, version, alias)
