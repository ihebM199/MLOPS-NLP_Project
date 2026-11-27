"""Backend configuration — port of ``backend/app/settings.py`` (12-factor).

The original used pydantic-settings; the behaviour is identical here: every
value comes from the environment first (see ``.env`` / ``.env.example``),
then from the project defaults. No Django settings are involved, so the
inference services stay framework-agnostic.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from src.utils.paths import MODEL_DIR, PROJECT_ROOT, REPORTS_DIR


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _absolute(value: Path) -> Path:
    return value if value.is_absolute() else (PROJECT_ROOT / value)


class Settings:
    """Settings read from environment variables / ``.env``."""

    def __init__(self) -> None:
        # .env support without an extra dependency (pydantic-settings in the
        # original): plain KEY=VALUE lines, quotes optional.
        self._load_dotenv(PROJECT_ROOT / ".env")

        self.app_name = "Deepfake Adaptive Fusion API"
        self.app_version = "1.0.0"

        # --- artefacts -------------------------------------------------------
        self.model_path = _absolute(Path(_env("MODEL_PATH", str(MODEL_DIR / "deepfake_fusion.onnx"))))
        self.pytorch_checkpoint_path = _absolute(
            Path(_env("PYTORCH_CHECKPOINT_PATH", str(MODEL_DIR / "AdaptiveFusion_best_model.pth")))
        )
        self.model_config_path = _absolute(
            Path(_env("MODEL_CONFIG_PATH", str(MODEL_DIR / "model_config.json")))
        )
        self.metrics_path = _absolute(Path(_env("METRICS_PATH", str(REPORTS_DIR / "metrics.json"))))
        # React app (npm run build) -> deepguard/frontend/dist/index.html. The
        # Vite dev server on :5173 proxies the API here; in the built
        # single-URL deployment Django serves the SPA itself.
        self.frontend_path = _absolute(
            Path(_env("FRONTEND_PATH", str(PROJECT_ROOT / "deepguard" / "frontend" / "dist" / "index.html")))
        )
        self.frontend_dist = self.frontend_path.parent
        self.media_dir = _absolute(
            Path(_env("MEDIA_DIR", str(PROJECT_ROOT / "deepguard" / "backend" / "media")))
        )
        self.monitoring_data_dir = _absolute(
            Path(_env("MONITORING_DATA_DIR", str(PROJECT_ROOT / "monitoring_data")))
        )

        # --- inference -------------------------------------------------------
        self.decision_threshold = _env_float("DECISION_THRESHOLD", 0.5)
        self.max_upload_mb = _env_float("MAX_UPLOAD_MB", 10.0)
        self.onnx_providers = _env("ONNX_PROVIDERS", "CPUExecutionProvider")
        self.enable_gradcam = _env_bool("ENABLE_GRADCAM", True)
        self.torch_device = _env("TORCH_DEVICE", "auto")

        # --- HTTP ------------------------------------------------------------
        self.cors_origins = _env(
            "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
        )
        self.log_level = _env("LOG_LEVEL", "INFO")
        self.model_version = _env("MODEL_VERSION", "1.0.0")

    @staticmethod
    def _load_dotenv(path: Path) -> None:
        if not path.exists():
            return
        try:
            for line in path.read_text("utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        except OSError:
            pass

    @property
    def cors_origin_list(self) -> List[str]:
        origins = [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
        return origins or ["*"]

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def provider_list(self) -> List[str]:
        return [p.strip() for p in self.onnx_providers.split(",") if p.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings (imported once by the views)."""
    return Settings()
