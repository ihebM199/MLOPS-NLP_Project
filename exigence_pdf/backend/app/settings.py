"""Backend configuration (12-factor: everything comes from the environment)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.paths import MODEL_DIR, PROJECT_ROOT, REPORTS_DIR


class Settings(BaseSettings):
    """Settings read from environment variables / ``.env`` (see ``.env.example``)."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Deepfake Adaptive Fusion API"
    app_version: str = "1.0.0"

    # --- artefacts -------------------------------------------------------
    model_path: Path = MODEL_DIR / "deepfake_fusion.onnx"
    pytorch_checkpoint_path: Path = MODEL_DIR / "AdaptiveFusion_best_model.pth"
    model_config_path: Path = MODEL_DIR / "model_config.json"
    metrics_path: Path = REPORTS_DIR / "metrics.json"
    frontend_dist: Path = PROJECT_ROOT / "frontend" / "dist"

    # --- inference -------------------------------------------------------
    decision_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_upload_mb: float = Field(default=10.0, gt=0)
    onnx_providers: str = "CPUExecutionProvider"
    enable_gradcam: bool = True
    torch_device: str = "auto"

    # --- HTTP ------------------------------------------------------------
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
    log_level: str = "INFO"
    model_version: str = "1.0.0"

    @field_validator(
        "model_path", "pytorch_checkpoint_path", "model_config_path", "metrics_path", "frontend_dist"
    )
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value)

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
    """Cached settings (imported by uvicorn as ``backend.app.main:app``)."""
    return Settings()
