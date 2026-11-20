"""Pydantic v2 contracts of the public API (strict types + examples for Swagger)."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE_PREDICTION = {
    "label": "fake",
    "prob_real": 0.03,
    "prob_fake": 0.97,
    "confidence": 0.97,
    "logit": 3.47,
    "threshold": 0.5,
    "fusion_weights": {"spatial": 0.74, "frequency": 0.26},
    "latency": {"preprocess_ms": 8.4, "inference_ms": 31.2, "total_server_ms": 41.7},
    "model_version": "1.0.0",
    "image_size": [224, 224],
}


class FusionWeights(BaseModel):
    """Per-image contribution of each stream (the gate outputs a softmax)."""

    spatial: float = Field(ge=0.0, le=1.0, description="Weight of the spatial (EfficientNetV2) stream")
    frequency: float = Field(ge=0.0, le=1.0, description="Weight of the FFT stream")

    model_config = ConfigDict(json_schema_extra={"example": {"spatial": 0.74, "frequency": 0.26}})


class LatencyBreakdown(BaseModel):
    """Server-side timing, all values in milliseconds."""

    preprocess_ms: float = Field(ge=0, description="PIL decode + resize + normalisation + FFT")
    inference_ms: float = Field(ge=0, description="ONNX Runtime session.run")
    total_server_ms: float = Field(ge=0, description="preprocess + inference + Python overhead")

    model_config = ConfigDict(
        json_schema_extra={"example": {"preprocess_ms": 8.4, "inference_ms": 31.2, "total_server_ms": 41.7}}
    )


class PredictionResponse(BaseModel):
    """Result of ``POST /predict``."""

    label: Literal["real", "fake"] = Field(description="Decision at the configured threshold")
    prob_real: float = Field(ge=0.0, le=1.0)
    prob_fake: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0, description="max(prob_real, prob_fake)")
    logit: float = Field(description="Raw model output; > 0 means FAKE")
    threshold: float = Field(ge=0.0, le=1.0)
    fusion_weights: FusionWeights
    latency: LatencyBreakdown
    model_version: str
    image_size: List[int] = Field(default_factory=lambda: [224, 224])

    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE_PREDICTION})


class HealthResponse(BaseModel):
    """``GET /health`` — also returned with HTTP 503 when the model is missing."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
    gpu_available: bool
    onnxruntime_providers: List[str]
    gradcam_available: bool
    uptime_s: float = Field(ge=0)
    detail: Optional[str] = Field(default=None, description="Error message when degraded")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "model_loaded": True,
                "model_version": "1.0.0",
                "gpu_available": False,
                "onnxruntime_providers": ["CPUExecutionProvider"],
                "gradcam_available": True,
                "uptime_s": 42.5,
                "detail": None,
            }
        }
    )


class ModelMetadata(BaseModel):
    """``GET /model/metadata`` — model card served to the frontend."""

    name: str
    architecture: str
    backbone: str
    mode: str
    image_size: int
    classes: Dict[str, str] = Field(description="0 = real, 1 = fake")
    decision_rule: str
    threshold: float
    parameters: Optional[int] = None
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Valid/test/cross-dataset metrics")
    trained_at: Optional[str] = None
    model_version: str
    git_sha: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "deepfake-adaptive-fusion",
                "architecture": "AdaptiveFusionDetector",
                "backbone": "efficientnetv2_rw_s",
                "mode": "fusion",
                "image_size": 224,
                "classes": {"0": "real", "1": "fake"},
                "decision_rule": "P(fake) >= threshold -> FAKE",
                "threshold": 0.5,
                "parameters": 31800000,
                "metrics": {"test": {"accuracy": 0.9912, "f1": 0.9913, "auc": 0.999}},
                "trained_at": "2026-01-01T12:00:00",
                "model_version": "1.0.0",
                "git_sha": "abc1234",
                "limitations": ["Lower accuracy on generators unseen during training"],
            }
        }
    )


class VisualizationResponse(BaseModel):
    """JSON variant of ``POST /predict/visualize`` (``format=json``)."""

    label: Literal["real", "fake"]
    prob_fake: float = Field(ge=0.0, le=1.0)
    prob_real: float = Field(ge=0.0, le=1.0)
    target: Literal["fake", "real"]
    alpha: float = Field(ge=0.0, le=1.0)
    image_base64: str = Field(description="PNG (Grad-CAM overlay) encoded in base64")
    latency: LatencyBreakdown

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "label": "fake",
                "prob_fake": 0.97,
                "prob_real": 0.03,
                "target": "fake",
                "alpha": 0.45,
                "image_base64": "iVBORw0KGgoAAAANSUhEUg...",
                "latency": {"preprocess_ms": 8.4, "inference_ms": 55.0, "total_server_ms": 72.1},
            }
        }
    )


class ErrorResponse(BaseModel):
    """Homogeneous error body for every 4xx/5xx."""

    error: str = Field(description="Machine-readable code, e.g. INVALID_IMAGE")
    detail: str = Field(description="Human-readable explanation")
    status_code: int = Field(ge=400, le=599)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"error": "INVALID_IMAGE", "detail": "The uploaded file is not a decodable image.",
                        "status_code": 400}
        }
    )
