"""API contract tests (no real weights, no dataset, no GPU)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import create_app
from backend.app.settings import Settings


@pytest.fixture()
def client(fake_onnx: Path, tmp_path: Path):
    settings = Settings(
        model_path=fake_onnx,
        pytorch_checkpoint_path=tmp_path / "missing.pth",
        model_config_path=tmp_path / "model_config.json",
        metrics_path=tmp_path / "metrics.json",
        frontend_dist=tmp_path / "no_frontend",
        enable_gradcam=False,
        max_upload_mb=10,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture()
def degraded_client(tmp_path: Path):
    settings = Settings(
        model_path=tmp_path / "absent.onnx",
        pytorch_checkpoint_path=tmp_path / "absent.pth",
        model_config_path=tmp_path / "missing.json",
        metrics_path=tmp_path / "missing_metrics.json",
        frontend_dist=tmp_path / "no_frontend",
        enable_gradcam=False,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #
def test_health_ok(client: TestClient, fake_onnx: Path):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]
    assert isinstance(body["onnxruntime_providers"], list) and body["onnxruntime_providers"]
    assert body["uptime_s"] >= 0
    assert "X-Process-Time-ms" in response.headers


def test_health_degraded_when_model_missing(degraded_client: TestClient):
    response = degraded_client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert body["detail"]


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #
def test_predict_returns_strict_schema(client: TestClient, image_bytes: bytes):
    response = client.post("/predict", files={"file": ("face.jpg", image_bytes, "image/jpeg")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) >= {
        "label", "prob_real", "prob_fake", "confidence", "logit", "threshold",
        "fusion_weights", "latency", "model_version",
    }
    assert body["label"] in {"real", "fake"}
    assert 0.0 <= body["prob_real"] <= 1.0
    assert 0.0 <= body["prob_fake"] <= 1.0
    assert body["prob_real"] + body["prob_fake"] == pytest.approx(1.0, abs=1e-6)
    assert body["confidence"] == pytest.approx(max(body["prob_real"], body["prob_fake"]))
    weights = body["fusion_weights"]
    assert weights["spatial"] + weights["frequency"] == pytest.approx(1.0, abs=1e-4)
    assert body["latency"]["total_server_ms"] >= 0
    assert body["latency"]["inference_ms"] >= 0
    assert body["latency"]["preprocess_ms"] >= 0


def test_predict_label_mapping_follows_logit_sign(client: TestClient, white_png_bytes: bytes):
    response = client.post("/predict", files={"file": ("white.png", white_png_bytes, "image/png")})
    body = response.json()
    assert response.status_code == 200
    # the fake graph returns logit = 10 * mean(normalised image); white is positive -> FAKE
    assert (body["logit"] > 0) is (body["label"] == "fake")


def test_predict_rejects_text_file(client: TestClient):
    response = client.post(
        "/predict", files={"file": ("notes.jpg", b"this is definitely not an image", "image/jpeg")}
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "INVALID_IMAGE"
    assert body["status_code"] == 400


def test_predict_rejects_unsupported_content_type(client: TestClient, image_bytes: bytes):
    response = client.post("/predict", files={"file": ("doc.pdf", image_bytes, "application/pdf")})
    assert response.status_code == 415
    assert response.json()["error"] == "UNSUPPORTED_FORMAT"


def test_predict_rejects_truncated_image(client: TestClient, image_bytes: bytes):
    truncated = image_bytes[:40]
    response = client.post("/predict", files={"file": ("truncated.jpg", truncated, "image/jpeg")})
    assert response.status_code == 400
    assert response.json()["error"] == "INVALID_IMAGE"


def test_predict_rejects_empty_file(client: TestClient):
    response = client.post("/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400
    assert response.json()["error"] == "EMPTY_FILE"


def test_predict_missing_file_is_422(client: TestClient):
    assert client.post("/predict").status_code == 422


def test_predict_file_too_large(fake_onnx: Path, image_bytes: bytes, tmp_path: Path):
    settings = Settings(
        model_path=fake_onnx,
        max_upload_mb=0.001,  # 1 kB: any real image is larger
        pytorch_checkpoint_path=tmp_path / "missing.pth",
        model_config_path=tmp_path / "missing.json",
        metrics_path=tmp_path / "missing_metrics.json",
        frontend_dist=tmp_path / "no_frontend",
        enable_gradcam=False,
    )
    with TestClient(create_app(settings)) as client:
        response = client.post("/predict", files={"file": ("big.jpg", image_bytes, "image/jpeg")})
    assert response.status_code == 413
    assert response.json()["error"] == "FILE_TOO_LARGE"


def test_predict_503_when_model_not_loaded(degraded_client: TestClient, image_bytes: bytes):
    response = degraded_client.post("/predict", files={"file": ("face.jpg", image_bytes, "image/jpeg")})
    assert response.status_code == 503
    assert response.json()["error"] == "MODEL_NOT_LOADED"


# --------------------------------------------------------------------------- #
# /model/metadata
# --------------------------------------------------------------------------- #
def test_metadata_defaults(client: TestClient):
    response = client.get("/model/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["architecture"] == "AdaptiveFusionDetector"
    assert body["classes"] == {"0": "real", "1": "fake"}
    assert 0.0 <= body["threshold"] <= 1.0
    assert isinstance(body["limitations"], list) and body["limitations"]


def test_metadata_reads_metrics_and_config(fake_onnx: Path, tmp_path: Path):
    config = {
        "architecture": "AdaptiveFusionDetector",
        "backbone": "efficientnetv2_rw_s",
        "image_size": 224,
        "parameters": 31_800_000,
        "mode": "fusion",
    }
    metrics = {
        "generated_at": "2026-01-01T00:00:00",
        "threshold": {"selected": 0.42},
        "test": {"accuracy": 0.99, "f1": 0.99, "auc": 0.999, "eer": 0.01},
        "cross_dataset": {"overall": {"accuracy": 0.8}},
        "robustness": {"jpeg_50": {"accuracy": 0.9, "accuracy_drop": 0.09}},
    }
    (tmp_path / "model_config.json").write_text(json.dumps(config), "utf-8")
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), "utf-8")
    settings = Settings(
        model_path=fake_onnx,
        model_config_path=tmp_path / "model_config.json",
        metrics_path=tmp_path / "metrics.json",
        pytorch_checkpoint_path=tmp_path / "missing.pth",
        frontend_dist=tmp_path / "no_frontend",
        enable_gradcam=False,
    )
    with TestClient(create_app(settings)) as client:
        body = client.get("/model/metadata").json()
    assert body["threshold"] == pytest.approx(0.42)
    assert body["parameters"] == 31_800_000
    assert body["metrics"]["test"]["accuracy"] == pytest.approx(0.99)
    assert body["metrics"]["cross_dataset"]["accuracy"] == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# /predict/visualize
# --------------------------------------------------------------------------- #
def test_visualize_503_without_pytorch_checkpoint(client: TestClient, image_bytes: bytes):
    response = client.post(
        "/predict/visualize", files={"file": ("face.jpg", image_bytes, "image/jpeg")}
    )
    assert response.status_code == 503
    assert response.json()["error"] == "GRADCAM_UNAVAILABLE"


def test_visualize_rejects_invalid_alpha(client: TestClient, image_bytes: bytes):
    response = client.post(
        "/predict/visualize?alpha=2",
        files={"file": ("face.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def test_root_hint_when_frontend_missing(client: TestClient):
    body = client.get("/").json()
    assert body["docs"] == "/docs"


def test_cors_headers(client: TestClient):
    response = client.options(
        "/predict",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert response.status_code in (200, 204)
    assert response.headers["access-control-allow-origin"] in {"http://localhost:5173", "*"}


def test_openapi_is_complete(client: TestClient):
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) >= {"/health", "/model/metadata", "/predict", "/predict/visualize"}
    prediction = schema["components"]["schemas"]["PredictionResponse"]
    assert "example" in prediction or "examples" in prediction
