"""Preprocessing, ONNX predictor and Grad-CAM service tests."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.app.inference import GradCamService, ONNXPredictor, Preprocessor
from backend.app.settings import Settings


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def test_preprocessor_shape_dtype_range():
    preprocessor = Preprocessor(224)
    image = Image.new("RGB", (320, 240), (128, 128, 128))
    array, freq = preprocessor(image)
    assert array.shape == (1, 3, 224, 224)
    assert array.dtype == np.float32
    assert freq.shape == (1, 3, 224, 224)
    assert freq.min() >= 0.0
    # grey 128 -> (0.502 - 0.485)/0.229 ~= 0.07 on the red channel
    assert -3.0 < array.min() and array.max() < 3.0


def test_preprocessor_keeps_rgb_order():
    preprocessor = Preprocessor(224)
    red = Image.new("RGB", (64, 64), (255, 0, 0))
    array, _ = preprocessor(red)
    channel_means = [float(array[0, c].mean()) for c in range(3)]
    # a pure red image must be strongest on channel 0 (RGB, not BGR)
    assert int(np.argmax(channel_means)) == 0
    assert channel_means[0] > channel_means[1]


# --------------------------------------------------------------------------- #
# ONNX predictor
# --------------------------------------------------------------------------- #
def test_onnx_predict_white_and_black(fake_onnx: Path, white_png_bytes: bytes, black_png_bytes: bytes):
    predictor = ONNXPredictor(fake_onnx, threshold=0.5)
    white = predictor.predict(Image.open(io.BytesIO(white_png_bytes)).convert("RGB"))
    black = predictor.predict(Image.open(io.BytesIO(black_png_bytes)).convert("RGB"))
    assert white["prob_fake"] > 0.5 and white["label"] == "fake"
    assert black["prob_fake"] < 0.5 and black["label"] == "real"
    assert white["prob_fake"] + white["prob_real"] == pytest.approx(1.0)
    assert 0.0 <= white["fusion_weights"]["spatial"] <= 1.0
    assert white["fusion_weights"]["spatial"] + white["fusion_weights"]["frequency"] == pytest.approx(1.0, abs=1e-4)
    for key in ("preprocess_ms", "inference_ms", "total_server_ms"):
        assert white["latency"][key] >= 0


def test_onnx_predictor_reports_threshold(fake_onnx: Path, white_png_bytes: bytes):
    predictor = ONNXPredictor(fake_onnx, threshold=0.99)
    body = predictor.predict(Image.open(io.BytesIO(white_png_bytes)).convert("RGB"))
    assert body["threshold"] == 0.99
    assert body["label"] == ("fake" if body["prob_fake"] >= 0.99 else "real")


def test_onnx_predictor_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ONNXPredictor(tmp_path / "nope.onnx")


def test_gradcam_service_unavailable_without_checkpoint(fake_onnx: Path, tmp_path: Path):
    settings = Settings(
        model_path=fake_onnx,
        pytorch_checkpoint_path=tmp_path / "nope.pth",
        enable_gradcam=True,
    )
    service = GradCamService(settings)
    assert service.available is False
    with pytest.raises(Exception):
        service.visualize(Image.new("RGB", (64, 64)))


# --------------------------------------------------------------------------- #
# Grad-CAM (needs torch; the model is a tiny stub with the same interface)
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch")


class TinyDetector(torch.nn.Module):
    """Minimal module exposing ``forward_detailed`` like AdaptiveFusionDetector."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.fc = torch.nn.Linear(8, 1)

    def forward_detailed(self, x):
        feature_map = self.conv(x)
        logits = self.fc(feature_map.mean(dim=(2, 3)))
        weights = torch.sigmoid(logits)
        return logits, (weights, 1.0 - weights), feature_map


@pytest.mark.slow
def test_gradcam_visualize_returns_png(fake_onnx: Path, tmp_path: Path, monkeypatch, image_bytes: bytes):
    checkpoint = tmp_path / "ckpt.pth"
    checkpoint.write_bytes(b"placeholder")
    settings = Settings(
        model_path=fake_onnx,
        pytorch_checkpoint_path=checkpoint,
        enable_gradcam=True,
        torch_device="cpu",
    )
    service = GradCamService(settings)

    def fake_load(self) -> None:
        from src.data.transforms import build_eval_transform
        from src.utils.gradcam import GradCAM

        self._explainer = GradCAM(TinyDetector(), device="cpu")
        self._transform = build_eval_transform(224)

    monkeypatch.setattr(GradCamService, "_load", fake_load)
    assert service.available is True
    png, meta = service.visualize(Image.open(io.BytesIO(image_bytes)).convert("RGB"), target="fake", alpha=0.5)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    decoded = Image.open(io.BytesIO(png))
    assert decoded.size == (224, 224)
    assert meta["target"] == "fake"
    assert 0.0 <= meta["prob_fake"] <= 1.0
    assert meta["latency_ms"] >= 0

    encoded, meta_b64 = service.visualize_base64(
        Image.open(io.BytesIO(image_bytes)).convert("RGB"), target="real", alpha=0.3
    )
    assert base64.b64decode(encoded)[:8] == b"\x89PNG\r\n\x1a\n"
    assert meta_b64["target"] == "real"


@pytest.mark.slow
def test_gradcam_preprocessing_matches_eval_transform():
    """Grad-CAM input must be the eval transform used during training."""
    from src.data.transforms import build_eval_transform

    transform = build_eval_transform(224)
    image = Image.open(io.BytesIO(_png((200, 60, 30)))).convert("RGB")
    expected = transform(image).unsqueeze(0).numpy()
    actual, _ = Preprocessor(224)(image)
    assert np.allclose(expected, actual, atol=2e-5)


def _png(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (128, 96), color).save(buffer, format="PNG")
    return buffer.getvalue()
