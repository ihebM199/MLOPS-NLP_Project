"""Inference services — port of ``backend/app/inference.py``.

* ``/predict``            -> ONNX Runtime (fast, no PyTorch import)
* ``/predict/visualize``  -> PyTorch, **lazily** loaded on first call (Grad-CAM
  requires gradients, which ONNX Runtime does not provide)

Both share the exact same preprocessing, so the label is identical between the
two endpoints. The logic is byte-for-byte the original one; only the imports
(``backend.app.*`` -> ``api.*``) and the settings source changed.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from api.config import Settings
from api.validation import GradCamUnavailableError, ModelNotLoadedError
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.models.onnx_export import frequency_input_numpy
from src.utils.io import read_json
from src.utils.logging import get_logger

LOGGER = get_logger("backend.inference")

LIMITATIONS = [
    "In-distribution metrics (140k test split) are optimistic: the training and "
    "test fakes come from StyleGAN.",
    "Generalisation to unseen generators is weaker (the reference run reported "
    "~31% of fakes missed on Face-VAE/GANsformer).",
    "Not validated on diffusion images, video frames, heavy compression, "
    "non-face images or demographic fairness.",
    "Label smoothing compresses the probabilities: do not read them as calibrated confidence.",
    "Never use this model as the sole evidence in a decision affecting a person.",
]


def torch_installed() -> bool:
    return importlib.util.find_spec("torch") is not None


def gpu_available() -> bool:
    if not torch_installed():
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - broken CUDA install must not crash the API
        return False


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
class Preprocessor:
    """PIL -> ``(3, H, W)`` float32 ImageNet-normalised array + FFT input."""

    def __init__(self, image_size: int = 224) -> None:
        self.image_size = int(image_size)
        self.mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)

    def image_array(self, image: Image.Image) -> np.ndarray:
        resized = image.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))  # HWC -> CHW, RGB order preserved
        return ((array - self.mean) / self.std).astype(np.float32)

    def __call__(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(image, freq_input)`` both shaped ``(1, 3, S, S)``."""
        array = self.image_array(image)[None, ...]
        return array, frequency_input_numpy(array)


# --------------------------------------------------------------------------- #
# ONNX predictor
# --------------------------------------------------------------------------- #
class ONNXPredictor:
    """Single-session ONNX Runtime predictor (loaded once at startup)."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        threshold: float = 0.5,
        providers: Optional[List[str]] = None,
        image_size: int = 224,
        model_version: str = "1.0.0",
    ) -> None:
        import onnxruntime as ort

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        available = ort.get_available_providers()
        requested = providers or ["CPUExecutionProvider"]
        self.providers = [p for p in requested if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=self.providers)
        self.model_path = model_path
        self.threshold = float(threshold)
        self.model_version = model_version
        self.preprocessor = Preprocessor(image_size)
        self.image_size = image_size

        inputs = self.session.get_inputs()
        if not inputs:
            raise ValueError("The ONNX model declares no input")
        self.image_input = inputs[0].name
        self.freq_input = inputs[1].name if len(inputs) > 1 else None
        self.output_names = [output.name for output in self.session.get_outputs()]
        LOGGER.info(
            "ONNX session ready: %s | providers=%s | inputs=%s",
            model_path.name,
            self.providers,
            [i.name for i in inputs],
        )

    def predict_arrays(self, image: np.ndarray, freq: np.ndarray) -> Dict[str, Any]:
        feed = {self.image_input: image}
        if self.freq_input:
            feed[self.freq_input] = freq
        started = time.perf_counter()
        outputs = self.session.run(self.output_names, feed)
        inference_ms = (time.perf_counter() - started) * 1000
        values = dict(zip(self.output_names, outputs))
        return {
            "logit": float(np.ravel(values["logit"])[0]),
            "w_spatial": float(np.ravel(values["w_spatial"])[0]),
            "w_frequency": float(np.ravel(values["w_frequency"])[0]),
            "inference_ms": inference_ms,
        }

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Full prediction with the latency breakdown."""
        total_started = time.perf_counter()
        preprocess_started = time.perf_counter()
        array, freq = self.preprocessor(image)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
        raw = self.predict_arrays(array, freq)
        prob_fake = float(1.0 / (1.0 + np.exp(-raw["logit"])))
        total_ms = (time.perf_counter() - total_started) * 1000
        label = "fake" if prob_fake >= self.threshold else "real"
        return {
            "label": label,
            "prob_fake": prob_fake,
            "prob_real": 1.0 - prob_fake,
            "confidence": max(prob_fake, 1.0 - prob_fake),
            "logit": raw["logit"],
            "threshold": self.threshold,
            "fusion_weights": {"spatial": raw["w_spatial"], "frequency": raw["w_frequency"]},
            "latency": {
                "preprocess_ms": round(preprocess_ms, 2),
                "inference_ms": round(raw["inference_ms"], 2),
                "total_server_ms": round(total_ms, 2),
            },
            "model_version": self.model_version,
            "image_size": [self.image_size, self.image_size],
        }


# --------------------------------------------------------------------------- #
# Grad-CAM service (lazy)
# --------------------------------------------------------------------------- #
class GradCamService:
    """PyTorch Grad-CAM, loaded on the first ``/predict/visualize`` call."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._explainer = None
        self._transform = None
        self._error: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.settings.enable_gradcam and torch_installed())

    @property
    def available(self) -> bool:
        return self.enabled and self.settings.pytorch_checkpoint_path.exists()

    @property
    def error(self) -> Optional[str]:
        return self._error

    def _load(self) -> None:
        """Import torch/timm and rebuild the model (once)."""
        from src.data.transforms import build_eval_transform
        from src.models.architecture import load_checkpoint
        from src.utils.device import resolve_device
        from src.utils.gradcam import GradCAM

        config = read_json(self.settings.model_config_path, default={}) or {}
        image_size = int(config.get("image_size", 224))
        backbone = config.get("backbone")
        device = resolve_device(self.settings.torch_device)
        model, info = load_checkpoint(
            self.settings.pytorch_checkpoint_path,
            backbone=backbone,
            mode="fusion",
            device=device,
        )
        self._explainer = GradCAM(model, device=device)
        self._transform = build_eval_transform(image_size)
        LOGGER.info(
            "Grad-CAM ready (backbone=%s, device=%s) — PyTorch loaded lazily.",
            info["backbone"],
            device,
        )

    def ensure_loaded(self) -> None:
        if not self.available:
            raise GradCamUnavailableError(
                "Grad-CAM is unavailable: enable ENABLE_GRADCAM, install torch and provide "
                "PYTORCH_CHECKPOINT_PATH."
            )
        if self._explainer is not None:
            return
        with self._lock:
            if self._explainer is not None:
                return
            try:
                self._load()
            except Exception as exc:  # noqa: BLE001 - convert to a clean 503
                self._error = f"{exc.__class__.__name__}: {exc}"
                LOGGER.exception("Grad-CAM initialisation failed")
                raise GradCamUnavailableError(f"Grad-CAM could not be initialised ({self._error}).")

    def visualize(
        self, image: Image.Image, *, target: str = "fake", alpha: float = 0.45
    ) -> tuple[bytes, Dict[str, Any]]:
        self.ensure_loaded()
        started = time.perf_counter()
        result = self._explainer.explain(
            image, self._transform, target=target, alpha=alpha
        )  # type: ignore[union-attr]
        elapsed = (time.perf_counter() - started) * 1000
        meta = {
            "label": (
                "fake" if result["prediction"]["prob_fake"] >= self.settings.decision_threshold else "real"
            ),
            "prob_fake": float(result["prediction"]["prob_fake"]),
            "target": target,
            "alpha": float(alpha),
            "latency_ms": round(elapsed, 2),
        }
        return result["png"], meta

    def visualize_base64(self, image: Image.Image, *, target: str = "fake", alpha: float = 0.45):
        png, meta = self.visualize(image, target=target, alpha=alpha)
        return base64.b64encode(png).decode("ascii"), meta


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
class MetadataService:
    """Merge ``model_config.json`` + ``metrics.json`` into the public model card."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load(self) -> Dict[str, Any]:
        config = read_json(self.settings.model_config_path, default={}) or {}
        metrics = read_json(self.settings.metrics_path, default={}) or {}
        threshold = (
            (metrics.get("threshold") or {}).get("selected")
            if isinstance(metrics.get("threshold"), dict)
            else self.settings.decision_threshold
        )
        cross = (
            (metrics.get("cross_dataset") or {}).get("overall")
            if isinstance(metrics.get("cross_dataset"), dict)
            else None
        )
        robustness = metrics.get("robustness") or {}
        return {
            "name": "deepfake-adaptive-fusion",
            "architecture": config.get("architecture", "AdaptiveFusionDetector"),
            "backbone": config.get("backbone", "efficientnetv2_rw_s"),
            "mode": config.get("mode", "fusion"),
            "image_size": int(config.get("image_size", 224)),
            "classes": {"0": "real", "1": "fake"},
            "decision_rule": "P(fake) >= threshold -> FAKE (label 1); P(fake) = sigmoid(logit)",
            "threshold": float(threshold if threshold is not None else self.settings.decision_threshold),
            "parameters": config.get("parameters"),
            "metrics": {
                "test": metrics.get("test"),
                "valid": metrics.get("valid"),
                "cross_dataset": cross,
                "robustness": {
                    key: {"accuracy": value.get("accuracy"), "drop": value.get("accuracy_drop")}
                    for key, value in robustness.items()
                    if isinstance(value, dict)
                },
                "fusion_weights": metrics.get("fusion_weights"),
            },
            "trained_at": (metrics.get("generated_at") or (config.get("training") or {}).get("saved_at")),
            "model_version": self.settings.model_version,
            "git_sha": (config.get("git_sha") or (metrics.get("model") or {}).get("git_sha")),
            "limitations": LIMITATIONS,
        }


# --------------------------------------------------------------------------- #
# Registry (application state)
# --------------------------------------------------------------------------- #
class InferenceRegistry:
    """Holds the loaded predictor and reports a clean ``503`` when it is absent."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.predictor: Optional[ONNXPredictor] = None
        self.gradcam = GradCamService(settings)
        self.metadata = MetadataService(settings)
        self.load_error: Optional[str] = None
        self.loaded_at: Optional[float] = None
        self.started_at = time.time()
        self._lock = threading.Lock()

    # -- lifecycle --------------------------------------------------------
    def load(self) -> None:
        with self._lock:
            try:
                config = read_json(self.settings.model_config_path, default={}) or {}
                metrics = read_json(self.settings.metrics_path, default={}) or {}
                image_size = int(config.get("image_size", 224))
                threshold = self.settings.decision_threshold
                calibrated = metrics.get("threshold")
                if isinstance(calibrated, dict) and calibrated.get("selected") is not None:
                    threshold = float(calibrated["selected"])
                self.predictor = ONNXPredictor(
                    self.settings.model_path,
                    threshold=threshold,
                    providers=self.settings.provider_list,
                    image_size=image_size,
                    model_version=self.settings.model_version,
                )
                self.load_error = None
                self.loaded_at = time.time()
                self.gradcam = GradCamService(self.settings)
            except Exception as exc:  # noqa: BLE001 - the API must still answer /health
                self.predictor = None
                self.load_error = f"{exc.__class__.__name__}: {exc}"
                LOGGER.error("Model could not be loaded: %s", self.load_error)

    def close(self) -> None:
        self.predictor = None

    # -- helpers ----------------------------------------------------------
    @property
    def model_loaded(self) -> bool:
        return self.predictor is not None

    @property
    def onnxruntime_providers(self) -> List[str]:
        if self.predictor is not None:
            return list(self.predictor.providers)
        try:
            import onnxruntime as ort

            return ort.get_available_providers()
        except Exception:  # noqa: BLE001
            return []

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        if self.predictor is None:
            raise ModelNotLoadedError(
                f"The model is not loaded ({self.load_error or 'unknown reason'}). "
                f"Expected ONNX file: {self.settings.model_path}"
            )
        return self.predictor.predict(image)

    def metadata_payload(self) -> Dict[str, Any]:
        payload = self.metadata.load()
        if self.predictor is not None and payload.get("threshold") is None:
            payload["threshold"] = self.predictor.threshold
        return payload

    @property
    def uptime(self) -> float:
        return round(time.time() - self.started_at, 2)


# --------------------------------------------------------------------------- #
# CLI smoke test (no server): python -m api.inference image.jpg
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import sys

    settings = Settings()
    registry = InferenceRegistry(settings)
    registry.load()
    if registry.predictor is None:
        print("Model not loaded:", registry.load_error)
        raise SystemExit(1)
    if len(sys.argv) > 1:
        with Image.open(sys.argv[1]) as handle:
            print(json.dumps(registry.predict(handle.convert("RGB")), indent=2))
