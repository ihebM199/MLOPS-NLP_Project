"""ONNX export, FP16/INT8 quantisation and PyTorch <-> ONNX parity.

Why two inputs? ``torch.fft.fft2`` cannot be exported reliably, so the spectrum
is computed **outside** the graph (``image`` + ``freq_input``). The very same
closed-form is used by the API:

    freq = log1p(|fftshift(fft2(x))|)          # x = ImageNet-normalised image

Documented in ``docs/DECISIONS.md`` (Option A). The parity test is part of the
pytest suite and of the final checklist.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from src.utils.io import human_size, write_json
from src.utils.logging import get_logger

LOGGER = get_logger("models.onnx")

ONNX_INPUT_NAMES = ("image", "freq_input")
ONNX_OUTPUT_NAMES = ("logit", "w_spatial", "w_frequency")


# --------------------------------------------------------------------------- #
# NumPy reference implementation (shared with the FastAPI backend)
# --------------------------------------------------------------------------- #
def frequency_input_numpy(images: np.ndarray) -> np.ndarray:
    """``log1p(|fftshift(fft2(x))|)`` for a ``(N, 3, H, W)`` float32 array."""
    images = np.asarray(images, dtype=np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(images, axes=(-2, -1)), axes=(-2, -1))
    return np.log1p(np.abs(spectrum)).astype(np.float32)


def softmax_gate(logits_gate: np.ndarray) -> np.ndarray:
    """Stable softmax (kept for symmetry with the torch gate)."""
    shifted = logits_gate - logits_gate.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# PyTorch wrapper
# --------------------------------------------------------------------------- #
def make_wrapper(model):
    """Return an ``nn.Module`` with the exportable signature (2 in / 3 out)."""
    import torch.nn as nn

    class FusionExportWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, image, freq_input):
            logits, (w_s, w_f), _ = self.inner.forward_detailed(image, freq_input=freq_input)
            return logits, w_s, w_f

    return FusionExportWrapper(model).eval()


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export_onnx(
    model,
    path: str | Path,
    *,
    image_size: int = 224,
    opset: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """Export to ONNX with a dynamic batch axis and verify the graph."""
    import onnx
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = make_wrapper(model)
    dummy_image = torch.randn(2, 3, image_size, image_size)
    dummy_freq = model.frequency_input(dummy_image)

    dynamic_axes: Dict[str, Dict[int, str]] = {}
    if dynamic_batch:
        dynamic_axes = {
            "image": {0: "batch"},
            "freq_input": {0: "batch"},
            "logit": {0: "batch"},
            "w_spatial": {0: "batch"},
            "w_frequency": {0: "batch"},
        }
    torch.onnx.export(
        wrapper,
        (dummy_image, dummy_freq),
        str(path),
        input_names=list(ONNX_INPUT_NAMES),
        output_names=list(ONNX_OUTPUT_NAMES),
        dynamic_axes=dynamic_axes or None,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    LOGGER.info("ONNX exported: %s (%s, opset %d)", path, human_size(path.stat().st_size), opset)
    return path


# --------------------------------------------------------------------------- #
# Quantisation
# --------------------------------------------------------------------------- #
def convert_fp16(onnx_path: str | Path, output_path: Optional[str | Path] = None) -> Path:
    """FP16 conversion (I/O kept float32 so the API is unchanged).

    Uses ONNX Runtime's converter (handles Cast insertion around mixed-precision
    branches correctly) and falls back to ``onnxconverter-common``.
    """
    import warnings

    import onnx

    onnx_path = Path(onnx_path)
    output_path = Path(output_path) if output_path else onnx_path.with_name(onnx_path.stem + "_fp16.onnx")
    model = onnx.load_model(str(onnx_path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from onnxruntime.transformers.float16 import convert_float_to_float16 as ort_convert

            model_fp16 = ort_convert(model, keep_io_types=True)
        except Exception:  # noqa: BLE001 - fall back to the generic converter
            from onnxconverter_common import float16

            model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        onnx.save_model(model_fp16, str(output_path))
    LOGGER.info("FP16 model: %s (%s)", output_path, human_size(output_path.stat().st_size))
    return output_path


def quantize_int8(
    onnx_path: str | Path,
    output_path: Optional[str | Path] = None,
    *,
    calibration_arrays: Optional[Sequence[np.ndarray]] = None,
) -> Path:
    """INT8 quantisation.

    * ``calibration_arrays`` provided -> **static** quantisation (best accuracy,
      needs ``onnxruntime.quantization`` calibrators on a few hundred images);
    * otherwise -> **dynamic** quantisation of the weights (no calibration data,
      slightly faster to produce, marginally less accurate).
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_path = Path(onnx_path)
    output_path = Path(output_path) if output_path else onnx_path.with_name(onnx_path.stem + "_int8.onnx")
    if calibration_arrays:
        from onnxruntime.quantization import CalibrationDataReader, QuantFormat, quantize_static

        class _Reader(CalibrationDataReader):
            def __init__(self, batches: Sequence[np.ndarray]):
                self._batches = list(batches)
                self._index = 0

            def get_next(self):
                if self._index >= len(self._batches):
                    return None
                image = self._batches[self._index]
                self._index += 1
                return {"image": image, "freq_input": frequency_input_numpy(image)}

        quantize_static(
            str(onnx_path),
            str(output_path),
            _Reader(calibration_arrays),
            quant_format=QuantFormat.QDQ,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
        )
    else:
        quantize_dynamic(str(onnx_path), str(output_path), weight_type=QuantType.QInt8)
    LOGGER.info("INT8 model: %s (%s)", output_path, human_size(output_path.stat().st_size))
    return output_path


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
class OnnxRunner:
    """Thin ONNX Runtime wrapper used by the parity test and the benchmark."""

    def __init__(self, path: str | Path, providers: Optional[Sequence[str]] = None) -> None:
        import onnxruntime as ort

        self.path = str(path)
        available = ort.get_available_providers()
        requested = list(providers) if providers else ["CPUExecutionProvider"]
        self.providers = [p for p in requested if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.path, providers=self.providers)
        inputs = self.session.get_inputs()
        self.image_input = inputs[0].name
        self.freq_input = inputs[1].name if len(inputs) > 1 else None

    def predict_arrays(self, images: np.ndarray, freq: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        images = np.asarray(images, dtype=np.float32)
        feed = {self.image_input: images}
        if self.freq_input:
            feed[self.freq_input] = frequency_input_numpy(images) if freq is None else np.asarray(
                freq, dtype=np.float32
            )
        names = [o.name for o in self.session.get_outputs()]
        outputs = self.session.run(names, feed)
        return dict(zip(names, outputs))

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Single image ``(3, H, W)`` -> label/probabilities/weights/latency."""
        batch = np.asarray(image, dtype=np.float32)[None, ...]
        start = time.perf_counter()
        outputs = self.predict_arrays(batch)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logit = float(np.ravel(outputs["logit"])[0])
        prob_fake = float(1.0 / (1.0 + np.exp(-logit)))
        return {
            "logit": logit,
            "prob_fake": prob_fake,
            "prob_real": 1.0 - prob_fake,
            "label": "fake" if prob_fake >= 0.5 else "real",
            "w_spatial": float(np.ravel(outputs["w_spatial"])[0]),
            "w_frequency": float(np.ravel(outputs["w_frequency"])[0]),
            "inference_ms": elapsed_ms,
        }


# --------------------------------------------------------------------------- #
# Parity test
# --------------------------------------------------------------------------- #
def check_parity(
    model,
    onnx_path: str | Path,
    *,
    n_images: int = 64,
    image_size: int = 224,
    tolerance: float = 1e-3,
    seed: int = 0,
) -> Dict[str, Any]:
    """Compare PyTorch and ONNX logits on ``n_images`` random images.

    Pass criteria: ``max|Δlogit| < tolerance`` **and** 100 % of the predicted
    classes agree. The result is written to ``reports/onnx_parity.json``.
    """
    import torch

    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(n_images, 3, image_size, image_size, generator=generator)
    model.eval()
    with torch.no_grad():
        logits, (w_s, w_f), _ = model.forward_detailed(images)

    runner = OnnxRunner(onnx_path)
    outputs = runner.predict_arrays(images.numpy())
    onnx_logits = outputs["logit"].ravel()

    torch_logits = logits.cpu().numpy().ravel()
    diff = np.abs(torch_logits - onnx_logits)
    agreement = float(np.mean((torch_logits > 0) == (onnx_logits > 0)))
    result = {
        "n_images": n_images,
        "max_abs_logit_diff": float(diff.max()),
        "mean_abs_logit_diff": float(diff.mean()),
        "class_agreement": agreement,
        "tolerance": tolerance,
        "passed": bool(diff.max() < tolerance and agreement == 1.0),
        "onnx_path": str(onnx_path),
        "max_abs_gate_diff": float(
            np.abs(w_s.cpu().numpy().ravel() - outputs["w_spatial"].ravel()).max()
        ),
    }
    from src.utils.paths import PROJECT_ROOT

    write_json(PROJECT_ROOT / "reports" / "onnx_parity.json", result)
    if result["passed"]:
        LOGGER.info(
            "Parity OK: max|Δlogit| = %.2e (< %s), class agreement = 100%%",
            result["max_abs_logit_diff"],
            tolerance,
        )
    else:
        LOGGER.error("Parity FAILED: %s", result)
    return result


def export_all_variants(
    model,
    cfg,
    *,
    report_dir: Optional[str | Path] = None,
    skip_int8: bool = False,
) -> Dict[str, Any]:
    """Export FP32, FP16 and (optionally) INT8, run the parity test, save a report."""
    from src.utils.paths import REPORTS_DIR

    report_dir = Path(report_dir or REPORTS_DIR)
    base = Path(cfg.export.path)
    if not base.is_absolute():
        from src.utils.paths import PROJECT_ROOT

        base = PROJECT_ROOT / base
    variants: Dict[str, Any] = {"fp32": {"path": str(export_onnx(
        model, base, image_size=cfg.data.image_size, opset=cfg.export.opset
    ))}}
    if cfg.export.fp16:
        variants["fp16"] = {"path": str(convert_fp16(base))}
    if cfg.export.int8 and not skip_int8:
        try:
            variants["int8"] = {"path": str(quantize_int8(base))}
        except Exception as exc:  # noqa: BLE001 - optional optimisation
            LOGGER.warning("INT8 quantisation skipped (%s)", exc)
    for name, info in variants.items():
        path = Path(info["path"])
        info["size_mb"] = round(path.stat().st_size / 1e6, 2) if path.exists() else None
    parity = check_parity(
        model,
        variants["fp32"]["path"],
        n_images=cfg.export.parity_images,
        image_size=cfg.data.image_size,
        tolerance=cfg.export.parity_tolerance,
    )
    # Informational only: INT8 is expected to deviate more than the FP32 tolerance.
    int8_parity = None
    if "int8" in variants:
        int8_parity = check_parity(
            model,
            variants["int8"]["path"],
            n_images=min(cfg.export.parity_images, 16),
            image_size=cfg.data.image_size,
            tolerance=cfg.export.parity_tolerance * 100,
        )
    write_json(
        report_dir / "onnx_variants.json",
        {"variants": variants, "parity": parity, "int8_parity": int8_parity},
    )
    return {"variants": variants, "parity": parity, "int8_parity": int8_parity}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m src.models.onnx_export --config params.yaml`` (DVC stage)."""
    import argparse
    import json

    from src.config import load_config
    from src.models.architecture import load_checkpoint
    from src.utils.device import resolve_device

    parser = argparse.ArgumentParser(description="Export the trained model to ONNX (+FP16/INT8).")
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--skip-int8", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    cfg = load_config(args.config).prepare_dirs()
    checkpoint = args.checkpoint or cfg.weights_path
    model, info = load_checkpoint(
        checkpoint, backbone=cfg.train.backbone, mode="fusion", device=resolve_device(args.device)
    )
    report = export_all_variants(model, cfg, skip_int8=args.skip_int8)
    report["checkpoint"] = str(checkpoint)
    report["backbone"] = info["backbone"]
    print(json.dumps(
        {
            "variants": {k: v.get("size_mb") for k, v in report["variants"].items()},
            "parity_passed": report["parity"]["passed"],
            "max_abs_logit_diff": report["parity"]["max_abs_logit_diff"],
        },
        indent=2,
    ))
    return 0 if report["parity"]["passed"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
