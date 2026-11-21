"""Efficiency benchmark: parameters, MACs, latency (p50/p95), memory, ONNX variants.

Latency is measured **with GPU synchronisation** (without it CUDA's async
scheduling inflates the throughput), after a warm-up, and reported as median +
p95 over several iterations — the numbers that actually matter for an API.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

import src.tracking as tracking
from src.utils.io import write_json
from src.utils.logging import get_logger

LOGGER = get_logger("models.benchmark")


def _sync(device: Any) -> None:
    import torch

    if getattr(device, "type", "cpu") == "cuda":
        torch.cuda.synchronize()


def count_parameters(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def macs(model: Any, image_size: int = 224) -> Optional[float]:
    """MACs in billions via ``thop`` (optional dependency, returns G-MACs)."""
    try:
        import copy

        import torch
        from thop import profile

        clone = copy.deepcopy(model).cpu().eval()
        value, _ = profile(clone, inputs=(torch.randn(1, 3, image_size, image_size),), verbose=False)
        return round(float(value) / 1e9, 3)
    except Exception as exc:  # noqa: BLE001 - benchmarking must not fail a run
        LOGGER.warning("MACs unavailable (%s)", exc)
        return None


def latency_benchmark(
    model: Any,
    *,
    device: Any = "cpu",
    batch_sizes: Sequence[int] = (1, 8, 32),
    warmup: int = 5,
    iterations: int = 30,
    image_size: int = 224,
) -> pd.DataFrame:
    """Median / p95 latency and throughput per batch size."""
    import torch

    model.eval()
    device_type = getattr(device, "type", str(device))
    rows = []
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)
        with torch.no_grad():
            for _ in range(warmup):
                model(x)
            _sync(device)
            timings = []
            for _ in range(iterations):
                start = time.perf_counter()
                model(x)
                _sync(device)
                timings.append((time.perf_counter() - start) * 1000)
        median = float(np.median(timings))
        rows.append(
            {
                "batch_size": batch_size,
                "latency_ms_p50": round(median, 2),
                "latency_ms_p95": round(float(np.percentile(timings, 95)), 2),
                "throughput_img_s": round(batch_size * 1000 / median, 1),
                "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
                if device_type == "cuda"
                else None,
            }
        )
    return pd.DataFrame(rows)


def onnx_latency(path: str | Path, *, iterations: int = 100, warmup: int = 10, image_size: int = 224) -> Dict[str, float]:
    """CPU latency of an ONNX variant (batch 1), median + p95 in ms."""
    import onnxruntime as ort

    from src.models.onnx_export import frequency_input_numpy

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    image = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    feed = {inputs[0].name: image}
    if len(inputs) > 1:
        feed[inputs[1].name] = frequency_input_numpy(image)
    names = [o.name for o in session.get_outputs()]

    for _ in range(warmup):
        session.run(names, feed)
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(names, feed)
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "latency_ms_p50": round(float(np.median(timings)), 2),
        "latency_ms_p95": round(float(np.percentile(timings, 95)), 2),
    }


def run_benchmark(cfg, model: Any, *, device: Any = "cpu", log_to_mlflow: bool = True) -> Dict[str, Any]:
    """Full efficiency report for the PyTorch model (written to ``reports/``)."""
    from src.utils.paths import REPORTS_DIR

    report = {
        "parameters": count_parameters(model),
        "active_parameters": model.active_parameters() if hasattr(model, "active_parameters") else None,
        "macs_g": macs(model, cfg.data.image_size),
        "device": str(device),
        "latency": latency_benchmark(model, device=device, image_size=cfg.data.image_size).to_dict("records"),
    }
    report["latency_ms_batch1"] = report["latency"][0]["latency_ms_p50"]
    write_json(Path(cfg.reports_dir) / "benchmark.json", report)
    if log_to_mlflow:
        tracking.log_metrics(
            cfg,
            {
                "parameters_m": report["parameters"] / 1e6,
                "macs_g": report["macs_g"] or 0.0,
                "latency_ms_batch1": report["latency_ms_batch1"],
            },
        )
    LOGGER.info(
        "Benchmark: %.2f M params | %s G MACs | %.1f ms (batch 1, %s)",
        report["parameters"] / 1e6,
        report["macs_g"],
        report["latency_ms_batch1"],
        device,
    )
    return report


def variants_table(
    rows: Iterable[Dict[str, Any]], path: Optional[str | Path] = None
) -> pd.DataFrame:
    """Build (and optionally persist) the FP32/FP16/INT8 trade-off table."""
    table = pd.DataFrame(list(rows))
    if not table.empty:
        columns = [
            c
            for c in (
                "variant", "size_mb", "latency_ms_p50", "latency_ms_p95",
                "accuracy", "f1", "auc", "eer", "n_samples",
            )
            if c in table.columns
        ]
        table = table[columns]
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        if not table.empty:
            try:
                markdown = table.to_markdown(index=False)
            except Exception:  # tabulate not installed
                markdown = "```\n" + table.to_string(index=False) + "\n```"
            path.with_suffix(".md").write_text(markdown, "utf-8")
    return table


def measure_variant_accuracy(
    session_path: str | Path,
    paths: Sequence[str],
    labels: Sequence[int],
    *,
    transform: Any,
    batch_size: int = 16,
) -> Dict[str, float]:
    """Accuracy/F1/AUC of an ONNX variant on a fixed list of images."""
    from PIL import Image

    from src.models.metrics import compute_metrics
    from src.models.onnx_export import OnnxRunner

    runner = OnnxRunner(session_path)
    probs, truth = [], []
    batch = []
    batch_labels = []
    for path, label in zip(paths, labels):
        try:
            with Image.open(path) as handle:
                tensor = transform(handle.convert("RGB")).numpy()
        except OSError:
            continue
        batch.append(tensor)
        batch_labels.append(int(label))
        if len(batch) == batch_size:
            outputs = runner.predict_arrays(np.stack(batch))
            probs.extend(1 / (1 + np.exp(-outputs["logit"].ravel())))
            truth.extend(batch_labels)
            batch, batch_labels = [], []
    if batch:
        outputs = runner.predict_arrays(np.stack(batch))
        probs.extend(1 / (1 + np.exp(-outputs["logit"].ravel())))
        truth.extend(batch_labels)
    if not truth:
        return {}
    probs_array = np.asarray(probs)
    truth_array = np.asarray(truth)
    metrics = compute_metrics(truth_array, (probs_array >= 0.5).astype(int), probs_array)
    metrics["n_samples"] = len(truth)
    return metrics
