"""Evaluation stage: metrics, calibration, robustness, cross-dataset, ONNX.

    python -m src.models.evaluate --config params.yaml
    python -m src.models.evaluate --config params.yaml --export-onnx --quantize
    python -m src.models.evaluate --quick-demo --robustness --cross-dataset

Outputs (all under ``reports/``):
    metrics.json            main metrics (DVC metric, read by the API)
    figures/*.png           confusion matrix, ROC, scores, fusion weights, Grad-CAM
    robustness.json         JPEG / blur / resize degradation
    cross_dataset.json      generalisation to unseen generators
    quantization_table.csv  PyTorch FP32 vs ONNX FP32/FP16/INT8
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

import src.tracking as tracking
from src.config import TrainConfig, load_config
from src.data.dataset import FaceDataset, build_dataframe, make_loader, subset_per_class
from src.data.transforms import build_eval_transform
from src.models.architecture import AdaptiveFusionDetector, load_checkpoint, model_config
from src.models.benchmark import (
    measure_variant_accuracy,
    onnx_latency,
    run_benchmark,
    variants_table,
)
from src.models.metrics import (
    bootstrap_ci,
    calibrate_threshold,
    compute_metrics,
    per_class_accuracy,
    threshold_metrics,
)
from src.models.train import predict_loader
from src.utils.device import describe_device, resolve_device
from src.utils.io import write_json
from src.utils.logging import get_logger
from src.utils.paths import dataset_root, ensure_dir
from src.utils.plots import (
    plot_evaluation,
    plot_fusion_weights,
    plot_gradcam_grid,
    plot_score_distribution,
)
from src.utils.seed import describe_environment, seed_everything

LOGGER = get_logger("models.evaluate")


# --------------------------------------------------------------------------- #
# Degradations (real deepfakes are re-compressed / resized / blurred)
# --------------------------------------------------------------------------- #
def degrade(image: Image.Image, kind: str, value: float) -> Image.Image:
    """Apply one degradation to a PIL image."""
    image = image.convert("RGB")
    if kind == "jpeg":
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(value))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(value)))
    if kind == "resize":
        width, height = image.size
        small = image.resize((max(8, int(width * value)), max(8, int(height * value))), Image.BILINEAR)
        return small.resize((width, height), Image.BILINEAR)
    raise ValueError(f"Unknown degradation: {kind}")


class DegradedDataset(Dataset):
    """Wrap another dataset and degrade each image (deterministically)."""

    def __init__(self, base: FaceDataset, transform, kind: str, value: float) -> None:
        self.base = base
        self.transform = transform
        self.kind = kind
        self.value = value

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        path = self.base.paths[index]
        try:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
        except OSError:
            image = Image.new("RGB", (self.base.image_size, self.base.image_size))
        return self.transform(degrade(image, self.kind, self.value)), self.base.labels[index]


# --------------------------------------------------------------------------- #
# Core evaluation
# --------------------------------------------------------------------------- #
def evaluate_loader(
    model: AdaptiveFusionDetector,
    loader: DataLoader,
    device: Any,
    *,
    threshold: float = 0.5,
    criterion: Optional[Any] = None,
) -> Dict[str, Any]:
    return predict_loader(model, loader, device=device, criterion=criterion, threshold=threshold)


def summarize(
    results: Dict[str, Any], *, threshold: float, name: str, bootstrap: bool = False
) -> Dict[str, Any]:
    y_true, y_prob = results["y_true"], results["y_prob"]
    metrics = threshold_metrics(y_true, y_prob, threshold)
    metrics["loss"] = float(results.get("loss", float("nan")))
    metrics.update(per_class_accuracy(y_true, (y_prob >= threshold).astype(int)))
    metrics["n_samples"] = int(len(y_true))
    metrics["threshold"] = float(threshold)
    if bootstrap and len(y_true) >= 50:
        metrics["accuracy_ci95"] = bootstrap_ci(y_true, y_prob, metric="accuracy")
        metrics["auc_ci95"] = bootstrap_ci(y_true, y_prob, metric="auc")
    LOGGER.info(
        "%s | acc %.2f%% | F1 %.4f | AUC %.4f | EER %.4f | thr %.3f | n=%d",
        name, 100 * metrics["accuracy"], metrics["f1"], metrics.get("auc", float("nan")),
        metrics.get("eer", float("nan")), threshold, metrics["n_samples"],
    )
    return metrics


def mean_spectrum(paths: Sequence[str], size: int = 224, max_images: int = 200) -> np.ndarray:
    accumulator = np.zeros((size, size), dtype=np.float64)
    used = 0
    for path in paths[:max_images]:
        try:
            with Image.open(path) as handle:
                gray = np.asarray(handle.convert("L").resize((size, size)), dtype=np.float32) / 255.0
        except OSError:
            continue
        accumulator += np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
        used += 1
    return accumulator / max(used, 1)


def run_robustness(
    model: AdaptiveFusionDetector, frame: pd.DataFrame, cfg: TrainConfig, device: Any, threshold: float
) -> Dict[str, Any]:
    """Degradation study on a fixed sample of the test split."""
    if frame.empty:
        return {}
    sample = frame.sample(min(len(frame), cfg.eval.max_robustness_samples), random_state=cfg.data.seed)
    base = FaceDataset(sample, build_eval_transform(cfg.data.image_size), image_size=cfg.data.image_size)
    report: Dict[str, Any] = {}
    degradations = (
        [("jpeg", q) for q in cfg.eval.jpeg_qualities]
        + [("blur", s) for s in cfg.eval.blur_sigmas]
        + [("resize", f) for f in cfg.eval.resize_factors]
    )
    reference = predict_loader(
        model, DataLoader(base, batch_size=cfg.train.batch_size, shuffle=False),
        device=device, threshold=threshold,
    )
    report["clean"] = summarize(reference, threshold=threshold, name="robustness/clean")
    for kind, value in degradations:
        dataset = DegradedDataset(base, build_eval_transform(cfg.data.image_size), kind, value)
        loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False)
        results = predict_loader(model, loader, device=device, threshold=threshold)
        key = f"{kind}_{int(value) if float(value).is_integer() else value}"
        metrics = summarize(results, threshold=threshold, name=f"robustness/{key}")
        metrics["accuracy_drop"] = round(report["clean"]["accuracy"] - metrics["accuracy"], 4)
        report[key] = metrics
    return report


def run_cross_dataset(
    model: AdaptiveFusionDetector, cfg: TrainConfig, device: Any, threshold: float
) -> Dict[str, Any]:
    """Evaluate on unseen generators (Celeb-DF v2, FF-C23, CelebA, LFW; FFHQ excluded)."""
    from src.data.external import build_external_manifest, manifest_summary, training_hashes

    hashes = training_hashes(dataset_root(), limit=2000, seed=cfg.data.seed)
    manifest = build_external_manifest(
        max_per_class=cfg.eval.external_max_per_class, seed=cfg.data.seed, exclude_hashes=hashes
    )
    if manifest.empty:
        LOGGER.warning(
            "No external dataset found under data/external/ — cross-dataset study skipped."
        )
        return {}
    eval_tf = build_eval_transform(cfg.data.image_size)
    report: Dict[str, Any] = {
        "sources": manifest_summary(manifest).to_dict(),
        "n_images": int(len(manifest)),
        "note": "FFHQ excluded (already present in the 140k 'real' class); MD5-deduplicated vs training.",
    }
    overall = predict_loader(
        model,
        DataLoader(FaceDataset(manifest, eval_tf), batch_size=cfg.train.batch_size, shuffle=False),
        device=device,
        threshold=threshold,
    )
    report["overall"] = summarize(
        overall, threshold=threshold, name="cross_dataset/overall", bootstrap=True
    )
    per_source = {}
    for source, part in manifest.groupby("source"):
        results = predict_loader(
            model,
            DataLoader(FaceDataset(part, eval_tf), batch_size=cfg.train.batch_size, shuffle=False),
            device=device,
            threshold=threshold,
        )
        per_source[source] = summarize(results, threshold=threshold, name=f"cross_dataset/{source}")
    report["per_source"] = per_source
    return report


def gradcam_artifacts(
    model: AdaptiveFusionDetector, frame: pd.DataFrame, cfg: TrainConfig, device: Any
) -> list[str]:
    """4 Grad-CAM examples (2 real / 2 fake) -> one figure + individual PNGs."""
    if frame.empty:
        return []
    from src.utils.gradcam import gradcam_examples, to_png_bytes

    picks = pd.concat(
        [
            frame[frame["label"] == label].sample(
                min(2, int((frame["label"] == label).sum())), random_state=cfg.data.seed
            )
            for label in (0, 1)
        ]
    )
    rows = gradcam_examples(
        model,
        picks["path"].tolist(),
        build_eval_transform(cfg.data.image_size),
        device=device,
        labels=picks["label"].tolist(),
    )
    figures = ensure_dir(Path(cfg.reports_dir) / "figures")
    paths = []
    if rows:
        plot_gradcam_grid(rows, figures / "gradcam_examples.png")
        paths.append(str(figures / "gradcam_examples.png"))
        for index, row in enumerate(rows):
            file_path = figures / f"gradcam_{index}.png"
            file_path.write_bytes(to_png_bytes(row["overlay"]))
            paths.append(str(file_path))
    return paths


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def load_model_for_evaluation(
    cfg: TrainConfig, checkpoint: Optional[str | Path], device: Any
) -> tuple[AdaptiveFusionDetector, Dict[str, Any]]:
    path = Path(checkpoint) if checkpoint else cfg.weights_path
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Train first (`python -m src.models.train --config params.yaml`) "
            "or pass --checkpoint <file.pth>."
        )
    config_path = Path(cfg.model_dir) / "model_config.json"
    meta_backbone = None
    meta_mode = cfg.train.mode
    if config_path.exists():
        try:
            saved = json.loads(config_path.read_text("utf-8"))
            meta_backbone = saved.get("backbone")
            meta_mode = saved.get("mode") or meta_mode
        except json.JSONDecodeError:
            pass
    return load_checkpoint(
        path,
        backbone=meta_backbone or cfg.train.backbone,
        mode=meta_mode,
        device=device,
    )


def run_evaluation(
    cfg: TrainConfig,
    *,
    checkpoint: Optional[str | Path] = None,
    device: Any = None,
    do_export: bool = False,
    do_quantize: bool = False,
    do_robustness: Optional[bool] = None,
    do_cross_dataset: Optional[bool] = None,
    do_benchmark: bool = True,
    root: Optional[str | Path] = None,
    run_name: str = "evaluate",
) -> Dict[str, Any]:
    cfg.prepare_dirs()
    seed_everything(cfg.data.seed)
    device = device or resolve_device()
    model, info = load_model_for_evaluation(cfg, checkpoint, device)
    LOGGER.info("Loaded %s on %s", model.backbone_name, describe_device(device))

    df = build_dataframe(root or dataset_root())
    if df.empty:
        raise FileNotFoundError("Dataset not found — run `python -m src.data.prepare` first.")
    frames = {
        "valid": subset_per_class(df, "valid", cfg.data.valid_per_class, seed=cfg.data.seed),
        "test": subset_per_class(df, "test", cfg.data.test_per_class, seed=cfg.data.seed),
    }
    eval_tf = build_eval_transform(cfg.data.image_size)
    loaders = {
        split: make_loader(
            frame, split if split in ("valid", "test") else "test", eval_tf,
            batch_size=cfg.train.batch_size, num_workers=cfg.data.num_workers,
            shuffle=False, image_size=cfg.data.image_size,
        )
        for split, frame in frames.items()
    }
    loaders = {k: v for k, v in loaders.items() if v is not None}

    report: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_config(model, image_size=cfg.data.image_size),
        "environment": describe_environment(),
        "threshold": {"default": cfg.eval.threshold},
    }
    figures = ensure_dir(Path(cfg.reports_dir) / "figures")

    with tracking.run(cfg, run_name, tags={"stage": "evaluate"}) as mlflow:
        results: Dict[str, Dict[str, Any]] = {}
        threshold = cfg.eval.threshold
        if "valid" in loaders:
            valid_results = evaluate_loader(model, loaders["valid"], device, threshold=threshold)
            if cfg.eval.calibration:
                calibration = calibrate_threshold(valid_results["y_true"], valid_results["y_prob"])
                report["threshold"] = {
                    "default": cfg.eval.threshold,
                    "selected": calibration["threshold"],
                    "calibration": {k: v for k, v in calibration.items() if k != "curve"},
                    "selected_on": "valid",
                }
                threshold = calibration["threshold"]
                plot_score_distribution(
                    valid_results["y_true"], valid_results["y_prob"],
                    threshold=threshold, path=figures / "valid_scores.png",
                )
                write_json(Path(cfg.reports_dir) / "threshold_calibration.json", calibration)
            results["valid"] = valid_results
            report["valid"] = summarize(valid_results, threshold=threshold, name="valid")
        else:
            report["valid"] = {}
            LOGGER.warning("No validation split available — threshold stays at %.2f", threshold)

        if "test" in loaders:
            test_results = evaluate_loader(model, loaders["test"], device, threshold=threshold)
            results["test"] = test_results
            report["test"] = summarize(test_results, threshold=threshold, name="test", bootstrap=True)
            report["test"]["accuracy_at_0.5"] = compute_metrics(
                test_results["y_true"], (test_results["y_prob"] >= 0.5).astype(int), test_results["y_prob"]
            )["accuracy"]
            plot_evaluation(
                test_results["y_true"], test_results["y_prob"], threshold=threshold,
                title=f"Test split — 140k real & fake faces (n={len(test_results['y_true'])})",
                path=figures / "test_evaluation.png", eer=report["test"].get("eer"),
            )
            plot_score_distribution(
                test_results["y_true"], test_results["y_prob"],
                threshold=threshold, path=figures / "test_scores.png",
            )
            if model.mode == "fusion" and len(test_results["w_spatial"]) == len(test_results["y_true"]):
                plot_fusion_weights(
                    test_results["y_true"], test_results["w_spatial"], test_results["w_frequency"],
                    path=figures / "fusion_weights.png",
                )
                report["fusion_weights"] = {
                    "mean_spatial_real": float(np.mean(test_results["w_spatial"][test_results["y_true"] == 0])),
                    "mean_spatial_fake": float(np.mean(test_results["w_spatial"][test_results["y_true"] == 1])),
                    "mean_frequency_real": float(np.mean(test_results["w_frequency"][test_results["y_true"] == 0])),
                    "mean_frequency_fake": float(np.mean(test_results["w_frequency"][test_results["y_true"] == 1])),
                }

        run_robustness_flag = do_robustness if do_robustness is not None else cfg.eval.robustness
        if run_robustness_flag:
            if "test" in frames:
                report["robustness"] = run_robustness(model, frames["test"], cfg, device, threshold)
                write_json(Path(cfg.reports_dir) / "robustness.json", report["robustness"])

        run_cross_flag = do_cross_dataset if do_cross_dataset is not None else cfg.eval.cross_dataset
        if run_cross_flag:
            report["cross_dataset"] = run_cross_dataset(model, cfg, device, threshold)
            write_json(Path(cfg.reports_dir) / "cross_dataset.json", report["cross_dataset"])

        if do_benchmark:
            report["benchmark"] = run_benchmark(cfg, model, device=device)

        if "test" in frames and len(frames["test"]):
            gradcam_files = gradcam_artifacts(model, frames["test"], cfg, device)
            report["gradcam_examples"] = gradcam_files
            for file_path in gradcam_files:
                tracking.log_artifact(cfg, file_path, artifact_path="gradcam")

        if do_export:
            from src.models.onnx_export import export_all_variants

            export = export_all_variants(model, cfg, report_dir=cfg.reports_dir)
            report["onnx"] = export
            if mlflow is not None:
                for name, variant in export["variants"].items():
                    tracking.log_metrics(
                        cfg, {f"onnx_{name}_size_mb": variant.get("size_mb") or 0.0}
                    )
                tracking.log_artifact(cfg, export["variants"]["fp32"]["path"], artifact_path="onnx")

        if do_quantize and "test" in frames and report.get("onnx"):
            rows = []
            sample = frames["test"].sample(
                min(len(frames["test"]), max(200, cfg.export.calibration_samples)),
                random_state=cfg.data.seed,
            )
            rows.append(
                {
                    "variant": "PyTorch FP32",
                    "size_mb": round(sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6, 2),
                    "latency_ms_p50": report.get("benchmark", {}).get("latency_ms_batch1"),
                    **{k: report.get("test", {}).get(k) for k in ("accuracy", "f1", "auc", "eer")},
                    "n_samples": len(frames["test"]),
                }
            )
            for name, variant in report["onnx"]["variants"].items():
                metrics = measure_variant_accuracy(
                    variant["path"], sample["path"].tolist(), sample["label"].tolist(),
                    transform=eval_tf,
                )
                rows.append(
                    {
                        "variant": f"ONNX {name.upper()}",
                        "size_mb": variant.get("size_mb"),
                        **onnx_latency(variant["path"]),
                        **{k: metrics.get(k) for k in ("accuracy", "f1", "auc", "eer")},
                        "n_samples": metrics.get("n_samples"),
                    }
                )
            table = variants_table(rows, Path(cfg.reports_dir) / "quantization_table.csv")
            report["quantization"] = rows
            LOGGER.info(
                "Quantisation trade-off:\n%s", table.to_string(index=False) if not table.empty else "n/a"
            )
            if mlflow is not None:
                tracking.log_artifact(cfg, Path(cfg.reports_dir) / "quantization_table.csv")

        report["artifacts"] = {
            "metrics": str(Path(cfg.reports_dir) / "metrics.json"),
            "figures": str(figures),
            "checkpoint": info["path"],
        }
        write_json(Path(cfg.reports_dir) / "metrics.json", report)
        tracking.log_metrics(
            cfg,
            {
                **{f"test_{k}": v for k, v in report.get("test", {}).items()
                   if isinstance(v, (int, float))},
                "threshold": threshold,
            },
        )
        for figure in figures.glob("*.png"):
            tracking.log_artifact(cfg, figure, artifact_path="figures")
        tracking.log_artifact(cfg, Path(cfg.reports_dir) / "metrics.json", artifact_path="metrics")

    LOGGER.info("Evaluation written to %s", Path(cfg.reports_dir) / "metrics.json")
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the model + export ONNX.")
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default=None, help="kept for compatibility (valid+test are evaluated)")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--robustness", dest="robustness", action="store_true", default=None)
    parser.add_argument("--no-robustness", dest="robustness", action="store_false")
    parser.add_argument("--cross-dataset", dest="cross_dataset", action="store_true", default=None)
    parser.add_argument("--no-cross-dataset", dest="cross_dataset", action="store_false")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--quick-demo", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, quick_demo=args.quick_demo).prepare_dirs()
    if args.threshold is not None:
        cfg.eval.threshold = args.threshold
        cfg.eval.calibration = False
    if args.no_mlflow:
        cfg.tracking.enabled = False
    report = run_evaluation(
        cfg,
        checkpoint=args.checkpoint,
        device=resolve_device(args.device),
        do_export=args.export_onnx or args.quantize,
        do_quantize=args.quantize,
        do_robustness=args.robustness,
        do_cross_dataset=args.cross_dataset,
        do_benchmark=not args.no_benchmark,
        root=args.dataset_root,
    )
    print(json.dumps({k: report.get(k) for k in ("threshold", "test", "onnx")}, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
