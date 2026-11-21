"""Training stage — CLI first, notebook second.

    python -m src.models.train --config params.yaml
    python -m src.models.train --config params.yaml --mode spatial_only --run-name R1_spatial
    python -m src.models.train --quick-demo          # 500 img/class, 2 epochs
    python -m src.models.train --init-from models/AdaptiveFusion_best_model.pth

What is logged (MLflow): hyper-parameters, per-epoch metrics, final validation
metrics, training history/curves, ``model_config.json`` and the checkpoint.
Resuming after a Colab disconnection is supported through ``last_checkpoint.pt``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import src.tracking as tracking
from src.config import MODES, TrainConfig, load_config
from src.data.dataset import build_dataframe, make_loader, subset_per_class
from src.data.transforms import build_eval_transform, build_train_transform
from src.models.architecture import (
    AdaptiveFusionDetector,
    build_model,
    load_checkpoint,
    model_config,
)
from src.models.augment import apply_mix, mix_criterion
from src.models.losses import criterion_for
from src.models.metrics import compute_metrics
from src.utils.device import amp_enabled, describe_device, resolve_device
from src.utils.io import write_json
from src.utils.logging import get_logger
from src.utils.paths import dataset_root, ensure_dir
from src.utils.plots import plot_training_history
from src.utils.seed import describe_environment, seed_everything

LOGGER = get_logger("models.train")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_loaders(
    cfg: TrainConfig, root: Optional[str | Path] = None
) -> tuple[Dict[str, Optional[DataLoader]], pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Official split + deterministic per-class subsampling (no augmentation leak)."""
    root = Path(root) if root else dataset_root()
    df = build_dataframe(root)
    if df.empty:
        raise FileNotFoundError(
            f"No images found under {root}. Run `python -m src.data.prepare --config params.yaml` first."
        )
    frames = {
        "train": subset_per_class(df, "train", cfg.data.train_per_class, seed=cfg.data.seed),
        "valid": subset_per_class(df, "valid", cfg.data.valid_per_class, seed=cfg.data.seed),
        "test": subset_per_class(df, "test", cfg.data.test_per_class, seed=cfg.data.seed),
    }
    train_tf = build_train_transform(
        cfg.data.image_size,
        advanced=cfg.data.advanced_augmentation,
        randaugment_num_ops=cfg.data.randaugment_num_ops,
        randaugment_magnitude=cfg.data.randaugment_magnitude,
    )
    eval_tf = build_eval_transform(cfg.data.image_size)
    loaders = {
        "train": make_loader(
            frames["train"],
            "train",
            train_tf,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=True,
            image_size=cfg.data.image_size,
            drop_last=True,
            seed=cfg.data.seed,
        ),
        "valid": make_loader(
            frames["valid"],
            "valid",
            eval_tf,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=False,
            image_size=cfg.data.image_size,
        ),
        "test": make_loader(
            frames["test"],
            "test",
            eval_tf,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=False,
            image_size=cfg.data.image_size,
        ),
    }
    LOGGER.info(
        "Data | train=%s valid=%s test=%s images",
        len(frames["train"]),
        len(frames["valid"]),
        len(frames["test"]),
    )
    return loaders, df, frames


def dataset_fingerprint(frames: Dict[str, pd.DataFrame]) -> str:
    """Stable id of the exact data used by a run (logged to MLflow as data_version)."""
    digest = hashlib.md5(usedforsecurity=False)
    for split in ("train", "valid", "test"):
        paths = sorted(frames.get(split, pd.DataFrame()).get("path", []))
        digest.update(split.encode())
        digest.update(str(len(paths)).encode())
        digest.update("\n".join(paths).encode())
    return digest.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Optimisation setup
# --------------------------------------------------------------------------- #
def build_optimizer(model: AdaptiveFusionDetector, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with the differentiated learning rates of the reference recipe."""
    t = cfg.train
    groups: list[dict[str, Any]] = [
        {"params": model.spatial_projection.parameters(), "lr": t.lr_fusion, "name": "spatial_projection"}
    ]
    if model.mode in ("fusion", "spatial_only", "concat_no_gate"):
        groups.append({"params": model.spatial_net.parameters(), "lr": t.lr_backbone, "name": "backbone"})
    if model.mode in ("fusion", "freq_only", "concat_no_gate"):
        groups.append({"params": model.freq_net.parameters(), "lr": t.lr_fusion, "name": "freq_net"})
        groups.append(
            {"params": model.frequency_projection.parameters(), "lr": t.lr_fusion, "name": "freq_projection"}
        )
    if model.mode == "fusion":
        groups.append({"params": model.fusion_gate.parameters(), "lr": t.lr_fusion, "name": "fusion_gate"})
        if t.learn_residual_scale:
            groups.append({"params": [model.residual_scale], "lr": t.lr_fusion, "name": "residual_scale"})
    groups.append({"params": model.classifier.parameters(), "lr": t.lr_classifier, "name": "classifier"})
    return torch.optim.AdamW(groups, weight_decay=t.weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.train.epochs, 1))


def set_backbone_trainable(model: AdaptiveFusionDetector, trainable: bool) -> None:
    for param in model.spatial_net.parameters():
        param.requires_grad = trainable


# --------------------------------------------------------------------------- #
# Epoch loops
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: AdaptiveFusionDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    *,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
    epoch: int = 1,
) -> Dict[str, float]:
    model.train()
    use_amp = amp_enabled(device, use_amp)
    loss_sum, seen = 0.0, 0
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.data.seed + epoch)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device)

        mixed, target_a, target_b, lam = apply_mix(
            images,
            labels,
            probability=cfg.data.mix_prob if cfg.data.advanced_augmentation else 0.0,
            mixup_alpha=cfg.data.mixup_alpha,
            cutmix_alpha=cfg.data.cutmix_alpha,
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(mixed)
        loss = mix_criterion(
            criterion,
            logits.float(),
            target_a * (1 - 2 * cfg.train.label_smoothing) + cfg.train.label_smoothing,
            target_b * (1 - 2 * cfg.train.label_smoothing) + cfg.train.label_smoothing,
            lam,
        )
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

        loss_sum += float(loss.detach()) * images.size(0)
        seen += images.size(0)
        if lam >= 0.999:  # only unmixed batches give an interpretable accuracy
            y_true_all.append(target_a.detach().cpu().numpy().ravel().astype(int))
            y_pred_all.append((logits.detach() > 0).float().cpu().numpy().ravel().astype(int))

    metrics: Dict[str, float] = {"loss": loss_sum / max(seen, 1)}
    if y_true_all:
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        batch_metrics = compute_metrics(y_true, y_pred)
        metrics.update(
            {k: v for k, v in batch_metrics.items() if k in ("accuracy", "balanced_accuracy", "f1")}
        )
        metrics["train_samples_scored"] = float(len(y_true))
    return metrics


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    use_amp: bool = False,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Collect labels, logits, probabilities and fusion weights for a loader."""
    model.eval()
    use_amp = amp_enabled(device, use_amp)
    logits_all, labels_all, ws_all, wf_all = [], [], [], []
    loss_sum = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits, (w_s, w_f) = model(images, return_weights=True)
        logits = logits.float()
        if criterion is not None:
            target = labels.float().unsqueeze(1).to(device)
            loss_sum += float(criterion(logits, target)) * images.size(0)
        logits_all.append(logits.squeeze(1).cpu())
        labels_all.append(labels.cpu())
        ws_all.append(w_s.squeeze(1).float().cpu())
        wf_all.append(w_f.squeeze(1).float().cpu())

    logits_t = torch.cat(logits_all) if logits_all else torch.empty(0)
    y_true = torch.cat(labels_all).numpy().astype(int) if labels_all else np.array([], dtype=int)
    prob = torch.sigmoid(logits_t).numpy()
    return {
        "y_true": y_true,
        "logits": logits_t.numpy(),
        "y_prob": prob,
        "y_pred": (prob >= threshold).astype(int),
        "w_spatial": torch.cat(ws_all).numpy() if ws_all else np.array([]),
        "w_frequency": torch.cat(wf_all).numpy() if wf_all else np.array([]),
        "loss": loss_sum / max(len(y_true), 1) if criterion is not None else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Full experiment
# --------------------------------------------------------------------------- #
def run_training(
    cfg: TrainConfig,
    *,
    device: Optional[Any] = None,
    root: Optional[str | Path] = None,
    run_name: Optional[str] = None,
    init_from: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Train one model variant and persist every artefact. Returns a summary dict."""
    cfg.prepare_dirs()
    seed_everything(cfg.train.seed, deterministic=cfg.train.deterministic)
    device = device or resolve_device()
    LOGGER.info("Device: %s | %s", describe_device(device), cfg.summary())

    loaders, full_df, frames = build_loaders(cfg, root)
    if loaders["train"] is None or len(loaders["train"].dataset) == 0:
        raise RuntimeError("The train split is empty — check the dataset and the subset sizes.")

    model = build_model(cfg).to(device)
    if init_from:
        source, info = load_checkpoint(
            init_from, backbone=cfg.train.backbone, mode=cfg.train.mode, device=device
        )
        model.load_state_dict(source.state_dict(), strict=True)
        LOGGER.info("Initialised from %s (backbone=%s)", init_from, info["backbone"])

    criterion = criterion_for(cfg.train)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    use_amp = amp_enabled(device, cfg.train.use_amp)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None

    if cfg.train.freeze_backbone_epochs > 0 and model.mode in ("fusion", "spatial_only", "concat_no_gate"):
        set_backbone_trainable(model, False)
        LOGGER.info("Backbone frozen for the first %d epoch(s).", cfg.train.freeze_backbone_epochs)

    best_path = ensure_dir(cfg.model_path) / "AdaptiveFusion_best_model.pth"
    last_path = cfg.model_path / "last_checkpoint.pt"
    history_path = cfg.model_path / "training_history.csv"
    run_name = run_name or cfg.train.run_name or f"{cfg.train.mode}"
    history: list[dict[str, Any]] = []
    best_f1, bad_epochs, start_epoch = -1.0, 0, 0

    if cfg.train.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        history = checkpoint.get("history", [])
        best_f1 = checkpoint.get("best_f1", -1.0)
        bad_epochs = checkpoint.get("bad_epochs", 0)
        start_epoch = checkpoint.get("epoch", 0)
        LOGGER.info("Resumed at epoch %d (best val F1 %.4f)", start_epoch, best_f1)

    params_for_tracking = {
        **{f"train.{k}": v for k, v in asdict(cfg.train).items()},
        **{f"data.{k}": v for k, v in asdict(cfg.data).items()},
        "data.fingerprint": dataset_fingerprint(frames),
        "n_train": len(frames["train"]),
        "n_valid": len(frames["valid"]),
        "n_test": len(frames["test"]),
    }
    tracking.log_params(cfg, params_for_tracking)
    tracking.log_params(cfg, {"environment": json.dumps(describe_environment(), default=str)})

    with tracking.run(cfg, run_name, tags={"mode": cfg.train.mode, "backbone": cfg.train.backbone}) as mlflow:
        tracking.log_params(cfg, {"model": json.dumps(model.describe(), default=str)})
        for epoch in range(start_epoch, cfg.train.epochs):
            started = time.time()
            if (
                cfg.train.freeze_backbone_epochs > 0
                and epoch == cfg.train.freeze_backbone_epochs
                and model.mode in ("fusion", "spatial_only", "concat_no_gate")
            ):
                set_backbone_trainable(model, True)
                LOGGER.info("Backbone unfrozen at epoch %d (gradual fine-tuning).", epoch + 1)

            train_metrics = train_one_epoch(
                model,
                loaders["train"],
                optimizer,
                criterion,
                cfg,
                device,
                scaler=scaler,
                use_amp=use_amp,
                epoch=epoch + 1,
            )
            val = None
            val_metrics: Dict[str, float] = {}
            if loaders["valid"] is not None:
                val = predict_loader(
                    model,
                    loaders["valid"],
                    device=device,
                    criterion=criterion,
                    use_amp=use_amp,
                    threshold=cfg.eval.threshold,
                )
                val_metrics = compute_metrics(val["y_true"], val["y_pred"], val["y_prob"])

            backbone_lr = next(
                (g["lr"] for g in optimizer.param_groups if g.get("name") == "backbone"),
                optimizer.param_groups[0]["lr"],
            )
            row: Dict[str, Any] = {
                "epoch": epoch + 1,
                "loss": train_metrics["loss"],
                "acc": train_metrics.get("accuracy", float("nan")),
                "f1": train_metrics.get("f1", float("nan")),
                "bal_acc": train_metrics.get("balanced_accuracy", float("nan")),
                "val_loss": val["loss"] if val else float("nan"),
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "lr_backbone": backbone_lr,
                "seconds": round(time.time() - started, 1),
            }
            history.append(row)
            pd.DataFrame(history).to_csv(history_path, index=False)
            tracking.log_metrics(cfg, {k: v for k, v in row.items() if k != "epoch"}, step=epoch + 1)

            improved = val_metrics.get("f1", -1.0) > best_f1
            if improved:
                best_f1 = val_metrics.get("f1", -1.0)
                bad_epochs = 0
                state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save(state, best_path)  # raw state_dict: same format as the source notebook
                write_json(
                    cfg.model_path / "model_config.json",
                    model_config(
                        model,
                        image_size=cfg.data.image_size,
                        training=asdict(cfg.train),
                        best_metrics={f"val_{k}": v for k, v in val_metrics.items()},
                        environment=describe_environment(),
                        data_version=dataset_fingerprint(frames),
                    ),
                )
            else:
                bad_epochs += 1

            scheduler.step()
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "history": history,
                    "best_f1": best_f1,
                    "bad_epochs": bad_epochs,
                    "epoch": epoch + 1,
                },
                last_path,
            )
            LOGGER.info(
                "Epoch %02d/%02d | loss %.4f/%.4f | acc %.2f%%/%.2f%% | F1 %.4f/%.4f | %.0fs %s",
                epoch + 1,
                cfg.train.epochs,
                row["loss"],
                row.get("val_loss", float("nan")),
                100 * row.get("acc", float("nan")),
                100 * row.get("val_accuracy", float("nan")),
                row.get("f1", float("nan")),
                row.get("val_f1", float("nan")),
                row["seconds"],
                "* best" if improved else f"(no gain {bad_epochs}/{cfg.train.patience})",
            )
            if bad_epochs >= cfg.train.patience:
                LOGGER.info("Early stopping at epoch %d.", epoch + 1)
                break

        if best_path.exists():
            state = torch.load(best_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            LOGGER.info("Best weights restored from %s", best_path.name)

        figures_dir = ensure_dir(cfg.reports_path / "figures")
        curve_path = figures_dir / f"training_curves_{cfg.train.mode}.png"
        plot_training_history(pd.DataFrame(history).to_dict("list"), curve_path)
        tracking.log_artifact(cfg, history_path, artifact_path="training")
        tracking.log_artifact(cfg, curve_path, artifact_path="figures")
        tracking.log_artifact(cfg, cfg.model_path / "model_config.json", artifact_path="model")
        if cfg.tracking.log_artifacts:
            tracking.log_artifact(cfg, best_path, artifact_path="model")

        summary = {
            "run_name": run_name,
            "mode": cfg.train.mode,
            "best_val_f1": float(best_f1),
            "epochs_run": len(history),
            "weights": str(best_path),
            "history": str(history_path),
            "model_config": str(cfg.model_path / "model_config.json"),
            "backbone": model.backbone_name,
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "active_parameters": model.active_parameters(),
            "device": str(device),
            "mlflow_run_id": mlflow.active_run().info.run_id if mlflow else None,
        }
        tracking.log_metrics(cfg, {"best_val_f1": best_f1, "epochs_run": len(history)})
        tracking.log_dict(cfg, summary, "run_summary.json")
        write_json(cfg.reports_path / f"train_summary_{cfg.train.mode}.json", summary)

    if cfg.train.mode == "fusion" and not (cfg.model_path / "AdaptiveFusion_best_model.pth").exists():
        LOGGER.warning("No fusion checkpoint produced — the API needs one (or an ONNX file).")
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Adaptive Fusion variant.")
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--mode", choices=MODES, default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr-backbone", type=float, default=None)
    parser.add_argument("--lr-fusion", type=float, default=None)
    parser.add_argument("--lr-classifier", type=float, default=None)
    parser.add_argument("--train-per-class", type=int, default=None)
    parser.add_argument("--valid-per-class", type=int, default=None)
    parser.add_argument("--test-per-class", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--init-from", default=None, help="checkpoint used to initialise the weights")
    parser.add_argument("--quick-demo", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--focal-loss", action="store_true")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    cfg = load_config(args.config, quick_demo=args.quick_demo)
    cfg.with_overrides(
        train__mode=args.mode,
        train__backbone=args.backbone,
        train__epochs=args.epochs,
        train__batch_size=args.batch_size,
        train__lr_backbone=args.lr_backbone,
        train__lr_fusion=args.lr_fusion,
        train__lr_classifier=args.lr_classifier,
        train__resume=args.resume or None,
        train__run_name=args.run_name,
    )
    cfg.data.train_per_class = args.train_per_class or cfg.data.train_per_class
    cfg.data.valid_per_class = args.valid_per_class or cfg.data.valid_per_class
    cfg.data.test_per_class = args.test_per_class or cfg.data.test_per_class
    cfg.data.num_workers = args.num_workers if args.num_workers is not None else cfg.data.num_workers
    if args.no_amp:
        cfg.train.use_amp = False
    if args.focal_loss:
        cfg.train.focal_loss = True
    if args.freeze_backbone_epochs is not None:
        cfg.train.freeze_backbone_epochs = args.freeze_backbone_epochs
    if args.no_mlflow:
        cfg.tracking.enabled = False
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = config_from_args(args)
    summary = run_training(
        cfg,
        device=resolve_device(args.device),
        root=args.dataset_root,
        run_name=args.run_name,
        init_from=args.init_from,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
