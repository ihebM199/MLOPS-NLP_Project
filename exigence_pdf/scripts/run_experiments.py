"""Run the mandatory comparison + ablations declared in ``experiments.yaml``.

    python scripts/run_experiments.py --quick-demo
    python scripts/run_experiments.py --only R1,R4 --no-mlflow
    python scripts/run_experiments.py            # full study (GPU recommended)

For every run the script:
  1. trains with the *same* data/seed/schedule (only the declared overrides differ);
  2. snapshots the checkpoint under ``models/runs/<run_name>.pth``;
  3. evaluates it (in-distribution test split + benchmark) and records the metrics;
  4. writes ``reports/comparison_table.csv``.

No number is invented: the table is built from the generated ``metrics.json``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.config import load_config  # noqa: E402
from src.models.evaluate import run_evaluation  # noqa: E402
from src.models.train import run_training  # noqa: E402

COMMON_OVERRIDES = {
    "seed": ("data__seed", "train__seed"),
    "epochs": ("train__epochs",),
    "batch_size": ("train__batch_size",),
    "image_size": ("data__image_size",),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline vs advanced study (MLflow-tracked).")
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--experiments", default="experiments.yaml")
    parser.add_argument("--only", default=None, help="comma-separated run ids, e.g. R1,R4")
    parser.add_argument("--include-ablations", action="store_true", default=True)
    parser.add_argument("--no-ablations", dest="include_ablations", action="store_false")
    parser.add_argument("--quick-demo", action="store_true")
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args(argv)


def load_runs(path: str | Path, include_ablations: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = yaml.safe_load(Path(path).read_text("utf-8")) or {}
    runs = list(payload.get("runs", []))
    if include_ablations:
        runs += list(payload.get("ablations", []))
    return payload.get("common", {}), runs


def build_config(base_config: str, common: dict[str, Any], run: dict[str, Any], *, quick_demo: bool, no_mlflow: bool):
    cfg = load_config(base_config, quick_demo=quick_demo)
    for key, value in common.items():
        for dotted in COMMON_OVERRIDES.get(key, ()):
            cfg.with_overrides(**{dotted: value})
    for key, value in (run.get("overrides") or {}).items():
        cfg.with_overrides(**{key: value})
    if no_mlflow:
        cfg.tracking.enabled = False
    cfg.prepare_dirs()
    return cfg


def snapshot_checkpoint(cfg, run_name: str) -> Path:
    source = cfg.weights_path
    if not source.exists():
        raise FileNotFoundError(f"No checkpoint produced for {run_name}: {source}")
    target = Path(cfg.model_dir) / "runs" / f"{run_name}.pth"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    config_source = Path(cfg.model_dir) / "model_config.json"
    if config_source.exists():
        shutil.copy2(config_source, target.with_suffix(".json"))
    return target


def evaluate_run(cfg, checkpoint: Path, run_name: str) -> dict[str, Any]:
    report = run_evaluation(
        cfg,
        checkpoint=checkpoint,
        do_export=False,
        do_quantize=False,
        do_robustness=False,
        do_cross_dataset=False,
        do_benchmark=True,
        run_name=f"{run_name}_eval",
    )
    test = report.get("test") or {}
    benchmark = report.get("benchmark") or {}
    return {
        "run": run_name,
        "mode": cfg.train.mode,
        "test_accuracy": test.get("accuracy"),
        "test_balanced_accuracy": test.get("balanced_accuracy"),
        "test_f1": test.get("f1"),
        "test_auc": test.get("auc"),
        "test_eer": test.get("eer"),
        "threshold": (report.get("threshold") or {}).get("selected", cfg.eval.threshold),
        "latency_ms_batch1": benchmark.get("latency_ms_batch1"),
        "parameters_m": (benchmark.get("parameters") or 0) / 1e6 if benchmark.get("parameters") else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    common, runs = load_runs(args.experiments, args.include_ablations)
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        runs = [run for run in runs if run.get("id") in wanted or run.get("name") in wanted]
    if not runs:
        print("No run selected.")
        return 1

    rows: list[dict[str, Any]] = []
    for run in runs:
        run_name = run.get("name") or run.get("id")
        print(f"\n=== {run_name} :: {run.get('description', '')} ===")
        cfg = build_config(
            args.config,
            common,
            run,
            quick_demo=args.quick_demo,
            no_mlflow=args.no_mlflow,
        )
        summary = run_training(cfg, run_name=run_name)
        checkpoint = snapshot_checkpoint(cfg, run_name)
        row = {
            "run": run_name,
            "mode": cfg.train.mode,
            "axis": run.get("axis"),
            "best_val_f1": summary.get("best_val_f1"),
            "parameters_m": (summary.get("parameters") or 0) / 1e6,
            "checkpoint": str(checkpoint),
        }
        if not args.no_evaluate:
            row.update(evaluate_run(cfg, checkpoint, run_name))
        rows.append(row)

    table = pd.DataFrame(rows)
    output = Path(load_config(args.config).reports_dir) / "comparison_table.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"\nComparison table -> {output}")
    if not table.empty:
        print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
