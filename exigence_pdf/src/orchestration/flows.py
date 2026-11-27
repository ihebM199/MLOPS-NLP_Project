"""Prefect flows for the deepfake detection pipeline.

Flows:
    - main_pipeline_flow: prepare -> train -> evaluate -> export_onnx
    - retrain_flow: triggered on data drift or metric degradation

Each task has retry logic, error handling, and logging for observability.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logging import get_logger

LOGGER = get_logger("orchestration.flows")

# --------------------------------------------------------------------------- #
# Lazy Prefect imports (optional dependency)
# --------------------------------------------------------------------------- #
_PREFECT_AVAILABLE = False
try:
    from prefect import flow, task, get_run_logger
    from prefect.tasks import task_input_hash
    from prefect.artifacts import create_markdown_artifact
    _PREFECT_AVAILABLE = True
except ImportError:
    # Provide no-op decorators so the module can be imported without Prefect
    def flow(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator if not args else args[0]
    
    def task(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator if not args else args[0]
    
    def get_run_logger():
        return LOGGER
    
    def task_input_hash(*args, **kwargs):
        return None
    
    def create_markdown_artifact(*args, **kwargs):
        return None


# --------------------------------------------------------------------------- #
# Task: Prepare data
# --------------------------------------------------------------------------- #
@task(
    name="prepare-data",
    retries=2,
    retry_delay_seconds=30,
    log_prints=True,
)
def prepare_data(config_path: str = "params.yaml") -> Dict[str, Any]:
    """Download and prepare the dataset.
    
    Retries: 2 attempts with 30s delay between retries.
    Error handling: Raises on failure after all retries exhausted.
    """
    logger = get_run_logger() if _PREFECT_AVAILABLE else LOGGER
    logger.info("Starting data preparation stage...")
    
    cmd = [sys.executable, "-m", "src.data.prepare", "--config", config_path]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour timeout
    )
    
    if result.returncode != 0:
        logger.error(f"Data preparation failed: {result.stderr}")
        raise RuntimeError(f"Data preparation failed with exit code {result.returncode}")
    
    logger.info("Data preparation completed successfully")
    return {"status": "success", "stage": "prepare"}


# --------------------------------------------------------------------------- #
# Task: Train model
# --------------------------------------------------------------------------- #
@task(
    name="train-model",
    retries=1,
    retry_delay_seconds=60,
    log_prints=True,
)
def train_model(config_path: str = "params.yaml") -> Dict[str, Any]:
    """Train the adaptive fusion model.
    
    Retries: 1 attempt (training is expensive, limited retries).
    Error handling: Raises on failure, logs training summary if available.
    """
    logger = get_run_logger() if _PREFECT_AVAILABLE else LOGGER
    logger.info("Starting model training stage...")
    
    cmd = [sys.executable, "-m", "src.models.train", "--config", config_path]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=14400,  # 4 hour timeout for training
    )
    
    if result.returncode != 0:
        logger.error(f"Training failed: {result.stderr}")
        raise RuntimeError(f"Training failed with exit code {result.returncode}")
    
    # Try to load training summary
    summary_path = Path("reports/train_summary_fusion.json")
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            logger.info(f"Training summary: {summary}")
        except Exception as e:
            logger.warning(f"Could not load training summary: {e}")
    
    logger.info("Model training completed successfully")
    return {"status": "success", "stage": "train", "summary": summary}


# --------------------------------------------------------------------------- #
# Task: Evaluate model
# --------------------------------------------------------------------------- #
@task(
    name="evaluate-model",
    retries=1,
    retry_delay_seconds=30,
    log_prints=True,
)
def evaluate_model(config_path: str = "params.yaml") -> Dict[str, Any]:
    """Evaluate the trained model on test data.
    
    Retries: 1 attempt with 30s delay.
    Error handling: Raises on failure, returns metrics on success.
    """
    logger = get_run_logger() if _PREFECT_AVAILABLE else LOGGER
    logger.info("Starting model evaluation stage...")
    
    cmd = [sys.executable, "-m", "src.models.evaluate", "--config", config_path]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour timeout
    )
    
    if result.returncode != 0:
        logger.error(f"Evaluation failed: {result.stderr}")
        raise RuntimeError(f"Evaluation failed with exit code {result.returncode}")
    
    # Load metrics
    metrics_path = Path("reports/metrics.json")
    metrics = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            logger.info(f"Evaluation metrics: {metrics}")
        except Exception as e:
            logger.warning(f"Could not load metrics: {e}")
    
    logger.info("Model evaluation completed successfully")
    return {"status": "success", "stage": "evaluate", "metrics": metrics}


# --------------------------------------------------------------------------- #
# Task: Export ONNX
# --------------------------------------------------------------------------- #
@task(
    name="export-onnx",
    retries=2,
    retry_delay_seconds=30,
    log_prints=True,
)
def export_onnx(config_path: str = "params.yaml") -> Dict[str, Any]:
    """Export the trained model to ONNX format.
    
    Retries: 2 attempts with 30s delay (export is fast, can retry).
    Error handling: Raises on failure after all retries exhausted.
    """
    logger = get_run_logger() if _PREFECT_AVAILABLE else LOGGER
    logger.info("Starting ONNX export stage...")
    
    cmd = [sys.executable, "-m", "src.models.onnx_export", "--config", config_path]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 minute timeout
    )
    
    if result.returncode != 0:
        logger.error(f"ONNX export failed: {result.stderr}")
        raise RuntimeError(f"ONNX export failed with exit code {result.returncode}")
    
    logger.info("ONNX export completed successfully")
    return {"status": "success", "stage": "export_onnx"}


# --------------------------------------------------------------------------- #
# Task: Register model in MLflow
# --------------------------------------------------------------------------- #
@task(
    name="register-model",
    retries=2,
    retry_delay_seconds=15,
    log_prints=True,
)
def register_model(config_path: str = "params.yaml") -> Dict[str, Any]:
    """Register the model in MLflow Model Registry.
    
    Retries: 2 attempts with 15s delay.
    Error handling: Logs warning but doesn't fail the flow (best-effort).
    """
    logger = get_run_logger() if _PREFECT_AVAILABLE else LOGGER
    logger.info("Registering model in MLflow...")
    
    try:
        from src.config import load_config
        from src.tracking import register_onnx_model
        
        cfg = load_config(config_path)
        onnx_path = cfg.onnx_path
        
        if not onnx_path.exists():
            logger.warning(f"ONNX model not found at {onnx_path}, skipping registration")
            return {"status": "skipped", "reason": "model_not_found"}
        
        version = register_onnx_model(cfg, onnx_path, alias="staging")
        
        if version:
            logger.info(f"Model registered as version {version}")
            return {"status": "success", "version": version}
        else:
            logger.warning("Model registration returned no version")
            return {"status": "failed", "reason": "no_version"}
            
    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        # Don't raise - registration is best-effort
        return {"status": "failed", "error": str(e)}


# --------------------------------------------------------------------------- #
# Task: Check model quality
# --------------------------------------------------------------------------- #
@task(
    name="check-model-quality",
    retries=0,  # No retries for quality checks
    log_prints=True,
)
def check_model_quality(min_auc: float = 0.85, min_f1: float = 0.80) -> Dict[str, Any]:
    """Validate that the model meets quality thresholds.
    
    This task checks if the trained model meets the minimum quality criteria.
    It does NOT retry - if quality is insufficient, the flow should fail.
    
    Args:
        min_auc: Minimum acceptable AUC score
        min_f1: Minimum acceptable F1 score
    """
    logger = get_run_logger() if _PREFECT_AVAILABLE else LOGGER
    
    metrics_path = Path("reports/metrics.json")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    metrics = json.loads(metrics_path.read_text())
    auc = metrics.get("auc", 0.0)
    f1 = metrics.get("f1", 0.0)
    
    logger.info(f"Model quality: AUC={auc:.4f}, F1={f1:.4f}")
    
    quality_passed = True
    issues = []
    
    if auc < min_auc:
        quality_passed = False
        issues.append(f"AUC {auc:.4f} < {min_auc}")
    
    if f1 < min_f1:
        quality_passed = False
        issues.append(f"F1 {f1:.4f} < {min_f1}")
    
    if not quality_passed:
        raise ValueError(f"Model quality check failed: {', '.join(issues)}")
    
    logger.info("Model quality check passed")
    return {"status": "success", "auc": auc, "f1": f1}


# --------------------------------------------------------------------------- #
# Main pipeline flow
# --------------------------------------------------------------------------- #
@flow(
    name="deepfake-adaptive-fusion-pipeline",
    description="Full MLOps pipeline: prepare -> train -> evaluate -> export -> register",
    retries=0,  # No flow-level retries; tasks handle their own retries
    on_failure=[lambda flow, flow_run, state: LOGGER.error(f"Pipeline failed: {state}")],
)
def main_pipeline_flow(
    config_path: str = "params.yaml",
    register: bool = False,
    quality_check: bool = True,
) -> Dict[str, Any]:
    """Main MLOps pipeline orchestration.
    
    This flow orchestrates the complete ML pipeline:
    1. Prepare data (download, preprocess)
    2. Train model (with MLflow tracking)
    3. Evaluate model (metrics, calibration, robustness)
    4. Check model quality (optional gate)
    5. Export to ONNX
    6. Register in MLflow Model Registry (optional)
    
    Error Handling:
        - Each task has its own retry policy
        - Tasks log errors before raising
        - Flow fails fast on quality gate failure
    
    Recovery:
        - Prefect automatically retries failed tasks
        - Failed tasks can be inspected in Prefect UI
        - Flow can be re-run from failure point
    
    Args:
        config_path: Path to params.yaml
        register: Whether to register the model in MLflow
        quality_check: Whether to enforce quality gates
        
    Returns:
        Dictionary with pipeline execution summary
    """
    logger = get_run_logger() if _PREFECT_AVAILABLE else LOGGER
    logger.info("=" * 60)
    logger.info("Starting Deepfake Adaptive Fusion Pipeline")
    logger.info(f"Config: {config_path}, Register: {register}, Quality Check: {quality_check}")
    logger.info("=" * 60)
    
    results = {}
    
    # Stage 1: Prepare data
    results["prepare"] = prepare_data(config_path)
    
    # Stage 2: Train model
    results["train"] = train_model(config_path)
    
    # Stage 3: Evaluate model
    results["evaluate"] = evaluate_model(config_path)
    
    # Stage 4: Quality check (optional gate)
    if quality_check:
        results["quality"] = check_model_quality()
    
    # Stage 5: Export ONNX
    results["export"] = export_onnx(config_path)
    
    # Stage 6: Register model (optional)
    if register:
        results["register"] = register_model(config_path)
    
    # Create pipeline summary artifact
    summary = {
        "pipeline": "deepfake-adaptive-fusion",
        "config": config_path,
        "stages": results,
        "status": "success",
    }
    
    if _PREFECT_AVAILABLE:
        markdown = f"""# Pipeline Execution Summary

**Status:** ✅ Success
**Config:** `{config_path}`

## Stages

| Stage | Status |
|-------|--------|
| Prepare | {results.get('prepare', {}).get('status', 'N/A')} |
| Train | {results.get('train', {}).get('status', 'N/A')} |
| Evaluate | {results.get('evaluate', {}).get('status', 'N/A')} |
| Quality Check | {results.get('quality', {}).get('status', 'N/A')} |
| Export ONNX | {results.get('export', {}).get('status', 'N/A')} |
| Register | {results.get('register', {}).get('status', 'N/A')} |
"""
        create_markdown_artifact(key="pipeline-summary", markdown=markdown)
    
    logger.info("Pipeline completed successfully")
    logger.info(f"Results: {json.dumps(summary, indent=2, default=str)}")
    
    return summary


# --------------------------------------------------------------------------- #
# Continuous Training flow
# --------------------------------------------------------------------------- #
@flow(
    name="deepfake-retrain-flow",
    description="Triggered retraining on data drift or metric degradation",
)
def retrain_flow(
    config_path: str = "params.yaml",
    trigger_reason: str = "manual",
    force: bool = False,
) -> Dict[str, Any]:
    """Retraining flow triggered by monitoring alerts.
    
    This flow is triggered when:
    - Data drift is detected (Evidently AI)
    - Model performance degrades below threshold
    - Manual trigger from CI/CD pipeline
    
    Args:
        config_path: Path to params.yaml
        trigger_reason: Why retraining was triggered
        force: Force retraining even if quality is acceptable
        
    Returns:
        Dictionary with retraining execution summary
    """
    logger = get_run_logger() if _PREFECT_AVAILABLE else LOGGER
    logger.info("=" * 60)
    logger.info("Starting Retraining Flow")
    logger.info(f"Trigger: {trigger_reason}, Force: {force}")
    logger.info("=" * 60)
    
    # Run the full pipeline with registration enabled
    results = main_pipeline_flow(
        config_path=config_path,
        register=True,
        quality_check=True,
    )
    
    results["trigger"] = trigger_reason
    results["force"] = force
    
    logger.info("Retraining completed successfully")
    return results


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main():
    """CLI entry point for running the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the MLOps pipeline")
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--register", action="store_true", help="Register model in MLflow")
    parser.add_argument("--no-quality-check", action="store_true", help="Skip quality gates")
    parser.add_argument("--retrain", action="store_true", help="Run retraining flow")
    parser.add_argument("--trigger", default="manual", help="Trigger reason for retraining")
    
    args = parser.parse_args()
    
    if args.retrain:
        retrain_flow(
            config_path=args.config,
            trigger_reason=args.trigger,
        )
    else:
        main_pipeline_flow(
            config_path=args.config,
            register=args.register,
            quality_check=not args.no_quality_check,
        )


if __name__ == "__main__":
    main()
