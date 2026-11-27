"""Prefect project configuration.

Defines the Prefect deployment for the deepfake detection pipeline.

Usage:
    prefect deployment build src/orchestration/deployment.py:main_pipeline_flow -n "deepfake-pipeline"
    prefect deployment run deepfake-adaptive-fusion-pipeline/deepfake-pipeline
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Prefect deployment configuration
# --------------------------------------------------------------------------- #
# When Prefect is available, this module is used to define deployments
# for the pipeline and retraining flows.

try:
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule, IntervalSchedule
    
    # Main pipeline deployment - runs on schedule
    pipeline_deployment = Deployment.build_from_flow(
        flow="src.orchestration.flows:main_pipeline_flow",
        name="deepfake-pipeline-daily",
        schedule=CronSchedule(cron="0 2 * * *"),  # Daily at 2 AM
        parameters={
            "config_path": "params.yaml",
            "register": True,
            "quality_check": True,
        },
        tags=["mlops", "deepfake", "pipeline"],
        description="Daily deepfake detection pipeline: prepare -> train -> evaluate -> export",
    )
    
    # Retraining deployment - triggered by monitoring alerts
    retrain_deployment = Deployment.build_from_flow(
        flow="src.orchestration.flows:retrain_flow",
        name="deepfake-retrain",
        tags=["mlops", "deepfake", "retrain"],
        description="On-demand retraining triggered by drift detection or metric degradation",
    )
    
    DEPLOYMENTS = [pipeline_deployment, retrain_deployment]
    
except ImportError:
    DEPLOYMENTS = []
