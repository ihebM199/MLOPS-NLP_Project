"""Prometheus /metrics endpoint and monitoring middleware for FastAPI.

Adds:
    GET /metrics - Prometheus exposition format
    Middleware that records every request/response for system monitoring
    POST /monitoring/drift - Trigger drift analysis
    GET /monitoring/stats - Prediction statistics
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.monitoring.metrics import MetricsCollector, get_metrics_collector
from src.utils.logging import get_logger

LOGGER = get_logger("monitoring.api")


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware that records Prometheus metrics for every HTTP request."""
    
    def __init__(self, app: FastAPI, metrics: Optional[MetricsCollector] = None):
        super().__init__(app)
        self.metrics = metrics or get_metrics_collector()
    
    async def dispatch(self, request: Request, call_next):
        started = time.perf_counter()
        endpoint = request.url.path
        
        # Skip metrics endpoint itself
        if endpoint == "/metrics":
            return await call_next(request)
        
        try:
            response = await call_next(request)
            duration = time.perf_counter() - started
            
            if self.metrics.enabled:
                self.metrics.record_request(
                    method=request.method,
                    endpoint=endpoint,
                    status_code=response.status_code,
                    duration=duration,
                )
                self.metrics.update_resource_metrics()
            
            return response
        except Exception as exc:
            duration = time.perf_counter() - started
            if self.metrics.enabled:
                self.metrics.record_request(
                    method=request.method,
                    endpoint=endpoint,
                    status_code=500,
                    duration=duration,
                )
            raise


def setup_monitoring(
    app: FastAPI,
    metrics: Optional[MetricsCollector] = None,
    prediction_log_dir: str = "monitoring_data",
) -> MetricsCollector:
    """Add monitoring endpoints and middleware to a FastAPI app.
    
    Endpoints added:
        GET  /metrics           -> Prometheus exposition
        GET  /monitoring/stats  -> Prediction statistics summary
        POST /monitoring/drift  -> Trigger drift analysis
    
    Args:
        app: FastAPI application
        metrics: Optional MetricsCollector instance
        prediction_log_dir: Directory for prediction logs
        
    Returns:
        The MetricsCollector instance used
    """
    metrics = metrics or get_metrics_collector()
    
    # Add Prometheus middleware (after CORS)
    app.add_middleware(PrometheusMiddleware, metrics=metrics)
    
    # Initialize prediction logger
    from src.monitoring.prediction_logger import PredictionLogger
    pred_logger = PredictionLogger(output_dir=prediction_log_dir)
    app.state.prediction_logger = pred_logger
    app.state.metrics_collector = metrics
    
    # --------------------------------------------------------------- #
    # Monitoring endpoints
    # --------------------------------------------------------------- #
    
    @app.get(
        "/metrics",
        tags=["monitoring"],
        summary="Prometheus metrics endpoint",
    )
    async def prometheus_metrics():
        """Expose metrics in Prometheus exposition format."""
        content = metrics.generate_latest()
        return Response(
            content=content,
            media_type=metrics.content_type(),
        )
    
    @app.get(
        "/monitoring/stats",
        tags=["monitoring"],
        summary="Prediction statistics and drift summary",
    )
    async def monitoring_stats():
        """Return prediction statistics from the local log."""
        stats = pred_logger.get_prediction_stats()
        return JSONResponse(content=stats)
    
    @app.post(
        "/monitoring/drift",
        tags=["monitoring"],
        summary="Run drift detection on recent predictions",
    )
    async def run_drift_detection(
        reference_path: Optional[str] = None,
        min_samples: int = 50,
    ):
        """Trigger drift analysis on recent predictions.
        
        Compares recent predictions against a reference distribution.
        Returns drift report with per-feature analysis.
        """
        import numpy as np
        from src.monitoring.drift import DriftDetector
        
        df = pred_logger.load_predictions()
        
        if len(df) < min_samples:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "insufficient_data",
                    "message": f"Need at least {min_samples} predictions, have {len(df)}",
                    "current_count": len(df),
                },
            )
        
        # Extract prediction confidences
        if "prediction" in df.columns:
            confidences = df["prediction"].apply(
                lambda x: x.get("confidence", x.get("prob_fake", 0.5))
                if isinstance(x, dict) else 0.5
            )
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "no_predictions", "message": "No prediction data logged"},
            )
        
        # Simple drift detection: compare recent half vs first half
        midpoint = len(confidences) // 2
        reference = np.array(confidences.iloc[:midpoint])
        current = np.array(confidences.iloc[midpoint:])
        
        detector = DriftDetector()
        report = detector.detect_prediction_drift(
            recent_predictions=current,
            baseline_predictions=reference,
        )
        
        # Record drift score in metrics
        if metrics.enabled:
            metrics.record_drift("prediction_confidence", report.drift_score)
        
        # Save report
        report_path = Path("reports/drift_report.json")
        report.to_json(report_path)
        
        return JSONResponse(content=report.to_dict())
    
    @app.post(
        "/monitoring/log-prediction",
        tags=["monitoring"],
        summary="Log a prediction for monitoring",
    )
    async def log_prediction(request: Request):
        """Manually log a prediction for monitoring purposes."""
        body = await request.json()
        pred_logger.log_prediction(
            features=body.get("features"),
            prediction=body.get("prediction"),
            metadata=body.get("metadata"),
        )
        return JSONResponse(content={"status": "logged"})
    
    LOGGER.info("Monitoring endpoints added: /metrics, /monitoring/stats, /monitoring/drift")
    
    return metrics
