"""Prometheus metrics for system and model monitoring.

Exposes:
    System metrics:
        - Request count, latency, error rate
        - CPU / memory usage
        - GPU utilization (if available)

    Model metrics:
        - Prediction count (by label)
        - Prediction confidence distribution
        - Fusion weight statistics
        - Model latency (preprocess + inference)
        - Data drift scores

Uses the `prometheus_client` library. Falls back to a no-op collector
when the library is not installed.
"""

from __future__ import annotations

import os
import time
import json
from pathlib import Path
from typing import Dict, Optional, Any

from src.utils.logging import get_logger

LOGGER = get_logger("monitoring.metrics")

# --------------------------------------------------------------------------- #
# Prometheus client (optional)
# --------------------------------------------------------------------------- #
_PROMETHEUS_AVAILABLE = False
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Summary,
        Info,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    LOGGER.info("prometheus_client not installed — metrics collection disabled")


# --------------------------------------------------------------------------- #
# Metrics Collector
# --------------------------------------------------------------------------- #
class MetricsCollector:
    """Singleton collector for all Prometheus metrics.
    
    Thread-safe. Lazy initialization. No-op when prometheus_client is absent.
    """
    
    _instance: Optional["MetricsCollector"] = None
    
    def __init__(self, prefix: str = "deepfake"):
        if not _PROMETHEUS_AVAILABLE:
            self._enabled = False
            return
        
        self._enabled = True
        self._prefix = prefix
        
        # --- System Metrics ---
        self.request_count = Counter(
            f"{prefix}_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"],
        )
        self.request_latency = Histogram(
            f"{prefix}_http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )
        self.request_errors = Counter(
            f"{prefix}_http_request_errors_total",
            "Total HTTP request errors",
            ["method", "endpoint", "error_type"],
        )
        
        # Resource usage
        self.cpu_usage = Gauge(
            f"{prefix}_cpu_usage_percent",
            "CPU usage percentage",
        )
        self.memory_usage = Gauge(
            f"{prefix}_memory_usage_bytes",
            "Memory usage in bytes",
        )
        self.memory_total = Gauge(
            f"{prefix}_memory_total_bytes",
            "Total memory in bytes",
        )
        self.gpu_utilization = Gauge(
            f"{prefix}_gpu_utilization_percent",
            "GPU utilization percentage",
            ["gpu_id"],
        )
        self.gpu_memory = Gauge(
            f"{prefix}_gpu_memory_bytes",
            "GPU memory usage in bytes",
            ["gpu_id"],
        )
        
        # --- Model Metrics ---
        self.prediction_count = Counter(
            f"{prefix}_predictions_total",
            "Total predictions",
            ["label"],
        )
        self.prediction_confidence = Histogram(
            f"{prefix}_prediction_confidence",
            "Prediction confidence distribution",
            ["label"],
            buckets=tuple(round(i * 0.05, 2) for i in range(21)),  # 0.0 to 1.0
        )
        self.prediction_latency = Histogram(
            f"{prefix}_prediction_latency_seconds",
            "Prediction latency in seconds",
            ["stage"],  # preprocess, inference, postprocess, total
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )
        
        # Fusion weights
        self.fusion_weight_spatial = Histogram(
            f"{prefix}_fusion_weight_spatial",
            "Spatial fusion weight distribution",
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )
        self.fusion_weight_frequency = Histogram(
            f"{prefix}_fusion_weight_frequency",
            "Frequency fusion weight distribution",
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )
        
        # Model quality / drift
        self.drift_score = Gauge(
            f"{prefix}_data_drift_score",
            "Data drift score (0-1, higher = more drift)",
            ["feature"],
        )
        self.model_version = Info(
            f"{prefix}_model",
            "Model metadata",
        )
        
        # Upload metrics
        self.upload_size = Histogram(
            f"{prefix}_upload_size_bytes",
            "Upload file size distribution",
            buckets=(1024, 10240, 102400, 1048576, 5242880, 10485760),
        )
        
        LOGGER.info("Prometheus metrics collector initialized")
    
    @classmethod
    def get_instance(cls, prefix: str = "deepfake") -> "MetricsCollector":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(prefix=prefix)
        return cls._instance
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    # ------------------------------------------------------------------ #
    # Recording methods
    # ------------------------------------------------------------------ #
    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
    ) -> None:
        """Record an HTTP request."""
        if not self._enabled:
            return
        self.request_count.labels(
            method=method, endpoint=endpoint, status_code=str(status_code)
        ).inc()
        self.request_latency.labels(method=method, endpoint=endpoint).observe(duration)
        if status_code >= 500:
            self.request_errors.labels(
                method=method, endpoint=endpoint, error_type="server_error"
            ).inc()
    
    def record_prediction(
        self,
        label: str,
        confidence: float,
        latency_total: float,
        latency_preprocess: float = 0.0,
        latency_inference: float = 0.0,
        latency_postprocess: float = 0.0,
        w_spatial: float = 0.5,
        w_frequency: float = 0.5,
    ) -> None:
        """Record a prediction with all its metadata."""
        if not self._enabled:
            return
        self.prediction_count.labels(label=label).inc()
        self.prediction_confidence.labels(label=label).observe(confidence)
        self.prediction_latency.labels(stage="preprocess").observe(latency_preprocess)
        self.prediction_latency.labels(stage="inference").observe(latency_inference)
        self.prediction_latency.labels(stage="postprocess").observe(latency_postprocess)
        self.prediction_latency.labels(stage="total").observe(latency_total)
        self.fusion_weight_spatial.observe(w_spatial)
        self.fusion_weight_frequency.observe(w_frequency)
    
    def record_drift(self, feature: str, score: float) -> None:
        """Record a data drift score."""
        if not self._enabled:
            return
        self.drift_score.labels(feature=feature).set(score)
    
    def record_upload(self, size_bytes: int) -> None:
        """Record an upload file size."""
        if not self._enabled:
            return
        self.upload_size.observe(size_bytes)
    
    def set_model_info(self, version: str, backbone: str, threshold: str) -> None:
        """Set model metadata."""
        if not self._enabled:
            return
        self.model_version.info({
            "version": version,
            "backbone": backbone,
            "threshold": threshold,
        })
    
    def update_resource_metrics(self) -> None:
        """Update system resource metrics."""
        if not self._enabled:
            return
        try:
            import psutil
            self.cpu_usage.set(psutil.cpu_percent(interval=0))
            mem = psutil.virtual_memory()
            self.memory_usage.set(mem.used)
            self.memory_total.set(mem.total)
        except ImportError:
            pass
        except Exception as e:
            LOGGER.debug(f"Could not update resource metrics: {e}")
    
    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    def generate_latest(self) -> bytes:
        """Generate Prometheus exposition format."""
        if not self._enabled:
            return b"# Metrics disabled (prometheus_client not installed)\n"
        return generate_latest()
    
    def content_type(self) -> str:
        """Return the Prometheus content type."""
        if not self._enabled:
            return "text/plain"
        return CONTENT_TYPE_LATEST


def get_metrics_collector() -> MetricsCollector:
    """Convenience accessor for the singleton collector."""
    return MetricsCollector.get_instance()
