"""Django integration for the project monitoring stack.

Re-uses ``src.monitoring`` (the same collectors as the FastAPI backend):

    GET  /metrics                    -> Prometheus exposition format
    GET  /monitoring/stats           -> prediction statistics (JSON)
    POST /monitoring/drift           -> drift report (JSON)
    POST /monitoring/log-prediction  -> manually append one prediction

Every ``/predict`` call records the prediction (label, confidence, fusion
weights, latency breakdown) and appends a JSONL entry for later drift analysis.

The collector degrades to a no-op when ``prometheus_client`` is not installed,
and a monitoring failure must never break inference.
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from api.config import get_settings
from src.utils.logging import get_logger

LOGGER = get_logger("deepguard.monitoring")


@lru_cache(maxsize=1)
def get_metrics():
    """Singleton Prometheus collector (no-op if the lib is missing)."""
    from src.monitoring.metrics import get_metrics_collector

    return get_metrics_collector()


@lru_cache(maxsize=1)
def get_prediction_logger():
    """JSONL prediction logger used for /monitoring/stats and drift."""
    import atexit

    from src.monitoring.prediction_logger import PredictionLogger

    logger = PredictionLogger(output_dir=str(get_settings().monitoring_data_dir))
    atexit.register(logger.flush)  # never lose the last buffered predictions
    return logger


def record_prediction(
    result: Dict[str, Any],
    total_ms: float,
    *,
    upload_bytes: Optional[int] = None,
) -> None:
    """Record one prediction in Prometheus + the JSONL log (never raises)."""
    try:
        metrics = get_metrics()
        prob_fake = float(result.get("prob_fake", 0.5))
        label = str(result.get("label", "unknown"))
        weights = result.get("fusion_weights") or {}
        latency = result.get("latency") or {}

        if metrics.enabled:
            metrics.record_prediction(
                label=label,
                confidence=prob_fake,
                latency_total=total_ms / 1000.0,
                latency_inference=float(latency.get("inference_ms", 0.0)) / 1000.0,
                latency_preprocess=float(latency.get("preprocess_ms", 0.0)) / 1000.0,
                w_spatial=float(weights.get("spatial", 0.5)),
                w_frequency=float(weights.get("frequency", 0.5)),
            )
            if upload_bytes is not None:
                metrics.record_upload(int(upload_bytes))

        try:
            get_prediction_logger().log_prediction(
                prediction={"label": label, "confidence": prob_fake, "prob_fake": prob_fake},
                metadata={
                    "latency_ms": round(total_ms, 2),
                    "model_version": get_settings().model_version,
                },
            )
        except Exception:  # noqa: BLE001 - logging must never break inference
            LOGGER.debug("Could not append prediction to the JSONL log", exc_info=True)
    except Exception:  # noqa: BLE001 - monitoring must never break inference
        LOGGER.debug("Could not record prediction metrics", exc_info=True)


class PrometheusMiddleware:
    """Record every HTTP request (status + latency + resource usage)."""

    def __init__(self, get_response) -> None:
        self.get_response = get_response

    def __call__(self, request):
        started = time.perf_counter()
        response = self.get_response(request)
        if request.path not in ("/metrics", "/metrics/"):
            try:
                metrics = get_metrics()
                if metrics.enabled:
                    metrics.record_request(
                        method=request.method,
                        endpoint=request.path,
                        status_code=response.status_code,
                        duration=time.perf_counter() - started,
                    )
                    metrics.update_resource_metrics()
            except Exception:  # noqa: BLE001
                LOGGER.debug("Could not record request metrics", exc_info=True)
        return response


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@require_GET
def metrics_view(request):
    metrics = get_metrics()
    if not metrics.enabled:
        return HttpResponse(
            b"# prometheus_client not installed - metrics disabled\n",
            content_type="text/plain",
        )
    return HttpResponse(metrics.generate_latest(), content_type=metrics.content_type())


@require_GET
def monitoring_stats(request):
    try:
        logger = get_prediction_logger()
        logger.flush()
        return JsonResponse(logger.get_prediction_stats())
    except Exception as exc:  # noqa: BLE001
        return JsonResponse({"error": "stats_unavailable", "detail": str(exc)}, status=503)


@csrf_exempt
@require_POST
def monitoring_drift(request):
    import numpy as np

    from src.monitoring.drift import DriftDetector

    try:
        min_samples = int(request.GET.get("min_samples", "50"))
    except ValueError:
        return JsonResponse(
            {"error": "validation_error", "detail": "min_samples must be an integer"},
            status=422,
        )

    try:
        logger = get_prediction_logger()
        logger.flush()
        frame = logger.load_predictions()
    except Exception as exc:  # noqa: BLE001
        return JsonResponse({"error": "logs_unavailable", "detail": str(exc)}, status=503)

    if len(frame) < min_samples:
        return JsonResponse(
            {
                "error": "insufficient_data",
                "message": f"Need at least {min_samples} predictions, have {len(frame)}",
                "current_count": int(len(frame)),
            },
            status=400,
        )
    if "prediction" not in frame.columns:
        return JsonResponse(
            {"error": "no_predictions", "message": "No prediction data logged"},
            status=400,
        )

    confidences = frame["prediction"].apply(
        lambda value: value.get("confidence", value.get("prob_fake", 0.5)) if isinstance(value, dict) else 0.5
    )
    midpoint = len(confidences) // 2
    reference = np.asarray(confidences.iloc[:midpoint], dtype=float)
    current = np.asarray(confidences.iloc[midpoint:], dtype=float)

    report = DriftDetector().detect_prediction_drift(
        recent_predictions=current,
        baseline_predictions=reference,
    )

    try:
        metrics = get_metrics()
        if metrics.enabled:
            metrics.record_drift("prediction_confidence", report.drift_score)
    except Exception:  # noqa: BLE001
        LOGGER.debug("Could not record drift metric", exc_info=True)

    try:
        report.to_json(Path(get_settings().monitoring_data_dir) / "drift_report.json")
    except Exception:  # noqa: BLE001
        LOGGER.debug("Could not persist the drift report", exc_info=True)

    return JsonResponse(report.to_dict())


@csrf_exempt
@require_POST
def log_prediction(request):
    try:
        body = json.loads(request.body or b"{}")
    except (TypeError, ValueError):
        return JsonResponse({"error": "validation_error", "detail": "Invalid JSON body"}, status=422)
    try:
        get_prediction_logger().log_prediction(
            features=body.get("features"),
            prediction=body.get("prediction"),
            metadata=body.get("metadata"),
        )
    except Exception as exc:  # noqa: BLE001
        return JsonResponse({"error": "logging_failed", "detail": str(exc)}, status=500)
    return JsonResponse({"status": "logged"})
