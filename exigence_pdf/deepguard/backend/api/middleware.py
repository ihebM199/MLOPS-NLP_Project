"""Small middlewares replicating the original FastAPI behaviours."""

from __future__ import annotations

import json
import time

from django.http import HttpResponse

from api.config import get_settings
from src.utils.logging import get_logger

LOGGER = get_logger("backend.api")


class TimingHeadersMiddleware:
    """X-Process-Time-ms on every response + structured request logging."""

    def __init__(self, get_response) -> None:
        self.get_response = get_response

    def __call__(self, request):
        started = time.perf_counter()
        response = self.get_response(request)
        duration_ms = (time.perf_counter() - started) * 1000
        if hasattr(response, "headers"):
            response.headers["X-Process-Time-ms"] = f"{duration_ms:.2f}"
        LOGGER.info(
            json.dumps(
                {
                    "method": request.method,
                    "path": request.path,
                    "status": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                }
            )
        )
        return response


class CorsAllowMiddleware:
    """CORS: same-origin by default; allow the configured origins for dev."""

    def __init__(self, get_response) -> None:
        self.get_response = get_response

    def __call__(self, request):
        if request.method == "OPTIONS":
            response = HttpResponse(status=204)
            self._apply(request, response)
            return response
        response = self.get_response(request)
        self._apply(request, response)
        return response

    def _apply(self, request, response) -> None:
        origin = request.headers.get("Origin")
        if not origin:
            return
        allowed = get_settings().cors_origin_list
        allow_all = "*" in allowed
        if allow_all or origin in allowed:
            response.headers["Access-Control-Allow-Origin"] = "*" if allow_all else origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Expose-Headers"] = (
                "X-Process-Time-ms, X-Prediction-Label, X-Prob-Fake, X-Gradcam-Target"
            )
