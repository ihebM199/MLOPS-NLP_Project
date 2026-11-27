"""Django settings for the DeepGuard platform.

The application is deliberately database-free: the model artifacts live on
disk (``models/``) and the web UI is a self-contained file. Only the
configuration values of the original FastAPI backend were ported
(12-factor: every value can be overridden with an environment variable,
see ``.env.example``).
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "django-insecure-demo-key-change-me")
DEBUG = os.environ.get("DEBUG", "1") not in ("0", "false", "False")
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "api",
]

MIDDLEWARE = [
    "api.middleware.TimingHeadersMiddleware",  # X-Process-Time-ms (as the original API)
    "api.middleware.CorsAllowMiddleware",  # CORS / OPTIONS preflight
    "api.monitoring.PrometheusMiddleware",  # /metrics: request + resource metrics
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "deepguard.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": False,
        "OPTIONS": {"context_processors": []},
    },
]

# No database: the ONNX artifacts, the front-end file and the user's browser
# (localStorage, as in the original platform) carry all the state.
DATABASES = {}

WSGI_APPLICATION = "deepguard.wsgi.application"

USE_I18N = False
USE_TZ = True

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
