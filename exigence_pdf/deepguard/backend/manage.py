#!/usr/bin/env python
"""Django's command-line utility (standard template, adapted).

The Django project lives in ``<repo>/deepguard/backend`` while the shared
model/training package is ``<repo>/src``. We add the repository root to
``sys.path`` so ``import src.*`` works exactly like the FastAPI backend.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # <repo>/exigence_pdf
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "deepguard.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
