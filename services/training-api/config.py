from __future__ import annotations

import os
from pathlib import Path

API_PREFIX = "/api/train"
ARTIFACT_ROOT = Path(os.getenv("TRAINING_API_ARTIFACT_ROOT", "outputs/training_api"))
DB_PATH = ARTIFACT_ROOT / "jobs.db"

WORKER_POLL_INTERVAL_SEC = 1.0

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
# Browser-reachable MLflow UI (host port), for links returned to API clients.
MLFLOW_EXTERNAL_UI_BASE = os.getenv("MLFLOW_EXTERNAL_UI_BASE", "http://127.0.0.1:5000")
