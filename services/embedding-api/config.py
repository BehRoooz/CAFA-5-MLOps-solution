from pathlib import Path

API_PREFIX = "/api/v1"
ARTIFACT_ROOT = Path("outputs/service_artifacts")
DB_PATH = ARTIFACT_ROOT / "jobs.db"

DEFAULT_STAGE = "test"
DEFAULT_BACKEND = "esm2"
DEFAULT_POOLING = "mean"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 1280

WORKER_POLL_INTERVAL_SEC = 1.0