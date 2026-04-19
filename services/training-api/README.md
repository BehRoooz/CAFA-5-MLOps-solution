# Training API

HTTP service that queues **model training** jobs and runs them in a background worker. Each job executes the same entrypoints as the CLI (`scripts/train.py` or `scripts/retrain_pipeline.py`), logs to **MLflow**, and can register the PyTorch model in the **Model Registry**.

## Base URL and paths

The route prefix is set in `config.py` as **`API_PREFIX`** (default: **`/api/train`**). All endpoints below are relative to the server root (e.g. `http://127.0.0.1:8002` when using Docker Compose port mapping).

| Method | Path | Description |
|--------|------|-------------|
| GET | `{API_PREFIX}/health` | Liveness check |
| POST | `{API_PREFIX}/train` | Submit a training job (202 + `job_id`) |
| GET | `{API_PREFIX}/jobs/{job_id}` | Poll job status, progress, errors, MLflow fields |

With **`API_PREFIX = "/api/train"`**, the submit URL is **`/api/train/train`** (prefix plus the fixed `/train` segment in code). Health is **`/api/train/health`**; job status is **`/api/train/jobs/{job_id}`**.

## Request body: `POST .../train`

JSON body:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `config` | string | `configs/config.yaml` | Path to the YAML config **inside the repo** (container: `/app/...`) |
| `mode` | `"train"` \| `"retrain"` | `"retrain"` | `train`: only `scripts/train.py`. `retrain`: `scripts/retrain_pipeline.py` (training, holdout eval, optional champion promotion) |

**Response (202 Accepted):**

```json
{
  "job_id": "<uuid>",
  "status": "queued",
  "poll_url": "/api/train/jobs/<uuid>"
}
```

Poll **`GET`** using the full URL: `{base}{poll_url}` (e.g. `http://127.0.0.1:8002/api/train/jobs/<uuid>`).

## Job status: `GET .../jobs/{job_id}`

`status` is one of: `queued` → `running` → `succeeded` | `failed`.

On success you may see:

- `train_run_id`, `registered_model_name`, `registered_model_version`, `model_uri`
- `mlflow`: `run_ui_url`, `model_registry_ui_url` (when the server can resolve experiment/version), plus `tracking_uri`

On failure, `error` contains structured details (e.g. `code`, `message`).

## Environment variables

| Variable | Purpose |
|----------|---------|
| `MLFLOW_TRACKING_URI` | MLflow tracking URI (e.g. `http://mlflow:5000` in Compose) |
| `MLFLOW_EXTERNAL_UI_BASE` | Host browser URL for MLflow UI links in JSON (e.g. `http://127.0.0.1:5000`) |
| `REGISTERED_MODEL_NAME` | Passed through to training scripts (registry name) |
| `PROMOTION_THRESHOLD` | Used by retrain pipeline / promotion when applicable |
| `TRAINING_API_ARTIFACT_ROOT` | Where the SQLite job DB lives (default `outputs/training_api`) |

## Prerequisites

Training expects the same artifacts as local CLI training: appropriate **config**, **data** under `data/`, **embeddings** and **splits** as required by `configs/config.yaml`, and writable **`outputs/`**. Prepare splits and embeddings using the main project README before relying on the API.

## How to run the Training API

### Option A: Docker Compose (recommended)

From the **repository root**:

```bash
docker compose --profile training up --build
```

This starts `trainer-api` (often on host port **8002**) and `mlflow` (**5000**), with `MLFLOW_TRACKING_URI` pointing at the MLflow service.

Check health:

```bash
curl -sS http://127.0.0.1:8002/api/train/health
```

### Option B: Local Uvicorn (development)

From the **repository root**, with dependencies installed and `PYTHONPATH` including the repo root:

```bash
export PYTHONPATH="$(pwd)"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
uvicorn main:app --app-dir services/training-api --host 0.0.0.0 --port 8002 --reload
```

Start an MLflow server separately if you use HTTP tracking, or use a `file:./mlruns` URI for a local file store.

## Run a training job via the API

**1. Submit training** (default `retrain` mode in the current schema; use `"mode": "train"` for train-only):

```bash
BASE="http://127.0.0.1:8002"
RESP=$(curl -sS -X POST "${BASE}/api/train/train" \
  -H "Content-Type: application/json" \
  -d '{"config": "configs/config.yaml", "mode": "train"}')
echo "$RESP"
```

**2. Extract `job_id` and poll until `succeeded` or `failed`:**

```bash
JOB_ID=$(python3 -c "import json,sys; print(json.loads(sys.stdin.read())['job_id'])" <<< "$RESP")
echo "JOB_ID=$JOB_ID"

while true; do
  ST=$(curl -sS "${BASE}/api/train/jobs/${JOB_ID}")
  echo "$ST" | python3 -m json.tool
  S=$(echo "$ST" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['status'])")
  if [ "$S" = "succeeded" ] || [ "$S" = "failed" ]; then
    break
  fi
  sleep 3
done
```

**3. Open MLflow** (if using the default Compose mapping): [http://127.0.0.1:5000](http://127.0.0.1:5000) — use `train_run_id` and registry fields from the final JSON to find the run and model version.

## Changing the URL prefix

Set `API_PREFIX` in `services/training-api/config.py`, or introduce an env-driven prefix there if you need different values per environment without editing code.

## Security note

The training API can execute long-running, resource-heavy work and write to shared volumes. Do not expose it on a public network without authentication and network controls.
