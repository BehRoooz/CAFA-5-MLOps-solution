# Embedding API (CPU-first, async jobs)

Minimal FastAPI service to generate protein language model embeddings and persist them as `.npy` artifacts.

## What this service does

- Accepts protein inputs as either:
  - JSON `sequence_list` (list of `{id, sequence}`), or
  - FASTA upload
- Runs embedding asynchronously via a background worker thread
- Writes outputs to disk as `.npy` under:
  - `outputs/service_artifacts/{job_id}/`
- Uses in-process model caching (models are loaded once and reused while the service runs)

Current scope (Milestone 2, v1):

- `stage` is effectively `test` only (embeds provided input; no labels)

## Requirements

- Python 3.10+
- Dependencies: `fastapi`, `uvicorn`, `python-multipart`, `torch`, `transformers`, `numpy`

## Run the service (local CPU)

From repo root (`CAFA-5-MLOps-solution/`):

```bash
uvicorn main:app --app-dir services/embedding-api --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

## Endpoints

### 1) Create a job (JSON sequence list)

`POST /api/v1/jobs`

Request body:

```json
{
  "stage": "test",
  "backend": "esm2",
  "pooling": "mean",
  "batch_size": 2,
  "max_length": 1280,
  "sequences": [
    { "id": "P1", "sequence": "MKTAYIAKQRQISFVKSHFSRQ" },
    { "id": "P2", "sequence": "GAVLIPFYWSTCMNQDEKRH" }
  ]
}
```

Response (`202 Accepted`):

- `job_id`
- `status` (queued)
- `poll_url` (where to check progress)

### 2) Create a job (FASTA upload)

`POST /api/v1/jobs/fasta` (multipart form)

Form fields:

- `fasta_file` (required)
- `backend` (default `esm2`)
- `pooling` (default `mean`)
- `batch_size` (default `8`)
- `max_length` (default `1280`)

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/jobs/fasta \
  -F "fasta_file=@/path/to/sample.fasta" \
  -F "backend=esm2" \
  -F "pooling=mean" \
  -F "batch_size=4" \
  -F "max_length=1280"
```

### 3) Poll job status

`GET /api/v1/jobs/{job_id}`

Returns:

- `status`: `queued | running | succeeded | failed`
- `progress`: `{embedded_sequences, total_sequences, percent}`
- `artifacts_manifest` (once artifacts exist)
- `error` (if failed)

### 4) Download artifacts

`GET /api/v1/jobs/{job_id}/artifacts/{name}`

Only available when the job is in `succeeded`.

Currently produced artifacts for `test` stage:

- `test_ids.npy`
- `test_embeddings.npy`

Example:

```bash
curl -o test_ids.npy \
  http://127.0.0.1:8000/api/v1/jobs/<JOB_ID>/artifacts/test_ids.npy
```

```bash
curl -o test_embeddings.npy \
  http://127.0.0.1:8000/api/v1/jobs/<JOB_ID>/artifacts/test_embeddings.npy
```

## Storage layout

- Job DB:
  - `outputs/service_artifacts/jobs.db`
- Artifacts:
  - `outputs/service_artifacts/{job_id}/test_ids.npy`
  - `outputs/service_artifacts/{job_id}/test_embeddings.npy`

## Model behavior (important)

- Backend options supported by the code: `esm2`, `protbert`, `t5`
- The current embedding runner uses CPU (`torch.device("cpu")`).
- Embeddings are saved as `float32`.
- Output shape is `(N, D)`, where `D` depends on backend:
  - `esm2`: `D=1280`
  - `protbert`: `D=1024`
  - `t5`: `D=1024`

## Troubleshooting

- First request may be slow: Hugging Face model/tokenizer downloads can take time.
- Ensure `outputs/service_artifacts/` is writable.
- If you get `Failed` jobs, check the API logs for the detailed exception message.

