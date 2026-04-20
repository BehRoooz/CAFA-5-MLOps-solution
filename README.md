# CAFA-5 MLOps Solution

Production-grade pipeline for CAFA-5 protein function prediction with:

- CLI workflow for preprocessing, embedding, training, evaluation, and submission.
- FastAPI microservices for embedding, GO prediction, and training orchestration.
- MLflow tracking + model registry promotion flow.
- NGINX gateway with TLS, basic auth, request limits, and routed access tiers.

## Architecture At A Glance

- `embedding-api` builds embeddings asynchronously from JSON sequences or FASTA uploads.
- `go-prediction-api` serves GO term inference from 1280-dim embeddings (ESM2 path).
- `trainer-api` queues and runs `train` or full `retrain` pipelines.
- `mlflow` stores runs, metrics, artifacts, and registry aliases (`champion`).
- `nginx` is the public entrypoint (`:80/:443`) and proxies all service routes.

## Project Structure

```text
CAFA-5-MLOps-solution/
├── configs/
│   └── config.yaml                       # Central config for data/model/train/predict/embed
├── data/                                 # Input data, embeddings, HF cache (mounted in containers)
├── docker/
│   ├── docker_embedding/
│   │   ├── Dockerfile.embedding-api      # Embedding API image
│   │   └── Dockerfile.embedding-cli      # Batch embedding CLI image
│   ├── docker_go_term/
│   │   ├── Dockerfile.api                # GO prediction API image
│   │   └── docker-compose.yml            # Local compose for go-term service dev
│   └── docker_training/
│       └── Dockerfile.training           # Training API image
├── examples/
│   └── small_sequences.fasta             # Tiny FASTA for quick tests
├── nginx/
│   ├── nginx.conf                        # TLS gateway, auth, rate limits, routing
│   ├── .htpasswd-admin                   # Admin credentials
│   ├── .htpasswd-user                    # User credentials
│   └── certs/                            # TLS cert/key mounted into nginx
├── outputs/                              # Labels, splits, checkpoints, submissions, service artifacts
├── scripts/
│   ├── preprocess.py
│   ├── split_train_holdout.py
│   ├── embed_sequences.py
│   ├── train.py
│   ├── evaluate_holdout.py
│   ├── promote_model.py
│   ├── retrain_pipeline.py
│   ├── predict.py
│   └── smoke_embedding_api.sh
├── services/
│   ├── embedding-api/
│   ├── go-prediction-api/
│   └── training-api/
├── src/                                  # Core model/data/training/inference modules
├── docker-compose.yml                    # Integrated stack orchestration
├── requirements.txt
├── pyproject.toml
└── README.md
```

## What Each Script Does

- `scripts/preprocess.py`: builds label matrix and GO term index from `train_terms.tsv`.
- `scripts/split_train_holdout.py`: deterministic train/holdout ID split.
- `scripts/embed_sequences.py`: generates embeddings from FASTA or raw sequence.
- `scripts/train.py`: trains model, logs to MLflow, registers model.
- `scripts/evaluate_holdout.py`: computes holdout BCE/F1 and logs evaluation run.
- `scripts/promote_model.py`: promotes model version to alias (default `champion`) if metric threshold passes.
- `scripts/retrain_pipeline.py`: train -> evaluate -> promote in one command.
- `scripts/predict.py`: writes CAFA-format submission TSV.
- `scripts/smoke_embedding_api.sh`: end-to-end embedding API sanity check.

## Data Layout Expectations

Download CAFA-5 input data from [Kaggle CAFA-5](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/data), then organize:

```text
data/
├── cafa-5-protein-function-prediction/
│   └── Train/
│       ├── train_sequences.fasta
│       └── train_terms.tsv
├── embeddings/
│   ├── hf_esm2/
│   ├── hf_protbert/
│   └── hf_prot_t5/
└── hf_cache/
```

## Local Python Setup (CLI workflow)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Core Configuration

Edit `configs/config.yaml` before running. Most important fields:

- `data.embeddings_source`: `ESM2 | ProtBERT | T5` (drives expected embedding dimension).
- `embedding.backend`: `esm2 | prot_bert | prot_t5` (encoder used for generation).
- `data.holdout_fraction`: split proportion for holdout evaluation.
- `model.type`: `mlp` or `cnn1d`.
- `training.*`: epochs, LR, scheduler, seed.
- `prediction.datatype`: `test` or `holdout`.

Important consistency rule: keep `data.embeddings_source` and `embedding.backend` aligned to avoid training/prediction on mismatched embedding spaces.

## End-To-End CLI Pipeline

### 1) Build labels and split IDs

```bash
python scripts/preprocess.py --config configs/config.yaml
python scripts/split_train_holdout.py --config configs/config.yaml
```

### 2) Generate embeddings for train and holdout

```bash
python scripts/embed_sequences.py --config configs/config.yaml \
  --ids-npy outputs/splits/train_ids.npy \
  --split train

python scripts/embed_sequences.py --config configs/config.yaml \
  --ids-npy outputs/splits/holdout_ids.npy \
  --split holdout
```

### 3) Train and evaluate

```bash
python scripts/train.py --config configs/config.yaml
python scripts/evaluate_holdout.py --config configs/config.yaml
```

### 4) Retrain + promote in one step

```bash
python scripts/retrain_pipeline.py --config configs/config.yaml \
  --promotion-threshold 0.35 \
  --model-name cafa-go-model
```

### 5) Generate CAFA submission

```bash
python scripts/predict.py --config configs/config.yaml
```

Output: `outputs/submission.tsv`.

## Docker Services And Compose

Main compose file: `docker-compose.yml`.

### Default stack (gateway + embedding + GO prediction + MLflow)

```bash
docker compose up --build
```

Public entrypoint is NGINX:

- `http://127.0.0.1` redirects to TLS.
- `https://127.0.0.1` serves routed APIs.

### Include training API profile

`trainer-api` is under Compose profile `training`:

```bash
docker compose --profile training up --build
```

Use this command when you need `/api/train/...` endpoints through NGINX.

## NGINX Gateway Routes

All routes are HTTPS on `https://127.0.0.1`:

- `/api/v1/*` -> `embedding-api` (admin basic auth).
- `/api/predict/*` -> `go-prediction-api` (user basic auth).
- `/api/train*` -> `trainer-api` (admin basic auth, requires `training` profile).
- `/mlflow/*` -> MLflow UI (admin basic auth).

Gateway also applies:

- per-path request body limits,
- request-rate limiting,
- upstream timeouts suitable for long jobs,
- forwarded trace headers for observability.

## Interactive Monitoring

- MLflow UI (interactive run comparison, metrics, artifacts, model registry):
  - `https://127.0.0.1/mlflow/`
- Async API workflows:
  - submit jobs, poll status, download artifacts.
- Compose/service logs:
  - `docker compose logs -f nginx embedding-api go-prediction-api mlflow`
  - add `trainer-api` when using training profile.

## Practical API Examples

Use your basic-auth credentials depending on route (`.htpasswd-admin` or `.htpasswd-user`).

### 1) Health checks via gateway

```bash
curl -sk -u USER:PASS https://127.0.0.1/api/v1/health
curl -sk -u USER:PASS https://127.0.0.1/api/predict/health
curl -sk -u USER:PASS https://127.0.0.1/api/train/health
```

### 2) Sequences -> embeddings (`embedding-api`)

```bash
curl -sk -u ADMIN:ADMIN_PASS -X POST https://127.0.0.1/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "stage": "test",
    "backend": "esm2",
    "pooling": "mean",
    "batch_size": 2,
    "max_length": 1280,
    "sequences": [
      {"id": "P1", "sequence": "MKTAYIAKQRQISFVKSHFSRQ"},
      {"id": "P2", "sequence": "GAVLIPFYWSTCMNQDEKRH"}
    ]
  }'
```

Poll:

```bash
curl -sk -u ADMIN:ADMIN_PASS https://127.0.0.1/api/v1/jobs/<JOB_ID>
```

Download artifacts:

```bash
curl -sk -u ADMIN:ADMIN_PASS -o test_ids.npy \
  https://127.0.0.1/api/v1/jobs/<JOB_ID>/artifacts/test_ids.npy
curl -sk -u ADMIN:ADMIN_PASS -o test_embeddings.npy \
  https://127.0.0.1/api/v1/jobs/<JOB_ID>/artifacts/test_embeddings.npy
```

### 3) Embeddings -> GO terms (`go-prediction-api`)

```bash
python - <<'PY'
import json
import numpy as np
import requests

emb = np.load("test_embeddings.npy")[0].astype(float).tolist()
resp = requests.post(
    "https://127.0.0.1/api/predict/predict",
    auth=("USER", "USER_PASS"),
    json={"embedding": emb, "top_k": 10},
    verify=False,
    timeout=60,
)
print(resp.status_code)
print(json.dumps(resp.json(), indent=2))
PY
```

### 4) Sequences -> GO terms in one call

```bash
curl -sk -u USER:USER_PASS -X POST \
  https://127.0.0.1/api/v1/predict-go-from-sequences \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "esm2",
    "pooling": "mean",
    "batch_size": 2,
    "max_length": 1280,
    "top_k": 10,
    "sequences": [
      {"id": "P1", "sequence": "MKTAYIAKQRQISFVKSHFSRQ"},
      {"id": "P2", "sequence": "GAVLIPFYWSTCMNQDEKRH"}
    ]
  }'
```

### 5) Trigger training job through API

```bash
curl -sk -u ADMIN:ADMIN_PASS -X POST https://127.0.0.1/api/train/train \
  -H "Content-Type: application/json" \
  -d '{"config":"configs/config.yaml","mode":"retrain"}'
```

Poll:

```bash
curl -sk -u ADMIN:ADMIN_PASS https://127.0.0.1/api/train/jobs/<JOB_ID>
```

## Standalone Docker Images

- `docker/docker_embedding/Dockerfile.embedding-api`
- `docker/docker_embedding/Dockerfile.embedding-cli`
- `docker/docker_go_term/Dockerfile.api`
- `docker/docker_training/Dockerfile.training`

Example CLI image run:

```bash
docker build -f docker/docker_embedding/Dockerfile.embedding-cli -t cafa5-embedding-cli:cpu .
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/outputs:/app/outputs" \
  cafa5-embedding-cli:cpu \
  --config configs/config.yaml \
  --fasta examples/small_sequences.fasta \
  --split test
```

## Reproducibility And QC Notes

- Fix seeds via `training.seed` for deterministic split/train behavior.
- Keep embedding backend and training metadata consistent across retrains.
- Use holdout evaluation before alias promotion to reduce optimistic bias.
- Inspect class imbalance and threshold sensitivity for multilabel GO prediction.
- Track dataset snapshot and term hash in MLflow for lineage auditing.

## Service-Specific Docs

- `services/embedding-api/README.md`
- `services/training-api/README.md`

## License

MIT
