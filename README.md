# CAFA-5 Protein Function Prediction

Production-ready ML pipeline for the [Kaggle CAFA-5 Protein Function Prediction competition](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction), predicting Gene Ontology (GO) terms from protein language model embeddings (ESM-2, ProtBERT, T5).

## Project Structure

```
CAFA-5-Protein-Function-Prediction-MLOps/
├── configs/
│   └── config.yaml                # All hyperparams, paths, model selection
├── src/
│   ├── __init__.py
│   ├── config.py                  # YAML config loading + dataclass validation
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                 # FastAPI application (health, embedding, GO-term endpoints)
│   │   ├── esm_embedder.py        # ESM-2 model wrapper for FASTA→embedding
│   │   └── schemas.py             # Pydantic request/response models
│   ├── preprocess/
│   │   ├── __init__.py
│   │   ├── dataset.py             # ProteinSequenceDataset (PyTorch Dataset)
│   │   └── preprocessing.py       # Build binary label matrix from train_terms.tsv
│   ├── models/
│   │   ├── __init__.py            # Factory function build_model()
│   │   ├── mlp.py                 # MultiLayerPerceptron
│   │   └── cnn1d.py               # CNN1D
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py             # Training loop, validation, checkpointing
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py           # Load model + generate submission
│   └── utils.py                   # Seed setting, logging setup, device selection
├── scripts/
│   ├── train.py                   # CLI: python scripts/train.py --config configs/config.yaml
│   ├── predict.py                 # CLI: python scripts/predict.py --config configs/config.yaml
│   ├── preprocess.py              # CLI: generate label matrix from raw data
│   └── serve.py                   # CLI: start the FastAPI inference server
├── tests/
│   ├── conftest.py                # Shared pytest fixtures
│   ├── test_config.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_dataset.py
│   └── test_api.py
├── preprocess/                    # .gitignored; user places data here
├── outputs/                       # .gitignored; checkpoints, logs, submissions
├── notebooks/
│   └── CAFA5-EMS2embeds-Pytorch.ipynb   # Archived original notebook
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

## Background

The Gene Ontology (GO) is a concept hierarchy describing biological function of genes and gene products at different levels of abstraction. This project frames GO term prediction as a **multi-label classification** problem: given a protein embedding, predict which of the top-N GO terms apply.

## Setup

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

### 2. Place data

Download data from the [Kaggle competition page](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/data) and embedding datasets : 
- EMS2 : [cafa-5-ems-2-embeddings-numpy](https://www.kaggle.com/datasets/viktorfairuschin/cafa-5-ems-2-embeddings-numpy)
- ProtBERT: [protbert-embeddings-for-cafa5](https://www.kaggle.com/datasets/henriupton/protbert-embeddings-for-cafa5)
- T5Embeds: [t5embeds](https://www.kaggle.com/datasets/kriukov/t5embeds)

Then organize under `preprocess/`:

```
preprocess/
├── cafa-5-protein-function-prediction/
│   └── Train/
│       ├── train_terms.tsv
│       ├── train_sequences.fasta
│       └── ...
├── cafa-5-ems-2-embeddings-numpy/
│   ├── train_embeddings.npy
│   ├── train_ids.npy
│   ├── test_embeddings.npy
│   └── test_ids.npy
└── ...
```

### 3. Configure

Edit `configs/config.yaml` to adjust paths, model type, hyperparameters, and embedding source.

## Usage

### Preprocess labels

```bash
python scripts/preprocess.py --config configs/config.yaml
```

Generates a binary label matrix (`.npy`) under `outputs/`.

### Train

```bash
python scripts/train.py --config configs/config.yaml
```

Trains the model, saves the best checkpoint (by val F1) to `outputs/checkpoints/best_model.pt`, and writes `outputs/training_history.json`.

### Predict

```bash
python scripts/predict.py --config configs/config.yaml [--checkpoint path/to/model.pt]
```

Produces `outputs/submission.tsv` in CAFA-5 format (Id, GO term, Confidence).

## Inference API

A two-step FastAPI service for real-time protein function prediction. The server exposes three endpoints: a health check, a sequence-to-embedding endpoint (ESM-2), and an embedding-to-GO-terms endpoint (trained MLP/CNN).

### Start the server

```bash
python scripts/serve.py --config configs/config.yaml --checkpoint outputs/checkpoints/best_model.pt
```

Additional flags:

```bash
python scripts/serve.py --host 0.0.0.0 --port 8000 --reload  # dev mode with auto-reload
```

Or with uvicorn directly (set env vars for config and checkpoint):

```bash
CONFIG_PATH=configs/config.yaml CHECKPOINT_PATH=outputs/checkpoints/best_model.pt \
  uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Interactive API documentation is auto-generated at [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI).

### Endpoints

**`GET /health`** — liveness probe

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "go_model_loaded": true, "esm_model_loaded": false}
```

The `esm_model_loaded` field becomes `true` after the first `/predict/embedding` request triggers lazy loading.

**`POST /predict/embedding`** — protein sequence → ESM-2 embedding (1280-dim)

Accepts a raw amino-acid sequence or a FASTA-formatted string.

```bash
curl -X POST http://localhost:8000/predict/embedding \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVL"}'
```

```json
{"protein_id": null, "embedding": [0.0123, -0.0456, ...]}
```

**`POST /predict/go-terms`** — embedding → GO-term predictions

Pass the 1280-dimensional embedding vector from the previous step. Returns GO terms with confidence scores above the configured threshold (default 0.5).

```bash
curl -X POST http://localhost:8000/predict/go-terms \
  -H "Content-Type: application/json" \
  -d '{"embedding": [0.0123, -0.0456, ...]}'
```

```json
{"predictions": [{"go_term": "GO:0005515", "confidence": 0.87}, {"go_term": "GO:0005634", "confidence": 0.62}]}
```

Returns `422` if the embedding length doesn't match the expected dimension, or `503` if the model checkpoint was not found at startup.

The ESM-2 model (~2.5 GB) is downloaded on the first `/predict/embedding` request and cached locally.

## Docker

### Build and run with docker-compose

```bash
docker compose up --build
```

The compose file mounts `preprocess/` and `outputs/` as read-only volumes and persists the ESM-2 weights in a named `esm_cache` volume so they survive container restarts.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `PROJECT_ROOT` | `/app` | Working directory inside the container |
| `CONFIG_PATH` | `configs/config.yaml` | Path to YAML config file |
| `CHECKPOINT_PATH` | *(auto-detected)* | Override path to model checkpoint |

Override at runtime:

```bash
docker compose run -e CHECKPOINT_PATH=outputs/checkpoints/epoch_3.pt api
```

### Build manually

```bash
docker build -t cafa5-api .
docker run -p 8000:8000 \
  -v $(pwd)/preprocess:/app/preprocess:ro \
  -v $(pwd)/outputs:/app/outputs:ro \
  -v esm_cache:/root/.cache/torch/hub \
  cafa5-api
```

## Testing

Tests use pytest with small dummy data fixtures — no real data or model checkpoints required.

### Install dev dependencies

```bash
pip install -e ".[dev]"
# or
pip install pytest httpx
```

### Run tests

```bash
pytest
```

Run a specific test file or with verbose output:

```bash
pytest tests/test_models.py -v
```

### Test modules

| File | Coverage |
|---|---|
| `tests/test_config.py` | `load_config()` with valid/invalid YAML, `Config` dataclass validation |
| `tests/test_preprocessing.py` | `build_label_matrix()` with mock `train_terms.tsv` (shape, value checks) |
| `tests/test_models.py` | `build_model()` factory, MLP and CNN1D forward-pass shape verification |
| `tests/test_dataset.py` | `ProteinSequenceDataset` `__len__`, `__getitem__` with dummy `.npy` files |
| `tests/test_api.py` | FastAPI endpoints via `TestClient`: `/health`, `/predict/go-terms` (valid + wrong-dimension embedding) |

Shared fixtures live in `tests/conftest.py` (temporary config, dummy embeddings, dummy label matrix).

## Configuration

All parameters live in `configs/config.yaml`:

```yaml
data:
  data_dir: "preprocess/cafa-5-protein-function-prediction"
  embeddings_dir: "preprocess/embeddings"
  embeddings_source: "ESM2"        # ESM2 | ProtBERT | T5
  num_labels: 500
  train_val_split: 0.9

model:
  type: "cnn1d"                    # mlp | cnn1d
  mlp_hidden_dims: [864, 712]
  cnn_out_channels: [3, 8]
  cnn_kernel_size: 3

training:
  epochs: 5
  batch_size: 128
  learning_rate: 0.001
  scheduler_factor: 0.1
  scheduler_patience: 1
  seed: 42

output:
  output_dir: "outputs"

api:
  host: "0.0.0.0"
  port: 8000
  prediction_threshold: 0.5
```

## Models

- **MLP** (`mlp`): Configurable hidden-layer sizes, ReLU activations.
- **CNN1D** (`cnn1d`): Two 1-D conv layers with tanh activations, max pooling, and fully-connected output.

Both output raw logits (no final sigmoid) — `BCEWithLogitsLoss` handles the sigmoid internally for numerical stability.

## License

MIT
