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
│   ├── data/
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
│   └── preprocess.py              # CLI: generate label matrix from raw data
├── data/                          # .gitignored; user places data here
├── outputs/                       # .gitignored; checkpoints, logs, submissions
├── notebooks/
│   └── CAFA5-EMS2embeds-Pytorch.ipynb   # Archived original notebook
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

Note: embedding generation can be memory-intensive (especially `prot_t5` / ProtT5-XL). Use a CUDA GPU and keep `embedding.fp16=true` for faster embedding on GPU.

### 2. Place data
Download CAFA-5 data from the [Kaggle competition page](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/data).

You can either:
- use precomputed embeddings (from Kaggle), or
- generate embeddings directly from `Train/train_sequences.fasta` with `scripts/embed_sequences.py`.

Then organize under `data/`:

```
data/
├── cafa-5-protein-function-prediction/
│   └── Train/
│       ├── train_terms.tsv
│       ├── train_sequences.fasta
│       └── ...
├── embeddings/                           # configured by `data.embeddings_dir`
│   ├── hf_esm2/
│   │   ├── train_embeddings.npy
│   │   ├── train_ids.npy
│   │   ├── holdout_embeddings.npy
│   │   └── holdout_ids.npy
│   ├── hf_protbert/
│   └── hf_prot_t5/
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

### Split train/holdout
```bash
python scripts/split_train_holdout.py --config configs/config.yaml
```

### Generate embeddings (train split)
```bash
python scripts/embed_sequences.py --config configs/config.yaml \
  --ids-npy outputs/splits/train_ids.npy \
  --split train
```

### Generate embeddings (holdout split)
```bash
python scripts/embed_sequences.py --config configs/config.yaml \
  --ids-npy outputs/splits/holdout_ids.npy \
  --split holdout
```

## Generate embedding Test
```bash
python scripts/embed_sequences.py --config configs/config.yaml \
  --ids-npy outputs/splits/test_ids.npy \
  --split test
```

### (Optional) Evaluate holdout
```bash
python scripts/evaluate_holdout.py --config configs/config.yaml
```

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

## Configuration

All parameters live in `configs/config.yaml`:

```yaml
data:
  data_dir: "data/cafa-5-protein-function-prediction"
  train_fasta: "data/cafa-5-protein-function-prediction/Train/train_sequences.fasta"
  embeddings_dir: "data/embeddings"
  embeddings_source: "ESM2"        # ESM2 | ProtBERT | T5
  num_labels: 500
  train_val_split: 0.9
  holdout_fraction: 0.1
  splits_dir: "outputs/splits"

embedding:
  backend: "esm2"                  # esm2 | prot_bert | prot_t5
  hf_cache_dir: "data/hf_cache"
  pooling: "mean"                  # mean | cls
  max_length: 1280
  batch_size: 8
  fp16: true
  num_workers: 0
  generated_subdir_prefix: "hf_"

model:
  type: "mlp"                      # mlp | cnn1d
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

prediction:
  datatype: "holdout"

output:
  output_dir: "outputs"
```

## Models

- **MLP** (`mlp`): Configurable hidden-layer sizes, ReLU activations.
- **CNN1D** (`cnn1d`): Two 1-D conv layers with tanh activations, max pooling, and fully-connected output.

Both output raw logits (no final sigmoid) — `BCEWithLogitsLoss` handles the sigmoid internally for numerical stability.

## License

MIT
