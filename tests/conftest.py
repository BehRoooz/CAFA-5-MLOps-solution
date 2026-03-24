"""Shared pytest fixtures: temporary configs, dummy embeddings, dummy labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

EMBED_DIM = 1280
NUM_LABELS = 10
NUM_SAMPLES = 20


@pytest.fixture()
def tmp_config_path(tmp_path: Path) -> Path:
    """Write a minimal valid config YAML and return its path."""
    cfg = {
        "data": {
            "data_dir": str(tmp_path / "preprocess" / "cafa-5-protein-function-prediction"),
            "embeddings_dir": str(tmp_path / "preprocess"),
            "embeddings_source": "ESM2",
            "num_labels": NUM_LABELS,
            "train_val_split": 0.9,
        },
        "model": {
            "type": "mlp",
            "mlp_hidden_dims": [64, 32],
        },
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "scheduler_factor": 0.1,
            "scheduler_patience": 1,
            "seed": 42,
        },
        "output": {
            "output_dir": str(tmp_path / "outputs"),
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "prediction_threshold": 0.5,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(cfg, default_flow_style=False))
    return config_path


@pytest.fixture()
def tmp_config(tmp_config_path: Path):
    """Return a loaded Config object from the temporary YAML."""
    from src.config import load_config

    return load_config(tmp_config_path)


@pytest.fixture()
def dummy_embeddings(tmp_path: Path) -> Path:
    """Create dummy ESM2-style .npy embedding files and return the embeddings directory."""
    rng = np.random.default_rng(42)
    embed_dir = tmp_path / "preprocess" / "cafa-5-ems-2-embeddings-numpy"
    embed_dir.mkdir(parents=True)

    ids = np.array([f"PROT{i:04d}" for i in range(NUM_SAMPLES)])
    embeddings = rng.standard_normal((NUM_SAMPLES, EMBED_DIM)).astype(np.float32)

    np.save(embed_dir / "train_embeddings.npy", embeddings)
    np.save(embed_dir / "train_ids.npy", ids)
    np.save(embed_dir / "test_embeddings.npy", embeddings[:5])
    np.save(embed_dir / "test_ids.npy", ids[:5])

    return tmp_path / "preprocess"


@pytest.fixture()
def dummy_label_matrix(tmp_path: Path) -> Path:
    """Create a dummy label matrix matching the dummy embeddings and return output dir."""
    rng = np.random.default_rng(42)
    out_dir = tmp_path / "outputs" / f"label_matrix_top{NUM_LABELS}"
    out_dir.mkdir(parents=True)

    ids = np.array([f"PROT{i:04d}" for i in range(NUM_SAMPLES)])
    label_matrix = rng.integers(0, 2, size=(NUM_SAMPLES, NUM_LABELS)).astype(np.float32)
    term_names = np.array([f"GO:{i:07d}" for i in range(NUM_LABELS)])

    np.save(out_dir / "label_matrix.npy", label_matrix)
    np.save(out_dir / "protein_ids.npy", ids)
    np.save(out_dir / "term_names.npy", term_names)

    return tmp_path / "outputs"
