"""Tests for the FastAPI prediction API using TestClient.

These tests mock the ML models to avoid loading large ESM-2 weights.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml

from tests.conftest import EMBED_DIM, NUM_LABELS


@pytest.fixture()
def api_env(tmp_path: Path):
    """Set up a minimal environment for the API: config, checkpoint, and term_names."""
    cfg = {
        "data": {
            "data_dir": str(tmp_path / "data"),
            "embeddings_dir": str(tmp_path / "data"),
            "embeddings_source": "ESM2",
            "num_labels": NUM_LABELS,
            "train_val_split": 0.9,
        },
        "model": {"type": "mlp", "mlp_hidden_dims": [64, 32]},
        "training": {"seed": 42, "epochs": 1, "batch_size": 4, "learning_rate": 0.001},
        "output": {"output_dir": str(tmp_path / "outputs")},
        "api": {"prediction_threshold": 0.3},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(cfg))

    from src.config import load_config
    from src.models import build_model

    loaded_cfg = load_config(config_path)
    model = build_model(loaded_cfg)

    ckpt_dir = tmp_path / "outputs" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": 0, "val_f1": 0.0},
        ckpt_dir / "best_model.pt",
    )

    label_dir = tmp_path / "outputs" / f"label_matrix_top{NUM_LABELS}"
    label_dir.mkdir(parents=True)
    term_names = np.array([f"GO:{i:07d}" for i in range(NUM_LABELS)])
    np.save(label_dir / "term_names.npy", term_names)

    os.environ["CONFIG_PATH"] = str(config_path)
    os.environ.pop("CHECKPOINT_PATH", None)

    yield

    os.environ.pop("CONFIG_PATH", None)


@pytest.fixture()
def client(api_env):
    from fastapi.testclient import TestClient
    from src.api.app import app

    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["go_model_loaded"] is True


class TestPredictGoTerms:
    def test_valid_embedding(self, client):
        embedding = [0.1] * EMBED_DIM
        resp = client.post("/predict/go-terms", json={"embedding": embedding})
        assert resp.status_code == 200
        body = resp.json()
        assert "predictions" in body
        assert isinstance(body["predictions"], list)

    def test_wrong_length_embedding(self, client):
        embedding = [0.1] * 100
        resp = client.post("/predict/go-terms", json={"embedding": embedding})
        assert resp.status_code == 422
        assert "1280" in resp.json()["detail"]

    def test_empty_embedding_rejected(self, client):
        resp = client.post("/predict/go-terms", json={"embedding": []})
        assert resp.status_code == 422

    def test_predictions_have_correct_fields(self, client):
        embedding = np.random.default_rng(0).standard_normal(EMBED_DIM).tolist()
        resp = client.post("/predict/go-terms", json={"embedding": embedding})
        assert resp.status_code == 200
        for pred in resp.json()["predictions"]:
            assert "go_term" in pred
            assert "confidence" in pred
            assert 0.0 <= pred["confidence"] <= 1.0


class TestPredictEmbeddingEndpoint:
    """Tests for /predict/embedding — mocked ESM-2 to avoid downloading weights."""

    def test_embedding_endpoint_returns_vector(self, client):
        fake_embedding = np.random.default_rng(0).standard_normal(EMBED_DIM).astype(np.float32)

        with patch("src.api.app._esm_embedder") as mock_esm:
            mock_esm.is_loaded = True
            mock_esm.generate_embedding.return_value = fake_embedding

            with patch("src.api.app._parse_fasta", return_value=("ACDEFG", "TEST_PROT")):
                resp = client.post(
                    "/predict/embedding", json={"sequence": "ACDEFGHIKLMNPQRSTVWY"}
                )

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["embedding"]) == EMBED_DIM
        assert body["protein_id"] == "TEST_PROT"
