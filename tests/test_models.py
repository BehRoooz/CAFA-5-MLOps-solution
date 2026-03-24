"""Tests for src.models — factory function and forward pass shapes."""

from __future__ import annotations

import pytest
import torch

from src.config import Config
from src.models import build_model
from src.models.cnn1d import CNN1D
from src.models.mlp import MultiLayerPerceptron

EMBED_DIM = 1280
NUM_LABELS = 10
BATCH_SIZE = 4


def _make_config(model_type: str = "mlp") -> Config:
    """Build a minimal Config for model tests."""
    return Config(
        data={"embeddings_source": "ESM2", "num_labels": NUM_LABELS},
        model={
            "type": model_type,
            "mlp_hidden_dims": [64, 32],
            "cnn_out_channels": [3, 8],
            "cnn_kernel_size": 3,
        },
        training={"seed": 42},
        output={"output_dir": "/tmp/test_out"},
    )


class TestBuildModel:
    def test_mlp_factory(self):
        model = build_model(_make_config("mlp"))
        assert isinstance(model, MultiLayerPerceptron)

    def test_cnn1d_factory(self):
        model = build_model(_make_config("cnn1d"))
        assert isinstance(model, CNN1D)

    def test_unknown_type_raises(self):
        cfg = _make_config("transformer")
        with pytest.raises(ValueError, match="Unknown model type"):
            build_model(cfg)


class TestMLPForwardPass:
    def test_output_shape(self):
        model = MultiLayerPerceptron(EMBED_DIM, [64, 32], NUM_LABELS)
        x = torch.randn(BATCH_SIZE, EMBED_DIM)
        out = model(x)
        assert out.shape == (BATCH_SIZE, NUM_LABELS)

    def test_no_sigmoid_applied(self):
        """Output should contain values outside [0, 1] (raw logits)."""
        model = MultiLayerPerceptron(EMBED_DIM, [64, 32], NUM_LABELS)
        x = torch.randn(BATCH_SIZE, EMBED_DIM)
        out = model(x)
        assert out.min() < 0 or out.max() > 1


class TestCNN1DForwardPass:
    def test_output_shape(self):
        model = CNN1D(EMBED_DIM, [3, 8], kernel_size=3, num_classes=NUM_LABELS)
        x = torch.randn(BATCH_SIZE, EMBED_DIM)
        out = model(x)
        assert out.shape == (BATCH_SIZE, NUM_LABELS)

    def test_single_channel_config(self):
        model = CNN1D(EMBED_DIM, [4], kernel_size=3, num_classes=NUM_LABELS)
        x = torch.randn(BATCH_SIZE, EMBED_DIM)
        out = model(x)
        assert out.shape == (BATCH_SIZE, NUM_LABELS)
