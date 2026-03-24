"""Tests for src.preprocess.dataset — ProteinSequenceDataset with dummy .npy files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tests.conftest import EMBED_DIM, NUM_LABELS, NUM_SAMPLES


class TestProteinSequenceDatasetTrain:
    def test_len(self, tmp_config, dummy_embeddings, dummy_label_matrix):
        from src.preprocess.dataset import ProteinSequenceDataset

        ds = ProteinSequenceDataset(tmp_config, datatype="train")
        assert len(ds) <= NUM_SAMPLES

    def test_getitem_shapes(self, tmp_config, dummy_embeddings, dummy_label_matrix):
        from src.preprocess.dataset import ProteinSequenceDataset

        ds = ProteinSequenceDataset(tmp_config, datatype="train")
        embed, label = ds[0]
        assert embed.shape == (EMBED_DIM,)
        assert label.shape == (NUM_LABELS,)
        assert isinstance(embed, torch.Tensor)
        assert isinstance(label, torch.Tensor)

    def test_embedding_dtype(self, tmp_config, dummy_embeddings, dummy_label_matrix):
        from src.preprocess.dataset import ProteinSequenceDataset

        ds = ProteinSequenceDataset(tmp_config, datatype="train")
        embed, _ = ds[0]
        assert embed.dtype == torch.float32


class TestProteinSequenceDatasetTest:
    def test_len(self, tmp_config, dummy_embeddings):
        from src.preprocess.dataset import ProteinSequenceDataset

        ds = ProteinSequenceDataset(tmp_config, datatype="test")
        assert len(ds) == 5

    def test_getitem_returns_id(self, tmp_config, dummy_embeddings):
        from src.preprocess.dataset import ProteinSequenceDataset

        ds = ProteinSequenceDataset(tmp_config, datatype="test")
        embed, prot_id = ds[0]
        assert embed.shape == (EMBED_DIM,)
        assert isinstance(prot_id, (str, np.str_))

    def test_invalid_datatype(self, tmp_config, dummy_embeddings):
        from src.preprocess.dataset import ProteinSequenceDataset

        with pytest.raises(ValueError, match="datatype must be"):
            ProteinSequenceDataset(tmp_config, datatype="validation")
