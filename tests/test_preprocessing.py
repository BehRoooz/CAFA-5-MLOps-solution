"""Tests for src.preprocess.preprocessing — label matrix construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.preprocess.preprocessing import build_label_matrix


@pytest.fixture()
def mock_train_terms(tmp_config_path: Path) -> Path:
    """Create a small train_terms.tsv matching the config's data_dir and return the config path."""
    cfg = load_config(tmp_config_path)
    data_dir = Path(cfg.data["data_dir"]) / "Train"
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(5):
        for j in range(3):
            rows.append({"EntryID": f"PROT{i:04d}", "term": f"GO:{j:07d}", "aspect": "BPO"})
    # Extra rows for some terms to ensure top-N selection works
    for i in range(5):
        rows.append({"EntryID": f"PROT{i:04d}", "term": "GO:9999999", "aspect": "BPO"})

    df = pd.DataFrame(rows)
    df.to_csv(data_dir / "train_terms.tsv", sep="\t", index=False)
    return tmp_config_path


class TestBuildLabelMatrix:
    def test_shapes(self, mock_train_terms: Path):
        cfg = load_config(mock_train_terms)
        label_matrix, protein_ids, term_names = build_label_matrix(cfg)

        assert label_matrix.ndim == 2
        assert label_matrix.shape[1] == cfg.num_labels
        assert len(protein_ids) == label_matrix.shape[0]
        assert len(term_names) == cfg.num_labels

    def test_values_are_binary(self, mock_train_terms: Path):
        cfg = load_config(mock_train_terms)
        label_matrix, _, _ = build_label_matrix(cfg)
        unique = np.unique(label_matrix)
        assert set(unique).issubset({0.0, 1.0})

    def test_protein_ids_sorted(self, mock_train_terms: Path):
        cfg = load_config(mock_train_terms)
        _, protein_ids, _ = build_label_matrix(cfg)
        assert list(protein_ids) == sorted(protein_ids)

    def test_missing_file_raises(self, tmp_config_path: Path):
        cfg = load_config(tmp_config_path)
        with pytest.raises(FileNotFoundError, match="train_terms.tsv not found"):
            build_label_matrix(cfg)
