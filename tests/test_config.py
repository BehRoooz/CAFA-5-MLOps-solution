"""Tests for src.config — YAML loading, validation, and derived fields."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import Config, load_config


class TestLoadConfig:
    def test_valid_config(self, tmp_config_path: Path):
        cfg = load_config(tmp_config_path)
        assert isinstance(cfg, Config)
        assert cfg.embedding_dim == 1280
        assert cfg.num_labels == 10

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(tmp_path / "nonexistent.yaml")

    def test_unknown_embedding_source(self, tmp_path: Path):
        bad_cfg = {
            "data": {"embeddings_source": "UNKNOWN"},
            "model": {"type": "mlp"},
            "training": {},
            "output": {},
        }
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump(bad_cfg))
        with pytest.raises(ValueError, match="Unknown embeddings_source"):
            load_config(path)


class TestConfigProperties:
    def test_seed(self, tmp_config: Config):
        assert tmp_config.seed == 42

    def test_epochs(self, tmp_config: Config):
        assert tmp_config.epochs == 1

    def test_batch_size(self, tmp_config: Config):
        assert tmp_config.batch_size == 4

    def test_learning_rate(self, tmp_config: Config):
        assert tmp_config.learning_rate == pytest.approx(0.001)

    def test_output_dir_is_path(self, tmp_config: Config):
        assert isinstance(tmp_config.output_dir, Path)

    def test_api_section(self, tmp_config: Config):
        assert tmp_config.api["prediction_threshold"] == 0.5
        assert tmp_config.api["port"] == 8000
