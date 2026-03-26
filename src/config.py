"""Configuration management: YAML loading and validation via dataclass."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

EMBEDDING_DIMS: dict[str, int] = {
    "esm2": 1280,
    "protbert": 1024,
    "t5": 1024,
}


@dataclass
class Config:
    """Typed project configuration loaded from a YAML file."""

    data: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    embedding: dict[str, Any] = field(default_factory=dict)
    prediction: dict[str, Any] = field(default_factory=dict)

    # Derived ---
    embedding_dim: int = field(init=False)
    project_root: Path = field(init=False)

    def __post_init__(self) -> None:
        source = self.data.get("embeddings_source", "ESM2").lower()
        if source not in EMBEDDING_DIMS:
            raise ValueError(
                f"Unknown embeddings_source '{source}'. "
                f"Choose from {list(EMBEDDING_DIMS.keys())}."
            )
        self.embedding_dim = EMBEDDING_DIMS[source]

        self.project_root = Path(os.environ.get("PROJECT_ROOT", Path.cwd()))
        self.output["output_dir"] = str(
            self.project_root / self.output.get("output_dir", "outputs")
        )

    # Convenience accessors ------------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.training.get("seed", 42))

    @property
    def epochs(self) -> int:
        return int(self.training.get("epochs", 5))

    @property
    def batch_size(self) -> int:
        return int(self.training.get("batch_size", 128))

    @property
    def learning_rate(self) -> float:
        return float(self.training.get("learning_rate", 1e-3))

    @property
    def num_labels(self) -> int:
        return int(self.data.get("num_labels", 500))

    @property
    def output_dir(self) -> Path:
        return Path(self.output["output_dir"])


def load_config(path: str | Path) -> Config:
    """Load a YAML config file and return a validated Config instance."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return Config(
        data=raw.get("data", {}),
        model=raw.get("model", {}),
        training=raw.get("training", {}),
        output=raw.get("output", {}),
        embedding=raw.get("embedding", {}),
        prediction=raw.get("prediction", {}),
    )
