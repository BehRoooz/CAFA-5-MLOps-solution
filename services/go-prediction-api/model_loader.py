from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.cnn1d import CNN1D
from src.models.mlp import MultiLayerPerceptron

EMBEDDING_DIMS: dict[str, int] = {
    "esm2": 1280,
    "protbert": 1024,
    "t5": 1024,
}


def load_model_meta(meta_path: str | Path) -> dict[str, Any]:
    path = Path(meta_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_model_from_meta(meta: dict[str, Any]) -> torch.nn.Module:
    model_type = meta["model_type"]
    embedding_dim = int(meta["embedding_dim"])
    num_labels = int(meta["num_labels"])

    if model_type == "cnn1d":
        model = CNN1D(
            input_dim=embedding_dim,
            num_classes=num_labels,
            out_channels=[3, 8],
            kernel_size=3,
        )
    elif model_type == "mlp":
        model = MultiLayerPerceptron(
            input_dim=embedding_dim,
            num_classes=num_labels,
            hidden_dims=[864, 712],
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return model


def load_term_names(term_names_path: str | Path) -> np.ndarray:
    return np.load(Path(term_names_path), allow_pickle=True)


def load_model(
    checkpoint_path: str | Path,
    meta_path: str | Path,
    device: str = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt = torch.load(Path(checkpoint_path), map_location=device)

    # Fallback for production containers where model_meta.json is absent.
    # We derive what is needed to reconstruct the model from checkpoint config.
    if Path(meta_path).exists():
        meta = load_model_meta(meta_path)
    else:
        cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})
        embeddings_source = str(data_cfg.get("embeddings_source", "esm2")).lower()
        meta = {
            "model_type": model_cfg.get("type", "mlp"),
            "embedding_dim": EMBEDDING_DIMS.get(embeddings_source, 1280),
            "num_labels": int(data_cfg.get("num_labels", 500)),
            "model_version": str(ckpt.get("epoch", "unknown")) if isinstance(ckpt, dict) else "unknown",
        }

    model = build_model_from_meta(meta)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, meta
