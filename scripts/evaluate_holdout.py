#!/usr/bin/env python
"""Evaluate the trained model on the holdout split.

This script uses:
 - `outputs/label_matrix_top{num_labels}/label_matrix.npy` (targets)
 - `data/embeddings/<backend_dir>/holdout_embeddings.npy` (inputs)
 - `data/embeddings/<backend_dir>/holdout_ids.npy` (alignment key)

Usage:
    python scripts/evaluate_holdout.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import MultilabelF1Score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.inference.predictor import load_checkpoint
from src.preprocess.dataset import EMBED_FILE_MAP
from src.utils import get_device, setup_logger
from src.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on holdout split")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Model checkpoint path (default: outputs/checkpoints/best_model.pt)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger("cafa5", log_dir=config.output_dir)
    device = get_device()

    checkpoint_path = args.checkpoint or str(config.output_dir / "checkpoints" / "best_model.pt")
    model = load_checkpoint(config, checkpoint_path)
    model.to(device)
    model.eval()

    label_dir = config.output_dir / f"label_matrix_top{config.num_labels}"
    label_matrix_path = label_dir / "label_matrix.npy"
    label_ids_path = label_dir / "protein_ids.npy"
    if not label_matrix_path.exists() or not label_ids_path.exists():
        raise FileNotFoundError(
            f"Label matrix artefacts not found in {label_dir}. "
            "Run scripts/preprocess.py first."
        )

    label_matrix = np.load(label_matrix_path)  # (N_labeled, num_labels)
    label_ids = np.load(label_ids_path, allow_pickle=True)
    label_ids = np.asarray(label_ids)

    embeddings_dir = Path(config.data.get("embeddings_dir", "./data/embeddings"))
    source = config.data.get("embeddings_source", "ESM2").lower()
    file_info = EMBED_FILE_MAP.get(source)
    if file_info is None:
        raise ValueError(f"Unknown embeddings_source '{source}'")

    embed_subdir = embeddings_dir / file_info["dir"]
    holdout_embeddings_path = embed_subdir / "holdout_embeddings.npy"
    holdout_ids_path = embed_subdir / "holdout_ids.npy"
    if not holdout_embeddings_path.exists() or not holdout_ids_path.exists():
        raise FileNotFoundError(
            "Holdout embeddings not found. Run scripts/embed_sequences.py with "
            "--split holdout and --ids-npy pointing to your holdout_ids.npy."
        )

    holdout_embeddings = np.load(holdout_embeddings_path).astype(np.float32)  # (N, D)
    holdout_ids = np.load(holdout_ids_path, allow_pickle=True)
    holdout_ids = np.asarray(holdout_ids)

    if holdout_embeddings.ndim != 2:
        raise ValueError(f"Expected holdout_embeddings 2-D, got shape {holdout_embeddings.shape}")
    if holdout_embeddings.shape[1] != config.embedding_dim:
        raise ValueError(
            "Embedding dim mismatch: "
            f"holdout_embeddings.shape[1]={holdout_embeddings.shape[1]} vs config.embedding_dim={config.embedding_dim}"
        )

    id_to_label_idx = {pid: i for i, pid in enumerate(label_ids)}
    kept_label_indices: list[int] = []
    kept_positions: list[int] = []
    for i, pid in enumerate(holdout_ids.tolist()):
        if pid in id_to_label_idx:
            kept_positions.append(i)
            kept_label_indices.append(id_to_label_idx[pid])

    if len(kept_positions) == 0:
        raise RuntimeError("No holdout IDs matched the label matrix IDs.")

    if len(kept_positions) != len(holdout_ids):
        logger.warning(
            "Some holdout IDs were not found in label matrix; using %d/%d samples.",
            len(kept_positions),
            len(holdout_ids),
        )

    holdout_embeddings = holdout_embeddings[kept_positions]
    targets = label_matrix[np.asarray(kept_label_indices)]

    embeds_t = torch.from_numpy(holdout_embeddings).float()
    targets_t = torch.from_numpy(targets).float()

    dataset = TensorDataset(embeds_t, targets_t)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    f1_metric = MultilabelF1Score(num_labels=config.num_labels, threshold=0.5).to(device)

    losses: list[float] = []
    f1s: list[float] = []

    with torch.no_grad():
        for embeds, y in loader:
            embeds = embeds.to(device)
            y = y.to(device)

            logits = model(embeds)
            loss = criterion(logits, y)

            probs = torch.sigmoid(logits)
            preds = probs  # metric applies threshold
            f1 = f1_metric(preds, y.int())

            losses.append(loss.item())
            f1s.append(f1.item())

    result = {
        "n_holdout_samples": int(len(dataset)),
        "holdout_loss_bce": float(np.mean(losses)),
        "holdout_f1_micro": float(np.mean(f1s)),
        "checkpoint": str(checkpoint_path),
        "embeddings_backend": source,
    }
    logger.info("Holdout evaluation: %s", result)
    out_path = config.output_dir / "holdout_evaluation.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Saved evaluation → %s", out_path)


if __name__ == "__main__":
    main()

