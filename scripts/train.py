#!/usr/bin/env python
"""CLI: Train a protein function prediction model.

Usage:
    python scripts/train.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.preprocess.dataset import ProteinSequenceDataset
from src.preprocess.preprocessing import build_label_matrix, save_label_matrix
from src.models import build_model
from src.training.trainer import Trainer
from src.utils import set_seed, setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CAFA-5 model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger("cafa5", log_dir=config.output_dir)
    set_seed(config.seed)

    # --- Preprocessing (build label matrix if not already present) ----------
    label_dir = config.output_dir / f"label_matrix_top{config.num_labels}"
    if not (label_dir / "label_matrix.npy").exists():
        logger.info("Label matrix not found — running preprocessing ...")
        lm, pids, terms = build_label_matrix(config)
        save_label_matrix(config, lm, pids, terms)
    else:
        logger.info("Using existing label matrix from %s", label_dir)

    # --- Dataset ------------------------------------------------------------
    logger.info("Loading training dataset ...")
    dataset = ProteinSequenceDataset(config, datatype="train", label_matrix_dir=label_dir)

    # --- Model --------------------------------------------------------------
    model = build_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s — %s parameters", config.model["type"], f"{total_params:,}")

    # --- Training -----------------------------------------------------------
    trainer = Trainer(config, model, dataset)
    history = trainer.train()

    # Save training history
    history_path = Path(config.output_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved → %s", history_path)


if __name__ == "__main__":
    main()
