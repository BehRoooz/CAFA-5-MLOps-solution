#!/usr/bin/env python
"""Split labeled training proteins into train/holdout.

Usage:
    python scripts/split_train_holdout.py --config configs/config.yaml

Outputs:
    <splits_dir>/train_ids.npy
    <splits_dir>/holdout_ids.npy
    <splits_dir>/split_meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Split labeled proteins into train/holdout")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger("cafa5", log_dir=config.output_dir)

    label_dir = config.output_dir / f"label_matrix_top{config.num_labels}"
    protein_ids_path = label_dir / "protein_ids.npy"
    if not protein_ids_path.exists():
        raise FileNotFoundError(
            f"protein_ids.npy not found at {protein_ids_path}. "
            "Run `python scripts/preprocess.py --config configs/config.yaml` first."
        )

    protein_ids = np.load(protein_ids_path, allow_pickle=True)
    protein_ids = np.asarray(protein_ids)
    if protein_ids.ndim != 1:
        raise ValueError(f"Expected 1-D protein_ids, got shape {protein_ids.shape}")

    holdout_fraction = float(config.data.get("holdout_fraction", 0.1))
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in (0,1), got {holdout_fraction}")

    seed = int(config.seed)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(protein_ids))
    rng.shuffle(idx)

    n_holdout = int(round(len(protein_ids) * holdout_fraction))
    n_holdout = max(1, min(n_holdout, len(protein_ids) - 1))

    holdout_idx = idx[:n_holdout]
    train_idx = idx[n_holdout:]

    train_ids = protein_ids[train_idx]
    holdout_ids = protein_ids[holdout_idx]

    splits_dir = Path(config.data.get("splits_dir", config.output_dir / "splits"))
    splits_dir.mkdir(parents=True, exist_ok=True)

    np.save(splits_dir / "train_ids.npy", train_ids)
    np.save(splits_dir / "holdout_ids.npy", holdout_ids)

    meta = {
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "n_proteins_labeled": int(len(protein_ids)),
        "n_train": int(len(train_ids)),
        "n_holdout": int(len(holdout_ids)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_labels": int(config.num_labels),
        "label_matrix_dir": str(label_dir),
    }
    (splits_dir / "split_meta.json").write_text(json.dumps(meta, indent=2))

    logger.info(
        "Split complete: %d train / %d holdout (fraction=%.3f) → %s",
        len(train_ids),
        len(holdout_ids),
        holdout_fraction,
        splits_dir,
    )


if __name__ == "__main__":
    main()

