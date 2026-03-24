"""Build binary label matrix from train_terms.tsv for the top-N GO terms."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config

logger = logging.getLogger("cafa5")


def build_label_matrix(config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read train_terms.tsv, select top-N GO terms, and construct a binary label matrix.

    Returns:
        Tuple of (label_matrix, protein_ids, term_names) where
        - label_matrix: shape (n_proteins, num_labels), dtype float32
        - protein_ids:  1-D array of EntryID strings
        - term_names:   1-D array of GO-term strings
    """
    num_labels = config.num_labels
    data_dir = Path(
        config.data.get("data_dir", "./data/cafa-5-protein-function-prediction")
    )
    labels_path = data_dir / "Train" / "train_terms.tsv"

    if not labels_path.exists():
        raise FileNotFoundError(f"train_terms.tsv not found at {labels_path}")

    logger.info("Reading %s", labels_path)
    df = pd.read_csv(labels_path, sep="\t")

    top_terms = (
        df.groupby("term")["EntryID"]
        .count()
        .sort_values(ascending=False)
        .head(num_labels)
    )
    term_names: np.ndarray = top_terms.index.values

    df_filtered = df[df["term"].isin(term_names)]
    protein_ids = df_filtered["EntryID"].unique()
    protein_ids.sort()

    pid_to_idx = {pid: i for i, pid in enumerate(protein_ids)}
    term_to_idx = {t: i for i, t in enumerate(term_names)}

    label_matrix = np.zeros((len(protein_ids), num_labels), dtype=np.float32)

    for _, row in df_filtered.iterrows():
        pid = row["EntryID"]
        term = row["term"]
        if pid in pid_to_idx and term in term_to_idx:
            label_matrix[pid_to_idx[pid], term_to_idx[term]] = 1.0

    logger.info(
        "Built label matrix: %d proteins x %d GO terms", len(protein_ids), num_labels
    )

    return label_matrix, protein_ids, term_names


def save_label_matrix(
    config: Config,
    label_matrix: np.ndarray,
    protein_ids: np.ndarray,
    term_names: np.ndarray,
) -> Path:
    """Persist the label matrix, protein IDs, and term names as .npy files.

    Returns:
        The directory where files were saved.
    """
    out_dir = config.output_dir / f"label_matrix_top{config.num_labels}"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "label_matrix.npy", label_matrix)
    np.save(out_dir / "protein_ids.npy", protein_ids)
    np.save(out_dir / "term_names.npy", term_names)

    logger.info("Saved label matrix artefacts to %s", out_dir)
    return out_dir
