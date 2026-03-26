"""PyTorch Dataset for protein embeddings + GO-term labels."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import Config

logger = logging.getLogger("cafa5")

EMBED_FILE_MAP: dict[str, dict[str, str]] = {
    "esm2": {
        # Written by scripts/embed_sequences.py
        "dir": "hf_esm2",
        "train_embeds": "train_embeddings.npy",
        "train_ids": "train_ids.npy",
        "holdout_embeds": "holdout_embeddings.npy",
        "holdout_ids": "holdout_ids.npy",
        "test_embeds": "test_embeddings.npy",
        "test_ids": "test_ids.npy",
    },
    "protbert": {
        # Written by scripts/embed_sequences.py
        "dir": "hf_protbert",
        "train_embeds": "train_embeddings.npy",
        "train_ids": "train_ids.npy",
        "holdout_embeds": "holdout_embeddings.npy",
        "holdout_ids": "holdout_ids.npy",
        "test_embeds": "test_embeddings.npy",
        "test_ids": "test_ids.npy",
    },
    "t5": {
        # Written by scripts/embed_sequences.py
        "dir": "hf_prot_t5",
        "train_embeds": "train_embeddings.npy",
        "train_ids": "train_ids.npy",
        "holdout_embeds": "holdout_embeddings.npy",
        "holdout_ids": "holdout_ids.npy",
        "test_embeds": "test_embeddings.npy",
        "test_ids": "test_ids.npy",
    },
}


class ProteinSequenceDataset(Dataset):
    """Dataset yielding (embedding, label) for training or (embedding, protein_id) for test.

    Loads numpy arrays directly into tensors — no intermediate DataFrame.

    Args:
        config: Project configuration.
        datatype: ``"train"`` or ``"test"``.
        label_matrix_dir: Directory containing precomputed label_matrix.npy / protein_ids.npy.
                          Required when ``datatype="train"``.
    """

    def __init__(
        self,
        config: Config,
        datatype: str,
        label_matrix_dir: Union[str, Path, None] = None,
    ) -> None:
        super().__init__()
        if datatype not in ("train", "test", "holdout"):
            raise ValueError(f"datatype must be 'train' or 'test', or 'holdout', got '{datatype}'")

        self.datatype = datatype
        source = config.data.get("embeddings_source", "ESM2").lower()
        embeds_dir = Path(config.data.get("embeddings_dir", "./data/embeddings"))

        file_info = EMBED_FILE_MAP.get(source)
        if file_info is None:
            raise ValueError(f"Unknown embeddings_source '{source}'")

        embed_subdir = embeds_dir / file_info["dir"]
        embed_file = file_info[f"{datatype}_embeds"]
        ids_file = file_info[f"{datatype}_ids"]

        logger.info("Loading %s embeddings from %s", datatype, embed_subdir)
        self.embeddings = torch.from_numpy(
            np.load(embed_subdir / embed_file)
        ).float()
        self.ids: np.ndarray = np.load(embed_subdir / ids_file, allow_pickle=True)

        self.labels: torch.Tensor | None = None
        if datatype == "train":
            if label_matrix_dir is None:
                label_matrix_dir = (
                    config.output_dir / f"label_matrix_top{config.num_labels}"
                )
            label_matrix_dir = Path(label_matrix_dir)
            lm_path = label_matrix_dir / "label_matrix.npy"
            lm_ids_path = label_matrix_dir / "protein_ids.npy"

            if not lm_path.exists():
                raise FileNotFoundError(
                    f"Label matrix not found at {lm_path}. "
                    "Run scripts/preprocess.py first."
                )

            label_matrix = np.load(lm_path)
            label_ids = np.load(lm_ids_path, allow_pickle=True)

            id_to_label_idx = {pid: i for i, pid in enumerate(label_ids)}

            keep_mask = np.array(
                [pid in id_to_label_idx for pid in self.ids], dtype=bool
            )
            kept_label_indices = np.array(
                [id_to_label_idx[pid] for pid in self.ids if pid in id_to_label_idx]
            )

            self.embeddings = self.embeddings[keep_mask]
            self.ids = self.ids[keep_mask]
            self.labels = torch.from_numpy(label_matrix[kept_label_indices]).float()

            logger.info(
                "Train dataset: %d samples, embedding dim %d, %d labels",
                len(self.embeddings),
                self.embeddings.shape[1],
                self.labels.shape[1],
            )

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, index: int):
        embed = self.embeddings[index]
        if self.datatype == "train":
            return embed, self.labels[index]
        return embed, self.ids[index]
