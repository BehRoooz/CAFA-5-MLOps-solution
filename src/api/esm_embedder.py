"""ESM-2 wrapper: load the model once, generate per-residue embeddings, mean-pool."""

from __future__ import annotations

import logging
import re

import numpy as np
import torch

logger = logging.getLogger("cafa5")

_ESM_MODEL_NAME = "esm2_t33_650M_UR50D"


class ESMEmbedder:
    """Lazy-loaded ESM-2 embedder.

    The heavy model is loaded on first call to :meth:`generate_embedding`
    so that import-time stays fast.
    """

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._alphabet = None
        self._batch_converter = None
        self._device: torch.device | None = None

    def _load_model(self) -> None:
        import esm

        logger.info("Loading ESM-2 model (%s) …", _ESM_MODEL_NAME)
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(self._device)
        model.eval()
        self._model = model
        self._alphabet = alphabet
        self._batch_converter = alphabet.get_batch_converter()
        logger.info("ESM-2 loaded on %s", self._device)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def generate_embedding(self, sequence: str) -> np.ndarray:
        """Generate a mean-pooled ESM-2 embedding for a protein sequence.

        Args:
            sequence: Raw amino-acid string **or** FASTA-formatted string.

        Returns:
            1-D numpy array of shape ``(1280,)``.
        """
        if not self.is_loaded:
            self._load_model()

        clean_seq, _ = _parse_fasta(sequence)
        data = [("protein", clean_seq)]
        _, _, batch_tokens = self._batch_converter(data)
        batch_tokens = batch_tokens.to(self._device)

        with torch.no_grad():
            results = self._model(batch_tokens, repr_layers=[33], return_contacts=False)

        token_embeddings = results["representations"][33]  # (1, seq_len+2, 1280)
        # Exclude BOS / EOS special tokens
        embedding = token_embeddings[0, 1 : len(clean_seq) + 1].mean(dim=0)
        return embedding.cpu().numpy()


def _parse_fasta(text: str) -> tuple[str, str | None]:
    """Extract a plain amino-acid sequence and optional protein ID from FASTA or raw text."""
    lines = text.strip().splitlines()
    protein_id: str | None = None

    if lines[0].startswith(">"):
        header = lines[0][1:].strip()
        match = re.search(r"\|(\w+)\|", header)
        protein_id = match.group(1) if match else header.split()[0]
        seq = "".join(l.strip() for l in lines[1:])
    else:
        seq = "".join(l.strip() for l in lines)

    seq = re.sub(r"[^A-Za-z]", "", seq).upper()
    return seq, protein_id
