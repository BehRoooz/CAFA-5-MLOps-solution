from __future__ import annotations

from pathlib import Path

import numpy as np

from config import ARTIFACT_ROOT


def get_job_dir(job_id: str) -> Path:
    out_dir = ARTIFACT_ROOT / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_test_artifacts(job_id: str, ids: list[str], embeddings: np.ndarray) -> list[dict]:
    out_dir = get_job_dir(job_id)
    ids_path = out_dir / "test_ids.npy"
    embeds_path = out_dir / "test_embeddings.npy"

    np.save(ids_path, np.asarray(ids, dtype=object))
    np.save(embeds_path, embeddings.astype(np.float32))

    return [
        {
            "name": "test_ids.npy",
            "path": str(ids_path),
            "dtype": "object",
            "shape": [len(ids)],
            "size_bytes": ids_path.stat().st_size,
        },
        {
            "name": "test_embeddings.npy",
            "path": str(embeds_path),
            "dtype": str(embeddings.dtype),
            "shape": list(embeddings.shape),
            "size_bytes": embeds_path.stat().st_size,
        },
    ]

