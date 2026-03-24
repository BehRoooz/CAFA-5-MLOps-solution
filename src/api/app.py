"""FastAPI application for CAFA-5 protein function prediction.

Endpoints
---------
GET  /health              — liveness probe
POST /predict/embedding   — FASTA sequence → ESM-2 embedding
POST /predict/go-terms    — embedding vector → GO-term predictions
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from src.api.esm_embedder import ESMEmbedder, _parse_fasta
from src.api.schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    GOTermPrediction,
    PredictionResponse,
    SequenceRequest,
)
from src.config import Config, load_config
from src.models import build_model
from src.utils import get_device

logger = logging.getLogger("cafa5")

_config: Config | None = None
_go_model: torch.nn.Module | None = None
_esm_embedder: ESMEmbedder | None = None
_term_names: np.ndarray | None = None
_prediction_threshold: float = 0.5


def _resolve_config() -> Config:
    """Load config from CONFIG_PATH env var or default location."""
    import os

    config_path = os.environ.get("CONFIG_PATH", "configs/config.yaml")
    return load_config(config_path)


def _resolve_checkpoint(config: Config) -> Path:
    """Return checkpoint path from env var or default."""
    import os

    cp = os.environ.get("CHECKPOINT_PATH")
    if cp:
        return Path(cp)
    return config.output_dir / "checkpoints" / "best_model.pt"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, release on shutdown."""
    global _config, _go_model, _esm_embedder, _term_names, _prediction_threshold

    _config = _resolve_config()
    _prediction_threshold = _config.api.get("prediction_threshold", 0.5)

    checkpoint_path = _resolve_checkpoint(_config)
    if checkpoint_path.exists():
        device = get_device()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_model(_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        _go_model = model
        logger.info("GO-term prediction model loaded from %s", checkpoint_path)
    else:
        logger.warning("Checkpoint not found at %s — /predict/go-terms will be unavailable", checkpoint_path)

    label_dir = _config.output_dir / f"label_matrix_top{_config.num_labels}"
    terms_path = label_dir / "term_names.npy"
    if terms_path.exists():
        _term_names = np.load(terms_path, allow_pickle=True)
        logger.info("Loaded %d GO term names", len(_term_names))
    else:
        logger.warning("term_names.npy not found at %s", terms_path)

    _esm_embedder = ESMEmbedder()
    logger.info("API ready")

    yield

    _go_model = None
    _esm_embedder = None
    _term_names = None


app = FastAPI(
    title="CAFA-5 Protein Function Prediction API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "go_model_loaded": _go_model is not None,
        "esm_model_loaded": _esm_embedder is not None and _esm_embedder.is_loaded,
    }


@app.post("/predict/embedding", response_model=EmbeddingResponse)
async def predict_embedding(request: SequenceRequest):
    """Generate an ESM-2 embedding from a protein sequence.

    The ESM-2 model is loaded lazily on the first request.
    """
    if _esm_embedder is None:
        raise HTTPException(status_code=503, detail="ESM embedder not initialised")

    _, protein_id = _parse_fasta(request.sequence)

    try:
        embedding = _esm_embedder.generate_embedding(request.sequence)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {exc}") from exc

    return EmbeddingResponse(
        protein_id=protein_id,
        embedding=embedding.tolist(),
    )


@app.post("/predict/go-terms", response_model=PredictionResponse)
async def predict_go_terms(request: EmbeddingRequest):
    """Predict GO terms from a pre-computed embedding vector."""
    if _go_model is None:
        raise HTTPException(status_code=503, detail="GO-term model not loaded (no checkpoint found)")
    if _term_names is None:
        raise HTTPException(status_code=503, detail="GO term names not loaded")
    if _config is None:
        raise HTTPException(status_code=503, detail="Config not loaded")

    expected_dim = _config.embedding_dim
    if len(request.embedding) != expected_dim:
        raise HTTPException(
            status_code=422,
            detail=f"Embedding must have length {expected_dim}, got {len(request.embedding)}",
        )

    device = get_device()
    tensor = torch.tensor([request.embedding], dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = _go_model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    predictions = [
        GOTermPrediction(go_term=str(_term_names[i]), confidence=round(float(probs[i]), 4))
        for i in range(len(probs))
        if probs[i] >= _prediction_threshold
    ]
    predictions.sort(key=lambda p: p.confidence, reverse=True)

    return PredictionResponse(predictions=predictions)
