from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from mlflow_logger import log_inference
from model_loader import load_model, load_model_from_registry, load_term_names
from predictor_service import predict_top_k
from schemas import HealthResponse, PredictRequest, PredictResponse

APP_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = APP_ROOT / "outputs" / "checkpoints" / "best_model.pt"
MODEL_URI = os.getenv("MODEL_URI", "models:/cafa-go-model@champion")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/mlflow-cache")
TERM_NAMES_PATH = APP_ROOT / "outputs" / "label_matrix_top500" / "term_names.npy"
META_PATH = APP_ROOT / "outputs" / "splits" / "model_meta.json"

app = FastAPI(title="CAFA Inference API")

MODEL = None
TERM_NAMES = None
MODEL_META = None


@app.on_event("startup")
def startup_event() -> None:
    global MODEL, TERM_NAMES, MODEL_META
    try:
        if MODEL_URI:
            MODEL, TERM_NAMES, MODEL_META = load_model_from_registry(
                MODEL_URI,
                device="cpu",
                cache_dir=MODEL_CACHE_DIR,
            )
        else:
            MODEL, MODEL_META = load_model(CHECKPOINT_PATH, META_PATH, device="cpu")
            TERM_NAMES = load_term_names(TERM_NAMES_PATH)
    except Exception:
        MODEL = None
        TERM_NAMES = None
        MODEL_META = None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=MODEL is not None and TERM_NAMES is not None and MODEL_META is not None,
        model_version=MODEL_META.get("model_version") if MODEL_META else None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, raw_request: Request) -> PredictResponse:
    if MODEL is None or TERM_NAMES is None or MODEL_META is None:
        raise HTTPException(status_code=500, detail="model artifacts could not be loaded")

    start = time.perf_counter()

    try:
        result = predict_top_k(
            model=MODEL,
            embedding=request.embedding,
            term_names=TERM_NAMES,
            top_k=request.top_k,
            apply_sigmoid=True,
            device="cpu",
            expected_dim=int(MODEL_META.get("embedding_dim", 1280)),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail="inference failure") from e

    runtime_ms = (time.perf_counter() - start) * 1000.0
    request_id = raw_request.headers.get("X-Request-ID")

    log_inference(
        model_version=MODEL_META.get("model_version", "unknown"),
        top_k=result["top_k"],
        runtime_ms=runtime_ms,
        prediction_count=len(result["predictions"]),
        request_id=request_id,
    )

    return PredictResponse(
        model_version=MODEL_META.get("model_version", "unknown"),
        top_k=result["top_k"],
        predictions=result["predictions"],
    )
