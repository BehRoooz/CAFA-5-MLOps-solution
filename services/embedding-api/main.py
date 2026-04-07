from __future__ import annotations

import threading
import uuid
import json
import time
from pathlib import Path
from typing import Literal
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from config import API_PREFIX, ARTIFACT_ROOT, DB_PATH, GO_PREDICTION_API_URL
from job_store import JobStore
from schemas import (
    CreateJobRequest,
    CreateJobResponse,
    JobStatusResponse,
    PredictGoFromSequencesRequest,
    PredictGoRequest,
    PredictGoResponse,
)
from worker import worker_loop

app = FastAPI(title="Embedding API", version="0.1.0")

store = JobStore(DB_PATH)
stop_event = threading.Event()
worker_thread: threading.Thread | None = None


@app.on_event("startup")
def startup_event() -> None:
    global worker_thread
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    worker_thread = threading.Thread(target=worker_loop, args=(store, stop_event), daemon=True)
    worker_thread.start()


@app.on_event("shutdown")
def shutdown_event() -> None:
    stop_event.set()
    if worker_thread is not None:
        worker_thread.join(timeout=2)


@app.get(API_PREFIX + "/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(API_PREFIX + "/jobs", response_model=CreateJobResponse, status_code=202)
def create_job(request: CreateJobRequest) -> CreateJobResponse:
    job_id = str(uuid.uuid4())
    store.create_job(job_id, request.model_dump())
    return CreateJobResponse(
        job_id=job_id,
        status="queued",
        poll_url=f"{API_PREFIX}/jobs/{job_id}",
    )


@app.post(API_PREFIX + "/jobs/fasta", response_model=CreateJobResponse, status_code=202)
async def create_fasta_job(
    fasta_file: UploadFile = File(...),
    backend: Literal["esm2", "protbert", "t5"] = Form(default="esm2"),
    pooling: Literal["mean", "cls"] = Form(default="mean"),
    batch_size: int = Form(default=8),
    max_length: int = Form(default=1280),
) -> CreateJobResponse:
    fasta_text = (await fasta_file.read()).decode("utf-8", errors="replace")
    if not fasta_text.strip():
        raise HTTPException(status_code=400, detail="Uploaded FASTA is empty.")

    job_id = str(uuid.uuid4())
    payload = {
        "stage": "test",
        "backend": backend,
        "pooling": pooling,
        "batch_size": batch_size,
        "max_length": max_length,
        "fasta_text": fasta_text,
    }
    store.create_job(job_id, payload)
    return CreateJobResponse(
        job_id=job_id,
        status="queued",
        poll_url=f"{API_PREFIX}/jobs/{job_id}",
    )


@app.get(API_PREFIX + "/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="JOB_NOT_FOUND")

    artifacts = store.list_artifacts(job_id)
    req = job["request"]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        stage=req.get("stage", "test"),
        backend=req.get("backend", "esm2"),
        progress=job["progress"],
        error=job["error"],
        artifacts_manifest=artifacts if artifacts else None,
    )


@app.get(API_PREFIX + "/jobs/{job_id}/artifacts/{name}")
def get_artifact(job_id: str, name: str):
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="JOB_NOT_FOUND")
    if job["status"] != "succeeded":
        raise HTTPException(status_code=409, detail="JOB_NOT_READY")

    artifacts = store.list_artifacts(job_id)
    match = next((a for a in artifacts if a["name"] == name), None)
    if match is None:
        raise HTTPException(status_code=404, detail="ARTIFACT_NOT_FOUND")

    path = Path(match["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="ARTIFACT_NOT_FOUND")
    return FileResponse(path=str(path), filename=name, media_type="application/octet-stream")


def _post_go_predict(embedding: list[float], top_k: int) -> dict:
    endpoint = f"{GO_PREDICTION_API_URL.rstrip('/')}/predict"
    payload = json.dumps({"embedding": embedding, "top_k": top_k}).encode("utf-8")
    req = urlrequest.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(
            status_code=502,
            detail=f"GO_API_HTTP_{exc.code}: {err_body}",
        ) from exc
    except URLError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"GO_API_UNREACHABLE: {exc.reason}",
        ) from exc


def _predict_go_for_job(job_id: str, request: PredictGoRequest) -> PredictGoResponse:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="JOB_NOT_FOUND")
    if job["status"] != "succeeded":
        raise HTTPException(status_code=409, detail="JOB_NOT_READY")

    artifacts = store.list_artifacts(job_id)
    ids_entry = next((a for a in artifacts if a["name"] == "test_ids.npy"), None)
    emb_entry = next((a for a in artifacts if a["name"] == "test_embeddings.npy"), None)
    if ids_entry is None or emb_entry is None:
        raise HTTPException(status_code=404, detail="EMBEDDING_ARTIFACTS_NOT_FOUND")

    ids = np.load(ids_entry["path"], allow_pickle=True)
    embeddings = np.load(emb_entry["path"])
    if embeddings.ndim != 2:
        raise HTTPException(status_code=500, detail="INVALID_EMBEDDINGS_SHAPE")
    if len(ids) != embeddings.shape[0]:
        raise HTTPException(status_code=500, detail="IDS_EMBEDDINGS_LENGTH_MISMATCH")
    if embeddings.shape[1] != 1280:
        raise HTTPException(
            status_code=400,
            detail=(
                "GO API expects 1280-dim embeddings. "
                f"Received dimension {embeddings.shape[1]} from embedding job."
            ),
        )

    if request.indices is None:
        selected_indices = list(range(embeddings.shape[0]))
    else:
        if not request.indices:
            raise HTTPException(status_code=400, detail="indices must not be empty")
        selected_indices = request.indices
        bad = [i for i in selected_indices if i < 0 or i >= embeddings.shape[0]]
        if bad:
            raise HTTPException(status_code=400, detail=f"indices out of range: {bad}")

    results: list[dict] = []
    failures: list[dict] = []
    model_version: str | None = None
    for idx in selected_indices:
        try:
            response = _post_go_predict(embeddings[idx].astype(float).tolist(), request.top_k)
            if model_version is None:
                model_version = response.get("model_version")
            results.append(
                {
                    "index": idx,
                    "sequence_id": str(ids[idx]),
                    "predictions": response.get("predictions", []),
                }
            )
        except HTTPException as exc:
            failure = {"index": idx, "sequence_id": str(ids[idx]), "error": exc.detail}
            if request.fail_fast:
                raise HTTPException(status_code=502, detail=failure) from exc
            failures.append(failure)

    return PredictGoResponse(
        job_id=job_id,
        status="succeeded",
        model_version=model_version,
        top_k=request.top_k,
        results=results,
        failures=failures,
    )


@app.post(API_PREFIX + "/jobs/{job_id}/predict-go", response_model=PredictGoResponse)
def predict_go_for_job(job_id: str, request: PredictGoRequest) -> PredictGoResponse:
    return _predict_go_for_job(job_id, request)


def _wait_for_job_completion(job_id: str, timeout_seconds: int, poll_interval_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    while True:
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="JOB_NOT_FOUND")
        if job["status"] == "succeeded":
            return
        if job["status"] == "failed":
            raise HTTPException(status_code=500, detail=job["error"] or "EMBEDDING_JOB_FAILED")
        if time.time() > deadline:
            raise HTTPException(status_code=504, detail="EMBEDDING_JOB_TIMEOUT")
        time.sleep(poll_interval_seconds)


@app.post(API_PREFIX + "/predict-go-from-sequences", response_model=PredictGoResponse)
def predict_go_from_sequences(request: PredictGoFromSequencesRequest) -> PredictGoResponse:
    job_id = str(uuid.uuid4())
    job_payload = {
        "stage": "test",
        "backend": request.backend,
        "pooling": request.pooling,
        "batch_size": request.batch_size,
        "max_length": request.max_length,
        "sequences": [seq.model_dump() for seq in request.sequences],
    }
    store.create_job(job_id, job_payload)
    _wait_for_job_completion(
        job_id=job_id,
        timeout_seconds=request.timeout_seconds,
        poll_interval_seconds=request.poll_interval_seconds,
    )
    return _predict_go_for_job(
        job_id=job_id,
        request=PredictGoRequest(top_k=request.top_k, indices=request.indices, fail_fast=request.fail_fast),
    )