from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from config import API_PREFIX, ARTIFACT_ROOT, DB_PATH
from job_store import JobStore
from schemas import CreateJobRequest, CreateJobResponse, JobStatusResponse
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