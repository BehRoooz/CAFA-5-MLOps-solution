from __future__ import annotations

import threading
import uuid

from fastapi import FastAPI, HTTPException

from config import API_PREFIX, ARTIFACT_ROOT, DB_PATH
from job_store import JobStore
from schemas import CreateTrainJobResponse, JobStatusResponse, MlflowLinks, TrainJobRequest, TrainingProgress
from worker import worker_loop

app = FastAPI(title="Training API", version="0.1.0")

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


@app.post(API_PREFIX + "/train", response_model=CreateTrainJobResponse, status_code=202)
def create_train_job(request: TrainJobRequest) -> CreateTrainJobResponse:
    job_id = str(uuid.uuid4())
    store.create_job(job_id, request.model_dump())
    return CreateTrainJobResponse(
        job_id=job_id,
        status="queued",
        poll_url=f"{API_PREFIX}/jobs/{job_id}",
    )


@app.get(API_PREFIX + "/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="JOB_NOT_FOUND")

    req = job["request"]
    prog = job["progress"]
    progress = TrainingProgress(
        percent=prog.get("percent"),
        message=str(prog.get("message", "")),
    )

    result = job.get("result") or {}
    mlflow_raw = result.get("mlflow")
    mlflow_model: MlflowLinks | None = None
    if isinstance(mlflow_raw, dict) and mlflow_raw.get("tracking_uri"):
        mlflow_model = MlflowLinks(
            tracking_uri=str(mlflow_raw["tracking_uri"]),
            train_run_id=mlflow_raw.get("train_run_id"),
            experiment_id=mlflow_raw.get("experiment_id"),
            run_ui_url=mlflow_raw.get("run_ui_url"),
            registered_model_name=mlflow_raw.get("registered_model_name"),
            registered_model_version=mlflow_raw.get("registered_model_version"),
            model_registry_ui_url=mlflow_raw.get("model_registry_ui_url"),
        )

    err = job["error"]
    if err is not None and not isinstance(err, dict):
        err = {"message": str(err)}

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        mode=str(req.get("mode", "train")),
        config=str(req.get("config", "configs/config.yaml")),
        progress=progress,
        error=err,
        train_run_id=result.get("train_run_id"),
        registered_model_name=result.get("registered_model_name"),
        registered_model_version=result.get("registered_model_version"),
        model_uri=result.get("model_uri"),
        mlflow=mlflow_model,
    )
