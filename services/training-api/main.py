from __future__ import annotations

import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest, CollectorRegistry

from config import API_PREFIX, ARTIFACT_ROOT, DB_PATH
from job_store import JobStore
from schemas import CreateTrainJobResponse, JobStatusResponse, MlflowLinks, TrainJobRequest, TrainingProgress
from worker import worker_loop

app = FastAPI(title="Training API", version="0.1.0")


# Prometheus metrics for the training API
registry = CollectorRegistry() # to store metrics

SERVICE_NAME = "trainer-api"

HTTP_REQUESTS_TOTAL = Counter(
    "cafa5_http_requests_total",
    "Total number of HTTP requests.",
    labelnames=("service", "route", "method", "status_code"),
    registry=registry,
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "cafa5_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    labelnames=("service", "route", "method", "status_code"),
    registry=registry,
)
HTTP_IN_FLIGHT_REQUESTS = Gauge(
    "cafa5_http_in_flight_requests",
    "Number of in-flight HTTP requests.",
    labelnames=("service",),
    registry=registry,
)

store = JobStore(DB_PATH)
stop_event = threading.Event()
worker_thread: threading.Thread | None = None


def _route_label(path: str) -> str:
    if path == "/metrics":
        return "/metrics"
    if path.startswith(API_PREFIX):
        return path[len(API_PREFIX) :] or "/"
    return path


@app.middleware("http")
async def prometheus_http_middleware(request, call_next):
    route = _route_label(request.url.path)
    method = request.method

    HTTP_IN_FLIGHT_REQUESTS.labels(service=SERVICE_NAME).inc()
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration = time.perf_counter() - start
        if route != "/metrics":
            labels = {
                "service": SERVICE_NAME,
                "route": route,
                "method": method,
                "status_code": str(status_code),
            }
            HTTP_REQUESTS_TOTAL.labels(**labels).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(**labels).observe(duration)
        HTTP_IN_FLIGHT_REQUESTS.labels(service=SERVICE_NAME).dec()


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


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


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
