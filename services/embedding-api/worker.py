from __future__ import annotations

import threading
import time
from typing import Any

from artifacts import save_test_artifacts
from config import WORKER_POLL_INTERVAL_SEC
from embedder import embed_sequence_batch
from job_store import JobStore
from prometheus_client import Counter, Gauge, Histogram

EMBEDDING_JOBS_TOTAL = Counter(
    "cafa5_embedding_jobs_total",
    "Total embedding jobs partitioned by terminal status and backend.",
    labelnames=("status", "backend"),
)
EMBEDDING_QUEUE_JOBS = Gauge(
    "cafa5_embedding_queue_jobs",
    "Embedding jobs currently in each lifecycle state.",
    labelnames=("status",),
)
EMBEDDING_JOB_DURATION_SECONDS = Histogram(
    "cafa5_embedding_job_duration_seconds",
    "Embedding job duration in seconds by terminal status and backend.",
    labelnames=("status", "backend"),
)
EMBEDDING_SEQUENCES_PROCESSED_TOTAL = Counter(
    "cafa5_embedding_sequences_processed_total",
    "Number of embedded sequences processed by backend.",
    labelnames=("backend",),
)
EMBEDDING_ARTIFACT_BYTES = Histogram(
    "cafa5_embedding_artifact_bytes",
    "Embedding artifact output size in bytes partitioned by artifact name.",
    labelnames=("artifact_name",),
    buckets=(1024, 10 * 1024, 100 * 1024, 1024**2, 5 * 1024**2, 10 * 1024**2, float("inf")),
)


def parse_fasta_text(fasta_text: str) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    sequences: list[str] = []
    current_id: str | None = None
    current_seq: list[str] = []

    for raw_line in fasta_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                ids.append(current_id)
                sequences.append("".join(current_seq))
            current_id = line[1:].split()[0]
            current_seq = []
        else:
            current_seq.append(line)

    if current_id is not None:
        ids.append(current_id)
        sequences.append("".join(current_seq))

    if not ids:
        raise ValueError("No FASTA records found in input.")
    return ids, sequences


def _extract_ids_sequences(request: dict[str, Any]) -> tuple[list[str], list[str]]:
    if "sequences" in request and request["sequences"]:
        ids = [item["id"] for item in request["sequences"]]
        sequences = [item["sequence"] for item in request["sequences"]]
        return ids, sequences

    if "fasta_text" in request and request["fasta_text"]:
        return parse_fasta_text(request["fasta_text"])

    raise ValueError("Request must include either `sequences` or `fasta_text`.")


def _sync_queue_gauges(store: JobStore) -> None:
    for status in ("queued", "running", "succeeded", "failed"):
        EMBEDDING_QUEUE_JOBS.labels(status=status).set(store.count_jobs_by_status(status))


def process_job(store: JobStore, job_id: str) -> None:
    job = store.get_job(job_id)
    if job is None:
        return

    req = job["request"]
    ids, sequences = _extract_ids_sequences(req)
    total = len(ids)
    store.update_progress(job_id, embedded=0, total=total)

    ids_out, embeds = embed_sequence_batch(
        ids,
        sequences,
        backend=req.get("backend", "esm2"),
        pooling=req.get("pooling", "mean"),
        max_length=int(req.get("max_length", 1280)),
        batch_size=int(req.get("batch_size", 8)),
    )
    store.update_progress(job_id, embedded=total, total=total)
    EMBEDDING_SEQUENCES_PROCESSED_TOTAL.labels(backend=req.get("backend", "esm2")).inc(total)

    artifacts = save_test_artifacts(job_id, ids_out, embeds)
    for art in artifacts:
        store.insert_artifact(
            job_id=job_id,
            name=art["name"],
            path=art["path"],
            dtype=art["dtype"],
            shape=art["shape"],
            size_bytes=art["size_bytes"],
        )
        EMBEDDING_ARTIFACT_BYTES.labels(artifact_name=art["name"]).observe(float(art["size_bytes"]))


def worker_loop(store: JobStore, stop_event: threading.Event) -> None:
    _sync_queue_gauges(store)
    while not stop_event.is_set():
        job_id = store.get_next_queued_job_id()
        if job_id is None:
            _sync_queue_gauges(store)
            time.sleep(WORKER_POLL_INTERVAL_SEC)
            continue

        job = store.get_job(job_id)
        backend = "unknown"
        if job is not None:
            backend = str(job["request"].get("backend", "esm2"))

        store.mark_running(job_id)
        _sync_queue_gauges(store)
        started = time.perf_counter()
        try:
            process_job(store, job_id)
            store.mark_succeeded(job_id)
            duration = time.perf_counter() - started
            EMBEDDING_JOBS_TOTAL.labels(status="succeeded", backend=backend).inc()
            EMBEDDING_JOB_DURATION_SECONDS.labels(status="succeeded", backend=backend).observe(duration)
        except Exception as exc:  # broad catch for stable worker loop
            store.mark_failed(
                job_id,
                {
                    "code": "EMBEDDING_RUNTIME_FAILURE",
                    "message": str(exc),
                },
            )
            duration = time.perf_counter() - started
            EMBEDDING_JOBS_TOTAL.labels(status="failed", backend=backend).inc()
            EMBEDDING_JOB_DURATION_SECONDS.labels(status="failed", backend=backend).observe(duration)
        finally:
            _sync_queue_gauges(store)
