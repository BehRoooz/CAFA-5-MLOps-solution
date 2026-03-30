from __future__ import annotations

import threading
import time
from typing import Any

from artifacts import save_test_artifacts
from config import WORKER_POLL_INTERVAL_SEC
from embedder import embed_sequence_batch
from job_store import JobStore


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


def worker_loop(store: JobStore, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        job_id = store.get_next_queued_job_id()
        if job_id is None:
            time.sleep(WORKER_POLL_INTERVAL_SEC)
            continue

        store.mark_running(job_id)
        try:
            process_job(store, job_id)
            store.mark_succeeded(job_id)
        except Exception as exc:  # broad catch for stable worker loop
            store.mark_failed(
                job_id,
                {
                    "code": "EMBEDDING_RUNTIME_FAILURE",
                    "message": str(exc),
                },
            )
