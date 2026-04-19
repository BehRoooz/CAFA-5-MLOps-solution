from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from config import MLFLOW_EXTERNAL_UI_BASE, MLFLOW_TRACKING_URI, WORKER_POLL_INTERVAL_SEC
from job_store import JobStore

APP_ROOT = Path(__file__).resolve().parents[2]


def _build_mlflow_links(train_run_id: str | None, summary: dict[str, Any]) -> dict[str, Any] | None:
    if not train_run_id:
        return None
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    external_base = (os.environ.get("MLFLOW_EXTERNAL_UI_BASE") or MLFLOW_EXTERNAL_UI_BASE).rstrip("/")
    registered_name = summary.get("registered_model_name")
    registered_ver = summary.get("registered_model_version")
    out: dict[str, Any] = {
        "tracking_uri": tracking_uri,
        "train_run_id": train_run_id,
        "experiment_id": None,
        "run_ui_url": None,
        "registered_model_name": registered_name,
        "registered_model_version": registered_ver,
        "model_registry_ui_url": None,
    }
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        run = client.get_run(train_run_id)
        exp_id = run.info.experiment_id
        out["experiment_id"] = str(exp_id)
        out["run_ui_url"] = f"{external_base}/#/experiments/{exp_id}/runs/{train_run_id}"
        if registered_name and registered_ver:
            out["model_registry_ui_url"] = (
                f"{external_base}/#/models/{registered_name}/versions/{registered_ver}"
            )
    except Exception:
        pass
    return out


def _run_training_subprocess(request: dict[str, Any]) -> tuple[int, str, str]:
    config_path = request.get("config") or "configs/config.yaml"
    mode = request.get("mode") or "train"
    if mode == "retrain":
        cmd = [
            os.environ.get("PYTHON_EXECUTABLE", "python"),
            str(APP_ROOT / "scripts" / "retrain_pipeline.py"),
            "--config",
            config_path,
        ]
    else:
        cmd = [
            os.environ.get("PYTHON_EXECUTABLE", "python"),
            str(APP_ROOT / "scripts" / "train.py"),
            "--config",
            config_path,
        ]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(APP_ROOT))
    proc = subprocess.run(
        cmd,
        cwd=str(APP_ROOT),
        capture_output=True,
        text=True,
        env=env,
        timeout=None,
    )
    return proc.returncode, proc.stdout, proc.stderr


def process_job(store: JobStore, job_id: str) -> None:
    job = store.get_job(job_id)
    if job is None:
        return
    req = job["request"]

    store.update_progress(job_id, percent=1.0, message="starting training subprocess")
    code, stdout, stderr = _run_training_subprocess(req)

    summary_path = APP_ROOT / "outputs" / "train_run_summary.json"
    if code != 0:
        err_text = (stderr or "").strip() or (stdout or "").strip() or f"exit code {code}"
        store.mark_failed(
            job_id,
            {
                "code": "TRAINING_SUBPROCESS_FAILED",
                "message": err_text[:8000],
                "exit_code": code,
            },
        )
        return

    if not summary_path.exists():
        store.mark_failed(
            job_id,
            {
                "code": "TRAIN_SUMMARY_MISSING",
                "message": "train.py reported success but outputs/train_run_summary.json not found",
            },
        )
        return

    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError as exc:
        store.mark_failed(
            job_id,
            {"code": "TRAIN_SUMMARY_INVALID", "message": str(exc)},
        )
        return

    train_run_id = summary.get("train_run_id")
    store.update_progress(job_id, percent=100.0, message="completed")

    mlflow_payload = _build_mlflow_links(str(train_run_id) if train_run_id else None, summary)

    result: dict[str, Any] = {
        "train_run_id": train_run_id,
        "registered_model_name": summary.get("registered_model_name"),
        "registered_model_version": summary.get("registered_model_version"),
        "model_uri": summary.get("model_uri"),
        "mlflow": mlflow_payload,
    }
    store.mark_succeeded(job_id, result)


def worker_loop(store: JobStore, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        job_id = store.get_next_queued_job_id()
        if job_id is None:
            time.sleep(WORKER_POLL_INTERVAL_SEC)
            continue

        store.mark_running(job_id)
        store.update_progress(job_id, percent=None, message="running")
        try:
            process_job(store, job_id)
        except Exception as exc:
            store.mark_failed(
                job_id,
                {
                    "code": "TRAINING_RUNTIME_FAILURE",
                    "message": str(exc),
                },
            )
