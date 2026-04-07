from __future__ import annotations

import os

try:
    import mlflow
    MLFLOW_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - defensive import for runtime environments
    mlflow = None
    MLFLOW_IMPORT_ERROR = str(exc)


def log_inference(
    *,
    model_version: str,
    top_k: int,
    runtime_ms: float,
    prediction_count: int,
    request_id: str | None = None,
) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        return
    if mlflow is None:
        return

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("cafa-inference")

        with mlflow.start_run(nested=False):
            mlflow.log_param("model_version", model_version)
            mlflow.log_param("top_k", top_k)
            mlflow.log_metric("runtime_ms", runtime_ms)
            mlflow.log_metric("prediction_count", prediction_count)

            if request_id:
                mlflow.log_param("request_id", request_id)
    except Exception:
        # best effort only: never break inference
        pass
