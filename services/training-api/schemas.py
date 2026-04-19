from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TrainJobRequest(BaseModel):
    """Body for POST /train."""

    config: str = Field(default="configs/config.yaml", description="Path to YAML config under repo root")
    mode: Literal["train", "retrain"] = Field(
        default="retrain",
        description="train: scripts/train.py only; retrain: full retrain_pipeline (holdout + promotion)",
    )


class TrainingProgress(BaseModel):
    percent: float | None = None
    message: str = ""


class MlflowLinks(BaseModel):
    tracking_uri: str
    train_run_id: str | None = None
    experiment_id: str | None = None
    run_ui_url: str | None = None
    registered_model_name: str | None = None
    registered_model_version: str | None = None
    model_registry_ui_url: str | None = None


class CreateTrainJobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    poll_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    mode: str
    config: str
    progress: TrainingProgress
    error: dict[str, Any] | str | None = None
    train_run_id: str | None = None
    registered_model_name: str | None = None
    registered_model_version: str | None = None
    model_uri: str | None = None
    mlflow: MlflowLinks | None = None
