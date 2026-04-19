from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    embedding: List[float] = Field(..., description="Embedding vector matching deployed model dimension")
    top_k: int = Field(default=10, ge=1, description="Number of top predictions to return")


class PredictionItem(BaseModel):
    go_term: str
    score: float


class PredictResponse(BaseModel):
    model_version: str
    top_k: int
    predictions: List[PredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
