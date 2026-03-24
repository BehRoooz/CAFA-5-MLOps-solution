"""Pydantic request/response models for the CAFA-5 prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SequenceRequest(BaseModel):
    """Request body for the /predict/embedding endpoint."""

    sequence: str = Field(
        ...,
        min_length=1,
        description="Protein amino-acid sequence (single-letter code) or FASTA-formatted string.",
        examples=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVL"],
    )


class EmbeddingResponse(BaseModel):
    """Response from the /predict/embedding endpoint."""

    protein_id: str | None = Field(
        None, description="Protein identifier extracted from FASTA header, if present."
    )
    embedding: list[float] = Field(
        ..., description="Mean-pooled ESM-2 embedding vector (length 1280)."
    )


class GOTermPrediction(BaseModel):
    """Single GO-term prediction with confidence score."""

    go_term: str = Field(..., description="Gene Ontology term identifier.", examples=["GO:0005515"])
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence (0-1).")


class EmbeddingRequest(BaseModel):
    """Request body for the /predict/go-terms endpoint."""

    embedding: list[float] = Field(
        ...,
        min_length=1,
        description="Protein embedding vector (length must match model input dimension).",
    )


class PredictionResponse(BaseModel):
    """Response from the /predict/go-terms endpoint."""

    predictions: list[GOTermPrediction] = Field(
        ..., description="GO-term predictions above the confidence threshold."
    )
