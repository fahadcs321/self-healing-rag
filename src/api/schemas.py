"""
schemas.py — Request/response models for the FastAPI app.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question.")

    model_config = {
        "json_schema_extra": {
            "example": {"question": "What does RAGAS measure?"}
        }
    }


class QueryResponse(BaseModel):
    answer: str = Field(..., description="The final answer (or an honest refusal).")
    grounded: bool = Field(..., description="True if the answer passed the critic.")
    critique: str = Field(..., description="grounded | hallucinated | insufficient")
    critique_reason: str = Field("", description="Why the critic reached its verdict.")
    retries: int = Field(0, description="Self-heal loops taken before answering.")
    sources: List[str] = Field(default_factory=list, description="Cited source docs.")


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "self-healing-rag"
