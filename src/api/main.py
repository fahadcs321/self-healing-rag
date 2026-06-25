"""
main.py — FastAPI app exposing the self-healing RAG graph over HTTP.

Run with:
    uvicorn src.api.main:app --reload

The graph (and therefore the heavy model stack) is imported lazily inside the
handler so the app process — and the test suite — can start without the full
dependency tree or API keys present.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from src.api.schemas import HealthResponse, QueryRequest, QueryResponse

logger = logging.getLogger("self_healing_rag.api")

app = FastAPI(
    title="Self-Healing RAG API",
    description=(
        "A RAG pipeline that detects its own hallucinations, rewrites failing "
        "queries, and refuses to answer when the context is insufficient."
    ),
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness probe — does not touch the model stack."""
    return HealthResponse()


@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Answer a question through the full retrieve→rerank→generate→critique loop."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Imported here so module import (and tests) don't require the model stack.
    from src.graph.graph import answer_query

    try:
        result = answer_query(question)
    except Exception as exc:  # noqa: BLE001 - surface a clean 500 to clients
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    return QueryResponse(
        answer=result["answer"],
        grounded=result["grounded"],
        critique=result["critique"],
        critique_reason=result["critique_reason"],
        retries=result["retries"],
        sources=result["sources"],
    )
