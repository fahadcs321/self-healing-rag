"""
edges.py — Conditional routing between nodes.

The single decision point of the graph: after the critic grades the answer, where
do we go next? This is what makes the pipeline *self-healing* rather than linear.
"""
from __future__ import annotations

from src.config import settings
from src.graph.state import RAGState

# Route names — must match the keys wired in graph.py's conditional edge map.
RETURN_ANSWER = "return_answer"
RETURN_IDK = "return_idk"
REWRITE_QUERY = "rewrite_query"


def route_after_critique(state: RAGState) -> str:
    """Decide the next step from the critic's verdict.

    - grounded     → return the answer to the user
    - insufficient → return an honest "I don't know"
    - hallucinated → rewrite the query and retry, until retries are exhausted,
                     then fall back to "I don't know" rather than keep guessing
    """
    verdict = state.get("critique", "hallucinated")
    retry_count = state.get("retry_count", 0)

    if verdict == "grounded":
        return RETURN_ANSWER
    if verdict == "insufficient":
        return RETURN_IDK
    if verdict == "hallucinated" and retry_count < settings.max_retries:
        return REWRITE_QUERY
    # Unknown verdict, or hallucinated with retries exhausted.
    return RETURN_IDK
