"""Tests for the conditional routing logic."""
from src.config import settings
from src.graph.edges import (
    RETURN_ANSWER,
    RETURN_IDK,
    REWRITE_QUERY,
    route_after_critique,
)


def test_grounded_returns_answer():
    assert route_after_critique({"critique": "grounded"}) == RETURN_ANSWER


def test_insufficient_returns_idk():
    assert route_after_critique({"critique": "insufficient"}) == RETURN_IDK


def test_hallucinated_within_budget_rewrites():
    state = {"critique": "hallucinated", "retry_count": 0}
    assert route_after_critique(state) == REWRITE_QUERY


def test_hallucinated_over_budget_gives_up_honestly():
    state = {"critique": "hallucinated", "retry_count": settings.max_retries}
    assert route_after_critique(state) == RETURN_IDK


def test_unknown_verdict_defaults_to_idk():
    assert route_after_critique({"critique": "???"}) == RETURN_IDK
