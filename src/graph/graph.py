"""
graph.py — Assemble and compile the self-healing RAG graph.

    retrieve → rerank → generate → critique → ┐
        ▲                                      ├─ grounded     → return_answer → END
        └──────── rewrite_query ◄──────────────┤─ hallucinated → rewrite_query (retry)
                                               └─ insufficient → return_idk     → END

Public entry point: ``answer_query(question)``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langgraph.graph import END, StateGraph

from src.graph.edges import (
    RETURN_ANSWER,
    RETURN_IDK,
    REWRITE_QUERY,
    route_after_critique,
)
from src.graph.nodes import (
    critique,
    generate,
    rerank,
    retrieve,
    return_answer,
    return_idk,
    rewrite_query,
)
from src.graph.state import RAGState


def build_graph() -> Any:
    """Wire the nodes and edges and compile the graph."""
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)
    graph.add_node("critique", critique)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("return_answer", return_answer)
    graph.add_node("return_idk", return_idk)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "critique")

    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {
            RETURN_ANSWER: "return_answer",
            RETURN_IDK: "return_idk",
            REWRITE_QUERY: "rewrite_query",
        },
    )

    # The self-healing loop: a rewritten query goes back through retrieval.
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("return_answer", END)
    graph.add_edge("return_idk", END)

    return graph.compile()


@lru_cache(maxsize=1)
def get_graph() -> Any:
    """Compile the graph once and reuse it."""
    return build_graph()


def _initial_state(question: str) -> RAGState:
    return {
        "query": question,
        "rewritten_query": None,
        "retrieved_docs": [],
        "reranked_docs": [],
        "answer": "",
        "critique": "",
        "critique_reason": "",
        "retry_count": 0,
        "final_answer": "",
        "sources": [],
        "contexts": [],
        "grounded": False,
    }


def answer_query(question: str) -> dict[str, Any]:
    """Run a question through the full graph and return a clean result dict."""
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    result = get_graph().invoke(_initial_state(question))
    return {
        "answer": result.get("final_answer", ""),
        "sources": result.get("sources", []),
        "contexts": result.get("contexts", []),
        "grounded": result.get("grounded", False),
        "critique": result.get("critique", ""),
        "critique_reason": result.get("critique_reason", ""),
        "retries": result.get("retry_count", 0),
    }


# Backwards-compatible alias used by earlier callers.
query = answer_query
