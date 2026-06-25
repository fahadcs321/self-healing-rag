"""
state.py — The shared state that flows through every node of the graph.

LangGraph passes one ``RAGState`` dict from node to node. Each node reads what it
needs and returns a partial update; think of it as the working memory for a single
user query as it travels Retrieve → Rerank → Generate → Critique → (loop/return).
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────
    query: str  # original user question

    # ── Retrieval ──────────────────────────────────────────────────────────
    rewritten_query: str | None  # query rewritten by the self-heal loop
    retrieved_docs: list[Document]  # raw chunks from Qdrant
    reranked_docs: list[Document]  # after Cohere Rerank

    # ── Generation ─────────────────────────────────────────────────────────
    answer: str  # generated answer (may be hallucinated)

    # ── Critique (LLM-as-judge) ────────────────────────────────────────────
    critique: str  # "grounded" | "hallucinated" | "insufficient"
    critique_reason: str  # one-line explanation from the critic

    # ── Loop control ───────────────────────────────────────────────────────
    retry_count: int  # retries taken so far (capped by MAX_RETRIES)

    # ── Output ─────────────────────────────────────────────────────────────
    final_answer: str  # answer returned to the user
    sources: list[str]  # de-duplicated source document names
    contexts: list[str]  # actual chunk texts used (for evaluation)
    grounded: bool  # True if the answer passed critique
