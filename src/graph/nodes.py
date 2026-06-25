"""
nodes.py — The graph's nodes, each a function: RAGState in, partial RAGState out.

Keeping nodes as small functions that delegate to the retrieval/LLM layers makes
every step independently unit-testable (inject fakes for the retriever, reranker
and LLM). The LLM is created lazily so importing this module needs no API key.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document

from src.config import settings
from src.graph.prompts import CRITIQUE_PROMPT, GENERATION_PROMPT, REWRITE_PROMPT
from src.graph.state import RAGState
from src.retrieval.reranker import get_reranker
from src.retrieval.retriever import get_retriever

VALID_VERDICTS = {"grounded", "hallucinated", "insufficient"}
IDK_MARKER = "I don't have enough information"


# Sensible default model per provider when LLM_MODEL is left unset.
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "google": "gemini-1.5-flash",
    "anthropic": "claude-3-5-haiku-latest",
}


@lru_cache(maxsize=1)
def get_llm() -> Any:
    """Lazily build the chat model for the configured provider.

    The provider is chosen by ``LLM_PROVIDER`` and the model by ``LLM_MODEL``
    (falling back to a per-provider default). Temperature is 0 for deterministic
    generation and grading. Provider SDKs are imported lazily so only the one you
    actually use needs to be installed.
    """
    provider = settings.llm_provider
    model = settings.llm_model or DEFAULT_MODELS.get(provider, "gpt-4o-mini")
    temperature = settings.llm_temperature

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=temperature)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature)

    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use one of: {', '.join(DEFAULT_MODELS)}.")


def _active_query(state: RAGState) -> str:
    """Use the rewritten query if the self-heal loop produced one."""
    return state.get("rewritten_query") or state["query"]


def _format_context(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


def _invoke_llm(prompt: str) -> str:
    return get_llm().invoke(prompt).content.strip()


# ── Nodes ────────────────────────────────────────────────────────────────────


def retrieve(state: RAGState) -> RAGState:
    """Dense search Qdrant for candidate chunks."""
    docs = get_retriever().search(_active_query(state), top_k=settings.retrieve_top_k)
    return {"retrieved_docs": docs}


def rerank(state: RAGState) -> RAGState:
    """Re-score and trim candidates with Cohere Rerank."""
    docs = state.get("retrieved_docs", [])
    reranked = get_reranker().rerank(_active_query(state), docs, top_n=settings.rerank_top_n)
    return {"reranked_docs": reranked}


def generate(state: RAGState) -> RAGState:
    """Synthesise an answer grounded in the reranked context."""
    docs = state.get("reranked_docs", [])
    prompt = GENERATION_PROMPT.format(context=_format_context(docs), query=state["query"])
    answer = _invoke_llm(prompt)

    sources = sorted({d.metadata.get("source", "unknown") for d in docs})
    contexts = [d.page_content for d in docs]
    return {"answer": answer, "sources": sources, "contexts": contexts}


def critique(state: RAGState) -> RAGState:
    """LLM-as-judge: is the answer grounded, hallucinated, or insufficient?"""
    docs = state.get("reranked_docs", [])
    prompt = CRITIQUE_PROMPT.format(
        context=_format_context(docs),
        query=state["query"],
        answer=state.get("answer", ""),
    )
    raw = _invoke_llm(prompt)
    verdict, reason = _parse_critique(raw)
    return {"critique": verdict, "critique_reason": reason}


def rewrite_query(state: RAGState) -> RAGState:
    """Rewrite the query to recover from a hallucinated answer."""
    prompt = REWRITE_PROMPT.format(
        original_query=state["query"],
        critique_reason=state.get("critique_reason", ""),
    )
    rewritten = _invoke_llm(prompt)
    retry_count = state.get("retry_count", 0) + 1
    return {"rewritten_query": rewritten, "retry_count": retry_count}


def return_answer(state: RAGState) -> RAGState:
    """Terminal node for a grounded answer."""
    return {"final_answer": state.get("answer", ""), "grounded": True}


def return_idk(state: RAGState) -> RAGState:
    """Terminal node: refuse honestly rather than emit an ungrounded answer."""
    reason = state.get("critique_reason", "insufficient context")
    return {
        "final_answer": (
            "I don't have enough reliable information to answer that confidently. "
            f"(Reason: {reason})"
        ),
        "grounded": False,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_critique(raw: str) -> tuple[str, str]:
    """Parse the critic's JSON, tolerating markdown fences and stray prose."""
    text = raw.strip()
    if text.startswith("```"):
        # Strip ```json ... ``` fences the model sometimes adds.
        text = text.strip("`")
        text = text[text.find("{") : text.rfind("}") + 1]

    try:
        parsed = json.loads(text)
        verdict = str(parsed.get("verdict", "")).lower().strip()
        reason = str(parsed.get("reason", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        return "hallucinated", f"Critic returned unparseable output: {raw[:120]}"

    if verdict not in VALID_VERDICTS:
        return "hallucinated", f"Critic returned unknown verdict '{verdict}'."
    return verdict, reason or "(no reason given)"
