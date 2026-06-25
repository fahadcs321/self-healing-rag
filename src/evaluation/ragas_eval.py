"""
ragas_eval.py — Score a set of RAG responses with RAGAS.

Computes the four core RAGAS metrics plus a derived hallucination rate. RAGAS and
its dataset/back-end deps are imported lazily so this module imports cheaply and
the rest of the test suite never pays for them.

A "sample" is a dict with these keys:
    question: str
    answer: str
    contexts: list[str]     # the actual retrieved chunk texts the answer used
    ground_truth: str
"""

from __future__ import annotations

from typing import Any

# A faithfulness score below this counts the sample as a hallucination.
HALLUCINATION_THRESHOLD = 0.5


def _aggregate(scores: Any, metric: str) -> float:
    """Pull a single mean float for ``metric`` out of a RAGAS result object."""
    try:
        df = scores.to_pandas()
    except Exception:  # pragma: no cover - depends on ragas version
        # Older/newer ragas may already behave like a mapping of scalars.
        value = scores[metric]
        return float(value)

    if metric not in df.columns:
        return 0.0
    series = df[metric].dropna()
    return float(series.mean()) if len(series) else 0.0


def _hallucination_rate(scores: Any) -> float:
    """Fraction of samples whose faithfulness falls below the threshold."""
    try:
        df = scores.to_pandas()
    except Exception:  # pragma: no cover
        return 0.0
    if "faithfulness" not in df.columns:
        return 0.0
    series = df["faithfulness"].dropna()
    if not len(series):
        return 0.0
    return float((series < HALLUCINATION_THRESHOLD).mean())


def _install_ragas_compat() -> None:
    """Shim Vertex AI modules ragas imports but langchain v1 removed.

    ragas 0.4.x unconditionally imports langchain_community's Vertex AI classes,
    which the langchain v1 community package no longer ships, so plain
    ``import ragas`` raises ModuleNotFoundError. We never use Vertex AI, so inject
    lightweight placeholders only when the real modules are genuinely absent.
    """
    import sys
    import types

    for modname, attrs in (
        ("langchain_community.chat_models.vertexai", ["ChatVertexAI"]),
        ("langchain_community.llms.vertexai", ["VertexAI"]),
    ):
        if modname in sys.modules:
            continue
        try:
            __import__(modname)
        except Exception:
            module = types.ModuleType(modname)
            for attr in attrs:
                setattr(module, attr, type(attr, (), {}))
            sys.modules[modname] = module


def _ragas_models():
    """Build the (llm, embeddings) RAGAS should use, from the project config.

    RAGAS defaults to OpenAI for its judge LLM and embeddings. We instead point it
    at the same chat model the pipeline uses (e.g. Groq) and a local
    sentence-transformers embedding model, so evaluation needs no OpenAI key.
    """
    _install_ragas_compat()
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from src.config import settings
    from src.graph.nodes import get_llm

    llm = LangchainLLMWrapper(get_llm())

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:  # older stacks expose it via langchain_community
        from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.embedding_model)
    )
    return llm, embeddings


def score_samples(samples: list[dict[str, Any]]) -> dict[str, float]:
    """Run RAGAS over ``samples`` and return aggregate metrics."""
    if not samples:
        raise ValueError("No samples to evaluate.")

    _install_ragas_compat()
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = Dataset.from_dict(
        {
            "question": [s["question"] for s in samples],
            "answer": [s["answer"] for s in samples],
            "contexts": [list(s.get("contexts") or []) for s in samples],
            "ground_truth": [s["ground_truth"] for s in samples],
        }
    )

    # Drive RAGAS with the project's configured LLM and a local embedding model,
    # so evaluation works on any provider (Groq, etc.) and never requires OpenAI.
    llm, embeddings = _ragas_models()
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm,
        embeddings=embeddings,
    )

    return {
        "faithfulness": _aggregate(scores, "faithfulness"),
        "answer_relevancy": _aggregate(scores, "answer_relevancy"),
        "context_recall": _aggregate(scores, "context_recall"),
        "context_precision": _aggregate(scores, "context_precision"),
        "hallucination_rate": _hallucination_rate(scores),
        "num_questions": float(len(samples)),
    }
