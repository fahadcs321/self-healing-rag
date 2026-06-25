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

from typing import Any, Dict, List

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


def score_samples(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    """Run RAGAS over ``samples`` and return aggregate metrics."""
    if not samples:
        raise ValueError("No samples to evaluate.")

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

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    return {
        "faithfulness": _aggregate(scores, "faithfulness"),
        "answer_relevancy": _aggregate(scores, "answer_relevancy"),
        "context_recall": _aggregate(scores, "context_recall"),
        "context_precision": _aggregate(scores, "context_precision"),
        "hallucination_rate": _hallucination_rate(scores),
        "num_questions": float(len(samples)),
    }
