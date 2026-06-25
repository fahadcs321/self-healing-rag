"""Tests for the RAGAS aggregation helpers (RAGAS itself is not invoked)."""

import pytest

from src.evaluation import ragas_eval

pd = pytest.importorskip("pandas")


class FakeScores:
    """Mimics a RAGAS result object's ``.to_pandas()`` surface."""

    def __init__(self, frame):
        self._frame = frame

    def to_pandas(self):
        return self._frame


def test_aggregate_takes_mean():
    frame = pd.DataFrame({"faithfulness": [0.8, 1.0, 0.6]})
    assert ragas_eval._aggregate(FakeScores(frame), "faithfulness") == pytest.approx(0.8)


def test_aggregate_missing_metric_is_zero():
    frame = pd.DataFrame({"faithfulness": [0.8]})
    assert ragas_eval._aggregate(FakeScores(frame), "answer_relevancy") == 0.0


def test_hallucination_rate_counts_low_faithfulness():
    frame = pd.DataFrame({"faithfulness": [0.9, 0.2, 0.4, 0.95]})
    # two of four samples are below the 0.5 threshold -> 0.5
    assert ragas_eval._hallucination_rate(FakeScores(frame)) == pytest.approx(0.5)


def test_score_samples_rejects_empty():
    with pytest.raises(ValueError):
        ragas_eval.score_samples([])
