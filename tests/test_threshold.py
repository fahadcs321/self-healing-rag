"""Tests for the CI quality gate."""
import json

from src.evaluation.threshold import GATES, check, evaluate_gates

PASSING = {
    "faithfulness": 0.87,
    "answer_relevancy": 0.83,
    "context_recall": 0.76,
    "context_precision": 0.71,
    "hallucination_rate": 0.03,
}


def test_passing_results_pass():
    assert evaluate_gates(PASSING) is True


def test_low_faithfulness_fails():
    bad = {**PASSING, "faithfulness": 0.5}
    assert evaluate_gates(bad) is False


def test_high_hallucination_rate_fails():
    bad = {**PASSING, "hallucination_rate": 0.2}
    assert evaluate_gates(bad) is False


def test_missing_metric_treated_as_zero_and_fails():
    assert evaluate_gates({"faithfulness": 0.9}) is False


def test_check_reads_file(tmp_path):
    p = tmp_path / "results.json"
    p.write_text(json.dumps(PASSING))
    assert check(str(p)) is True


def test_gates_cover_all_scope_metrics():
    assert {"faithfulness", "answer_relevancy", "hallucination_rate"} <= set(GATES)
