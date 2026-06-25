"""
threshold.py — The CI quality gate.

Reads the RAGAS results JSON and exits non-zero if any metric is out of bounds,
which fails the GitHub Actions job and blocks the merge. ``min`` metrics must be
at least their threshold; ``max`` metrics must be at most their threshold.

Usage:
    python -m src.evaluation.threshold --results results/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import sys

# metric -> (direction, threshold). "min" = floor, "max" = ceiling.
GATES: dict[str, tuple[str, float]] = {
    "faithfulness": ("min", 0.80),
    "answer_relevancy": ("min", 0.75),
    "context_recall": ("min", 0.70),
    "context_precision": ("min", 0.65),
    "hallucination_rate": ("max", 0.05),
}


def evaluate_gates(results: dict[str, float]) -> bool:
    """Print a report and return True iff every gate passes."""
    passed = True
    print("\n── CI Quality Gate ───────────────────────────────────")
    for metric, (direction, threshold) in GATES.items():
        value = float(results.get(metric, 0.0))
        ok = value >= threshold if direction == "min" else value <= threshold
        comparator = ">=" if direction == "min" else "<="
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {metric:>20}: {value:.3f}  ({comparator} {threshold:.2f})")
        passed = passed and ok
    print("───────────────────────────────────────────────────────")
    return passed


def check(results_path: str) -> bool:
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)
    return evaluate_gates(results)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check RAGAS quality gates.")
    parser.add_argument("--results", default="results/eval_results.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if check(args.results):
        print("All quality gates passed. Safe to merge.\n")
        sys.exit(0)
    print("Quality gate FAILED. Merge blocked.\n")
    sys.exit(1)
