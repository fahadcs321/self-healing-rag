"""
run_eval.py — Run the pipeline over the golden dataset and score it with RAGAS.

For every golden question we run the *full* graph (retrieve → rerank → generate →
critique → self-heal), capture the answer and the **actual retrieved contexts**,
then hand the batch to RAGAS. Capturing real contexts (not just source filenames)
is what makes faithfulness / context-recall meaningful.

Usage:
    python -m src.evaluation.run_eval --output results/eval_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import settings
from src.evaluation.ragas_eval import score_samples


def load_golden_dataset(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Golden dataset at {path} must be a non-empty JSON list.")
    return data


def collect_samples(golden: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run each golden question through the graph and gather RAGAS inputs."""
    from src.graph.graph import answer_query

    samples: list[dict[str, Any]] = []
    total = len(golden)
    for i, item in enumerate(golden, start=1):
        question = item["question"]
        print(f"  [{i}/{total}] {question[:70]}")
        result = answer_query(question)
        samples.append(
            {
                "question": question,
                "answer": result["answer"],
                "contexts": result["contexts"],  # real retrieved chunk texts
                "ground_truth": item["ground_truth"],
            }
        )
    return samples


def run_evaluation(golden_path: str, output_path: str) -> dict[str, float]:
    settings.require("openai_api_key", "cohere_api_key")

    golden = load_golden_dataset(golden_path)
    print(f"Running pipeline over {len(golden)} golden questions...")
    samples = collect_samples(golden)

    print("Scoring with RAGAS...")
    results = score_samples(samples)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n── RAGAS Results ─────────────────────────────")
    for key, value in results.items():
        print(f"  {key:>20}: {value:.3f}")
    print("──────────────────────────────────────────────")
    print(f"Written to {output_path}\n")
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation.")
    parser.add_argument("--golden", default="data/golden_dataset.json")
    parser.add_argument("--output", default="results/eval_results.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(args.golden, args.output)
