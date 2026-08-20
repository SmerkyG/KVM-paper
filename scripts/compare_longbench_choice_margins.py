#!/usr/bin/env python3
"""Compare candidate LongBench choice margins against a cached reference."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
from pathlib import Path

from eval_longbench_choice_margins import (
    CHOICES,
    compare_records,
    correct_margin,
    load_jsonl,
)


def load_runs(paths: list[Path]) -> list[list[dict]]:
    runs = [load_jsonl(path) for path in paths]
    expected = {str(record["_id"]) for record in runs[0]}
    for path, run in zip(paths[1:], runs[1:]):
        observed = {str(record["_id"]) for record in run}
        if observed != expected:
            raise ValueError(f"repeat IDs differ in {path}")
    return runs


def average_runs(runs: list[list[dict]]) -> list[dict]:
    indexed = [
        {str(record["_id"]): record for record in run}
        for run in runs
    ]
    averaged = []
    for record_id, first in indexed[0].items():
        probabilities = {
            choice: sum(run[record_id]["choice_probabilities"][choice] for run in indexed)
            / len(indexed)
            for choice in CHOICES
        }
        logprobs = {
            choice: math.log(max(probabilities[choice], 1e-300))
            for choice in CHOICES
        }
        prediction = max(CHOICES, key=probabilities.__getitem__)
        record = first.copy()
        record.update(
            choice_probabilities=probabilities,
            choice_logprobs=logprobs,
            prediction=prediction,
            correct=prediction == first["answer"],
            correct_margin=correct_margin(logprobs, first["answer"]),
            repeat_count=len(runs),
        )
        averaged.append(record)
    return averaged


def repeat_stability(runs: list[list[dict]]) -> dict | None:
    comparisons = [
        compare_records(left, right) for left, right in combinations(runs, 2)
    ]
    if not comparisons:
        return None
    numeric_keys = (
        "prediction_agreement_rate",
        "mean_js_divergence",
        "max_js_divergence",
        "mean_correct_margin_delta",
        "mean_correct_margin_drop",
        "rms_correct_margin_delta",
    )
    return {
        "pair_count": len(comparisons),
        **{
            key: sum(comparison[key] for comparison in comparisons)
            / len(comparisons)
            for key in numeric_keys
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference_runs = load_runs(args.reference)
    candidate_runs = load_runs(args.candidate)
    summary = {
        "reference_runs": [str(path.resolve()) for path in args.reference],
        "candidate_runs": [str(path.resolve()) for path in args.candidate],
        "mean_distribution_comparison": compare_records(
            average_runs(reference_runs), average_runs(candidate_runs)
        ),
        "reference_repeat_stability": repeat_stability(reference_runs),
        "candidate_repeat_stability": repeat_stability(candidate_runs),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
