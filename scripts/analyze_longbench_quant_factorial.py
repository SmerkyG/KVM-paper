#!/usr/bin/env python3
"""Aggregate a two-repeat K/V precision factorial for the LBv2 sentinel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_longbench_choice_margins import (
    average_runs,
    load_runs,
    repeat_stability,
)
from eval_longbench_choice_margins import compare_records


PRECISIONS = (0, 8, 4)


def format_name(bits: int) -> str:
    return "BF16" if bits == 0 else f"INT{bits}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, nargs=2, required=True)
    parser.add_argument("--actual-int4", type=Path, nargs=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs: dict[tuple[int, int], list[list[dict]]] = {
        (0, 0): load_runs(list(args.baseline))
    }
    for key_bits in PRECISIONS:
        for value_bits in PRECISIONS:
            if (key_bits, value_bits) == (0, 0):
                continue
            paths = [
                args.directory / f"k{key_bits}_v{value_bits}_a.jsonl",
                args.directory / f"k{key_bits}_v{value_bits}_b.jsonl",
            ]
            if all(path.exists() for path in paths):
                runs[(key_bits, value_bits)] = load_runs(paths)

    averaged = {pair: average_runs(pair_runs) for pair, pair_runs in runs.items()}
    baseline = averaged[(0, 0)]
    formats = {}
    for (key_bits, value_bits), records in averaged.items():
        formats[f"k{key_bits}_v{value_bits}"] = {
            "key_precision": format_name(key_bits),
            "value_precision": format_name(value_bits),
            "vs_bf16": compare_records(baseline, records),
            "repeat_stability": repeat_stability(runs[(key_bits, value_bits)]),
        }

    conditional = {}
    for target_bits in (8, 4):
        for fixed_bits in PRECISIONS:
            key_pair = (target_bits, fixed_bits)
            key_reference = (0, fixed_bits)
            if key_pair in averaged and key_reference in averaged:
                conditional[
                    f"K_BF16_to_INT{target_bits}_at_V_{format_name(fixed_bits)}"
                ] = compare_records(
                    averaged[key_reference], averaged[key_pair]
                )
            value_pair = (fixed_bits, target_bits)
            value_reference = (fixed_bits, 0)
            if value_pair in averaged and value_reference in averaged:
                conditional[
                    f"V_BF16_to_INT{target_bits}_at_K_{format_name(fixed_bits)}"
                ] = compare_records(
                    averaged[value_reference], averaged[value_pair]
                )

    report: dict[str, object] = {
        "formats": formats,
        "conditional_precision_effects": conditional,
    }
    if args.actual_int4 is not None:
        actual_runs = load_runs(list(args.actual_int4))
        actual = average_runs(actual_runs)
        report["actual_packed_int4"] = {
            "vs_bf16": compare_records(baseline, actual),
            "repeat_stability": repeat_stability(actual_runs),
            "vs_simulated_int4": (
                compare_records(averaged[(4, 4)], actual)
                if (4, 4) in averaged
                else None
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
