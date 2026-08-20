#!/usr/bin/env python3
"""Select a tiny, length-stratified sentinel from paired choice margins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_longbench_choice_margins import (
    compare,
    js_divergence,
    load_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sensitive-count", type=int, default=6)
    parser.add_argument("--control-count", type=int, default=2)
    return parser.parse_args()


def length_bin(tokens: int) -> str:
    if tokens < 64 * 1024:
        return "lt64k"
    if tokens < 128 * 1024:
        return "64k_128k"
    if tokens < 192 * 1024:
        return "128k_192k"
    return "ge192k"


def stratified_extreme(
    records: list[dict], count: int, *, largest: bool
) -> list[dict]:
    if count <= 0:
        return []
    buckets: dict[str, list[dict]] = {}
    for record in records:
        buckets.setdefault(record["sentinel_length_bin"], []).append(record)
    for values in buckets.values():
        values.sort(
            key=lambda record: record["calibration_js_divergence"],
            reverse=largest,
        )

    representatives = [values[0] for values in buckets.values() if values]
    representatives.sort(
        key=lambda record: record["calibration_js_divergence"],
        reverse=largest,
    )
    selected = representatives[:count]
    selected_ids = {str(record["_id"]) for record in selected}
    remaining = [
        record for record in records if str(record["_id"]) not in selected_ids
    ]
    remaining.sort(
        key=lambda record: record["calibration_js_divergence"],
        reverse=largest,
    )
    selected.extend(remaining[: count - len(selected)])
    return selected


def main() -> None:
    args = parse_args()
    if args.sensitive_count < 1 or args.control_count < 0:
        raise ValueError("invalid sentinel counts")

    reference = {
        str(record["_id"]): record for record in load_jsonl(args.reference)
    }
    candidate = {
        str(record["_id"]): record for record in load_jsonl(args.candidate)
    }
    common = sorted(reference.keys() & candidate.keys())
    if len(common) < args.sensitive_count + args.control_count:
        raise ValueError("not enough paired records for requested sentinel")

    paired = []
    for record_id in common:
        left = reference[record_id]
        right = candidate[record_id]
        record = right.copy()
        record.update(
            sentinel_length_bin=length_bin(int(right["sent_input_tokens"])),
            calibration_js_divergence=js_divergence(
                left["choice_probabilities"], right["choice_probabilities"]
            ),
            calibration_margin_delta=(
                right["correct_margin"] - left["correct_margin"]
            ),
            calibration_prediction_flip=(
                right["prediction"] != left["prediction"]
            ),
        )
        paired.append(record)

    sensitive = stratified_extreme(
        paired, args.sensitive_count, largest=True
    )
    sensitive_ids = {str(record["_id"]) for record in sensitive}
    controls = stratified_extreme(
        [record for record in paired if str(record["_id"]) not in sensitive_ids],
        args.control_count,
        largest=False,
    )
    for record in sensitive:
        record["sentinel_role"] = "quant_sensitive"
    for record in controls:
        record["sentinel_role"] = "stability_control"
    selection = sorted(
        sensitive + controls,
        key=lambda record: (int(record["sent_input_tokens"]), str(record["_id"])),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_dir / "selection.jsonl"
    with selection_path.open("w", encoding="utf-8") as handle:
        for record in selection:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "purpose": (
            "Sensitivity-enriched BF16-versus-quantized sentinel; use JS and "
            "margin drift, not its raw accuracy as a benchmark estimate."
        ),
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "selection": str(selection_path.resolve()),
        "count": len(selection),
        "input_tokens": sum(int(record["sent_input_tokens"]) for record in selection),
        "length_bins": {
            name: sum(record["sentinel_length_bin"] == name for record in selection)
            for name in ("lt64k", "64k_128k", "128k_192k", "ge192k")
        },
        "full_calibration": compare(args.reference, list(candidate.values())),
        "sentinel_calibration": compare(args.reference, selection),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
