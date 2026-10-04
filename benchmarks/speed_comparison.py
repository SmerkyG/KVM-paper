"""Validate and summarize matched end-to-end ProLong speed runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .attention_timing import validate_pair


def summarize_speed_comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Return speedups only after enforcing the canonical pairing contract."""

    validate_pair(baseline, candidate, candidate_name="candidate")
    baseline_source = baseline["benchmark_identity"].get("source")
    candidate_source = candidate["benchmark_identity"].get("source")
    if baseline_source != candidate_source:
        raise ValueError(
            "end-to-end speed comparisons require the exact same source "
            f"identity: {baseline_source!r} != {candidate_source!r}"
        )
    rows = {}
    for length in sorted(baseline["measurements"], key=int):
        base = baseline["measurements"][length]
        other = candidate["measurements"][length]
        base_prefill = float(base["prefill_seconds"])
        other_prefill = float(other["prefill_seconds"])
        base_decode = float(base["decode_ms_per_batch_step"])
        other_decode = float(other["decode_ms_per_batch_step"])
        base_prefill_samples = list(map(float, base["prefill_timings_seconds"]))
        other_prefill_samples = list(map(float, other["prefill_timings_seconds"]))
        decode_steps = int(baseline["decode_tokens"]) - 1
        base_decode_samples = [
            1_000.0 * float(value) / decode_steps
            for value in base["decode_timings_seconds"]
        ]
        other_decode_samples = [
            1_000.0 * float(value) / decode_steps
            for value in other["decode_timings_seconds"]
        ]
        rows[length] = {
            "baseline_prefill_seconds": base_prefill,
            "candidate_prefill_seconds": other_prefill,
            "prefill_speedup": base_prefill / other_prefill,
            "prefill_speedup_observed_bounds": [
                min(base_prefill_samples) / max(other_prefill_samples),
                max(base_prefill_samples) / min(other_prefill_samples),
            ],
            "baseline_decode_ms_per_batch_step": base_decode,
            "candidate_decode_ms_per_batch_step": other_decode,
            "decode_speedup": base_decode / other_decode,
            "decode_speedup_observed_bounds": [
                min(base_decode_samples) / max(other_decode_samples),
                max(base_decode_samples) / min(other_decode_samples),
            ],
            "baseline_cohort_unattributed_wall_seconds": base.get(
                "cohort_unattributed_wall_seconds"
            ),
            "candidate_cohort_unattributed_wall_seconds": other.get(
                "cohort_unattributed_wall_seconds"
            ),
            "generated_tokens_identical": (
                base.get("output_token_sha256")
                == other.get("output_token_sha256")
            ),
        }
    return {
        "method": "matched-prolong-end-to-end-v2",
        "baseline": baseline["mode"],
        "candidate": candidate["mode"],
        "source_identity_equal": True,
        "timing_protocol": baseline["timing_protocol"],
        "measurements": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize_speed_comparison(
        json.loads(args.baseline.read_text()),
        json.loads(args.candidate.read_text()),
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(args.output)


if __name__ == "__main__":
    main()
