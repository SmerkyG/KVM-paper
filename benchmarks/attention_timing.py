"""Estimate attention-core time with a matched dummy-attention control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MATCHED_FIELDS = (
    "checkpoint",
    "dataset",
    "dataset_revision",
    "batch_size",
    "speed_samples",
    "tensor_parallel_size",
    "decode_tokens",
    "seed",
    "scheduler_chunk_tokens",
    "scheduler_budget_tokens",
    "scheduler_cls",
    "speculative_model",
    "num_speculative_tokens",
)


def _runtime_identity(result: dict[str, Any], *, name: str) -> dict[str, Any]:
    identity = result.get("benchmark_identity")
    if not identity or not identity.get("runtime"):
        raise ValueError(
            f"{name} must contain benchmark_identity.runtime; "
            "legacy results cannot be validated for subtraction"
        )
    return identity["runtime"]


def _prompt_hashes(measurement: dict[str, Any]) -> list[str]:
    return [prompt["token_sha256"] for prompt in measurement["prompts"]]


def _validate_pair(
    dummy: dict[str, Any],
    real: dict[str, Any],
    *,
    real_name: str,
) -> None:
    dummy_runtime = _runtime_identity(dummy, name="dummy")
    real_runtime = _runtime_identity(real, name=real_name)
    if dummy_runtime != real_runtime:
        raise ValueError(
            f"dummy and {real_name} runtime identities differ: "
            f"{dummy_runtime!r} != {real_runtime!r}"
        )
    mismatches = {
        field: (dummy.get(field), real.get(field))
        for field in MATCHED_FIELDS
        if dummy.get(field) != real.get(field)
    }
    if mismatches:
        details = ", ".join(
            f"{field}={dummy_value!r}/{real_value!r}"
            for field, (dummy_value, real_value) in mismatches.items()
        )
        raise ValueError(f"dummy and {real_name} configurations differ: {details}")

    dummy_lengths = set(dummy["measurements"])
    real_lengths = set(real["measurements"])
    if dummy_lengths != real_lengths:
        raise ValueError(
            f"dummy and {real_name} context lengths differ: "
            f"{sorted(dummy_lengths)} != {sorted(real_lengths)}"
        )
    for length in sorted(dummy_lengths, key=int):
        dummy_hashes = _prompt_hashes(dummy["measurements"][length])
        real_hashes = _prompt_hashes(real["measurements"][length])
        if dummy_hashes != real_hashes:
            raise ValueError(
                f"dummy and {real_name} prompt token hashes differ at {length}"
            )


def summarize_attention_time(
    dummy: dict[str, Any],
    reals: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Return strict matched real-minus-dummy timing estimates."""
    if not dummy.get("dummy_attention"):
        raise ValueError("dummy result was not recorded with --dummy-attention")
    if dummy.get("measure") != "speed" or dummy.get("mode") != "full":
        raise ValueError("dummy result must be a full-mode speed benchmark")
    if not reals:
        raise ValueError("at least one real result is required")
    _runtime_identity(dummy, name="dummy")

    result: dict[str, Any] = {
        "method": "matched-real-minus-dummy-attention",
        "benchmark_identity": dummy["benchmark_identity"],
        "checkpoint": dummy["checkpoint"],
        "batch_size": dummy["batch_size"],
        "tensor_parallel_size": dummy["tensor_parallel_size"],
        "decode_tokens": dummy["decode_tokens"],
        "measurements": {},
    }
    for real_name, real in reals:
        if real.get("dummy_attention", False):
            raise ValueError(f"{real_name} is another dummy result")
        _validate_pair(dummy, real, real_name=real_name)

    full_runs = [(name, value) for name, value in reals if value["mode"] == "full"]
    if len(full_runs) != 1:
        raise ValueError("exactly one real full-attention result is required")

    for length in sorted(dummy["measurements"], key=int):
        dummy_measurement = dummy["measurements"][length]
        dummy_prefill = float(dummy_measurement["prefill_seconds"])
        dummy_decode = float(dummy_measurement["decode_ms_per_batch_step"])
        rows: list[dict[str, Any]] = []
        for real_name, real in reals:
            measurement = real["measurements"][length]
            real_prefill = float(measurement["prefill_seconds"])
            real_decode = float(measurement["decode_ms_per_batch_step"])
            attention_prefill = real_prefill - dummy_prefill
            attention_decode = real_decode - dummy_decode
            if attention_prefill <= 0.0 or attention_decode <= 0.0:
                raise ValueError(
                    f"{real_name} at {length} is not slower than the dummy control: "
                    f"prefill delta={attention_prefill}, decode delta={attention_decode}"
                )
            rows.append(
                {
                    "name": real_name,
                    "mode": real["mode"],
                    "benchmark_identity": real["benchmark_identity"],
                    "real_prefill_seconds": real_prefill,
                    "real_decode_ms_per_batch_step": real_decode,
                    "attention_prefill_seconds": attention_prefill,
                    "attention_decode_ms_per_batch_step": attention_decode,
                }
            )
        full_row = next(row for row in rows if row["mode"] == "full")
        for row in rows:
            row["attention_prefill_speedup_vs_full"] = (
                full_row["attention_prefill_seconds"]
                / row["attention_prefill_seconds"]
            )
            row["attention_decode_speedup_vs_full"] = (
                full_row["attention_decode_ms_per_batch_step"]
                / row["attention_decode_ms_per_batch_step"]
            )
            row["end_to_end_prefill_speedup_vs_full"] = (
                full_row["real_prefill_seconds"] / row["real_prefill_seconds"]
            )
            row["end_to_end_decode_speedup_vs_full"] = (
                full_row["real_decode_ms_per_batch_step"]
                / row["real_decode_ms_per_batch_step"]
            )
        result["measurements"][length] = {
            "dummy_prefill_seconds": dummy_prefill,
            "dummy_decode_ms_per_batch_step": dummy_decode,
            "runs": rows,
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dummy", type=Path, required=True)
    parser.add_argument(
        "--real",
        type=Path,
        action="append",
        required=True,
        help="matched real benchmark JSON; repeat for each attention mode",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dummy = json.loads(args.dummy.read_text())
    reals = [(path.stem, json.loads(path.read_text())) for path in args.real]
    result = summarize_attention_time(dummy, reals)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(args.output)


if __name__ == "__main__":
    main()
