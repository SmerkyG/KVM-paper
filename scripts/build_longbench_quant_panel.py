#!/usr/bin/env python3
"""Build a small paired LongBench-v2 quantization regression panel.

The panel is deliberately sensitivity-enriched.  It is meant to answer whether
a new KV format recovers failures seen with an older quantized format; its raw
accuracy is not an unbiased estimate of full-suite LongBench accuracy.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="BF16 LOD JSONL file or directory",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="INT4 LOD JSONL file or directory",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        help="Optional independent repeat used only to report panel stability",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sensitive-count", type=int, default=32)
    parser.add_argument("--control-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def jsonl_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    paths = [
        candidate
        for candidate in sorted(path.glob("*.jsonl"))
        if "warmup" not in candidate.name
    ]
    if not paths:
        raise ValueError(f"no JSONL files found under {path}")
    return paths


def load_records(path: Path) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for source in jsonl_paths(path):
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                grouped[str(record["_id"])].append(record)

    records: dict[str, dict] = {}
    for record_id, variants in grouped.items():
        prediction_counts = Counter(record.get("prediction") for record in variants)
        modal_prediction, modal_count = prediction_counts.most_common(1)[0]
        chosen = next(
            record
            for record in reversed(variants)
            if record.get("prediction") == modal_prediction
        ).copy()
        chosen["repeat_count"] = len(variants)
        chosen["prediction_consistency"] = modal_count / len(variants)
        records[record_id] = chosen
    return records


def stable_random(seed: int, *parts: object) -> random.Random:
    digest = hashlib.sha256(
        "\0".join([str(seed), *(str(part) for part in parts)]).encode()
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "little"))


def diverse_sample(records: Iterable[dict], count: int, seed: int) -> list[dict]:
    records = list(records)
    if count >= len(records):
        return records
    buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        key = (
            str(record.get("length", "")),
            str(record.get("domain", "")),
            str(record.get("difficulty", "")),
        )
        buckets[key].append(record)
    for key, values in buckets.items():
        stable_random(seed, *key).shuffle(values)
    keys = sorted(buckets)
    stable_random(seed, "bucket-order").shuffle(keys)
    selected: list[dict] = []
    while len(selected) < count:
        made_progress = False
        for key in keys:
            if buckets[key]:
                selected.append(buckets[key].pop())
                made_progress = True
                if len(selected) == count:
                    break
        if not made_progress:
            break
    return selected


def paired_record(reference: dict, candidate: dict) -> dict:
    record = reference.copy()
    reference_correct = bool(reference.get("correct"))
    candidate_correct = bool(candidate.get("correct"))
    if reference_correct and not candidate_correct:
        role = "reference_win"
    elif candidate_correct and not reference_correct:
        role = "candidate_win"
    elif reference.get("prediction") != candidate.get("prediction"):
        role = "prediction_disagreement"
    else:
        role = "agreement_control"
    record.update(
        panel_role=role,
        reference_prediction=reference.get("prediction"),
        reference_correct=reference_correct,
        candidate_prediction=candidate.get("prediction"),
        candidate_correct=candidate_correct,
        reference_prediction_consistency=reference.get(
            "prediction_consistency", 1.0
        ),
        candidate_prediction_consistency=candidate.get(
            "prediction_consistency", 1.0
        ),
    )
    return record


def paired_summary(records: Iterable[dict], *, prefix: str = "candidate") -> dict:
    records = list(records)
    reference_correct = sum(bool(record["reference_correct"]) for record in records)
    candidate_correct = sum(bool(record[f"{prefix}_correct"]) for record in records)
    prediction_agreement = sum(
        record["reference_prediction"] == record[f"{prefix}_prediction"]
        for record in records
    )
    return {
        "count": len(records),
        "reference_correct": reference_correct,
        f"{prefix}_correct": candidate_correct,
        f"{prefix}_minus_reference_correct": candidate_correct - reference_correct,
        f"{prefix}_prediction_agreement": prediction_agreement,
        f"{prefix}_prediction_agreement_rate": (
            prediction_agreement / len(records) if records else None
        ),
    }


def main() -> None:
    args = parse_args()
    if args.sensitive_count < 1 or args.control_count < 0:
        raise ValueError("panel counts must be nonnegative and sensitive-count positive")

    reference = load_records(args.reference)
    candidate = load_records(args.candidate)
    common_ids = sorted(reference.keys() & candidate.keys())
    if not common_ids:
        raise ValueError("reference and candidate have no common examples")
    paired = [paired_record(reference[key], candidate[key]) for key in common_ids]

    by_role: dict[str, list[dict]] = defaultdict(list)
    for record in paired:
        by_role[record["panel_role"]].append(record)

    # Half the sensitive panel targets observed regressions, one quarter guards
    # against apparent candidate wins disappearing, and one quarter catches
    # answer changes hidden by binary correctness.
    loss_quota = (args.sensitive_count + 1) // 2
    win_quota = args.sensitive_count // 4
    disagreement_quota = args.sensitive_count - loss_quota - win_quota
    sensitive: list[dict] = []
    for role, quota, seed_offset in (
        ("reference_win", loss_quota, 1),
        ("candidate_win", win_quota, 2),
        ("prediction_disagreement", disagreement_quota, 3),
    ):
        sensitive.extend(
            diverse_sample(by_role[role], quota, args.seed + seed_offset)
        )
    if len(sensitive) < args.sensitive_count:
        selected_ids = {record["_id"] for record in sensitive}
        remaining = [
            record
            for record in paired
            if record["panel_role"] != "agreement_control"
            and record["_id"] not in selected_ids
        ]
        sensitive.extend(
            diverse_sample(
                remaining, args.sensitive_count - len(sensitive), args.seed + 4
            )
        )

    controls = diverse_sample(
        by_role["agreement_control"], args.control_count, args.seed + 5
    )
    panel = sensitive + controls
    panel.sort(key=lambda record: (int(record["sent_input_tokens"]), record["_id"]))

    validation_summary = None
    if args.validation is not None:
        validation = load_records(args.validation)
        for record in panel:
            validation_record = validation.get(record["_id"])
            if validation_record is None:
                raise ValueError(
                    f"validation is missing panel example {record['_id']}"
                )
            record["validation_prediction"] = validation_record.get("prediction")
            record["validation_correct"] = bool(validation_record.get("correct"))
        validation_summary = paired_summary(panel, prefix="validation")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_dir / "selection.jsonl"
    with selection_path.open("w", encoding="utf-8") as handle:
        for record in panel:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    full_paired = paired_summary(paired)
    panel_paired = paired_summary(panel)
    role_counts = Counter(record["panel_role"] for record in panel)
    summary = {
        "purpose": (
            "Sensitivity-enriched regression sentinel; panel accuracy is not an "
            "unbiased estimate of full LongBench-v2 accuracy."
        ),
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "validation": str(args.validation.resolve()) if args.validation else None,
        "seed": args.seed,
        "full_paired": full_paired,
        "panel_paired": panel_paired,
        "validation_panel_paired": validation_summary,
        "panel_role_counts": dict(sorted(role_counts.items())),
        "selection": str(selection_path.resolve()),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
