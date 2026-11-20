"""Audit the matched Qwen RULER and first-token retrieval experiments."""

import argparse
import gzip
import json
import time
from pathlib import Path
from statistics import mean

from .amd_k2_threshold_ablation import validate_requests, verified_cohort
from .amd_threshold_prefill_quality_panel import job_state, write
from .amd_threshold_ruler_panel import MODELS, TASKS, digest, paired_interval, read

BLOCKS = ("1024", "16384")
MODES = ("dense", *BLOCKS)
BANDS = ("local-both", "local-16k", "remote-both")


def collect_ruler(root, manifest, runs):
    prior = Path(manifest["prior"])
    source = Path(manifest["source"])
    source_manifest = read(root / "source-manifest.json")
    for name, checksum in source_manifest["source_files"].items():
        if digest(source / name) != checksum:
            raise ValueError(f"Changed source file {name}")
    if digest(root / "settings.json") != manifest["settings_sha256"]:
        raise ValueError("Changed RULER settings")
    settings = read(root / "settings.json")
    cohorts, tokens = verified_cohort(prior, model="qwen")
    groups = {}
    for job, data in runs:
        task, block = job["task"], job["variant"]
        if (block, task) in groups:
            raise ValueError(f"Duplicate RULER arm {block}/{task}")
        groups[block, task] = validate_requests(
            data, task, cohorts[task], tokens[task], settings[block],
            manifest["source_sha256"], model="qwen", prefill_chunk=int(block),
        )
    if set(groups) != {(block, task) for block in BLOCKS for task in TASKS}:
        raise ValueError("Incomplete Qwen RULER panel")
    scores = {
        block: {task: 100 * mean(r["score"] for r in groups[block, task].values())
                for task in TASKS} for block in BLOCKS
    }
    changes = [[groups["1024", task][i]["score"] - groups["16384", task][i]["score"]
                for i in sorted(cohorts[task])] for task in TASKS]
    means = {block: mean(values.values()) for block, values in scores.items()}
    low, high = paired_interval(changes)
    delta = means["1024"] - means["16384"]
    lines = [
        "# Qwen3.8 top-8 two-tier RULER at 64K", "",
        "Both arms use the same 32 matched requests per task, source, FP8 checkpoint, BF16 KV, "
        "top-eight prefill and decode, and 256-token prefill lookback. The scheduler chunk, "
        "initial exact block, and prefill catch-up interval move together.", "",
        f"The 1K minus 16K macro-average is {delta:+.2f} points "
        f"(paired 95% bootstrap interval {low:+.2f} to {high:+.2f}).", "",
        "| Task | 16K | 1K | 1K minus 16K (pp) |", "|---|---:|---:|---:|",
    ]
    for task in TASKS:
        a, b = scores["16384"][task], scores["1024"][task]
        lines.append(f"| {task} | {a:.2f}% | {b:.2f}% | {b-a:+.2f} |")
    lines += [f'| **Macro-average** | **{means["16384"]:.2f}%** | **{means["1024"]:.2f}%** | **{delta:+.2f}** |',
              "", "[Commands](jobs.json) · [First-token retrieval](prefill-retrieval/REPORT.md)", ""]
    return lines, {"means": means, "tasks": scores, "delta_points": delta,
                   "paired_bootstrap_95_interval_points": [low, high]}


def collect_retrieval(root, manifest, runs):
    path = root / "cohort.json.gz"
    cohort_hash = digest(path)
    if cohort_hash != manifest["cohort_sha256"]:
        raise ValueError("Changed retrieval cohort")
    with gzip.open(path, "rt") as handle:
        cohort = json.load(handle)
    expected = {(row["example"], row["band"]): row for row in cohort["records"]}
    if len(expected) != len(cohort["records"]):
        raise ValueError("Duplicate prompts")
    groups = {mode: {} for mode in MODES}
    for job, data in runs:
        mode = job["variant"]
        expected_block = 16384 if mode == "dense" else int(mode)
        if (data["checkpoint"] != MODELS["qwen"] or data["cohort_sha256"] != cohort_hash
                or data["benchmark_identity"]["source"]["source_sha256"] != manifest["source_sha256"]
                or data["scheduler_chunk_tokens"] != expected_block
                or data["mode"] != ("full" if mode == "dense" else "two-tier")):
            raise ValueError(f"Mismatched benchmark identity for {job['id']}")
        for row in data["records"]:
            key = (row["example"], row["band"])
            ref = expected[key]
            if (key in groups[mode] or not job["start"] <= key[0] < job["stop"]
                    or any(row[k] != ref[k] for k in ref if k != "prompt_token_ids")):
                raise ValueError(f"Mismatched or duplicate input {mode}/{key}")
            groups[mode][key] = row
    if any(set(group) != set(expected) for group in groups.values()):
        raise ValueError("Incomplete paired retrieval results")
    lines = [
        "# Qwen3.8 retrieval from prefill logits at 64K", "",
        "Dense, 1K top-eight LoD, and 16K top-eight LoD score identical prompts. The query "
        "asks for the codeword matching two identifiers. Each example shares its haystack, "
        "question, options, and randomized answer label across all three placements. The "
        "document table is local under both settings, local only under 16K, or remote under both. "
        "The table ends 512, 8,192, or 32,768 tokens before the last prompt position, respectively. "
        "The label with the highest next-token log probability is the prediction. Only one "
        "new token is requested, so there are no decoder attention steps.", "",
        "| Evidence placement | Arm | Correct / total | Mean gold-vs-best-distractor log-odds |",
        "|---|---|---:|---:|",
    ]
    results = {}
    for band in BANDS:
        keys = sorted(key for key in expected if key[1] == band)
        results[band] = {}
        for mode in MODES:
            records = [groups[mode][key] for key in keys]
            correct = sum(row["correct"] for row in records)
            margin = mean(row["gold_margin"] for row in records)
            results[band][mode] = {"correct": correct, "total": len(keys), "mean_gold_margin": margin}
            lines.append(f"| {band} | {mode} | {correct}/{len(keys)} | {margin:+.3f} |")
    lines += ["", "| Placement | Comparison | Accuracy change (pp) | Paired 95% interval | Mean margin change | Paired 95% margin interval |",
              "|---|---|---:|---:|---:|---:|"]
    comparisons = (("1024", "16384"), ("1024", "dense"), ("16384", "dense"))
    for band in BANDS:
        keys = sorted(key for key in expected if key[1] == band)
        results[band]["comparisons"] = {}
        for a, b in comparisons:
            differences = [int(groups[a][k]["correct"]) - int(groups[b][k]["correct"]) for k in keys]
            margins = [groups[a][k]["gold_margin"] - groups[b][k]["gold_margin"] for k in keys]
            delta = 100 * mean(differences)
            interval = paired_interval([differences]) if any(differences) else None
            interval_text = f"[{interval[0]:+.2f}, {interval[1]:+.2f}]" if interval else "No discordances"
            margin_delta = mean(margins)
            margin_interval = [value / 100 for value in paired_interval([margins])]
            results[band]["comparisons"][f"{a}-{b}"] = {
                "delta_points": delta, "paired_bootstrap_95_interval_points": interval,
                "discordant_examples": sum(x != 0 for x in differences),
                "mean_margin_delta": margin_delta, "paired_bootstrap_95_margin_interval": margin_interval,
            }
            lines.append(f"| {band} | {a} − {b} | {delta:+.2f} | {interval_text} | {margin_delta:+.3f} | "
                         f"[{margin_interval[0]:+.3f}, {margin_interval[1]:+.3f}] |")
    lines += ["", "[Commands](jobs.json) · [Canonical RULER](../REPORT.md)", ""]
    return lines, {"results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("ruler", "retrieval"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    manifest = read(root / "jobs.json")
    expected_count = 26 if args.study == "ruler" else 12
    if len(manifest["jobs"]) != expected_count:
        raise ValueError("Incomplete job manifest")
    while True:
        complete, failures, count = [], [], 0
        for job in manifest["jobs"]:
            fields, data = job_state(job)
            count += len(data.get("records", []))
            if fields and fields[7:9] == ["finished", "0"] and data.get("stage") == "complete":
                complete.append((job, data))
            elif fields:
                failures.append(job["id"])
        status = {"jobs": expected_count, "complete": len(complete), "records": count, "failures": failures}
        write(root / "status.json", status)
        print(status, flush=True)
        if failures:
            raise RuntimeError(f"Failed runs {failures}")
        if len(complete) == expected_count:
            collect = collect_ruler if args.study == "ruler" else collect_retrieval
            lines, result = collect(root, manifest, complete)
            write(root / "REPORT.json", {"stage": "complete", **result})
            (root / "REPORT.md").write_text("\n".join(lines))
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
