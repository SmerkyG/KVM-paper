"""Run and verify a disjoint expansion of the untruncated LongBench cohort."""

import argparse
import os
from pathlib import Path
import re
import shutil
from statistics import mean
import subprocess
import time

from ._identity import source_identity
from .amd_threshold_fast_submit import freeze
from .amd_threshold_prefill_quality_panel import job_state, write
from .amd_threshold_ruler_panel import digest, paired_interval, read
from .reviewer2_inputs import CAP, token_hash
from .reviewer2_summary_ablation import (
    ARMS, COMPARISONS, LABELS, LONG_BENCH, collect, launch, read_task,
    verify_untruncated, work_and_latency,
)


ADDED = 64
SHARDS = 8
RUNNER = "benchmarks/reviewer2_summary_ablation.py"


def prepare(root, prior):
    old = read(prior / "jobs.json")
    initial = read(prior / "longbench-selection.json")
    extra = read(root / "cohort.json")
    prior_ids = {row["id"] for row in initial["rows"]}
    rows = extra["rows"]
    extra_ids = {row["id"] for row in rows}
    if (len(prior_ids) != 32 or len(rows) != ADDED or len(extra_ids) != ADDED
            or prior_ids & extra_ids or extra["protocol"]["exclude_sha256"] != digest(prior / "longbench-selection.json")
            or extra["protocol"]["eligible"] != verify_untruncated(prior) - len(prior_ids)):
        raise ValueError("The expansion must contain 64 unique prompts outside the original cohort")
    for row in rows:
        if (row["truncated"] or row["original_tokens"] > CAP or len(row["prompt_tokens"]) > CAP
                or token_hash(row["prompt_tokens"]) != row["input_sha256"]):
            raise ValueError(f"Invalid untruncated expansion prompt {row['id']}")

    parent = Path(old["source"])
    baseline = read(parent.parent / "source-manifest.json")["source_files"]
    source = freeze(parent, root)
    manifest_path = root / "source-manifest.json"
    frozen = read(manifest_path)
    if frozen["source_files"] != baseline:
        raise ValueError("The parent frozen source changed")
    shutil.copy2(Path(__file__).with_name("reviewer2_summary_ablation.py"), source / RUNNER)
    frozen["source_files"][RUNNER] = digest(source / RUNNER)
    write(manifest_path, frozen)
    identity = source_identity(source)
    for index in range(SHARDS):
        shard = root / "shards" / f"{index:02d}"
        part = rows[index::SHARDS]
        write(shard / "cohort" / f"{LONG_BENCH}.json", {**extra, "count": len(part), "rows": part})
        write(shard / "jobs.json", {"identity": identity, "source": str(source), "jobs": [], "smokes": []})
    selection = [
        {**{key: value for key, value in row.items() if key not in ("answer", "prompt_tokens")},
         "rendered_tokens": len(row["prompt_tokens"])} for row in rows
    ]
    write(root / "longbench-selection.json", {
        "protocol": extra["protocol"], "count": 32 + ADDED, "original_count": 32, "added_count": ADDED,
        "rows": [*initial["rows"], *selection],
    })
    write(root / "expansion.json", {"prior": str(prior), "source": str(source),
                                   "source_identity": identity, "prior_identity": old["identity"],
                                   "changed_source_files": [RUNNER], "cohort_sha256": digest(root / "cohort.json")})


def submit(root):
    status = subprocess.run(["cluster-run", "status"], check=True, capture_output=True, text=True, timeout=15)
    lines = status.stdout.splitlines()
    free = re.search(r"\s+\d+\s+\d+\s+(\d+)\s*$", lines[1])
    if free is None or int(free[1]) < SHARDS or any(line.startswith(("queued ", "submitted ")) for line in lines):
        raise ValueError("Wait for eight free GPUs and a clear submission queue")
    for index in range(SHARDS):
        shard = root / "shards" / f"{index:02d}"
        if not read(shard / "jobs.json")["jobs"]:
            launch(shard, LONG_BENCH, False, f"amd-reviewer2-longbench-add64-{index:02d}")


def combined_results(root, metadata):
    prior = Path(metadata["prior"])
    initial_manifest = read(prior / "jobs.json")
    initial_job, = initial_manifest["jobs"]
    initial = read_task(prior, initial_job, initial_manifest)
    added = {arm: {} for arm in ARMS}
    for shard in sorted((root / "shards").iterdir()):
        manifest = read(shard / "jobs.json")
        job, = manifest["jobs"]
        rows = read_task(shard, job, manifest)
        for arm in ARMS:
            if set(rows[arm]) & (set(initial[arm]) | set(added[arm])):
                raise ValueError("Overlap in LongBench result IDs")
            added[arm].update(rows[arm])
    if any(len(added[arm]) != ADDED or set(added[arm]) != set(added["weighted"]) for arm in ARMS):
        raise ValueError("Incomplete or unpaired LongBench expansion")
    combined = {arm: {**initial[arm], **added[arm]} for arm in ARMS}
    selection = read(root / "longbench-selection.json")["rows"]
    if {r["id"] for r in selection} != set(combined["weighted"]) or any(r["truncated"] for r in selection):
        raise ValueError("Combined results differ from the untruncated selection")
    return initial, added, combined


def summarize(root, metadata, initial, added, combined):
    count = len(combined["weighted"])
    lengths = [r["input_tokens"] for r in combined["weighted"].values()]
    lines = [
        "# LongBench v2 untruncated Qwen3.8 summary ablation", "",
        f"The original 32 and additional {ADDED} examples form a matched {count}-example subset of "
        "the 298 prompts that fit intact under the paper’s 131,072-token cap. "
        f"Rendered inputs range from {min(lengths):,} to {max(lengths):,} tokens. "
        "Sampling used domain and length before decoding. Each shard ran its four arms on the same GPU "
        "with the same answer constraint and BF16 attention settings as the original comparison.", "",
        f"| Arm | Original correct / 32 | Added correct / {ADDED} | Combined correct / {count} | Combined accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    scores = {}
    for arm in ("dense", *ARMS[:3]):
        counts = [int(sum(row["score"] for row in group[arm].values())) for group in (initial, added, combined)]
        scores[arm] = dict(original=counts[0], added=counts[1], combined=counts[2])
        lines.append(f"| {LABELS[arm]} | {counts[0]} | {counts[1]} | {counts[2]} | {100 * counts[2] / count:.2f}% |")
    lines += ["", "| Paired contrast | Combined score difference (pp) | Paired 95% bootstrap interval |", "|---|---:|---:|"]
    comparisons = {}
    for a, b in COMPARISONS:
        changes = [combined[a][i]["score"] - combined[b][i]["score"] for i in combined[a]]
        delta = 100 * mean(changes)
        lo, hi = paired_interval([changes])
        comparisons[f"{a}-{b}"] = dict(delta_pp=delta, ci95_pp=[lo, hi])
        lines.append(f"| {LABELS[a]} minus {LABELS[b]} | {delta:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += [
        "", "Population correction is retained in routing for all three LoD arms. The pure-sparse prefill "
        "reuses the fused routing pass and drops its summary output. Unweighted prefill uses an additional "
        "centroid aggregation pass. LoD exact work is instrumented. Dense exact work is derived from lengths.", "",
        "| Arm | Prefill exact KV / head | Prefill remote exact / head | Decode exact KV / head | Decode remote exact / head "
        "| Median TTFT (s) | Median request (s) | Median decode (ms/token) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    timings = {}
    for arm in ("dense", *ARMS[:3]):
        work = work_and_latency(list(combined[arm].values()))
        timings[arm] = work
        values = [work[key] for key in ("prefill_total", "prefill_remote", "decode_total", "decode_remote",
                                        "ttft_s", "elapsed_s", "decode_ms")]
        lines.append(f"| {LABELS[arm]} | " + " | ".join("—" if value is None else f"{value:.2f}" for value in values) + " |")
    lines += ["", "[All 96 frozen IDs](longbench-selection.json) · [Expansion runs](shards/) · "
              f"[Original 32]({os.path.relpath(Path(metadata['prior']) / 'REPORT.md', root)})", ""]
    write(root / "REPORT.json", dict(stage="complete", count=count, scores=scores,
                                    comparisons=comparisons, work_and_latency=timings))
    (root / "REPORT.md").write_text("\n".join(lines))


def monitor(root, reviewer):
    metadata = read(root / "expansion.json")
    while True:
        finished, examples, failures = 0, 0, []
        for shard in sorted((root / "shards").iterdir()):
            job, = read(shard / "jobs.json")["jobs"]
            arms = [job_state({"id": job["id"], "output": str(shard / "runs" / LONG_BENCH / f"{arm}.json")})
                    for arm in ARMS]
            fields = arms[0][0]
            examples += sum(len(data.get("records", [])) for _, data in arms)
            if fields and fields[7:9] == ["finished", "0"]:
                finished += 1
            elif fields:
                failures.append(job["id"])
        write(root / "status.json", dict(completed_jobs=finished, jobs=SHARDS, examples=examples,
                                        expected_examples=len(ARMS) * ADDED, failures=failures))
        if failures:
            raise RuntimeError(f"Failed LongBench shards: {failures}")
        if finished == SHARDS:
            initial, added, combined = combined_results(root, metadata)
            summarize(root, metadata, initial, added, combined)
            break
        time.sleep(30)
    if reviewer is not None:
        report_path = reviewer / "REPORT.json"
        while not report_path.is_file() or "LongBench v2 untruncated subset" not in read(report_path)["outcomes"]:
            time.sleep(20)
        collect(reviewer, Path(metadata["prior"]), expanded_longbench=(root, combined))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "monitor"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--reviewer", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        if args.prior is None:
            parser.error("prepare requires --prior")
        prepare(root, args.prior.resolve())
    elif args.action == "submit":
        submit(root)
    else:
        monitor(root, args.reviewer.resolve() if args.reviewer else None)


if __name__ == "__main__":
    main()
