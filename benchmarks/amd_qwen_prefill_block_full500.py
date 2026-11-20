"""Freeze, submit, and verify the full paired Qwen prefill block sweep."""

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
from statistics import mean
import subprocess
import time

from .amd_k2_threshold_ablation import validate_requests, verified_cohort
from .amd_qwen_prefill_block_results import BLOCKS
from .amd_threshold_prefill_quality_panel import NODES, job_state, write
from .amd_threshold_ruler import score
from .amd_threshold_ruler_panel import TASKS, digest, paired_interval, read


COUNT = 500
EXPECTED_JOBS = len(TASKS)
IDENTITY_FIELDS = ("doc_hash", "prompt_hash", "target_hash", "arguments")
TOKEN_FIELDS = ("input_tokens", "input_token_sha256")


def archived_rows(item):
    checksum = hashlib.sha256()
    rows = {}
    with Path(item["path"]).open("rb") as stream:
        for line in stream:
            checksum.update(line)
            row = json.loads(line)
            doc_id = row["doc_id"]
            if doc_id in rows or abs(score(row, row["filtered_resps"][0]) - row["65536"]) > 1e-12:
                raise ValueError(f"Duplicate or invalid archived score in {item['path']}/{doc_id}")
            rows[doc_id] = {field: row[field] for field in (
                "doc_id", *IDENTITY_FIELDS, "filtered_resps", "65536",
            )}
            rows[doc_id]["doc"] = {"outputs": row["doc"]["outputs"]}
    if checksum.hexdigest() != item["sha256"] or set(rows) != set(range(COUNT)):
        raise ValueError(f"Changed archive or incomplete cohort in {item['path']}")
    return rows


def verify_source(root, manifest):
    if digest(root / "settings.json") != manifest["settings_sha256"]:
        raise ValueError("Changed Qwen settings")
    source_manifest = read(root / "source-manifest.json")
    if source_manifest["source_sha256"] != manifest["source_sha256"]:
        raise ValueError("Unexpected source identity")
    source = Path(manifest["source"])
    for name, checksum in source_manifest["source_files"].items():
        if digest(source / name) != checksum:
            raise ValueError(f"Changed source file {name}")


def prepare(root, pilot):
    root.mkdir(parents=True, exist_ok=False)
    previous = read(pilot / "jobs.json")
    prior = Path(previous["prior"])
    old_cohort, _ = verified_cohort(prior, model="qwen")
    for name in ("settings.json", "source-manifest.json"):
        (root / name).write_bytes((pilot / name).read_bytes())
    manifest = {key: previous[key] for key in ("source", "source_sha256", "settings_sha256")}
    manifest.update(pilot=str(pilot), prior=str(prior), jobs=[], stage="prepared")
    verify_source(root, manifest)
    archive = read(prior / "cohort-manifest.json")["models"]["qwen"]
    plan = []
    for task in TASKS:
        two_tier = archived_rows(archive[task]["source"])
        dense = archived_rows(archive[task]["dense_source"])
        for doc_id, row in two_tier.items():
            reference = dense[doc_id]
            if any(row[field] != reference[field] for field in (*IDENTITY_FIELDS, "doc")):
                raise ValueError(f"Different dense request {task}/{doc_id}")
            row["archived_dense_score"] = reference["65536"]
            if doc_id in old_cohort[task] and row != old_cohort[task][doc_id]:
                raise ValueError(f"Different pilot request {task}/{doc_id}")
        path = root / "cohort" / task / "samples_frozen.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as stream:
            for doc_id in range(COUNT):
                stream.write(json.dumps(two_tier[doc_id], ensure_ascii=False) + "\n")
        plan.append({"task": task, "path": str(path), "sha256": digest(path), **{
            field: archive[task][field] for field in ("source", "dense_source")
        }})
        print(f"Frozen {task} ({COUNT} prompts)", flush=True)
    write(root / "cohort-manifest.json", {"examples_per_task": COUNT, "tasks": plan})
    manifest["cohort_sha256"] = digest(root / "cohort-manifest.json")
    write(root / "jobs.json", manifest)


def launch(root, max_new):
    path = root / "jobs.json"
    manifest = read(path)
    verify_source(root, manifest)
    if digest(root / "cohort-manifest.json") != manifest["cohort_sha256"]:
        raise ValueError("Changed cohort manifest")
    pilot = read(Path(manifest["pilot"]) / "jobs.json")
    templates = {(job["variant"], job["task"]): job for job in pilot["jobs"]}
    durations = dict.fromkeys(TASKS, 0.0)
    for key, job in templates.items():
        fields, data = job_state(job)
        if not fields or fields[7:9] != ["finished", "0"] or data.get("stage") != "complete":
            raise ValueError(f"Incomplete pilot run {job['id']}")
        durations[key[1]] = max(durations[key[1]], mean(row["elapsed_s"] for row in data["records"]))
    plan = sorted(read(root / "cohort-manifest.json")["tasks"], key=lambda item: -durations[item["task"]])
    seen = {j["task"] for j in manifest["jobs"]}
    todo = [info for info in plan if info["task"] not in seen]
    for info in todo[:max_new]:
        task = info["task"]
        frozen = Path(info["path"])
        if digest(frozen) != info["sha256"]:
            raise ValueError(f"Changed frozen request file {frozen}")
        template = templates[BLOCKS[0], task]["command"]
        uv_start = template.index("uv")
        outputs = {block: str(root / "runs" / task / f"{block}.json") for block in BLOCKS}
        arms = {}
        for block, output in outputs.items():
            template = templates[block, task]["command"]
            ar = template[template.index("--") + 1:]
            for key, value in (("--samples-root", str(frozen.parent.parent)),
                               ("--limit", str(COUNT)), ("--output", output)):
                ar[ar.index(key) + 1] = value
            arms[block] = ar
        spec = root / "runs" / task / "commands.json"
        spec.parent.mkdir(parents=True, exist_ok=True)
        write(spec, arms)
        command = [
            "cluster-run", "--detach", "--num-gpus", "2", "--nodes", NODES,
            "--name", f"amd-qwen-prefill-500-{task}", "--",
            *template[uv_start:template.index("python", uv_start) + 1],
            "-m", "benchmarks.amd_qwen_prefill_block_full500", "worker",
            "--output", str(root), "--task", task,
        ]
        submission = subprocess.run(command, text=True, capture_output=True)
        match = re.search(r"^Submitted (\S+)", submission.stdout, flags=re.MULTILINE)
        if submission.returncode or not match:
            raise RuntimeError(f"Cluster submission failed: {submission.stdout} {submission.stderr}")
        job = dict(id=match[1], task=task, outputs=outputs,
                   command=command, worker_commands=arms, worker_spec_sha256=digest(spec))
        manifest["jobs"].append(job)
        write(path, manifest)
        print(job["id"], flush=True)
    manifest["stage"] = "submitted" if len(manifest["jobs"]) == EXPECTED_JOBS else "submitting"
    write(path, manifest)


def worker(root, task):
    import torch

    work_dir = root / "runs" / task
    commands = read(work_dir / "commands.json")
    visible = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids = visible.split(",") if visible else ["0", "1"]
    if set(commands) != set(BLOCKS) or len(set(gpu_ids)) != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError(f"Expected a two-GPU allocation, got visibility {visible!r} and "
                           f"{torch.cuda.device_count()} GPUs")
    with ExitStack() as stack:
        processes = {}
        for block, gpu in zip(BLOCKS, gpu_ids, strict=True):
            env = {**os.environ, "HIP_VISIBLE_DEVICES": gpu}
            env.pop("CUDA_VISIBLE_DEVICES", None)
            log = stack.enter_context((work_dir / f"{block}.log").open("w"))
            print(f"{task} {block}-token blocks on allocation GPU {gpu}", flush=True)
            processes[block] = subprocess.Popen(commands[block], env=env, stdout=log, stderr=subprocess.STDOUT)
        failures = [block for block, process in processes.items() if process.wait()]
        if failures:
            raise RuntimeError(f"Failed block sizes for {task}: {failures}")


def collect(root, manifest):
    verify_source(root, manifest)
    if digest(root / "cohort-manifest.json") != manifest["cohort_sha256"]:
        raise ValueError("Changed cohort manifest")
    settings = read(root / "settings.json")
    _, old_tokens = verified_cohort(Path(manifest["prior"]), model="qwen")
    pilot = read(Path(manifest["pilot"]) / "jobs.json")
    prior_data = {(j["variant"], j["task"]): read(Path(j["output"])) for j in pilot["jobs"]}
    by_task = {j["task"]: j for j in manifest["jobs"]}
    if set(by_task) != set(TASKS) or len(manifest["jobs"]) != EXPECTED_JOBS:
        raise ValueError("Duplicate or missing job")
    means_by_task = {"archived_dense": {}, **{block: {} for block in BLOCKS}}
    deltas, paired, pilot_drifts = [], {}, {}
    for item in read(root / "cohort-manifest.json")["tasks"]:
        task = item["task"]
        path = Path(item["path"])
        if digest(path) != item["sha256"]:
            raise ValueError(f"Changed frozen request file {path}")
        with path.open() as stream:
            cohort = {r["doc_id"]: r for r in map(json.loads, stream)}
        if set(cohort) != set(range(COUNT)):
            raise ValueError(f"Incomplete cohort {task}")
        job = by_task[task]
        verify_commands(root, job)
        groups, verified = {}, None
        for block in reversed(BLOCKS):
            fields, data = job_state({"id": job["id"], "output": job["outputs"][block]})
            previous = prior_data[block, task]
            if (not fields or fields[7:9] != ["finished", "0"]
                    or data["benchmark_identity"]["runtime"] != previous["benchmark_identity"]["runtime"]):
                raise ValueError(f"Incomplete job or changed runtime {job['id']} / {block}")
            if verified is None:
                verified = {r["doc_id"]: {field: r[field] for field in TOKEN_FIELDS}
                            for r in data["records"]}
                for doc_id, row in old_tokens[task].items():
                    if verified[doc_id] != {field: row[field] for field in TOKEN_FIELDS}:
                        raise ValueError(f"Different pilot tokens {task}/{doc_id}")
            groups[block] = validate_requests(
                data, task, cohort, verified, settings[block], manifest["source_sha256"],
                model="qwen", prefill_chunk=int(block),
            )
        means_by_task["archived_dense"][task] = 100 * mean(r["archived_dense_score"] for r in cohort.values())
        for block, rows in groups.items():
            means_by_task[block][task] = 100 * mean(r["score"] for r in rows.values())
            pilot_drifts.setdefault(block, {})[task] = sum(
                r["score"] != rows[r["doc_id"]]["score"] for r in prior_data[block, task]["records"]
            )
        changes = [groups["1024"][i]["score"] - groups["16384"][i]["score"] for i in range(COUNT)]
        deltas.append(changes)
        paired[task] = {
            "delta_points": 100 * mean(changes),
            "paired_bootstrap_95_interval_points": paired_interval([changes]),
            "improved": sum(d > 0 for d in changes),
            "declined": sum(d < 0 for d in changes),
        }
    means = {block: mean(values.values()) for block, values in means_by_task.items()}
    lo, hi = interval = paired_interval(deltas)
    delta = means["1024"] - means["16384"]
    lines = [
        "# Qwen3.8 top-8 two-tier RULER at 64K", "",
        "Both arms use all 500 matched archived prompts on each of the 13 tasks with the frozen "
        "source and serving settings from the first-32 pilot. Both use FP8 weights, BF16 KV, "
        "greedy generation at batch one, native top-eight prefill and decode, and a "
        "256-token prefill lookback. The initial exact block, scheduler chunk, and cache "
        "catch-up interval change together. Dense scores come from the same archived requests "
        "in the original serving runtime.", "",
        f"The 1K minus 16K macro-average is {delta:+.2f} points "
        f"(paired 95% bootstrap interval {lo:+.2f} to {hi:+.2f}).", "",
        "| Task | Archived dense | 16K | 1K | 1K minus 16K (pp) | Paired 95% interval (pp) | Improved / declined |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        dense, long, short = (means_by_task[block][task] for block in ("archived_dense", "16384", "1024"))
        row = paired[task]
        lower, upper = row["paired_bootstrap_95_interval_points"]
        lines.append(f"| {task} | {dense:.2f}% | {long:.2f}% | {short:.2f}% | {row['delta_points']:+.2f} | "
                     f"[{lower:+.2f}, {upper:+.2f}] | {row['improved']} / {row['declined']} |")
    lines += [
        f"| **Macro-average** | **{means['archived_dense']:.2f}%** | **{means['16384']:.2f}%** | "
        f"**{means['1024']:.2f}%** | **{delta:+.2f}** | **[{lo:+.2f}, {hi:+.2f}]** | — |", "",
        f"Scores on the retested first 32 prompts per task differed from the pilot on "
        f"{sum(pilot_drifts['1024'].values())} prompts at 1K and "
        f"{sum(pilot_drifts['16384'].values())} at 16K.", "",
        "[Frozen inputs](cohort-manifest.json) · [Commands](jobs.json) · [Pilot](../qwen-prefill-block-ruler-v1/REPORT.md)", "",
    ]
    write(root / "REPORT.json", {
        "stage": "complete", "examples_per_task": COUNT, "tasks": means_by_task, "means": means,
        "delta_points": delta, "paired_bootstrap_95_interval_points": interval,
        "per_task_pairing": paired, "pilot_score_changes": pilot_drifts,
    })
    (root / "REPORT.md").write_text("\n".join(lines))
    print(root / "REPORT.md", flush=True)


def verify_commands(root, job):
    spec = root / "runs" / job["task"] / "commands.json"
    if digest(spec) != job["worker_spec_sha256"] or read(spec) != job["worker_commands"]:
        raise ValueError(f"Changed worker commands for {job['task']}")


def monitor(root):
    while True:
        manifest = read(root / "jobs.json")
        complete, failures = 0, []
        examples = dict.fromkeys(BLOCKS, 0)
        for job in manifest["jobs"]:
            stages = []
            for block, output in job["outputs"].items():
                fields, data = job_state({"id": job["id"], "output": output})
                stages.append(data.get("stage"))
                examples[block] += len(data.get("records", []))
            if fields and fields[7:9] == ["finished", "0"] and stages == ["complete"] * len(BLOCKS):
                complete += 1
            elif fields:
                failures.append(job["id"])
        status = {"jobs": len(manifest["jobs"]), "complete": complete,
                  "examples": sum(examples.values()), "by_block": examples, "failures": failures}
        write(root / "status.json", status)
        print(status, flush=True)
        if complete == EXPECTED_JOBS:
            collect(root, manifest)
            return
        time.sleep(45)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "worker", "monitor"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, default=Path(
        "results/amd-threshold-ports/qwen-prefill-block-ruler-v1",
    ))
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--max-new", type=int, default=EXPECTED_JOBS)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.action == "prepare":
        prepare(root, args.pilot.resolve())
    elif args.action == "launch":
        launch(root, args.max_new)
    elif args.action == "worker":
        if args.task is None:
            parser.error("worker requires --task")
        worker(root, args.task)
    else:
        monitor(root)


if __name__ == "__main__":
    main()
