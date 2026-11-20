"""Run paired BF16 Qwen summary ablations on the frozen reviewer cohorts."""

import argparse
import json
import os
from pathlib import Path
import re
from statistics import mean, median
import subprocess
import sys
import time

from ._identity import benchmark_identity, source_identity
from ._vllm import close_llm, default_gpu_memory_utilization, llm_kwargs
from .amd_threshold_fast_submit import freeze
from .amd_threshold_prefill_quality_panel import NODES, job_state, write
from .amd_threshold_ruler import score
from .amd_threshold_ruler_panel import PYTHON, TASKS, digest, paired_interval, read
from .longbench_v2 import extract_answer
from .reviewer2_inputs import CAP, CHECKPOINT, COUNT, MAX_MODEL_LEN, token_hash


LONG_BENCH = "longbench-v2"
TASK_NAMES = (*TASKS, LONG_BENCH)
ARMS = ("weighted", "sparse", "unweighted", "dense")
LABELS = dict(weighted="Standard LoD", sparse="Pure sparse", unweighted="Unweighted summaries", dense="Dense")
COMPARISONS = (("weighted", "sparse"), ("weighted", "unweighted"), ("dense", "weighted"))


def measure_score(task, row, response):
    if task == LONG_BENCH:
        return float(extract_answer(response) == row["answer"])
    return score({"doc": {"outputs": row["targets"]}}, response)


def reset_work(model):
    for module in model.modules():
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is not None:
            pool.engine._lod_bench_exact_work.zero_()


def snapshot_work(model):
    layers = []
    for module in model.modules():
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is not None:
            layers.append((pool.query_heads, pool.engine._lod_bench_exact_work.tolist()))
    return layers


def exact_work(layer_counts, length, generated, finish_reason):
    prefill, decode = [], []
    for heads, counts in layer_counts:
        remote, total, queries, d_remote, d_total, d_queries = counts
        prefix = min(length, 16384)
        expected_decode = max(0, generated - 1) * heads
        # A stop can leave one more attention step than the reported token count.
        valid_decode = d_queries == expected_decode or (
            finish_reason == "stop" and d_queries == expected_decode + heads
        )
        if queries != (length - prefix) * heads or not valid_decode:
            raise ValueError(f"Incomplete exact-work coverage for {length}/{generated}/{finish_reason}: {heads}, {counts}")
        prefill.append((remote, total + heads * prefix * (prefix + 1) // 2, heads * length))
        decode.append((d_remote, d_total, d_queries))
    if not prefill:
        raise ValueError("No LoD layers reported their exact work")
    return dict(prefill=list(map(sum, zip(*prefill))), decode=list(map(sum, zip(*decode))))


def run_arm(root, task, arm, *, smoke):
    os.environ["LOD_BENCH_SUMMARY_MODE"] = "weighted" if arm == "dense" else arm
    if arm != "dense":
        os.environ["LOD_BENCH_EXACT_WORK"] = "1"
    else:
        os.environ.pop("LOD_BENCH_EXACT_WORK", None)
    mode = "full" if arm == "dense" else "two-tier"
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    input_path = root / "cohort" / f"{task}.json"
    cohort = read(input_path)
    if (cohort["context_setting"] != CAP or cohort["checkpoint"] != CHECKPOINT
            or not cohort["rows"] or len(cohort["rows"]) != cohort["count"]):
        raise ValueError("Unexpected frozen cohort")
    rows = cohort["rows"]
    path = root / ("smoke" if smoke else "runs") / task / f"{arm}.json"
    result = {
        "task": task, "arm": arm, "mode": mode, "input_sha256": digest(input_path),
        "checkpoint": CHECKPOINT, "benchmark_identity": benchmark_identity(),
        "stage": "running", "records": [],
    }
    output_choices = (
        StructuredOutputsParams(choice=[f"The correct answer is ({answer})" for answer in "ABCD"])
        if task == LONG_BENCH else None
    )
    kwargs = llm_kwargs(
        checkpoint=CHECKPOINT, mode=mode, max_model_len=MAX_MODEL_LEN, batch_size=1,
        tensor_parallel_size=1, gpu_memory_utilization=default_gpu_memory_utilization(CHECKPOINT, mode),
        full_attention_backend="ROCM_AITER_UNIFIED_ATTN",
    )
    if kwargs["long_prefill_token_threshold"] != 16384 or kwargs["kv_cache_dtype"] != "bfloat16":
        raise ValueError("Unexpected release serving settings")
    llm = LLM(**kwargs)

    def generate(row):
        params = SamplingParams(
            max_tokens=row["max_tokens"], temperature=0, seed=1234,
            stop=row.get("stop"), structured_outputs=output_choices,
        )
        return llm.generate({"prompt_token_ids": row["prompt_tokens"]}, params, use_tqdm=False)[0]

    longest = max(rows, key=lambda row: len(row["prompt_tokens"]))
    try:
        if not smoke:
            generate(longest)
        for row in ([longest] if smoke else rows):
            if arm != "dense":
                llm.apply_model(reset_work)
            start = time.perf_counter()
            generated = generate(row)
            elapsed = time.perf_counter() - start
            answer = generated.outputs[0]
            if token_hash(generated.prompt_token_ids) != row["input_sha256"]:
                raise ValueError(f"Prompt tokens changed for {task}/{row['id']}")
            if not answer.token_ids or not generated.finished:
                raise ValueError(f"Incomplete generation for {task}/{row['id']}")
            if task == LONG_BENCH and extract_answer(answer.text) is None:
                raise ValueError(f"Invalid constrained answer for {task}/{row['id']}: {answer.text!r}")
            prompt_len = len(generated.prompt_token_ids)
            metrics = generated.metrics
            if metrics is None or metrics.is_corrupted or metrics.first_token_latency <= 0:
                raise ValueError(f"Unusable request metrics for {task}/{row['id']}")
            output_len = metrics.num_generation_tokens
            if arm == "dense":
                # One copy per query head. Dense averages are invariant to the
                # number of identical head copies.
                steps = output_len - 1
                work = {
                    "prefill": [None, prompt_len * (prompt_len + 1) // 2, prompt_len],
                    "decode": [None, steps * prompt_len + steps * (steps + 1) // 2, steps],
                }
            else:
                layers, = llm.apply_model(snapshot_work)
                work = exact_work(layers, prompt_len, output_len, answer.finish_reason)
            record = {
                "id": row["id"], "input_sha256": row["input_sha256"],
                "input_tokens": prompt_len, "output_tokens": output_len, "prediction": answer.text,
                "returned_tokens": len(answer.token_ids), "finish_reason": answer.finish_reason,
                "score": measure_score(task, row, answer.text), "work": work,
                "elapsed_s": elapsed, "ttft_s": metrics.first_token_latency,
                "decode_ms_per_token": (
                    1000 * (metrics.last_token_ts - metrics.first_token_ts) / (output_len - 1)
                    if output_len > 1 else None
                ),
            }
            result["records"].append(record)
            write(path, result)
            print(json.dumps({key: record[key] for key in (
                "id", "input_tokens", "score", "ttft_s", "elapsed_s",
            )}), flush=True)
        result["stage"] = "complete"
        write(path, result)
    finally:
        close_llm(llm)


def run_task(root, task, smoke):
    # Rotate the order across tasks, with all four arms of a given task on
    # the same GPU. The choice is fixed before the benchmark starts.
    offset = TASK_NAMES.index(task) % len(ARMS)
    arms = ARMS[offset:] + ARMS[:offset]
    for arm in arms:
        dest = root / ("smoke" if smoke else "runs") / task / f"{arm}.log"
        dest.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-P", "-m", "benchmarks.reviewer2_summary_ablation",
                   "arm", "--root", str(root), "--task", task, "--arm", arm]
        if smoke:
            command.append("--smoke")
        with dest.open("w") as log:
            print(f"Starting {task} / {arm}", flush=True)
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def launch(root, task, smoke, name=None):
    manifest_file = root / "jobs.json"
    if manifest_file.exists():
        manifest = read(manifest_file)
        source = Path(manifest["source"])
    else:
        source = freeze(Path(__file__).resolve().parents[1], root)
        manifest = {"source": str(source), "identity": source_identity(source), "jobs": [], "smokes": []}
    cohort = root / "cohort" / f"{task}.json"
    jobs = manifest["smokes" if smoke else "jobs"]
    if not cohort.is_file() or any(job["task"] == task for job in jobs):
        raise ValueError(f"Missing cohort or duplicate submission for {task}")
    if source_identity(source) != manifest["identity"]:
        raise ValueError("Changed frozen source")
    command = [
        "cluster-run", "--detach", "--num-gpus", "1", "--nodes", NODES,
        "--name", name or f"amd-reviewer2-{'smoke-' if smoke else ''}{task}", "--",
        "env", f"PYTHONPATH={source}:{source}/integrations/vllm_lod",
        "HF_HUB_OFFLINE=1", "HF_DATASETS_OFFLINE=1", "LOD_AMD_STATE_PACKED_MAX=1",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "VLLM_ALLOW_INSECURE_SERIALIZATION=1",
        "uv", "run", "--offline", "--no-project", "--python", PYTHON,
        "python", "-P", "-m", "benchmarks.reviewer2_summary_ablation",
        "task", "--root", str(root), "--task", task,
    ]
    if smoke:
        command.append("--smoke")
    run = subprocess.run(command, check=True, text=True, capture_output=True)
    match = re.search(r"^Submitted (\S+)", run.stdout, flags=re.MULTILINE)
    if not match:
        raise ValueError(run.stdout + run.stderr)
    jobs.append({"id": match[1], "task": task, "command": command, "sha256": digest(cohort)})
    write(manifest_file, manifest)
    print(match[1], flush=True)


def read_task(root, job, manifest):
    task = job["task"]
    input_path = root / "cohort" / f"{task}.json"
    if digest(input_path) != job["sha256"]:
        raise ValueError(f"Changed input for {task}")
    cohort = read(input_path)
    refs = {row["id"]: row for row in cohort["rows"]}
    if len(refs) != len(cohort["rows"]) or len(refs) != cohort["count"] or not refs:
        raise ValueError(f"Unexpected cohort size for {task}")
    outputs = {}
    for arm in ARMS:
        path = root / "runs" / task / f"{arm}.json"
        fields, data = job_state({"id": job["id"], "output": str(path)})
        if (not fields or fields[7:9] != ["finished", "0"] or data["stage"] != "complete"
                or data["task"] != task or data["arm"] != arm or data["checkpoint"] != CHECKPOINT
                or data["input_sha256"] != job["sha256"]
                or data["benchmark_identity"]["source"] != manifest["identity"]):
            raise ValueError(f"Invalid run for {task}/{arm}")
        rows = {row["id"]: row for row in data["records"]}
        if len(data["records"]) != len(refs) or set(rows) != set(refs):
            raise ValueError(f"Incomplete output for {task}/{arm}")
        for i, row in rows.items():
            if row["input_sha256"] != refs[i]["input_sha256"] or row["score"] != measure_score(
                task, refs[i], row["prediction"],
            ):
                raise ValueError(f"Bad pairing for {task}/{arm}/{i}")
        outputs[arm] = rows
    return outputs


def work_and_latency(rows):
    work = {}
    for phase in ("prefill", "decode"):
        query_heads = sum(r["work"][phase][2] for r in rows)
        for i, name in enumerate(("remote", "total")):
            values = [r["work"][phase][i] for r in rows]
            work[f"{phase}_{name}"] = sum(values) / query_heads if None not in values and query_heads else None
    decode_times = [r["decode_ms_per_token"] for r in rows if r["decode_ms_per_token"] is not None]
    return {**work, "ttft_s": median(r["ttft_s"] for r in rows),
            "elapsed_s": median(r["elapsed_s"] for r in rows),
            "decode_ms": median(decode_times) if decode_times else None}


def verify_untruncated(root):
    cohort = read(root / "cohort" / f"{LONG_BENCH}.json")
    if (not cohort["protocol"]["untruncated_only"] or any(
        r["truncated"] or r["original_tokens"] > CAP for r in cohort["rows"]
    )):
        raise ValueError("Expected a LongBench subset requiring no prompt truncation")
    return cohort["protocol"]["eligible"]


def collect(root, longbench_root=None, *, expanded_longbench=None):
    manifest = read(root / "jobs.json")
    jobs = {job["task"]: job for job in manifest["jobs"]}
    if len(manifest["jobs"]) != len(TASK_NAMES) or set(jobs) != set(TASK_NAMES):
        raise ValueError("Incomplete paired panel")
    by_task = {task: read_task(root, jobs[task], manifest) for task in TASK_NAMES}
    longbench_root = root if longbench_root is None else longbench_root
    if longbench_root != root:
        eligible = verify_untruncated(longbench_root)
        replacement = read(longbench_root / "jobs.json")
        if replacement["identity"] != manifest["identity"]:
            raise ValueError("LongBench rerun and RULER source identities differ")
        job, = replacement["jobs"]
        by_task[LONG_BENCH] = read_task(longbench_root, job, replacement)
    if expanded_longbench is not None:
        expansion_root, expanded_rows = expanded_longbench
        by_task[LONG_BENCH] = expanded_rows
    longbench_count = len(by_task[LONG_BENCH]["weighted"])
    per_task = {
        task: {arm: 100 * mean(row["score"] for row in records.values()) for arm, records in runs.items()}
        for task, runs in by_task.items()
    }
    longbench_label = "LongBench v2 untruncated subset" if longbench_root != root else "LongBench v2 subset"
    groups = {"128K RULER": TASKS, longbench_label: (LONG_BENCH,)}
    selection_path = (expansion_root if expanded_longbench is not None else longbench_root) / "longbench-selection.json"
    selection = read(selection_path)["rows"]
    truncated = sum(row["truncated"] for row in selection)
    cap_note = (f"All {longbench_count} LongBench prompts fit intact under the cap. The sample was drawn from {eligible} eligible "
                "examples." if longbench_root != root else
                f"The subset contains {truncated} truncated prompts. The rendered input includes chat-template "
                "tokens above the content cap where applicable.")
    lines = [
        "# Qwen3.8 BF16 summary ablation", "",
        "The 128K RULER setting uses 32 matched prompts per task from lm-eval 0.4.13. "
        f"LongBench v2 uses {longbench_count} frozen examples sampled across domain and length from the 503-example split. "
        "Prompts over 131,072 input tokens retain the first and last halves before chat templating. "
        "LongBench retains original shorter prompts, constrains the answer to A/B/C/D and permits 32 output tokens.", "",
        cap_note, "",
        "All LoD arms keep the BF16 two-tier cache, 16K exact first prefill block, local fields, and "
        "top-eight population-corrected region router. The sparse arm omits closed summaries. "
        "The unweighted arm retains each closed summary with zero population bias in aggregation. "
        "Dense is a fresh BF16 control. All runs use batch one and a shared prompt and output policy.", "",
        "| Task | Dense | Standard LoD | Pure sparse | Unweighted summaries |", "|---|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        lines.append(f"| {task} | " + " | ".join(f"{per_task[task][arm]:.2f}%" for arm in ("dense", *ARMS[:3])) + " |")
    outcomes = {}
    for label, tasks in groups.items():
        scores = {arm: mean(per_task[task][arm] for task in tasks) for arm in ARMS}
        table_label = f"{label} macro-average" if label == "128K RULER" else label
        lines.append(f"| **{table_label}** | " + " | ".join(f"**{scores[arm]:.2f}%**" for arm in ("dense", *ARMS[:3])) + " |")
        outcomes[label] = {"scores": scores, "comparisons": {}, "work_and_latency": {}}
    lines += ["", "| Benchmark | Paired score contrast | Difference (pp) | Paired 95% bootstrap interval |",
              "|---|---|---:|---:|"]
    for label, tasks in groups.items():
        for a, b in COMPARISONS:
            changes = [[by_task[t][a][i]["score"] - by_task[t][b][i]["score"] for i in by_task[t][a]] for t in tasks]
            delta = 100 * mean(mean(values) for values in changes)
            lo, hi = paired_interval(changes)
            outcomes[label]["comparisons"][f"{a}-{b}"] = {"delta_pp": delta, "ci95_pp": [lo, hi]}
            lines.append(f"| {label} | {LABELS[a]} minus {LABELS[b]} | {delta:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += [
        "", "Exact work averages individual exact KV entries per query head, including the causal first "
        "prefill block. LoD decode counters record the actual executed attention queries, including "
        "any additional terminal step. Dense exact work is calculated from sequence lengths. "
        "The median request "
        "latencies include LoD cache construction, exclude model load and warmup, and use the same GPU "
        "within each task. LoD work counters run during generation. Unweighted prefill currently uses "
        "a second centroid aggregation pass.", "",
        "| Benchmark | Arm | Prefill exact KV / head | Prefill remote exact / head | "
        "Decode exact KV / head | Decode remote exact / head | Median TTFT (s) | Median request (s) | Median decode (ms/token) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, tasks in groups.items():
        for arm in ARMS:
            rows = [row for task in tasks for row in by_task[task][arm].values()]
            metrics = work_and_latency(rows)
            outcomes[label]["work_and_latency"][arm] = metrics
            values = [metrics[k] for k in (
                "prefill_total", "prefill_remote", "decode_total", "decode_remote", "ttft_s", "elapsed_s", "decode_ms",
            )]
            lines.append(f"| {label} | {LABELS[arm]} | " + " | ".join("—" if x is None else f"{x:.2f}" for x in values) + " |")
    lines += ["", "| Benchmark | Paired timing contrast | Mean TTFT difference (s) | Mean request difference (s) |",
              "|---|---|---:|---:|"]
    for label, tasks in groups.items():
        outcomes[label]["paired_latency"] = {}
        for a, b in COMPARISONS:
            stats, cells = {}, []
            for metric in ("ttft_s", "elapsed_s"):
                changes = [
                    [by_task[t][a][i][metric] - by_task[t][b][i][metric] for i in by_task[t][a]]
                    for t in tasks
                ]
                delta = mean(mean(values) for values in changes)
                lo, hi = (v / 100 for v in paired_interval(changes))
                stats[metric] = {"delta_s": delta, "ci95_s": [lo, hi]}
                cells.append(f"{delta:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            outcomes[label]["paired_latency"][f"{a}-{b}"] = stats
            lines.append(f"| {label} | {LABELS[a]} minus {LABELS[b]} | " + " | ".join(cells) + " |")
    lines += ["", "[RULER commands and source](jobs.json) · "
              f"[Selected LongBench IDs and truncation metadata]({os.path.relpath(selection_path, root)})"]
    if longbench_root != root:
        lines += [f"[LongBench rerun commands]({os.path.relpath(longbench_root / 'jobs.json', root)})"]
    if expanded_longbench is not None:
        lines += [f"[Additional 64 LongBench examples]({os.path.relpath(expansion_root / 'REPORT.md', root)})"]
    lines.append("")
    write(root / "REPORT.json", {"stage": "complete", "tasks": per_task, "outcomes": outcomes})
    (root / "REPORT.md").write_text("\n".join(lines))


def collect_untruncated(root, job, manifest):
    eligible = verify_untruncated(root)
    rows = read_task(root, job, manifest)
    lengths = [r["input_tokens"] for r in rows["weighted"].values()]
    lines = [
        "# LongBench v2 untruncated summary ablation on Qwen3.8", "",
        f"The 32 examples were selected across domain and length from the {eligible} of 503 prompts "
        "that fit intact under the paper’s 131,072-token input cap. "
        f"After chat templating, inputs ranged from {min(lengths):,} to {max(lengths):,} tokens. "
        "The four arms used the same tokens, two-tier BF16 for LoD, and an A/B/C/D constrained answer "
        "with an output limit of 32 tokens.", "",
        "| Arm | Correct / 32 | Accuracy | Prefill exact KV / head | Prefill remote exact / head | "
        "Decode exact KV / head | Decode remote exact / head | Median TTFT (s) | Median request (s) | Median decode (ms/token) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    results = {}
    for arm in ("dense", *ARMS[:3]):
        arm_rows = list(rows[arm].values())
        hits = int(sum(r["score"] for r in arm_rows))
        work = work_and_latency(arm_rows)
        results[arm] = {"correct": hits, "accuracy": hits / COUNT, **work}
        metrics = [work[k] for k in ("prefill_total", "prefill_remote", "decode_total",
                                     "decode_remote", "ttft_s", "elapsed_s", "decode_ms")]
        lines.append(f"| {LABELS[arm]} | {hits} | {100 * hits / COUNT:.2f}% | " +
                     " | ".join("—" if x is None else f"{x:.2f}" for x in metrics) + " |")
    lines += ["", "| Paired contrast | Score difference (pp) | Paired 95% bootstrap interval |",
              "|---|---:|---:|"]
    comparisons = {}
    for a, b in COMPARISONS:
        changes = [rows[a][i]["score"] - rows[b][i]["score"] for i in rows[a]]
        delta = 100 * mean(changes)
        lo, hi = paired_interval([changes])
        comparisons[f"{a}-{b}"] = {"delta_pp": delta, "ci95_pp": [lo, hi]}
        lines.append(f"| {LABELS[a]} minus {LABELS[b]} | {delta:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ["", "The three LoD arms retain population correction for routing. "
              "Unweighted prefill uses a second centroid aggregation pass. LoD work counters run "
              "during generation. The four arms ran on the same GPU after warmup.", "",
              "[Frozen IDs and lengths](longbench-selection.json) · [Commands](jobs.json) · [Raw outputs](runs/longbench-v2/)", ""]
    write(root / "REPORT.json", {"stage": "complete", "eligible": eligible,
                                "results": results, "comparisons": comparisons})
    (root / "REPORT.md").write_text("\n".join(lines))


def monitor_untruncated(root, reviewer_root):
    manifest = read(root / "jobs.json")
    job, = manifest["jobs"]
    while True:
        arms = [job_state({"id": job["id"], "output": str(root / "runs" / LONG_BENCH / f"{arm}.json")})
                for arm in ARMS]
        fields = arms[0][0]
        write(root / "status.json", {"job": job["id"], "expected_examples": len(ARMS) * COUNT,
                                    "scored_examples": sum(len(data.get("records", [])) for _, data in arms),
                                    "exit": fields[7:9] if fields else None})
        if fields:
            if fields[7:9] != ["finished", "0"]:
                raise RuntimeError(f"LongBench exited unsuccessfully: {job['id']}")
            collect_untruncated(root, job, manifest)
            break
        time.sleep(30)
    if reviewer_root is not None:
        while not (reviewer_root / "REPORT.md").is_file():
            time.sleep(30)
        collect(reviewer_root, root)


def monitor(root):
    previous = None
    while True:
        manifest = read(root / "jobs.json")
        smoke_jobs = manifest["smokes"]
        if {job["task"] for job in smoke_jobs} != {LONG_BENCH, "niah_single_1"}:
            raise ValueError("Expected the LongBench and 128K RULER smoke runs")
        fixture = read(root / "check-job.json")
        gated = [
            (job["id"], [root / "smoke" / job["task"] / f"{arm}.json" for arm in ARMS], 1)
            for job in smoke_jobs
        ] + [(fixture["id"], [root / "check.json"], 0)]
        scored = [
            (job["id"], [root / "runs" / job["task"] / f"{arm}.json" for arm in ARMS], COUNT)
            for job in manifest["jobs"]
        ]
        completed, failures, examples = set(), [], 0
        for run_id, paths, expected in (*gated, *scored):
            rows = [job_state({"id": run_id, "output": str(path)}) for path in paths]
            if expected == COUNT:
                examples += sum(len(data.get("records", [])) for _, data in rows)
            fields = rows[0][0]
            if not fields:
                continue
            if fields[7:9] == ["finished", "0"] and all(
                data.get("stage") == "complete" and len(data.get("records", [])) == expected
                for _, data in rows
            ):
                completed.add(run_id)
            else:
                failures.append(run_id)
        gate_pass = all(run_id in completed for run_id, _, _ in gated)
        complete_tasks = sum(run_id in completed for run_id, _, _ in scored)
        status = {
            "gate_pass": gate_pass, "submitted_tasks": len(scored), "complete_tasks": complete_tasks,
            "scored_examples": examples, "expected_examples": len(ARMS) * COUNT * len(TASK_NAMES),
            "failures": failures,
        }
        if status != previous:
            write(root / "status.json", status)
            print(status, flush=True)
            previous = status
        if failures:
            raise RuntimeError(f"Review failed runs before continuing: {failures}")
        if complete_tasks == len(TASK_NAMES):
            collect(root)
            return
        if gate_pass and len(scored) < len(TASK_NAMES):
            placement = subprocess.run(["cluster-run", "status"], capture_output=True, text=True, timeout=15)
            lines = placement.stdout.splitlines()
            free = re.search(r"\s+\d+\s+\d+\s+(\d+)\s*$", lines[1]) if len(lines) > 1 else None
            waiting = any(line.startswith(("submitted ", "queued ")) for line in lines)
            if placement.returncode == 0 and free and int(free[1]) > 0 and not waiting:
                seen = {job["task"] for job in manifest["jobs"]}
                order = (LONG_BENCH, "niah_multiquery", "ruler_cwe", *TASK_NAMES)
                launch(root, next(task for task in order if task not in seen), False)
                time.sleep(6)
                continue
        time.sleep(35)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "task", "arm", "collect", "monitor", "monitor-untruncated"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=TASK_NAMES)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--longbench-root", type=Path)
    parser.add_argument("--reviewer-root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "collect":
        collect(root, args.longbench_root.resolve() if args.longbench_root else None)
    elif args.action == "monitor":
        monitor(root)
    elif args.action == "monitor-untruncated":
        monitor_untruncated(root, args.reviewer_root.resolve() if args.reviewer_root else None)
    else:
        if args.task is None:
            parser.error("The selected action requires --task")
        if args.action == "launch":
            launch(root, args.task, args.smoke, args.name)
        elif args.action == "task":
            run_task(root, args.task, args.smoke)
        else:
            if args.arm is None:
                parser.error("arm requires --arm")
            run_arm(root, args.task, args.arm, smoke=args.smoke)


if __name__ == "__main__":
    main()
