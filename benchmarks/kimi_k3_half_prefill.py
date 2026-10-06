"""Sequential, matched prefill comparisons using the half-K3 weight daemon."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from benchmarks.kimi_k3_decode_power2 import LOD_ENV

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/kimi-k3-half-model"


def length_suffix(lengths):
    """Keep original 64K artifacts; never overwrite them with a longer sweep."""
    return "" if lengths == [65536] else "-" + "-".join(f"{n // 1024}k" for n in lengths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-id", default="kimi-k3-first48-int4-v1")
    parser.add_argument("--cohorts", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--lengths", type=int, nargs="+", default=[65536])
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=3 << 30,
                        help="matched native-cache reservation per rank in both modes")
    args = parser.parse_args()
    if any(n not in (1, 2, 4, 8) for n in args.cohorts):
        parser.error("half-model comparison supports one, two, four or eight 16K rows")
    if any(n < 16384 or n % 16384 for n in args.lengths):
        parser.error("prefill lengths must be positive multiples of 16K")
    if args.kv_cache_memory_bytes <= 0:
        parser.error("native-cache reservation must be positive")
    suffix = length_suffix(args.lengths)
    ready = RESULTS / "residency-audit.json"
    deadline = time.monotonic() + 3600
    while not ready.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("half-model residency audit not ready")
        time.sleep(5)
    residency = json.loads(ready.read_text())
    if (residency["checkpoint"] != args.checkpoint
            or residency["weight_cache_id"] != args.cache_id
            or residency["trained_prefix_layers"] != 48
            or len(residency["audits"]) != 8):
        raise ValueError("wrong half-model daemon residency audit")
    # The audit artifact is written before the preload client closes. Give
    # its normal engine shutdown a moment; never run two benchmark clients.
    time.sleep(10)
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(("LOD_KIMI_", "LOD_BENCHMARK_")):
            env.pop(name)
    env.update(LOD_ENV)
    env.update(LOD_BENCHMARK_ADMISSION_COHORT="8",
        LOD_BENCHMARK_SYNCHRONIZED_DECODE="1",
        VLLM_ALLOW_INSECURE_SERIALIZATION="1", VLLM_USE_TRITON_AWQ="1",
        TRITON_CACHE_AUTOTUNING="1", HSA_NO_SCRATCH_RECLAIM="1",
        AITER_CONFIG_FMOE=str(ROOT / "results/kimi-k3-full-model-current/kimik3_i4_tuned_fmoe_b2x16k_merged.csv"))
    plans = []
    summary = {"trained_prefix_layers": 48, "batch_size": 8,
        "lengths": args.lengths, "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        "scope": "one-node half-model stage; not complete-model quality or pipeline throughput",
        "plans": plans}
    panel_name = ("prefill-panel.json" if args.cohorts == [1, 2] and not suffix else
                  "prefill-panel-c" + "-".join(map(str, args.cohorts)) + suffix + ".json")
    references = [ROOT / "results/kimi-k3-full-model-current" / name for name in (
        "oct4-full-warm-prefill-b8-32k64k.json",
        "oct4-full-prefill-scale-b8-5g-128k.json",
        "oct4-full-b8-decode-256k512k-four-updates.json",
    )]
    references = [p for p in references if p.exists()]
    for cohort in args.cohorts:
        env["LOD_BENCHMARK_PREFILL_COHORT"] = str(cohort)
        for mode in ("full", "two-tier"):
            output = RESULTS / f"oct5-{mode}-b8-cohort{cohort}-prefill{suffix}.json"
            command = ["bash", str(ROOT / "benchmarks/run_kimi_k3_v10_direct.sh"),
                "-m", "benchmarks.kimi_k3_prefill_sweep", "--checkpoint", args.checkpoint,
                "--mode", mode, "--lengths", *map(str, args.lengths),
                "--batch-size", "8", "--decode-tokens", "1", "--repeats", "1",
                "--tensor-parallel-size", "8", "--decode-context-parallel-size", "8",
                "--gpu-memory-utilization", "0.8", "--kv-cache-memory-bytes", str(args.kv_cache_memory_bytes),
                "--weight-cache-id", args.cache_id,
                "--real-token-cache", str(ROOT / "results/kimi-k3-full-model-current/prolong-kimi-k3-speed-token-cache.pt"),
                "--reference-baselines", *map(str, references),
                "--audit-prefill-batches", "--report-memory", "--output", str(output)]
            plan = dict(mode=mode, cohort=cohort, command=command,
                environment={k:v for k,v in env.items() if k.startswith(("LOD_", "AITER_CONFIG", "HSA_", "TRITON_CACHE_AUTOTUNING", "VLLM_USE_TRITON_AWQ"))},
                output=str(output), status="running")
            plans.append(plan)
            summary_path = RESULTS / panel_name
            if output.exists():
                previous = json.loads(output.read_text())
                if previous.get("measurement_status") == "complete":
                    if (previous["checkpoint"] != args.checkpoint
                            or previous["mode"] != mode
                            or previous["weight_cache_id"] != args.cache_id
                            or previous["kv_cache_memory_bytes"] != args.kv_cache_memory_bytes
                            or previous["scheduler_total_budget"] != 16384 * cohort + 8
                            or set(previous["measurements"]) != set(map(str, args.lengths))):
                        raise ValueError("completed half-model result has different settings")
                    plan.update(status="reused_complete", measurements=previous["measurements"])
                    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
                    print("KIMI_HALF_PANEL_REUSED " + json.dumps(dict(mode=mode, cohort=cohort)), flush=True)
                    continue
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            print("KIMI_HALF_PANEL_BEGIN " + json.dumps(plan), flush=True)
            try:
                subprocess.run(command, env=env, cwd=ROOT, check=True)
            except subprocess.CalledProcessError as error:
                plan.update(status="failed", returncode=error.returncode)
                summary_path.write_text(json.dumps(summary, indent=2) + "\n")
                raise
            data = json.loads(output.read_text())
            if data["measurement_status"] != "complete":
                raise RuntimeError("half-model timing did not pass its audits")
            plan.update(status="complete", measurements=data["measurements"])
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            print("KIMI_HALF_PANEL_POINT " + json.dumps(dict(mode=mode, cohort=cohort,
                prefill_seconds={n:m["prefill_seconds"] for n,m in data["measurements"].items()})), flush=True)


if __name__ == "__main__":
    main()
