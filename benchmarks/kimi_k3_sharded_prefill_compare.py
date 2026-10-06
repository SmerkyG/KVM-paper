"""Sequential, allocation-matched K3 fixture comparison on one idle TP8 node.

Run inside the K3 v10 userspace. This is an attention-only fixture benchmark,
not trained-model quality evidence or an amortized decode benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--kv-cache-memory-bytes", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=("full", "replicated", "distributed", "reconstructed", "shared"),
                        default=("full", "replicated", "distributed", "reconstructed"))
    args = parser.parse_args()
    if min(*args.lengths, args.batch_size, args.kv_cache_memory_bytes) < 1:
        raise ValueError("lengths, batch size and cache reservation must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Do not inherit another experiment's routing/storage/cadence variants.
    common = {k: v for k, v in os.environ.items() if not k.startswith("LOD_KIMI_")}
    common.update({
        "LOD_KIMI_SUBTILE64": "score", "LOD_KIMI_CHUNK_TILE_PACK": "1",
        "LOD_KIMI_TILE_PACK_QUERY_BLOCK": "1024", "LOD_KIMI_SORT_LEAF_ROUTES": "1",
        "LOD_KIMI_LEAF_BLOCK_M": "64", "LOD_KIMI_LEAF_WARPS": "1",
        "LOD_KIMI_PREFILL_RECLAIM_INTERVAL": "0", "LOD_KIMI_REUSE_PREFILL_ALLOCATOR": "1",
        "LOD_KIMI_PREFILL_MIN_FREE_GIB": "4", "LOD_KIMI_TILE_REFINE": "1",
        "LOD_KIMI_DIRECT_LEAF_RESULT": "1", "TRITON_CACHE_AUTOTUNING": "1",
        "LOD_KIMI_COMPACT_SELECTED_PROJECTION": "1",
        "HSA_NO_SCRATCH_RECLAIM": "0",
    })
    # Each process exits completely before the next starts. All variants use
    # identical native-cache reservation, request capacity and scheduler budget.
    results = {}
    for variant in args.variants:
        env = dict(common)
        if variant in ("distributed", "reconstructed"):
            env["LOD_KIMI_DCP_SHARDED_LEAVES"] = "1"
        if variant == "reconstructed":
            env["LOD_KIMI_DCP_PREFILL_WORKSPACE"] = "1"
        if variant == "shared":
            # Each rank builds one eighth of the centroid budget from its
            # owned sequence slice; global top-eight sees their summaries.
            env["LOD_KIMI_DCP_SHARED_PREFILL"] = "1"
        path = args.output_dir / f"{variant}.json"
        command = [sys.executable, "-m", "benchmarks.kimi_k3_prefill_sweep",
                   "--checkpoint", "tests/fixtures/kimi-k3-mla-stack",
                   "--mode", "full" if variant == "full" else "two-tier",
                   "--lengths", *map(str, args.lengths), "--batch-size", str(args.batch_size),
                   "--decode-tokens", "2", "--tensor-parallel-size", "8",
                   "--decode-context-parallel-size", "8", "--kv-cache-memory-bytes",
                   str(args.kv_cache_memory_bytes), "--report-memory", "--output", str(path)]
        print("KIMI_STORAGE_VARIANT " + json.dumps({"variant": variant, "command": command}), flush=True)
        subprocess.run(command, env=env, check=True)
        result = json.loads(path.read_text())
        if (result["measurement_status"] != "complete"
                or result["worker_attention_audit_status"] != "passed"):
            raise RuntimeError(f"{variant} did not complete and pass its loaded-kernel audit")
        results[variant] = result
    comparison = {
        "fixture_only": True, "batch_size": args.batch_size,
        "native_cache_reservation_bytes": args.kv_cache_memory_bytes,
        "peak_definition": "maximum worker peak across warmup and measured pass",
        "decode_note": "Two outputs test handoff only; not amortized decode timing.",
        "measurements": {},
    }
    for length in args.lengths:
        row = {}
        reference = results.get("replicated", {}).get("measurements", {}).get(str(length))
        for variant, result in results.items():
            point = result["measurements"][str(length)]
            memories = point["worker_memory"] + point["warmup_worker_memory"]
            item = {
                "prefill_seconds": point["prefill_seconds"],
                "peak_allocated_gib": max(x["peak_torch_allocated_bytes"] for x in memories) / 2**30,
                "peak_reserved_gib": max(x["peak_torch_reserved_bytes"] for x in memories) / 2**30,
            }
            if reference and variant != "full":
                item["generated_ids_match_replicated"] = (
                    point["generated_token_ids"] == reference["generated_token_ids"])
                item["relative_prefill_time"] = point["prefill_seconds"] / reference["prefill_seconds"]
            row[variant] = item
        comparison["measurements"][str(length)] = row
    destination = args.output_dir / "comparison.json"
    destination.write_text(json.dumps(comparison, indent=2) + "\n")
    print("KIMI_STORAGE_COMPARISON " + json.dumps(comparison), flush=True)


if __name__ == "__main__":
    main()
