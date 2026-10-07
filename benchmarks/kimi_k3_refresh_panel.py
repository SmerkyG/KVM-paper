"""Sequential current K3 sweeps on a resident-weight node; no cluster API.

Run dense and LoD on different nodes, never multiple engines on one node.
Each child saves completed points atomically before attempting a longer one.
The full 1020K cohort ledger contains only frozen-input identities, not times.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmarks.kimi_k3_current_timings import (
    CURRENT_PREFIX, EXCLUDED_POINTS, LENGTHS, RESULTS, ROOT,
    current_sources, render, validate_current_point,
)


def completed_lengths(mode, batch, directory=RESULTS):
    """Resume by audited points, not the final status of a whole long block."""
    completed = set()
    for path in current_sources(directory):
        data = json.loads(path.read_text())
        if data.get("mode") != mode or data.get("batch_size") != batch:
            continue
        completed.update(int(length) for length, point in data.get("measurements", {}).items()
                         if (path.name, int(length)) not in EXCLUDED_POINTS
                         and validate_current_point(data, point))
    return completed


def remaining_output(output, lengths):
    """Never overwrite a previously saved point on a second resumption."""
    if not output.exists():
        return output
    remaining = output.with_name(output.stem + f"-remaining-{max(lengths)//1024}k.json")
    attempt = 2
    candidate = remaining
    while candidate.exists():
        candidate = remaining.with_name(remaining.stem + f"-retry{attempt}.json")
        attempt += 1
    return candidate


def plans(mode, batch, upper):
    lengths = [n for n in LENGTHS if n <= upper]
    boundary = 262144 if batch == 1 or mode == "two-tier" else 131072
    short = [n for n in lengths if n <= boundary]
    long = [n for n in lengths if n > boundary]
    output = []
    if short:
        output.append(("short", short, 1 if mode == "two-tier" or batch == 1 else 5, False))
    if long:
        if mode == "two-tier" and batch == 8:
            # A failed million-token reservation must not prevent the smaller
            # 512K engine from producing a real measured result first.
            output.extend((f"long-{length // 1024}k", [length], 1, False) for length in long)
        else:
            output.append(("long", long, 1 if mode == "two-tier" else 5 if batch == 1 else
                           31 if max(long) > 524288 else 17, mode == "two-tier" and batch == 1))
    return output


def lod_memory_environment(batch, block, sharded):
    """Exact-storage capacity choices; no routing or cadence overrides."""
    env = {"HSA_NO_SCRATCH_RECLAIM": "0"}
    if sharded:
        env["LOD_KIMI_DCP_SHARDED_LEAVES"] = "1"
    if batch == 8 and block in ("long-512k", "long-1020k"):
        # The 512K path completed both trained generations. The 1020K path
        # has passed a short live-prefix preflight with full reservation;
        # require the actual full-length run before claiming support.
        env.update(LOD_KIMI_COMPACT_PAGE_DIRECTORY="1", LOD_KIMI_OWNER_SHARD_RESIDUAL="1",
                   LOD_KIMI_OWNER_PREFILL_HEAD_GROUP="4", LOD_KIMI_OWNER_MOE_CHUNK="4096")
        if block == "long-1020k":
            env.update(LOD_KIMI_OWNER_PREFILL_HEAD_GROUP="2", LOD_KIMI_OWNER_MOE_CHUNK="1024",
                       LOD_KIMI_LOCAL_PREFILL_HEAD_GROUP="8", LOD_KIMI_COARSE_PREFILL_HEAD_GROUP="16",
                       HSA_SCRATCH_SINGLE_LIMIT_ASYNC="268435456")
    return env


def wait_for_preceding(path):
    """Wait without allocating a model; fail promptly on a failed dependency."""
    while not path.exists() or "==> cluster-run completed:" not in path.read_text()[-10000:]:
        time.sleep(5)
    if "==> cluster-run completed: status=finished exit_code=0" not in path.read_text()[-10000:]:
        raise RuntimeError("preceding engine did not complete successfully; inspect it before starting another")


def million_token_cohort(checkpoint, token_cache):
    """Make one small, reusable identity ledger for the previously missing B8 cohort."""
    import torch
    from transformers import AutoTokenizer
    from benchmarks.prolong import DATASET, DATASET_REVISION, SPEED_SHUFFLE_SEED, SEPARATOR, token_digest

    path = RESULTS / "oct7-frozen-b8-1020k-cohort.json"
    if path.exists():
        return path
    cached = torch.load(token_cache, map_location="cpu", weights_only=False)
    if (cached["dataset"], cached["revision"], cached["seed"]) != (
            DATASET, DATASET_REVISION, SPEED_SHUFFLE_SEED):
        raise ValueError("not the frozen ProLong corpus")
    docs = cached["documents"]
    separator = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)(
        SEPARATOR, add_special_tokens=False)["input_ids"]
    records = []
    length = 1044480
    for row in range(8):
        stream, indices, cursor = [], [], row
        while len(stream) < length + 1026:
            index = cursor % len(docs)
            if stream:
                stream.extend(separator)
            stream.extend(docs[index])
            indices.append(index)
            cursor += 8
        records.append(dict(panel_source_stream_indices=indices,
            token_sha256=token_digest(stream[:length]), trace_tokens=1026,
            trace_token_sha256=token_digest(stream[length:length + 1026])))
    value = dict(scope="frozen input identities only; not a measured dense baseline",
                 dataset=DATASET, revision=DATASET_REVISION, seed=SPEED_SHUFFLE_SEED,
                 measurements={str(length): dict(prompts=records)})
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "two-tier"), required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 8), required=True)
    parser.add_argument("--upper-context", type=int, choices=LENGTHS, default=1044480)
    parser.add_argument("--block", choices=("short", "long", "all"), default="all")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weight-cache-id", required=True)
    parser.add_argument("--token-cache", type=Path,
                        default=RESULTS / "prolong-kimi-k3-speed-token-cache.pt")
    parser.add_argument("--after-log", type=Path,
                        help="wait for this specific preceding job to exit; never overlap engines")
    args = parser.parse_args()
    if args.after_log:
        wait_for_preceding(args.after_log)
    references = ([RESULTS / "oct4-full-b1-decode-power2-four-updates.json",
                   RESULTS / "oct4-full-b1-512k1020k-decode1025.json"] if args.batch_size == 1 else
                  [RESULTS / f"oct4-full-b8-decode-{label}-four-updates.json"
                   for label in ("16k64k", "128k128k", "256k512k")])
    if args.batch_size == 8 and args.upper_context == 1044480:
        references.append(million_token_cohort(args.checkpoint, args.token_cache))
    for block, lengths, cache_gib, sharded in plans(args.mode, args.batch_size, args.upper_context):
        if args.block not in (block.split("-", 1)[0], "all"):
            continue
        label = "full" if args.mode == "full" else "lod"
        output = RESULTS / f"{CURRENT_PREFIX}-{label}-b{args.batch_size}-{block}.json"
        completed = completed_lengths(args.mode, args.batch_size)
        lengths = [length for length in lengths if length not in completed]
        if not lengths:
            continue
        output = remaining_output(output, lengths)
        env = dict(os.environ, PYTORCH_ALLOC_CONF="expandable_segments:True",
                   TRITON_CACHE_AUTOTUNING="1", VLLM_USE_TRITON_AWQ="1",
                   AITER_CONFIG_FMOE=str(RESULTS / "kimik3_i4_tuned_fmoe_b2x16k_merged.csv"))
        if args.mode == "full":
            env["HSA_NO_SCRATCH_RECLAIM"] = "1"
        if args.mode == "two-tier":
            from benchmarks.kimi_k3_decode_power2 import LOD_ENV
            env.update(LOD_ENV)
            env.update(lod_memory_environment(args.batch_size, block, sharded))
        command = [sys.executable, "-m", "benchmarks.kimi_k3_prefill_sweep",
            "--checkpoint", args.checkpoint, "--weight-cache-id", args.weight_cache_id,
            "--mode", args.mode, "--batch-size", str(args.batch_size),
            "--lengths", *map(str, lengths), "--decode-tokens", "1026",
            "--reference-decode-trace", "--repeats", "1", "--report-memory",
            "--kv-cache-memory-bytes", str(cache_gib << 30),
            "--real-token-cache", str(args.token_cache),
            "--reference-baselines", *map(str, references), "--output", str(output)]
        print("K3_CURRENT_COMMAND " + json.dumps(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        print(render(), flush=True)


if __name__ == "__main__":
    main()
