"""Inspect exported IPC storage extents without mapping GPU tensors.

This reads existing manifests through the supported fetch operation. Handles
are grouped locally and never written to the report. Extents are lower bounds:
IPC metadata does not describe unused trailing allocator-block space.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import socket


def extent_summary(entries):
    intervals = defaultdict(list)
    for entry in entries.values():
        if entry["transport"] != "cuda_ipc":
            continue
        args = entry["ipc_args"]
        if len(args) != 15:
            raise ValueError("Unsupported PyTorch IPC argument layout")
        device, handle, size, offset = args[6:10]
        if size < 0 or offset < 0:
            raise ValueError("Negative IPC storage extent")
        if size:
            intervals[(device, handle)].append((offset, offset + size))
    occupied = extent = 0
    for spans in intervals.values():
        spans.sort()
        start, end = spans[0]
        extent += max(stop for _, stop in spans)
        for lower, upper in spans[1:]:
            if lower > end:
                occupied += end - start
                start, end = lower, upper
            else:
                end = max(end, upper)
        occupied += end - start
    return dict(ipc_allocation_handles=len(intervals), exported_storage_union_bytes=occupied,
        minimum_pinned_allocation_extent_bytes=extent,
        minimum_unexported_internal_gap_bytes=extent - occupied,
        trailing_allocation_slack="not available in IPC metadata")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-id", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    from benchmarks.kimi_k3_weight_residency import snapshot
    from vllm_lod_plugin.weight_cache_protocol import receive_message, send_message
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("Run IPC inspection in a fresh process without a GPU context")
    metadata = snapshot(args.cache_id, args.cache_dir)
    records = []
    for worker in metadata["workers"]:
        rank = worker["fingerprint"]["tp_rank"]
        if rank not in args.ranks:
            continue
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(30)
            client.connect(worker["socket"])
            send_message(client, {"type": "fetch", "fingerprint": worker["fingerprint"]})
            payload = receive_message(client)
        if payload.get("status") != "ok":
            raise RuntimeError(f"IPC metadata fetch failed: {payload.get('status')}")
        records.append(dict(rank=rank, **extent_summary(payload["entries"])))
    if sorted(record["rank"] for record in records) != sorted(set(args.ranks)):
        raise ValueError("Requested ranks not present exactly once in resident weight group")
    if torch.cuda.is_initialized():
        raise RuntimeError("Inspection unexpectedly initialized a GPU context")
    report = dict(scope="read-only IPC extents, not a GPU memory snapshot", cuda_context_initialized=False,
        cache_id=args.cache_id, records=records)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
