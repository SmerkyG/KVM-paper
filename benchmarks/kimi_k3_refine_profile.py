"""Attribute serial route-refinement GPU stages on trained K3 inputs.

Event probes are benchmark-only; they are not installed in serving or used
to claim model wall-time speedups. Changed-input correctness and whole-path
ordinary/candidate controls remain required for any proposed optimization.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch

from benchmarks.kimi_k3_update_graph import reconstruct_centroids
from lod_attention.kernels import aiter_mla_prefill_attention as attention
from lod_attention.kernels import kimi_route_tile_refine as refine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["LOD_KIMI_TILE_REFINE"] = "1"
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    sums, counts, _ = reconstruct_centroids(payload)
    q = payload["q"].cuda().contiguous()
    keys, counts = sums[None, None].cuda().bfloat16(), counts[None, None].cuda()
    lengths = counts[..., 0].int()
    uk, uv = payload["w_uk_t"].cuda(), payload["w_uv"].cuda()
    workspace = {}
    probes = []

    def run():
        output = attention.aiter_kimi_expanded_prefill_route_coarse_attention(
            q, keys, keys[..., :512], counts, uk, uv,
            state_len=sums.size(0), scale=payload["scale"],
            normalize_route_query=False, slot_lengths=lengths,
            max_open_leaf_tokens=1024, buffers=workspace)
        torch.cuda.current_stream().wait_stream(output[1].ready_stream)
        return output

    def probe(name, function):
        def wrapped(*args, **kwargs):
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            result = function(*args, **kwargs)
            end.record()
            probes.append((name, begin, end))
            return result
        return wrapped

    class KernelProbe:
        def __init__(self, name, kernel):
            self.name, self.kernel = name, kernel

        def __getitem__(self, grid):
            return probe(self.name, self.kernel[grid])

    with torch.inference_mode():
        run()
        run()
        torch.cuda.synchronize()
        refine._select_centroid_tiles = KernelProbe("select_tiles", refine._select_centroid_tiles)
        refine._pack_expert_routes = probe("pack_tile_queries", refine._pack_expert_routes)
        refine._rescore_centroid_tiles = KernelProbe("rescore_tiles_top8", refine._rescore_centroid_tiles)
        attention._reduce_route_candidates = probe("merge_top8_candidates", attention._reduce_route_candidates)
        for _ in range(5):
            run()
        torch.cuda.synchronize()
    phases = {}
    for name, begin, end in probes:
        phases.setdefault(name, []).append(begin.elapsed_time(end))
    result = {"scope": "instrumented serial GPU routing-stage diagnostic, not model latency",
              "source": str(args.input), "query_shape": list(q.shape),
              "state_len": sums.size(0),
              "phases": {name: {"median_ms": statistics.median(samples), "samples_ms": samples}
                         for name, samples in phases.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
