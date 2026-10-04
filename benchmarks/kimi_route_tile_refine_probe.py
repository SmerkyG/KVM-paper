"""Matched route/coarse kernel probe; never report as full-model timing."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from lod_attention.kernels.aiter_prefill_attention import (
    _reduce_route_candidates, _specialized_kimi_coarse_mha_fwd,
)
from lod_attention.kernels.kimi_route_tile_refine import refine_kimi_centroid_tiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=16384)
    parser.add_argument("--states", type=int, default=3547)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(83)
    dtype, device = torch.bfloat16, "cuda"
    q = torch.randn(1, args.heads, args.queries, 192, dtype=dtype, device=device)
    k = torch.randn(1, args.states, args.heads, 192, dtype=dtype, device=device)
    v = torch.randn(1, args.states, args.heads, 128, dtype=dtype, device=device)
    logs = torch.randint(1, 65, (1, args.states), device=device).float().log().bfloat16()
    bias = logs[:, None, None].expand(1, args.heads, 1, args.states)
    output = q.new_empty(1, args.queries, args.heads, 128)
    scale = 192**-0.5
    original = _specialized_kimi_coarse_mha_fwd(192, async_bias=True, fused_route=True)
    tile_max = _specialized_kimi_coarse_mha_fwd(192, async_bias=True, fused_route=True, tile_max_probe=True)
    workspaces = {"original": {}, "tile_refine": {}}

    def run(variant):
        refined = variant == "tile_refine"
        op = tile_max if refined else original
        coarse = op(q.permute(0, 2, 1, 3), k, v, 0.0, scale, False,
                    -1, -1, 0, True, True, None, None, output, bias,
                    None, None, None, None, None)
        candidates = coarse[2]
        buffers = workspaces[variant]
        if refined:
            candidates = refine_kimi_centroid_tiles(
                candidates, q, k, logs, state_len=args.states,
                scale=scale, buffers=buffers,
            )
        routes, _, _, scores = _reduce_route_candidates(
            candidates, state_len=args.states, head_dim=192,
            emit_metadata=False, buffers=buffers,
        )
        return coarse[0], coarse[1], routes, scores

    baseline = tuple(x.clone() for x in run("original"))
    candidate = tuple(x.clone() for x in run("tile_refine"))
    torch.cuda.synchronize()
    torch.testing.assert_close(candidate[0], baseline[0], atol=0.002, rtol=0.002)
    torch.testing.assert_close(candidate[1], baseline[1], atol=0.002, rtol=0.002)
    # Both kernels retain FP32 logits. Near-ties can cross their boundary due
    # to different MMA accumulation order; record and bound this explicitly.
    matching = float((candidate[2].sort(-1).values == baseline[2].sort(-1).values).all(-1).float().mean())
    if matching < 0.995:
        raise RuntimeError(f"tile-refined routes disagree: {matching:.6f}")
    subset = min(args.queries, 128)
    exact = torch.einsum("bhqd,bshd->bhqs", q[:, :, :subset].float(), k.float()) * scale
    exact += logs.float()[:, None, None]
    for result in (baseline, candidate):
        selected = exact.gather(-1, result[2][:, :, :subset])
        torch.testing.assert_close(result[3][:, :, :subset], selected, atol=0.025, rtol=0.005)
        best = exact.topk(8, dim=-1).values
        torch.testing.assert_close(selected.sort(-1).values, best.sort(-1).values, atol=0.025, rtol=0.005)
    result = {"scope": "warmed route/coarse including exact global top-eight; synthetic Q/K",
              "queries": args.queries, "states": args.states, "heads": args.heads,
              "route_set_matching_fraction": matching, "checks": "passed", "measurements": []}
    for variant in ("original", "tile_refine", "original", "tile_refine"):
        run(variant)
        torch.cuda.synchronize()
        wall, gpu = [], []
        for _ in range(args.iterations):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            begin.record()
            run(variant)
            end.record()
            end.synchronize()
            wall.append((time.perf_counter() - started) * 1000)
            gpu.append(begin.elapsed_time(end))
        result["measurements"].append({"variant": variant, "wall_median_ms": statistics.median(wall),
                                        "gpu_median_ms": statistics.median(gpu), "gpu_samples_ms": gpu})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
