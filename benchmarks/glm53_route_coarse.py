"""Validate and time NoPE MLA route/coarse layouts, outside model timings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmarks._vllm import write_json
from lod_attention.kernels.aiter_mla_prefill_attention import aiter_mla_prefill_route_coarse_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--slots", type=int, default=2048)
    parser.add_argument("--gluon", action="store_true")
    parser.add_argument("--early-exit", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(27)
    q = torch.randn(1, 64, args.queries, 512, device="cuda", dtype=torch.bfloat16) * 0.3
    counts = torch.randint(1, 20, (1, 1, args.slots, 1), device="cuda").float()
    sums = (torch.randn(1, 1, args.slots, 512, device="cuda") * counts * 0.4).bfloat16()
    indices = sorted({0, min(37, args.queries - 1), args.queries - 1})
    means = (sums.float() / counts).bfloat16().float()
    scores = q[:, :, indices].float() @ means.transpose(-1, -2) * 0.0625 + counts[..., 0].log().unsqueeze(2)
    expected = scores.softmax(-1) @ means
    result = dict(scope="random-input GPU route/coarse layout diagnostic",
                  query_shape=list(q.shape), slots=args.slots, candidates=[])
    with torch.inference_mode():
        candidates = (
                (False, 32, 16, 8), (False, 16, 32, 4),
                (True, 32, 32, 4), (True, 64, 32, 4),
                (True, 32, 64, 4), (True, 64, 64, 4))
        candidates = [(*tile, False) for tile in candidates] if not args.gluon else (
            (True, 64, 32, 4, False), (True, 32, 32, 4, True),
            (True, 64, 32, 4, True), (True, 32, 64, 4, True),
            (True, 64, 64, 4, True))
        for tiled, bm, bn, warps, gluon_layout in candidates:
            entry = dict(query_tiled=tiled, block_m=bm, block_n=bn, warps=warps,
                         gluon_layout=gluon_layout)
            buffers = {}
            try:
                def run():
                    return aiter_mla_prefill_route_coarse_attention(q, sums, sums, counts,
                        state_len=args.slots, kv_group_size=64, scale=0.0625,
                        normalize_route_query=False, buffers=buffers,
                        query_tiled=tiled, block_m=bm, block_n=bn, num_warps=warps,
                        gluon_layout=gluon_layout,
                        early_exit=args.early_exit)
                routes, coarse, _, _ = run()
                torch.cuda.synchronize()
                torch.testing.assert_close(coarse.output_0.permute(0, 2, 1, 3)[:, :, indices].float(),
                                           expected, atol=0.004, rtol=0.025)
                torch.testing.assert_close(coarse.lse_0[:, :, indices], scores.logsumexp(-1),
                                           atol=0.003, rtol=0.003)
                assert torch.equal(routes[:, :, indices], scores.topk(8, dim=-1).indices)
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                run()
                end.record()
                end.synchronize()
                entry.update(status="complete", gpu_ms=begin.elapsed_time(end))
                if gluon_layout:
                    from lod_attention.kernels.latent_route_coarse import latent_route_coarse_gluon
                    compiled = list(latent_route_coarse_gluon.device_caches[
                        torch.cuda.current_device()][0].values())[-1]
                    entry.update(registers=compiled.n_regs, spills=compiled.n_spills,
                                 shared_bytes=compiled.metadata.shared)
            except Exception as exc:
                entry.update(status="failed", error=str(exc))
            print(json.dumps(entry), flush=True)
            result["candidates"].append(entry)
            write_json(args.output, result)


if __name__ == "__main__":
    main()
