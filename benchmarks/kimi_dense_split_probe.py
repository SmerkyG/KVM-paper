"""Isolate dense live-split occupancy; not an end-to-end speed measurement."""

from __future__ import annotations

import argparse
import json
import math

import torch

from lod_attention.kernels.kimi_gluon_decode import absorbed_mla_decode_gfx942


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192, 32768])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    args = parser.parse_args()
    for batch in args.batches:
        page_size, heads = 768, 96
        blocks = math.ceil(max(args.lengths) / page_size)
        kv = torch.randn(batch * blocks, page_size, 576, dtype=torch.bfloat16, device="cuda")
        q = torch.randn(batch, heads, 576, dtype=torch.bfloat16, device="cuda")
        table = torch.arange(batch * blocks, dtype=torch.int32, device="cuda").view(batch, blocks)
        lengths = torch.empty(batch, dtype=torch.int32, device="cuda")
        out = torch.empty(batch, heads, 512, dtype=q.dtype, device="cuda")
        lse = torch.empty(batch, heads, device="cuda")
        for length in args.lengths:
            lengths.fill_(length)
            for splits, adaptive in [(n, False) for n in (4, 8, 16, 32, 64, 128)] + [(128, True)]:
                partial = torch.empty(batch, heads, splits, 512, device="cuda", dtype=q.dtype)
                partial_lse = torch.empty(batch, heads, splits, device="cuda")
                def run():
                    absorbed_mla_decode_gfx942(q, kv, out, table, lengths, 576**-0.5,
                        num_splits=splits, partial=partial, partial_lse=partial_lse,
                        final_lse=lse, adaptive_splits=adaptive)
                run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                for _ in range(10):
                    graph.replay()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(200):
                    graph.replay()
                end.record()
                end.synchronize()
                print(json.dumps(dict(batch=batch, local_length=length, splits=splits,
                    adaptive=adaptive, ms=start.elapsed_time(end) / 200)), flush=True)


if __name__ == "__main__":
    main()
