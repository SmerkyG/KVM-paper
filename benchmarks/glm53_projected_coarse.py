"""Projection-inclusive coarse timing, matched inputs, allocation/JIT excluded.

Random NoPE geometry only: this is not trained-model quality or model latency.
"""

import argparse
from pathlib import Path

import torch

from benchmarks._vllm import write_json
from lod_attention.kernels.aiter_mla_prefill_attention import aiter_mla_prefill_route_coarse_attention
from lod_attention.kernels.glm_projected_prefill import projected_route_coarse


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(79)
    heads, queries = 64, 16384
    q = (torch.randn(1, heads, queries, 256, device="cuda") * .3).bfloat16()
    uk = (torch.randn(heads, 256, 512, device="cuda") / 512**.5).bfloat16()
    uv = (torch.randn(heads, 512, 256, device="cuda") / 512**.5).bfloat16()
    absorbed = torch.bmm(q[0], uk).unsqueeze(0)
    results = dict(scope="random-input TP1 coarse-only GPU interval; not model latency",
                   queries=queries, heads=heads, projection_included=True, rows=[])
    for slots in (2048, 4096):
        counts = torch.randint(1, 64, (1, 1, slots, 1), device="cuda").float()
        state = (torch.randn(1, 1, slots, 512, device="cuda") * counts * .4).bfloat16()
        lengths = counts[..., 0].int().contiguous()
        for projected in (False, True):
            buffers = {}
            def run():
                if projected:
                    return projected_route_coarse(q, state, counts, uk, uv,
                        state_len=slots, scale=.0625, slot_lengths=lengths,
                        max_open_leaf_tokens=1024, buffers=buffers)
                return aiter_mla_prefill_route_coarse_attention(absorbed, state, state, counts,
                    state_len=slots, kv_group_size=heads, scale=.0625,
                    normalize_route_query=False, buffers=buffers)
            run()
            torch.cuda.synchronize()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            run()
            end.record()
            end.synchronize()
            row = dict(slots=slots, path="projected256" if projected else "absorbed512",
                       milliseconds=begin.elapsed_time(end))
            results["rows"].append(row)
            print(row, flush=True)
            write_json(args.output, results)


if __name__ == "__main__":
    main()
