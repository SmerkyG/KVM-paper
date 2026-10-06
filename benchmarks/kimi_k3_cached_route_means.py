"""Freeze K3 route inputs and compare sum division against cached means.

Isolated graph-replay timings, not model latency. Cached means are produced
by the same materializer used at serving cache installation/catch-up.
"""

import argparse
import json
from pathlib import Path

import torch
import triton

from benchmarks.kimi_k3_decode_route_tune import graph_us, compiler_resources
from lod_attention.kernels.paged_decode_kernels import materialize_absorbed_mla_coarse_means
from lod_attention.kernels.paged_routing import _decode_route_coarse_gqa_groups_kernel


def run_case(batch, slots, *, ties=False):
    torch.manual_seed(41)
    capacity = slots + 23
    q = torch.randn(batch, 96, 1, 576, device="cuda", dtype=torch.bfloat16)
    if ties:
        q.zero_()
    count = torch.randint(1, 80, (batch, 1, capacity, 1), device="cuda").float()
    count[:, :, ::37] = 0
    count[:, :, 1::41] = 1024
    count[:, :, 2::43] = 1025
    key = (torch.randn(batch, 1, capacity, 576, device="cuda") * count).bfloat16()
    means = torch.full_like(key, float("nan"))
    bias = torch.full((batch, 1, capacity), float("nan"), device="cuda", dtype=torch.float16)
    materialize_absorbed_mla_coarse_means(key, count, means, bias, active_state_len=slots)
    index = torch.arange(batch, device="cuda", dtype=torch.int32).flip(0)
    lengths = torch.arange(batch, device="cuda", dtype=torch.int32) * -3 + slots
    groups = triton.cdiv(capacity, 64)
    dummy = torch.empty(1, device="cuda")
    results = []
    before = None
    for cached in (False, True):
        k = (means if cached else key).expand(batch, 6, capacity, 576)
        counts = count.expand(batch, 6, capacity, 1)
        cached_bias = bias.expand(batch, 6, capacity)
        scores = torch.empty(batch, 96, groups, 8, device="cuda")
        indices = torch.empty_like(scores, dtype=torch.int64)
        def launch():
            return _decode_route_coarse_gqa_groups_kernel[batch*6, groups](
                q, k, dummy, counts, index, scores, indices, dummy, dummy,
                *k.stride()[:3], 0, 0, 0, *counts.stride()[:3], slots,
                lengths, dummy, dummy, dummy, cached_bias, dummy, dummy, dummy,
                dummy, dummy, dummy, *cached_bias.stride(),
                QUERY_HEADS=96, KV_HEADS=6, KV_GROUP_SIZE=16, HEAD_DIM=576,
                VALUE_DIM=512, SCALE=576**-.5, GROUP_N=64, MAX_GROUPS=groups,
                PROTECTED_LEN=0, MAX_LEAF_TOKENS=1024, USE_DOT=True,
                KEYS_ARE_MEANS=cached, SCORE_ONLY=True, USE_STATE_LENS=True,
                # Attention's cache bias is FP16. Keep fresh FP32 log(count)
                # for routing, preserving every original candidate score bit.
                USE_LOG_COUNT_BIAS=False, CANDIDATES_PER_GROUP=8,
                num_warps=4, num_stages=2, waves_per_eu=1)
        compiled = launch()
        if before is None:
            before = scores.clone(), indices.clone()
        else:
            torch.testing.assert_close(scores, before[0], atol=0, rtol=0)
            torch.testing.assert_close(indices, before[1], atol=0, rtol=0)
        micros, samples = graph_us(launch)
        results.append(dict(cached_means=cached, graph_kernel_us=micros,
                            samples_us=samples, resources=compiler_resources(compiled)))
    return dict(batch=batch, active_slots=slots, ties=ties,
                scores_and_indices_bit_identical=True, variants=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(scope="isolated frozen centroid router; excludes collectives/model", rows=[])
    for batch, slots, ties in ((1, 512, False), (1, 4096, False), (8, 512, False),
                               (8, 4096, False), (1, 4096, True), (8, 512, True)):
        row = run_case(batch, slots, ties=ties)
        result["rows"].append(row)
        print(json.dumps(row), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n")
    result["status"] = "complete"
    args.output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
