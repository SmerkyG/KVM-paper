"""Tune fused CUDA prefill tiling at Llama's real 16K query/GQA4 geometry.

Run through cluster-run. Each timed configuration must match the independently
verified default's top-eight set, coarse output, and partition mass first.
"""

import argparse
import json
import os
from pathlib import Path

from benchmarks._identity import benchmark_identity
from benchmarks._vllm import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from triton.testing import do_bench_cudagraph
    from lod_attention.kernels.cuda_prefill_attention import (
        cuda_prefill_route_coarse_attention,
    )

    torch.manual_seed(741)
    result = dict(
        benchmark="h100-fused-prefill-tiles",
        benchmark_identity=benchmark_identity(),
        gpu=torch.cuda.get_device_name(),
        node=os.environ["CLUSTER_RUN_NODE"],
        measurements=[],
    )
    for states in (2048, 4096):
        q = torch.randn(1, 32, 16384, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 8, states, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        counts = torch.randint(1, 65, (1, 8, states, 1), device="cuda").float()
        common = dict(
            state_len=states,
            kv_group_size=4,
            scale=128**-0.5,
            normalize_route_query=True,
        )
        default, coarse, _, _ = cuda_prefill_route_coarse_attention(
            q, k, v, counts, **common
        )
        expected_routes = default.sort(-1).values.clone()
        expected_output = coarse.output_0.clone()
        expected_lse = coarse.lse_0.clone()
        for m, n, warps in (
            (32, 64, 4),
            (32, 128, 4),
            (32, 256, 4),
            (64, 64, 4),
            (64, 128, 4),
            (64, 128, 8),
            (64, 32, 4),
            (128, 32, 4),
            (128, 32, 8),
            (128, 64, 4),
            (128, 64, 8),
            (128, 128, 8),
        ):
            buffers = {}

            def run():
                return cuda_prefill_route_coarse_attention(
                    q,
                    k,
                    v,
                    counts,
                    **common,
                    buffers=buffers,
                    block_m=m,
                    block_n=n,
                    num_warps=warps,
                )

            routes, coarse, _, _ = run()
            if not torch.equal(routes.sort(-1).values, expected_routes):
                raise AssertionError(f"Top-eight mismatch for {(states, m, n, warps)}")
            torch.testing.assert_close(
                coarse.output_0, expected_output, atol=0.005, rtol=0.02
            )
            torch.testing.assert_close(
                coarse.lse_0, expected_lse, atol=2.0e-4, rtol=2.0e-4
            )
            milliseconds = do_bench_cudagraph(run, rep=300)
            row = dict(
                states=states,
                queries=16384,
                block_m=m,
                block_n=n,
                num_warps=warps,
                milliseconds=milliseconds,
            )
            result["measurements"].append(row)
            write_json(args.output, result)
            print(json.dumps(row), flush=True)
    result["stage"] = "complete"
    write_json(args.output, result)


if __name__ == "__main__":
    main()
