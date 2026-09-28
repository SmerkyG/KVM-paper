"""Benchmark and verify the patched AITER top-eight route/coarse kernel."""

from __future__ import annotations

import argparse
import json
import math
import statistics

import torch

from lod_attention.kernels.aiter_prefill_attention import (
    _reduce_route_candidates,
    _specialized_route_mha_fwd,
    aiter_prefill_route_coarse_attention,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-len", type=int, default=512)
    parser.add_argument("--state-len", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--components", action="store_true")
    parser.add_argument("--oracle-queries", type=int, default=32)
    parser.add_argument("--head-dim", type=int, choices=(128, 256), default=128)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--raw-route-query", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim = args.head_dim
    group_size = args.group_size
    q = torch.randn(1, group_size, args.query_len, head_dim, device=device, dtype=dtype)
    counts = torch.randint(1, 17, (1, 1, args.state_len, 1), device=device).to(dtype)
    state_k = (
        torch.randn(1, 1, args.state_len, head_dim, device=device, dtype=dtype) * counts
    )
    state_v = torch.randn_like(state_k) * counts
    workspaces: dict[str, torch.Tensor] = {}

    def run():
        return aiter_prefill_route_coarse_attention(
            q,
            state_k,
            state_v,
            counts,
            state_len=args.state_len,
            kv_group_size=group_size,
            scale=1.0 / math.sqrt(head_dim),
            normalize_route_query=not args.raw_route_query,
            buffers=workspaces,
        )

    slots8, _, _, _ = run()
    torch.cuda.synchronize()
    oracle_queries = min(args.query_len, args.oracle_queries)
    mean_k = (state_k / counts).to(dtype)[0, 0].float()
    sample_q = q[0, :, :oracle_queries].float()
    scores = torch.einsum("hqd,sd->hqs", sample_q, mean_k)
    scores *= 1.0 / math.sqrt(head_dim)
    if not args.raw_route_query:
        scores /= sample_q.square().mean(dim=-1, keepdim=True).sqrt()
    scores += counts.log().to(dtype)[0, 0, :, 0].float()
    oracle_boundary = scores.topk(8, dim=-1).values[..., -1]
    selected_min = scores.gather(-1, slots8[0, :, :oracle_queries]).min(dim=-1).values
    max_route_regret = (oracle_boundary - selected_min).clamp_min(0).max().item()
    if max_route_regret > 0.05:
        raise AssertionError(
            f"top-eight missed a higher-scoring slot: {max_route_regret}"
        )

    run()
    torch.cuda.synchronize()

    def measure(call) -> float:
        samples = []
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop))
        return statistics.median(samples)

    timing = measure(run)
    components = {}
    if args.components:
        mean_k = (state_k / counts).to(dtype)
        mean_v = (state_v / counts).to(dtype)
        q_aiter = q.permute(0, 2, 1, 3)
        k_aiter = mean_k.permute(0, 2, 1, 3)
        v_aiter = mean_v.permute(0, 2, 1, 3)
        bias = (
            counts.log()
            .transpose(-1, -2)
            .expand(1, group_size, 1, args.state_len)
            .contiguous()
        )
        native = _specialized_route_mha_fwd(not args.raw_route_query)
        output = torch.empty(
            1, args.query_len, group_size, head_dim, device=device, dtype=dtype
        )

        def native_call():
            return native(
                q_aiter,
                k_aiter,
                v_aiter,
                0.0,
                1.0 / math.sqrt(head_dim),
                False,
                -1,
                -1,
                0,
                True,
                True,
                None,
                None,
                output,
                bias,
                None,
                None,
                None,
                None,
                None,
            )

        candidates = native_call()[2]
        reducer_buffers = {}

        def reduce_call():
            return _reduce_route_candidates(
                candidates,
                state_len=args.state_len,
                head_dim=head_dim,
                buffers=reducer_buffers,
            )

        reduce_call()
        torch.cuda.synchronize()
        components = {
            "native_ms": measure(native_call),
            "reducer_ms": measure(reduce_call),
        }
    print(
        json.dumps(
            {
                "query_len": args.query_len,
                "state_len": args.state_len,
                "head_dim": head_dim,
                "group_size": group_size,
                "raw_route_query": args.raw_route_query,
                "top8_ms": timing,
                "max_route_regret": max_route_regret,
                "components": components,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
