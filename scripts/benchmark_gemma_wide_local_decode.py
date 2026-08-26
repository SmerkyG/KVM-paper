#!/usr/bin/env python3
"""Verify and time the D=512 GQA local decoder against its scalar predecessor."""

from __future__ import annotations

import argparse
import math

import torch
import triton

from model.kernels.paged_leaf_attention import (
    _mask_decode_routes_residual_mass_kernel,
    _wide_gqa_local_scores_kernel,
    _wide_gqa_local_value_kernel,
)


def elapsed(call, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        call()
    end.record()
    torch.cuda.synchronize()
    return begin.elapsed_time(end) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-len", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--query-fp32", action="store_true")
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--query-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    args = parser.parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch = 8
    q_heads, kv_heads, dim = args.query_heads, args.kv_heads, args.head_dim
    if q_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    group = q_heads // kv_heads
    if not 1 < group <= 16:
        raise ValueError("this benchmark requires a GQA group in [2, 16]")
    local_len = args.local_len
    q = torch.randn(
        batch,
        q_heads,
        1,
        dim,
        dtype=torch.float32 if args.query_fp32 else torch.bfloat16,
        device=device,
    )
    k = torch.randn(batch, kv_heads, local_len + 1, dim, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    new_k = torch.randn(batch, kv_heads, 1, dim, dtype=torch.bfloat16, device=device)
    new_v = torch.randn_like(new_k)
    cache_indices = torch.arange(batch, dtype=torch.int64, device=device)
    local_lens = torch.full((batch,), local_len, dtype=torch.int32, device=device)
    scores = torch.empty(batch, q_heads, local_len + 1, dtype=torch.float32, device=device)
    wide_out = torch.empty(batch, q_heads, dim, dtype=torch.float32, device=device)
    wide_lse = torch.empty(batch, q_heads, dtype=torch.float32, device=device)
    old_out = torch.empty_like(wide_out)
    old_lse = torch.empty_like(wide_lse)
    top_slots = torch.zeros(batch, q_heads, 8, dtype=torch.int64, device=device)
    top_scores = torch.zeros(batch, q_heads, 8, dtype=torch.float32, device=device)
    coarse_lse = torch.zeros(batch, q_heads, dtype=torch.float32, device=device)
    scale = 1.0 / math.sqrt(dim)

    def old() -> None:
        _mask_decode_routes_residual_mass_kernel[(batch * q_heads,)](
            q, k, v, cache_indices, local_lens, new_k, new_v,
            top_slots, top_scores, coarse_lse, old_out, old_lse, 1.0,
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            new_k.stride(0), new_k.stride(1), new_v.stride(0), new_v.stride(1),
            local_len, QUERY_HEADS=q_heads, KV_GROUP_SIZE=group, HEAD_DIM=dim,
            ROUTE_COUNT=8, LOCAL_BLOCK_N=32, SCALE=scale, INCLUDE_NEW=True,
            COMPUTE_LOCAL_OUTPUT=True, APPLY_ROUTE_MASK=False, num_warps=1,
        )

    def wide() -> None:
        _wide_gqa_local_scores_kernel[(batch * kv_heads, triton.cdiv(local_len + 1, 32))](
            q, cache_indices, local_lens, k, new_k, scores,
            k.stride(0), k.stride(1), k.stride(2), new_k.stride(0), new_k.stride(1),
            scores.stride(0), scores.stride(1), local_len,
            QUERY_HEADS=q_heads, KV_HEADS=kv_heads, KV_GROUP_SIZE=group,
            HEAD_DIM=dim, SCALE=scale, BLOCK_M=16, BLOCK_N=32,
            INCLUDE_NEW=True, num_warps=4,
        )
        _wide_gqa_local_value_kernel[(batch * kv_heads, triton.cdiv(dim, 32))](
            cache_indices, local_lens, k, v, new_k, new_v, scores, wide_out, wide_lse,
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            new_k.stride(0), new_k.stride(1), new_v.stride(0), new_v.stride(1),
            scores.stride(0), scores.stride(1), local_len,
            QUERY_HEADS=q_heads, KV_HEADS=kv_heads, KV_GROUP_SIZE=group,
            HEAD_DIM=dim, BLOCK_M=16, BLOCK_D=32, BLOCK_K=32, INCLUDE_NEW=True,
            num_warps=4,
        )

    old()
    wide()
    torch.cuda.synchronize()
    print(f"output_max_abs_error={(old_out - wide_out).abs().max().item():.8f}")
    print(f"lse_max_abs_error={(old_lse - wide_lse).abs().max().item():.8f}")
    old_ms = elapsed(old, args.warmup, args.repeats)
    wide_ms = elapsed(wide, args.warmup, args.repeats)
    print(f"scalar_lod_ms={old_ms:.6f}")
    print(f"wide_gqa_mfma_ms={wide_ms:.6f}")
    print(f"speedup={old_ms / wide_ms:.3f}")


if __name__ == "__main__":
    main()
