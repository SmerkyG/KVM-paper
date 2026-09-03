#!/usr/bin/env python3
"""Time Gemma-4's native D=256 sliding-window AITER decode kernel."""

import argparse
import math

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    from aiter.ops.triton.unified_attention import unified_attention

    batch, query_heads, kv_heads, dim = 8, 16, 8, 256
    window, block_size = 1024, 64
    blocks_per_sequence = window // block_size
    total_blocks = batch * blocks_per_sequence
    device = torch.device("cuda")
    q = torch.randn(batch, query_heads, dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(
        total_blocks, block_size, kv_heads, dim,
        dtype=torch.bfloat16, device=device,
    )
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device=device)
    seq_lens = torch.full((batch,), window, dtype=torch.int32, device=device)
    block_table = torch.arange(
        total_blocks, dtype=torch.int32, device=device
    ).view(batch, blocks_per_sequence)

    def attention() -> None:
        unified_attention(
            q=q, k=k, v=v, out=out, cu_seqlens_q=cu_q, max_seqlen_q=1,
            seqused_k=seq_lens, max_seqlen_k=window,
            softmax_scale=1.0 / math.sqrt(dim), causal=True,
            window_size=(window - 1, 0), block_table=block_table,
            softcap=0.0, q_descale=None, k_descale=None, v_descale=None,
        )

    for _ in range(args.warmup):
        attention()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(args.repeats):
        attention()
    end.record()
    torch.cuda.synchronize()
    print(f"aiter_bswa_ms={begin.elapsed_time(end) / args.repeats:.6f}")
    print(f"finite={bool(torch.isfinite(out).all())}")


if __name__ == "__main__":
    main()
