#!/usr/bin/env python3
"""Check AITER packed-query, paged-KV varlen attention semantics."""

from __future__ import annotations

import argparse
import math

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("public", "ck", "kvcache", "batch_prefill"),
        default="public",
    )
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()
    from aiter.ops.mha import flash_attn_varlen_func, mha_varlen_fwd

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    page_size = args.page_size
    head_dim = args.head_dim
    q_lengths = torch.tensor([7, 3, 19], device=device, dtype=torch.int32)
    k_lengths = torch.tensor([5, 22, 31], device=device, dtype=torch.int32)
    block_table = torch.tensor(
        [[5, -1], [1, 3], [0, 2]], device=device, dtype=torch.int32
    )
    page_k = torch.randn(
        6, page_size, 1, head_dim, device=device, dtype=dtype
    )
    page_v = torch.randn_like(page_k)
    packed_q = torch.randn(
        int(q_lengths.sum().item()), 1, head_dim, device=device, dtype=dtype
    )
    cu_q = torch.nn.functional.pad(q_lengths.cumsum(0), (1, 0)).to(torch.int32)
    cu_k = torch.nn.functional.pad(k_lengths.cumsum(0), (1, 0)).to(torch.int32)
    scale = 1.0 / math.sqrt(head_dim)

    with torch.inference_mode():
        if args.backend == "public":
            actual_out, actual_lse = flash_attn_varlen_func(
                packed_q,
                page_k,
                page_v,
                cu_q,
                cu_k,
                int(q_lengths.max().item()),
                int(k_lengths.max().item()),
                softmax_scale=scale,
                causal=False,
                return_lse=True,
                block_table=block_table,
            )
        elif args.backend == "ck":
            actual_out, actual_lse, _, _ = mha_varlen_fwd(
                packed_q,
                page_k,
                page_v,
                cu_q,
                cu_k,
                int(q_lengths.max().item()),
                int(k_lengths.max().item()),
                0,
                0.0,
                scale,
                0.0,
                False,
                False,
                -1,
                -1,
                0,
                True,
                False,
                block_table=block_table,
            )
        elif args.backend == "kvcache":
            from aiter.ops.triton.attention.mha import flash_attn_with_kvcache

            outputs = []
            lses = []
            for expert in range(int(q_lengths.numel())):
                q_begin = int(cu_q[expert].item())
                q_end = int(cu_q[expert + 1].item())
                local_out, local_lse = flash_attn_with_kvcache(
                    packed_q[q_begin:q_end].unsqueeze(0),
                    page_k,
                    page_v,
                    cache_seqlens=k_lengths[expert : expert + 1],
                    softmax_scale=scale,
                    causal=False,
                    block_table=block_table[expert : expert + 1],
                    return_softmax_lse=True,
                )
                outputs.append(local_out.squeeze(0))
                lses.append(local_lse.reshape(-1))
            actual_out = torch.cat(outputs)
            actual_lse = torch.cat(lses)
        else:
            from aiter.ops.mha import mha_batch_prefill_func

            page_counts = torch.div(
                k_lengths + page_size - 1,
                page_size,
                rounding_mode="floor",
            )
            kv_indptr = torch.nn.functional.pad(
                page_counts.cumsum(0), (1, 0)
            ).to(torch.int32)
            page_mask = (
                torch.arange(block_table.size(1), device=device)[None, :]
                < page_counts[:, None]
            )
            kv_page_indices = block_table[page_mask].contiguous()
            kv_last_page_lens = ((k_lengths - 1) % page_size + 1).to(
                torch.int32
            )
            actual_out, actual_lse = mha_batch_prefill_func(
                packed_q,
                page_k,
                page_v,
                cu_q,
                kv_indptr,
                kv_page_indices,
                int(q_lengths.max().item()),
                int(k_lengths.max().item()),
                softmax_scale=scale,
                causal=False,
                return_lse=True,
                kv_last_page_lens=kv_last_page_lens,
                seqlen_k=k_lengths,
            )

    expected_out = torch.empty_like(actual_out)
    expected_lse = torch.empty(
        packed_q.size(0), device=device, dtype=torch.float32
    )
    for expert in range(int(q_lengths.numel())):
        q_begin = int(cu_q[expert].item())
        q_end = int(cu_q[expert + 1].item())
        key_count = int(k_lengths[expert].item())
        page_count = (key_count + page_size - 1) // page_size
        local_k = page_k[
            block_table[expert, :page_count].long(), :, 0
        ].reshape(-1, head_dim)[:key_count]
        local_v = page_v[
            block_table[expert, :page_count].long(), :, 0
        ].reshape(-1, head_dim)[:key_count]
        scores = (
            packed_q[q_begin:q_end, 0].float() @ local_k.float().T
        ) * scale
        probabilities = torch.softmax(scores, dim=-1)
        expected_out[q_begin:q_end, 0] = (
            probabilities @ local_v.float()
        ).to(dtype)
        expected_lse[q_begin:q_end] = torch.logsumexp(scores, dim=-1)

    if actual_lse.ndim == 2:
        actual_lse = actual_lse[0]
    output_error = float(
        (actual_out.float() - expected_out.float()).abs().max().item()
    )
    lse_error = float((actual_lse.float() - expected_lse).abs().max().item())
    print(f"output_max_abs_error={output_error:.8f}")
    print(f"lse_max_abs_error={lse_error:.8f}")
    if output_error > 0.02 or lse_error > 0.02:
        raise AssertionError("AITER paged-varlen attention disagrees with reference")


if __name__ == "__main__":
    main()
