#!/usr/bin/env python3
"""Verify and microbenchmark AITER's compact LOD score-tile predicate."""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F


def run_aiter(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    *,
    max_q: int,
    max_k: int,
    scale: float,
    query_route_masks: torch.Tensor | None = None,
    kv_query_masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.mha import mha_batch_prefill_func

    kwargs: dict[str, torch.Tensor] = {}
    if query_route_masks is not None or kv_query_masks is not None:
        if query_route_masks is None or kv_query_masks is None:
            raise ValueError("query and KV route metadata must be provided together")
        kwargs["block_table"] = query_route_masks[:1]
        kwargs["seqlen_k"] = kv_query_masks
    out, lse = mha_batch_prefill_func(
        q,
        k,
        v,
        qo_indptr,
        kv_indptr,
        kv_page_indices,
        max_q,
        max_k,
        softmax_scale=scale,
        causal=False,
        return_lse=True,
        **kwargs,
    )
    return out, lse.reshape(-1)


def reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    query_route_masks: torch.Tensor,
    kv_query_masks: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(q)
    lse = torch.empty(q.size(0), dtype=torch.float32, device=q.device)
    for sequence in range(qo_indptr.numel() - 1):
        qb = int(qo_indptr[sequence].item())
        qe = int(qo_indptr[sequence + 1].item())
        kb = int(kv_indptr[sequence].item())
        ke = int(kv_indptr[sequence + 1].item())
        physical = kv_page_indices[kb:ke].long()
        local_k = k[physical, 0, 0].float()
        local_v = v[physical, 0, 0].float()
        local_query_masks = kv_query_masks[kb:ke].long()
        for row in range(qb, qe):
            local_row = row - qb
            selected = (
                torch.bitwise_right_shift(local_query_masks, local_row) & 1
            ).bool()
            scores = q[row, 0].float() @ local_k[selected].T * scale
            probabilities = torch.softmax(scores, dim=-1)
            out[row, 0] = (probabilities @ local_v[selected]).to(out.dtype)
            lse[row] = torch.logsumexp(scores, dim=-1)
    return out, lse


def elapsed_ms(call, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        call()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-sequences", type=int, default=0)
    parser.add_argument("--query-tile", type=int, default=16)
    parser.add_argument("--union-tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim = 128
    scale = 1.0 / math.sqrt(head_dim)

    q_lengths = torch.tensor([16, 11, 5], dtype=torch.int32, device=device)
    k_lengths = torch.tensor([37, 53, 29], dtype=torch.int32, device=device)
    qo_indptr = F.pad(q_lengths.cumsum(0), (1, 0)).to(torch.int32)
    kv_indptr = F.pad(k_lengths.cumsum(0), (1, 0)).to(torch.int32)
    total_q = int(q_lengths.sum().item())
    total_k = int(k_lengths.sum().item())
    q = torch.randn(total_q, 1, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_k, 1, 1, head_dim, dtype=dtype, device=device)
    v = torch.randn_like(k)
    pages = torch.arange(total_k, dtype=torch.int32, device=device)
    kv_union_ranks = torch.arange(total_k, dtype=torch.int32, device=device) % 12
    query_routes = torch.randint(
        0, 12, (total_q, 8), dtype=torch.int32, device=device
    )
    # Every row must select at least one route represented in its sequence.
    query_routes[:, 0] = 0
    query_route_masks = torch.zeros(
        total_q, 4, dtype=torch.int32, device=device
    )
    for rank in range(query_routes.size(1)):
        route = query_routes[:, rank]
        word = torch.div(route, 32, rounding_mode="floor").long()
        bit = route % 32
        query_route_masks[
            torch.arange(total_q, device=device), word
        ] |= torch.bitwise_left_shift(torch.ones_like(bit), bit)
    kv_query_masks = torch.zeros(total_k, dtype=torch.int32, device=device)
    for sequence in range(q_lengths.numel()):
        qb = int(qo_indptr[sequence].item())
        qe = int(qo_indptr[sequence + 1].item())
        kb = int(kv_indptr[sequence].item())
        ke = int(kv_indptr[sequence + 1].item())
        ranks = kv_union_ranks[kb:ke].long()
        for local_row, row in enumerate(range(qb, qe)):
            words = torch.div(ranks, 32, rounding_mode="floor")
            bits = ranks % 32
            selected = (
                torch.bitwise_right_shift(
                    query_route_masks[row, words].long(), bits
                )
                & 1
            ).to(torch.int32)
            kv_query_masks[kb:ke] |= selected << local_row

    with torch.inference_mode():
        actual_out, actual_lse = run_aiter(
            q,
            k,
            v,
            qo_indptr,
            kv_indptr,
            pages,
            max_q=int(q_lengths.max().item()),
            max_k=int(k_lengths.max().item()),
            scale=scale,
            query_route_masks=query_route_masks,
            kv_query_masks=kv_query_masks,
        )
        unmasked_out, unmasked_lse = run_aiter(
            q,
            k,
            v,
            qo_indptr,
            kv_indptr,
            pages,
            max_q=int(q_lengths.max().item()),
            max_k=int(k_lengths.max().item()),
            scale=scale,
        )
        expected_out, expected_lse = reference(
            q,
            k,
            v,
            qo_indptr,
            kv_indptr,
            pages,
            query_route_masks,
            kv_query_masks,
            scale,
        )
        torch.cuda.synchronize()

    output_error = float((actual_out.float() - expected_out.float()).abs().max())
    lse_error = float((actual_lse - expected_lse).abs().max())
    print(f"output_max_abs_error={output_error:.8f}")
    print(f"lse_max_abs_error={lse_error:.8f}")
    print(
        "masked_vs_unmasked_output_max_abs="
        f"{float((actual_out.float() - unmasked_out.float()).abs().max()):.8f}"
    )
    print(
        "masked_vs_unmasked_lse_max_abs="
        f"{float((actual_lse - unmasked_lse).abs().max()):.8f}"
    )
    if output_error > 0.02 or lse_error > 0.02:
        raise AssertionError("AITER compact LOD route mask disagrees with reference")

    if args.benchmark_sequences <= 0:
        return

    sequences = args.benchmark_sequences
    query_tile = args.query_tile
    union_tokens = args.union_tokens
    bench_q = torch.randn(
        sequences * query_tile, 1, head_dim, dtype=dtype, device=device
    )
    bench_k = torch.randn(union_tokens, 1, 1, head_dim, dtype=dtype, device=device)
    bench_v = torch.randn_like(bench_k)
    bench_qo = torch.arange(
        0,
        (sequences + 1) * query_tile,
        query_tile,
        dtype=torch.int32,
        device=device,
    )
    bench_kv = torch.arange(
        0,
        (sequences + 1) * union_tokens,
        union_tokens,
        dtype=torch.int32,
        device=device,
    )
    bench_pages = torch.arange(union_tokens, dtype=torch.int32, device=device).repeat(
        sequences
    )
    bench_kv_query_masks = torch.full(
        (sequences * union_tokens,),
        (1 << query_tile) - 1,
        dtype=torch.int32,
        device=device,
    )
    bench_query_masks = torch.zeros(
        sequences * query_tile, 4, dtype=torch.int32, device=device
    )
    bench_query_masks[:, 0] = 0xFF

    def unmasked():
        return run_aiter(
            bench_q,
            bench_k,
            bench_v,
            bench_qo,
            bench_kv,
            bench_pages,
            max_q=query_tile,
            max_k=union_tokens,
            scale=scale,
        )

    def masked():
        return run_aiter(
            bench_q,
            bench_k,
            bench_v,
            bench_qo,
            bench_kv,
            bench_pages,
            max_q=query_tile,
            max_k=union_tokens,
            scale=scale,
            query_route_masks=bench_query_masks,
            kv_query_masks=bench_kv_query_masks,
        )

    with torch.inference_mode():
        unmasked_ms = elapsed_ms(unmasked, warmup=args.warmup, repeats=args.repeats)
        masked_ms = elapsed_ms(masked, warmup=args.warmup, repeats=args.repeats)
    print(f"unmasked_aiter_ms={unmasked_ms:.6f}")
    print(f"masked_aiter_ms={masked_ms:.6f}")
    print(f"masked_overhead={(masked_ms / unmasked_ms - 1.0) * 100.0:.2f}%")


if __name__ == "__main__":
    main()
