#!/usr/bin/env python3
"""Benchmark page-size-1 centroid argmax against Qwen's AITER decode."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from model.kernels.aiter_centroid_argmax import (
    centroid_argmax_page1,
    flat_centroid_argmax_reduce,
    flat_page1_qk_scores,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192, 16384, 32768, 65536])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--state-multiplier", type=float, default=16.0)
    parser.add_argument("--summary-page-size", type=int, default=1)
    parser.add_argument(
        "--leaf-distribution", choices=("balanced", "zipf"), default="balanced"
    )
    parser.add_argument("--zipf-exponent", type=float, default=0.7)
    parser.add_argument("--shuffle-counts", action="store_true")
    parser.add_argument("--block-k", type=int, choices=(64, 128), default=128)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--centroids-per-block", type=int, default=16)
    parser.add_argument("--reduce-layout", choices=("packed", "stream"), default="packed")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _elapsed_ms(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / iterations


def _centroid_lists(
    batch: int,
    kv_heads: int,
    context: int,
    centroids: int,
    device: torch.device,
    distribution: str,
    zipf_exponent: float,
    shuffle_counts: bool,
    summary_page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if distribution == "balanced":
        token_offsets = torch.div(
            torch.arange(centroids + 1, device=device, dtype=torch.int64) * context,
            centroids,
            rounding_mode="floor",
        )
        token_counts = token_offsets[1:] - token_offsets[:-1]
    else:
        if context < centroids:
            raise ValueError("Zipf lists require at least one leaf per centroid")
        rank = torch.arange(1, centroids + 1, device=device, dtype=torch.float64)
        weight = rank.pow(-zipf_exponent)
        expected = weight / weight.sum() * (context - centroids)
        extra = expected.floor().to(torch.int64)
        remainder = context - centroids - int(extra.sum())
        if remainder:
            extra[(expected - extra).topk(remainder).indices] += 1
        token_counts = extra + 1
        if shuffle_counts:
            token_counts = token_counts[torch.randperm(centroids, device=device)]
    counts = torch.div(
        token_counts + summary_page_size - 1,
        summary_page_size,
        rounding_mode="floor",
    )
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.int64, device=device), counts.cumsum(0))
    ).to(torch.int32)
    item_count = int(offsets[-1])
    physical_stride = math.ceil(item_count / 16) * 16
    offsets = offsets.view(1, 1, -1).expand(batch, kv_heads, -1).contiguous()
    counts = offsets[0, 0, 1:] - offsets[0, 0, :-1]
    owners = torch.repeat_interleave(
        torch.arange(centroids, device=device, dtype=torch.int32), counts.long()
    )
    owners = owners.view(1, 1, -1).expand(batch, kv_heads, -1).contiguous()
    indices = torch.empty(
        (batch, kv_heads, item_count), dtype=torch.int32, device=device
    )
    for batch_index in range(batch):
        page_base = batch_index * physical_stride
        for kv_head in range(kv_heads):
            indices[batch_index, kv_head] = (
                torch.randperm(item_count, device=device, dtype=torch.int64) + page_base
            ).to(torch.int32)
    return indices, offsets, owners


def _check_argmax(
    query: torch.Tensor,
    keys: torch.Tensor,
    leaf_indices: torch.Tensor,
    offsets: torch.Tensor,
    winners: torch.Tensor,
    winner_scores: torch.Tensor,
) -> None:
    batch, query_heads, _ = query.shape
    _, kv_heads, centroids, query_group = winners.shape
    if query_heads != kv_heads * query_group:
        raise AssertionError("winner shape does not preserve GQA grouping")
    reference = torch.empty_like(winners)
    reference_scores = torch.empty_like(winner_scores)
    for batch_index in range(batch):
        for kv_head in range(kv_heads):
            q = query[
                batch_index,
                kv_head * query_group : (kv_head + 1) * query_group,
            ].float()
            for centroid in range(centroids):
                begin = int(offsets[batch_index, kv_head, centroid])
                end = int(offsets[batch_index, kv_head, centroid + 1])
                pages = leaf_indices[batch_index, kv_head, begin:end].long()
                scores = q @ keys[pages, kv_head].float().T
                score, position = scores.max(dim=1)
                reference[batch_index, kv_head, centroid] = pages[position].to(
                    torch.int32
                )
                reference_scores[batch_index, kv_head, centroid] = score
    if not torch.equal(winners, reference):
        mismatch = int((winners != reference).sum())
        raise AssertionError(f"centroid argmax has {mismatch} wrong winner indices")
    torch.testing.assert_close(
        winner_scores,
        reference_scores,
        rtol=2e-2,
        atol=2e-1,
    )


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires a ROCm/CUDA GPU")
    from aiter.ops.triton.unified_attention import unified_attention

    if args.query_heads % args.kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if args.summary_page_size < 1:
        raise ValueError("summary page size must be positive")
    if any(context % 16 for context in args.contexts):
        raise ValueError("AITER comparison contexts must be divisible by 16")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    results: list[dict[str, float | int]] = []

    # A small independent parity case compiles the same Qwen head geometry.
    check_context = 512
    check_centroids = 32
    check_query = torch.randn(
        (1, args.query_heads, args.head_dim), device=device, dtype=dtype
    )
    check_indices, check_offsets, check_owners = _centroid_lists(
        1,
        args.kv_heads,
        check_context,
        check_centroids,
        device,
        args.leaf_distribution,
        args.zipf_exponent,
        args.shuffle_counts,
        args.summary_page_size,
    )
    check_item_count = check_indices.size(2)
    check_physical_stride = math.ceil(check_item_count / 16) * 16
    check_keys = torch.randn(
        (check_physical_stride, args.kv_heads, args.head_dim),
        device=device,
        dtype=dtype,
    )
    check_winners, check_scores = centroid_argmax_page1(
        check_query,
        check_keys,
        check_indices,
        check_offsets,
        leaf_owners=check_owners if args.reduce_layout == "stream" else None,
        block_k=args.block_k,
        block_n=args.block_n,
        centroids_per_block=args.centroids_per_block,
    )
    _check_argmax(
        check_query,
        check_keys,
        check_indices,
        check_offsets,
        check_winners,
        check_scores,
    )
    print("centroid argmax parity: passed", flush=True)

    for batch in args.batch_sizes:
        for context in args.contexts:
            centroids = min(context, math.ceil(args.state_multiplier * math.sqrt(context)))
            leaf_indices, centroid_offsets, leaf_owners = _centroid_lists(
                batch,
                args.kv_heads,
                context,
                centroids,
                device,
                args.leaf_distribution,
                args.zipf_exponent,
                args.shuffle_counts,
                args.summary_page_size,
            )
            item_count = leaf_indices.size(2)
            physical_stride = math.ceil(item_count / 16) * 16
            total_tokens = batch * physical_stride
            keys = torch.randn(
                (total_tokens, args.kv_heads, args.head_dim),
                device=device,
                dtype=dtype,
            )
            values = torch.randn_like(keys)
            query = torch.randn(
                (batch, args.query_heads, args.head_dim), device=device, dtype=dtype
            )
            blocks_per_sequence = physical_stride // 16
            key_blocks = keys.view(
                batch * blocks_per_sequence,
                16,
                args.kv_heads,
                args.head_dim,
            )
            value_blocks = values.view_as(key_blocks)
            block_table = (
                torch.arange(
                    batch * blocks_per_sequence, device=device, dtype=torch.int32
                )
                .view(batch, blocks_per_sequence)
                .contiguous()
            )
            seq_lens = torch.full(
                (batch,), item_count, device=device, dtype=torch.int32
            )
            cu_seqlens_q = torch.arange(
                batch + 1, device=device, dtype=torch.int32
            )
            full_output = torch.empty_like(query)
            query_group = args.query_heads // args.kv_heads
            score_workspace = torch.empty(
                (batch, args.kv_heads, item_count, query_group),
                device=device,
                dtype=torch.float32,
            )
            winner_indices = torch.empty(
                (batch, args.kv_heads, centroids, query_group),
                device=device,
                dtype=torch.int32,
            )
            winner_scores = torch.empty(
                (batch, args.kv_heads, centroids, query_group),
                device=device,
                dtype=torch.float32,
            )

            def run_qk():
                return flat_page1_qk_scores(
                    query,
                    keys,
                    leaf_indices,
                    scores=score_workspace,
                    block_k=args.block_k,
                )

            def run_reduce():
                return flat_centroid_argmax_reduce(
                    score_workspace,
                    leaf_indices,
                    centroid_offsets,
                    leaf_owners=(
                        leaf_owners if args.reduce_layout == "stream" else None
                    ),
                    winner_indices=winner_indices,
                    winner_scores=winner_scores,
                    block_n=args.block_n,
                    centroids_per_block=args.centroids_per_block,
                )

            def run_argmax():
                return centroid_argmax_page1(
                    query,
                    keys,
                    leaf_indices,
                    centroid_offsets,
                    leaf_owners=(
                        leaf_owners if args.reduce_layout == "stream" else None
                    ),
                    scores=score_workspace,
                    winner_indices=winner_indices,
                    winner_scores=winner_scores,
                    block_k=args.block_k,
                    block_n=args.block_n,
                    centroids_per_block=args.centroids_per_block,
                )

            def run_aiter():
                return unified_attention(
                    q=query,
                    k=key_blocks,
                    v=value_blocks,
                    out=full_output,
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=1,
                    seqused_k=seq_lens,
                    max_seqlen_k=item_count,
                    softmax_scale=args.head_dim**-0.5,
                    causal=True,
                    window_size=(-1, -1),
                    block_table=block_table,
                    softcap=0.0,
                    q_descale=None,
                    k_descale=None,
                    v_descale=None,
                )

            # Compile both passes before timing either one in isolation.
            run_qk()
            run_reduce()
            torch.cuda.synchronize()
            qk_ms = _elapsed_ms(run_qk, args.warmup, args.iterations)
            reduce_ms = _elapsed_ms(run_reduce, args.warmup, args.iterations)
            argmax_ms = _elapsed_ms(run_argmax, args.warmup, args.iterations)
            aiter_ms = _elapsed_ms(run_aiter, args.warmup, args.iterations)
            row: dict[str, float | int] = {
                "batch": batch,
                "context": context,
                "centroids": centroids,
                "summary_page_size": args.summary_page_size,
                "search_items": item_count,
                "mean_items_per_centroid": item_count / centroids,
                "max_items_per_centroid": int(
                    (centroid_offsets[0, 0, 1:] - centroid_offsets[0, 0, :-1]).max()
                ),
                "winner_table_kib": batch
                * args.query_heads
                * centroids
                * 4
                / 1024,
                "score_workspace_mib": score_workspace.numel()
                * score_workspace.element_size()
                / (1024 * 1024),
                "qk_ms": qk_ms,
                "reduce_ms": reduce_ms,
                "sum_of_passes_ms": qk_ms + reduce_ms,
                "argmax_ms": argmax_ms,
                "aiter_full_ms": aiter_ms,
                "argmax_over_aiter": argmax_ms / aiter_ms,
            }
            results.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            del (
                keys,
                values,
                query,
                leaf_indices,
                centroid_offsets,
                leaf_owners,
                key_blocks,
                value_blocks,
                block_table,
                seq_lens,
                cu_seqlens_q,
                full_output,
                score_workspace,
                winner_indices,
                winner_scores,
            )
            torch.cuda.empty_cache()

    payload = {
        "device": torch.cuda.get_device_name(),
        "query_heads": args.query_heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "state_multiplier": args.state_multiplier,
        "summary_page_size": args.summary_page_size,
        "leaf_distribution": args.leaf_distribution,
        "zipf_exponent": args.zipf_exponent,
        "shuffle_counts": args.shuffle_counts,
        "block_k": args.block_k,
        "block_n": args.block_n,
        "centroids_per_block": args.centroids_per_block,
        "reduce_layout": args.reduce_layout,
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
