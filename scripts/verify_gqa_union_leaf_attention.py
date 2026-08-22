#!/usr/bin/env python3
"""Verify and microbenchmark GQA-union indexed leaf attention."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from model.kernels.gqa_union_leaf_attention import (
    gqa_union_aiter_attention,
    gqa_union_indexed_attention,
)
from model.kernels.lod_kernels import (
    merge_attention_branches_with_aiter_stats,
    merge_attention_branches_with_sink,
    remove_state_slots_from_attention,
)
from model.kernels.paged_leaf_attention import (
    masked_decode_coarse_local_attention,
    new_fused_decode_buffers,
    persistent_decode_route_coarse_gqa,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192, 16384, 32768, 65536])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--union-group-size", type=int)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--block-n", type=int)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--max-slot-leaves", type=int, default=0)
    parser.add_argument("--hot-slot-leaves", type=int, default=0)
    parser.add_argument("--aiter", action="store_true")
    parser.add_argument("--capture-empty-aiter", action="store_true")
    parser.add_argument("--capture-empty-custom", action="store_true")
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


def _metadata(
    batch: int,
    kv_heads: int,
    context: int,
    centroids: int,
    device: torch.device,
    *,
    hot_slots: int = 0,
    hot_slot_leaves: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hot_slot_leaves:
        if hot_slots <= 0 or hot_slots * hot_slot_leaves > context:
            raise ValueError("hot-slot leaf geometry exceeds the context")
        counts = torch.full(
            (centroids,),
            (context - hot_slots * hot_slot_leaves) // (centroids - hot_slots),
            device=device,
            dtype=torch.int64,
        )
        counts[:hot_slots] = hot_slot_leaves
        counts[hot_slots : hot_slots + (context - int(counts.sum()))] += 1
        offsets = torch.cat(
            (torch.zeros(1, device=device, dtype=torch.int64), counts.cumsum(0))
        ).to(torch.int32)
    else:
        offsets = torch.div(
            torch.arange(centroids + 1, device=device, dtype=torch.int64)
            * context,
            centroids,
            rounding_mode="floor",
        ).to(torch.int32)
    offsets = offsets.view(1, 1, -1).expand(batch, kv_heads, -1).contiguous()
    packed = torch.empty(batch, kv_heads, context, dtype=torch.int32, device=device)
    for batch_index in range(batch):
        for kv_head in range(kv_heads):
            packed[batch_index, kv_head] = torch.randperm(
                context, device=device, dtype=torch.int32
            )
    return offsets, packed


def _reference(
    q: torch.Tensor,
    leaf_k: torch.Tensor,
    leaf_v: torch.Tensor,
    top_slots: torch.Tensor,
    offsets: torch.Tensor,
    packed: torch.Tensor,
    scale: float,
    union_group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, query_heads, _, _ = q.shape
    kv_heads = leaf_k.size(1)
    groups = query_heads // kv_heads
    if union_group_size is None:
        union_group_size = groups
    if groups % union_group_size:
        raise ValueError("union group size must divide the physical GQA group")
    route_count = top_slots.size(-1)
    union_count = union_group_size * route_count
    output = torch.empty_like(q)
    lse = torch.empty(batch, query_heads, 1, dtype=torch.float32, device=q.device)
    union_routes = torch.full(
        (batch, query_heads, 1, union_count),
        -1,
        dtype=torch.int64,
        device=q.device,
    )
    for batch_index in range(batch):
        for kv_head in range(kv_heads):
            for subgroup_begin in range(0, groups, union_group_size):
                head_begin = kv_head * groups + subgroup_begin
                head_end = head_begin + union_group_size
                routes = top_slots[batch_index, head_begin:head_end, 0].flatten()
                slots = torch.unique(routes[routes >= 0], sorted=False)
                union_routes[
                    batch_index, head_begin:head_end, 0, : slots.numel()
                ] = slots
                leaves = torch.cat(
                    [
                        packed[
                            batch_index,
                            kv_head,
                            offsets[batch_index, kv_head, slot] : offsets[
                                batch_index, kv_head, slot + 1
                            ],
                        ]
                        for slot in slots
                    ]
                ).long()
                query = q[batch_index, head_begin:head_end, 0].float()
                scores = (
                    query @ leaf_k[batch_index, kv_head, leaves].float().T * scale
                )
                probability = torch.softmax(scores, dim=-1)
                output[batch_index, head_begin:head_end, 0] = (
                    probability.to(leaf_v.dtype)
                    @ leaf_v[batch_index, kv_head, leaves]
                )
                lse[batch_index, head_begin:head_end, 0] = torch.logsumexp(
                    scores, dim=-1
                )
    return output, lse, union_routes


def _verify_capacity_bound(
    *,
    query_heads: int,
    kv_heads: int,
    union_group_size: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Ensure routed leaves plus a local range never exceed physical storage."""
    leaf_capacity = 512
    centroids = 64
    groups = query_heads // kv_heads
    route_count = 8
    q = torch.randn(1, query_heads, 1, head_dim, device=device, dtype=dtype)
    leaf_k = torch.randn(
        1, kv_heads, leaf_capacity, head_dim, device=device, dtype=dtype
    )
    leaf_v = torch.randn_like(leaf_k)
    offsets, packed = _metadata(
        1, kv_heads, leaf_capacity, centroids, device
    )
    top_slots = torch.empty(
        1, query_heads, 1, route_count, device=device, dtype=torch.int64
    )
    for query_head in range(query_heads):
        subgroup_head = query_head % union_group_size
        top_slots[0, query_head, 0] = (
            torch.arange(route_count, device=device)
            + subgroup_head * route_count
        )
    _, _, _, buffers = gqa_union_indexed_attention(
        q,
        leaf_k,
        leaf_v,
        top_slots,
        offsets,
        packed,
        kv_group_size=groups,
        union_group_size=union_group_size,
        scale=head_dim**-0.5,
        local_begin=0,
        local_len=400,
        max_slot_leaves=512,
        block_n=32 if head_dim > 256 else 128,
    )
    torch.cuda.synchronize()
    lengths = buffers["lengths"]
    if int(lengths.max()) != leaf_capacity:
        raise AssertionError(
            f"capacity clamp produced max length {int(lengths.max())}, "
            f"expected {leaf_capacity}"
        )
    packed_rows = buffers["leaf_indices"]
    for logical_group in range(packed_rows.size(1)):
        row = packed_rows[0, logical_group, : int(lengths[0, logical_group])]
        if int(row.min()) < 0 or int(row.max()) >= kv_heads * leaf_capacity:
            raise AssertionError("capacity-clamped row contains an invalid leaf index")
    print("GQA-union physical-capacity bound: passed", flush=True)


def _lse_from_aiter_stats(
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    lengths: torch.Tensor,
    *,
    partition_size: int = 256,
) -> torch.Tensor:
    partition = torch.arange(exp_sums.size(-1), device=exp_sums.device)
    valid = partition[None, None, :] * partition_size < lengths[:, None, None]
    maxima = torch.where(valid, max_logits, -torch.inf)
    maximum = maxima.max(dim=-1).values
    denominator = torch.where(
        valid,
        exp_sums * torch.exp(maxima - maximum.unsqueeze(-1)),
        0.0,
    ).sum(dim=-1)
    return maximum.add(denominator.log())


def _verify_persistent_route_coarse(
    *,
    query_heads: int,
    kv_heads: int,
    route_group_size: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    batch = 2
    state_len = 64
    physical_group_size = query_heads // kv_heads
    scale = head_dim**-0.5
    q = torch.randn(batch, query_heads, 1, head_dim, device=device, dtype=dtype)
    mean_k = torch.randn(
        batch, kv_heads, state_len, head_dim, device=device, dtype=dtype
    )
    mean_v = torch.randn_like(mean_k)
    counts = torch.randint(
        1,
        9,
        (batch, kv_heads, state_len, 1),
        device=device,
        dtype=torch.int32,
    )
    state_k = mean_k * counts.to(dtype)
    state_v = mean_v * counts.to(dtype)
    buffers = new_fused_decode_buffers(
        q,
        splits=1,
        state_capacity=state_len,
    )
    top_slots, _, coarse_out, coarse_lse = persistent_decode_route_coarse_gqa(
        q,
        state_k,
        state_v,
        counts,
        state_len=state_len,
        kv_group_size=physical_group_size,
        route_group_size=route_group_size,
        scale=scale,
        buffers=buffers,
        group_size=128,
    )
    top_slots = top_slots.clone()
    coarse_out = coarse_out.clone()
    coarse_lse = coarse_lse.clone()
    normalized_slots, _, normalized_out, normalized_lse = (
        persistent_decode_route_coarse_gqa(
            q,
            mean_k,
            mean_v,
            counts.float().log(),
            state_len=state_len,
            kv_group_size=physical_group_size,
            route_group_size=route_group_size,
            scale=scale,
            buffers=buffers,
            group_size=128,
            state_is_normalized=True,
        )
    )
    reference_out = torch.empty_like(q)
    reference_lse = torch.empty_like(coarse_lse)
    reference_slots = torch.empty_like(top_slots)
    groups_per_kv = physical_group_size // route_group_size
    for batch_index in range(batch):
        for kv_head in range(kv_heads):
            count = counts[batch_index, kv_head, :, 0].float()
            keys = state_k[batch_index, kv_head].float() / count[:, None]
            values = state_v[batch_index, kv_head].float() / count[:, None]
            for subgroup in range(groups_per_kv):
                first_head = (
                    kv_head * physical_group_size + subgroup * route_group_size
                )
                last_head = first_head + route_group_size
                query = q[batch_index, first_head:last_head, 0].float()
                scores = query @ keys.T * scale + count.log()[None, :]
                routes = scores.topk(8, dim=-1).indices
                reference_slots[batch_index, first_head:last_head, 0] = routes
                union = torch.unique(routes.flatten())
                remainder_scores = scores.clone()
                remainder_scores[:, union] = -torch.inf
                probability = torch.softmax(remainder_scores, dim=-1)
                reference_out[batch_index, first_head:last_head, 0] = (
                    probability.to(dtype) @ values.to(dtype)
                )
                reference_lse[batch_index, first_head:last_head, 0] = (
                    torch.logsumexp(remainder_scores, dim=-1)
                )
    if not torch.equal(
        top_slots.sort(dim=-1).values,
        reference_slots.sort(dim=-1).values,
    ):
        raise AssertionError("persistent GQA top-8 routes differ from reference")
    torch.testing.assert_close(
        coarse_out,
        reference_out,
        rtol=3e-2,
        atol=3e-2,
        check_dtype=False,
    )
    torch.testing.assert_close(coarse_lse, reference_lse, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(
        normalized_out, coarse_out, rtol=3e-2, atol=3e-2, check_dtype=False
    )
    torch.testing.assert_close(
        normalized_lse, coarse_lse, rtol=3e-3, atol=3e-3
    )
    if not torch.equal(normalized_slots, top_slots):
        raise AssertionError("normalized persistent GQA routes differ")
    print("Persistent GQA route/coarse parity: passed", flush=True)


def _verify_masked_coarse(
    *,
    query_heads: int,
    kv_heads: int,
    route_group_size: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    batch = 2
    state_len = 64
    physical_group_size = query_heads // kv_heads
    scale = head_dim**-0.5
    q = torch.randn(batch, query_heads, 1, head_dim, device=device, dtype=dtype)
    mean_k = torch.randn(
        batch, kv_heads, state_len, head_dim, device=device, dtype=dtype
    )
    mean_v = torch.randn_like(mean_k)
    counts = torch.randint(
        1, 9, (batch, kv_heads, state_len, 1), device=device, dtype=torch.int32
    )
    state_k = mean_k * counts.to(dtype)
    state_v = mean_v * counts.to(dtype)
    buffers = new_fused_decode_buffers(
        q, splits=1, state_capacity=state_len, route_group_size=32
    )
    union_count = route_group_size * 8
    top_slots = torch.full(
        (batch, query_heads, 1, union_count),
        -1,
        device=device,
        dtype=torch.int64,
    )
    groups_per_kv = physical_group_size // route_group_size
    for batch_index in range(batch):
        for kv_head in range(kv_heads):
            for subgroup in range(groups_per_kv):
                first_head = (
                    kv_head * physical_group_size + subgroup * route_group_size
                )
                slots = torch.tensor(
                    [subgroup + 1, subgroup + 5, subgroup + 11],
                    device=device,
                )
                top_slots[
                    batch_index,
                    first_head : first_head + route_group_size,
                    0,
                    : slots.numel(),
                ] = slots
    dummy_local = torch.empty(
        batch, kv_heads, 1, head_dim, device=device, dtype=dtype
    )
    output, lse, _, _ = masked_decode_coarse_local_attention(
        q,
        state_k,
        state_v,
        counts,
        dummy_local,
        dummy_local,
        top_slots,
        state_len=state_len,
        local_len=0,
        kv_group_size=physical_group_size,
        scale=scale,
        buffers=buffers,
        group_size=32,
        compute_local=False,
    )
    reference_output = torch.empty_like(q)
    reference_lse = torch.empty_like(lse)
    for batch_index in range(batch):
        for kv_head in range(kv_heads):
            count = counts[batch_index, kv_head, :, 0].float()
            keys = state_k[batch_index, kv_head].float() / count[:, None]
            values = state_v[batch_index, kv_head].float() / count[:, None]
            for subgroup in range(groups_per_kv):
                first_head = (
                    kv_head * physical_group_size + subgroup * route_group_size
                )
                last_head = first_head + route_group_size
                query = q[batch_index, first_head:last_head, 0].float()
                scores = query @ keys.T * scale + count.log()[None, :]
                union = top_slots[batch_index, first_head, 0]
                scores[:, union[union >= 0]] = -torch.inf
                probability = torch.softmax(scores, dim=-1)
                reference_output[batch_index, first_head:last_head, 0] = (
                    probability.to(dtype) @ values.to(dtype)
                )
                reference_lse[batch_index, first_head:last_head, 0] = (
                    torch.logsumexp(scores, dim=-1)
                )
    torch.testing.assert_close(
        output, reference_output, rtol=3e-2, atol=3e-2, check_dtype=False
    )
    torch.testing.assert_close(lse, reference_lse, rtol=3e-3, atol=3e-3)
    print("GQA subgroup-masked coarse parity: passed", flush=True)


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("verification requires a ROCm/CUDA GPU")
    torch.manual_seed(19)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    query_heads = args.query_heads
    kv_heads = args.kv_heads
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    groups = query_heads // kv_heads
    union_group_size = args.union_group_size or groups
    if groups % union_group_size:
        raise ValueError("union group size must divide the physical GQA group")
    head_dim = args.head_dim
    block_n = args.block_n or (32 if head_dim > 256 else 128)
    route_count = 8
    scale = head_dim**-0.5

    _verify_capacity_bound(
        query_heads=query_heads,
        kv_heads=kv_heads,
        union_group_size=union_group_size,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )

    # These legacy route/coarse parity kernels deliberately materialize the
    # whole GQA tile and exceed MI325X LDS at Gemma's 512-wide global heads.
    # They are independent of the AITER exact-leaf path tested below.
    if (
        union_group_size == groups
        and union_group_size in {4, 5, 8, 16}
        and head_dim <= 256
    ):
        _verify_persistent_route_coarse(
            query_heads=query_heads,
            kv_heads=kv_heads,
            route_group_size=union_group_size,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )
    if head_dim <= 256:
        _verify_masked_coarse(
            query_heads=query_heads,
            kv_heads=kv_heads,
            route_group_size=union_group_size,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )

    parity_context = 512
    parity_centroids = 64
    q = torch.randn(1, query_heads, 1, head_dim, device=device, dtype=dtype)
    leaf_k = torch.randn(1, kv_heads, parity_context, head_dim, device=device, dtype=dtype)
    leaf_v = torch.randn_like(leaf_k)
    offsets, packed = _metadata(1, kv_heads, parity_context, parity_centroids, device)
    top_slots = torch.randint(
        parity_centroids,
        (1, query_heads, 1, route_count),
        device=device,
        dtype=torch.int64,
    )
    reference_output, reference_lse, reference_routes = _reference(
        q,
        leaf_k,
        leaf_v,
        top_slots,
        offsets,
        packed,
        scale,
        union_group_size,
    )
    output, lse, union_routes, _ = gqa_union_indexed_attention(
        q,
        leaf_k,
        leaf_v,
        top_slots,
        offsets,
        packed,
        kv_group_size=groups,
        union_group_size=union_group_size,
        scale=scale,
        max_slot_leaves=args.max_slot_leaves,
        block_n=block_n,
        num_warps=args.num_warps,
    )
    if head_dim > 256:
        _["output"].zero_()
        _["lse"].zero_()
        output, lse, union_routes, _ = gqa_union_indexed_attention(
            q,
            leaf_k,
            leaf_v,
            top_slots,
            offsets,
            packed,
            kv_group_size=groups,
            union_group_size=union_group_size,
            scale=scale,
            buffers=_,
            max_slot_leaves=args.max_slot_leaves,
            block_n=block_n,
            num_warps=args.num_warps,
        )
    try:
        torch.testing.assert_close(output, reference_output, rtol=2e-2, atol=2e-2)
    except AssertionError:
        head_errors = (
            output.float().sub(reference_output.float()).abs().amax(dim=(0, 2, 3))
        )
        print(
            "GQA-union per-head max errors:", head_errors.cpu().tolist(),
            flush=True,
        )
        print(
            "GQA-union output max magnitudes:",
            output.float().abs().amax(dim=(0, 2, 3)).cpu().tolist(),
            flush=True,
        )
        print(
            "GQA-union reference max magnitudes:",
            reference_output.float().abs().amax(dim=(0, 2, 3)).cpu().tolist(),
            flush=True,
        )
        print(
            "GQA-union packed lengths:",
            _.get("lengths", torch.empty(0, device=device)).cpu().tolist(),
            flush=True,
        )
        packed_rows = _.get("leaf_indices")
        if packed_rows is not None:
            packed_ranges = []
            for logical_group, length in enumerate(_["lengths"][0].tolist()):
                row = packed_rows[0, logical_group, :length]
                packed_ranges.append(
                    (int(row.min().item()), int(row.max().item()))
                    if row.numel()
                    else None
                )
            print("GQA-union packed index ranges:", packed_ranges, flush=True)
            flat_k = leaf_k.view(-1, head_dim).float()
            flat_v = leaf_v.view(-1, head_dim).float()
            packed_reference = torch.empty_like(output)
            for logical_group, length in enumerate(_["lengths"][0].tolist()):
                indices = packed_rows[0, logical_group, :length].long()
                first_head = logical_group * union_group_size
                last_head = first_head + union_group_size
                scores = q[0, first_head:last_head, 0].float() @ flat_k[indices].T
                weights = torch.softmax(scores * scale, dim=-1)
                packed_reference[0, first_head:last_head, 0] = (
                    weights @ flat_v[indices]
                ).to(dtype)
            packed_errors = (
                packed_reference.float()
                .sub(reference_output.float())
                .abs()
                .amax(dim=(0, 2, 3))
            )
            print(
                "GQA-union packed-reference errors:",
                packed_errors.cpu().tolist(),
                flush=True,
            )
        print(
            "GQA-union route parity:",
            torch.equal(
                union_routes.sort(dim=-1).values,
                reference_routes.sort(dim=-1).values,
            ),
            flush=True,
        )
        for head in range(query_heads):
            actual = union_routes[0, head, 0].sort().values
            expected = reference_routes[0, head, 0].sort().values
            if not torch.equal(actual, expected):
                print(
                    "GQA-union first route mismatch:",
                    head,
                    union_routes[0, head, 0].cpu().tolist(),
                    reference_routes[0, head, 0].cpu().tolist(),
                    flush=True,
                )
                break
        raise
    torch.testing.assert_close(lse, reference_lse, rtol=2e-3, atol=2e-3)
    if not torch.equal(
        union_routes.sort(dim=-1).values,
        reference_routes.sort(dim=-1).values,
    ):
        raise AssertionError("GQA union route set differs from reference")
    print("GQA-union parity: passed", flush=True)

    if args.capture_empty_custom:
        empty_slots = torch.full_like(top_slots, -1)
        empty_offsets = torch.zeros_like(offsets)
        empty_buffers = None

        def empty_custom():
            nonlocal empty_buffers
            result = gqa_union_indexed_attention(
                q,
                leaf_k,
                leaf_v,
                empty_slots,
                empty_offsets,
                packed,
                kv_group_size=groups,
                union_group_size=union_group_size,
                scale=scale,
                buffers=empty_buffers,
                block_n=block_n,
                num_warps=args.num_warps,
            )
            empty_buffers = result[3]
            return result

        empty_custom()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            empty_result = empty_custom()
        graph.replay()
        torch.cuda.synchronize()
        if torch.count_nonzero(empty_result[3]["lengths"]).item():
            raise AssertionError("captured empty custom rows lost their true length")
        print("Custom empty-row graph capture: passed", flush=True)

    if args.aiter:
        aiter_reference_output, aiter_reference_lse, aiter_reference_routes = (
            _reference(
                q,
                leaf_k,
                leaf_v,
                top_slots,
                offsets,
                packed,
                scale,
                union_group_size,
            )
        )
        (
            aiter_output,
            exp_sums,
            max_logits,
            lengths,
            aiter_routes,
            aiter_buffers,
        ) = gqa_union_aiter_attention(
            q,
            leaf_k,
            leaf_v,
            top_slots,
            offsets,
            packed,
            kv_group_size=groups,
            union_group_size=union_group_size,
            scale=scale,
        )
        aiter_lse = _lse_from_aiter_stats(
            exp_sums, max_logits, lengths
        ).view(1, query_heads, 1)
        torch.testing.assert_close(
            aiter_output, aiter_reference_output, rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            aiter_lse, aiter_reference_lse, rtol=2e-3, atol=2e-3
        )
        if not torch.equal(
            aiter_routes.sort(dim=-1).values,
            aiter_reference_routes.sort(dim=-1).values,
        ):
            raise AssertionError("AITER GQA union route set differs from reference")
        primary_output = torch.randn_like(q)
        primary_lse = torch.randn_like(reference_lse)
        tertiary_output = torch.randn_like(q)
        tertiary_lse = torch.randn_like(reference_lse)
        sink_k = torch.randn(
            1, kv_heads, 2, head_dim, device=device, dtype=dtype
        )
        sink_v = torch.randn_like(sink_k)
        reference_merged = merge_attention_branches_with_sink(
            q,
            sink_k,
            sink_v,
            primary_output,
            primary_lse,
            aiter_output,
            aiter_lse,
            tertiary_output,
            tertiary_lse,
            kv_group_size=groups,
            scale=scale,
        )
        fused_merged = merge_attention_branches_with_aiter_stats(
            q,
            primary_output,
            primary_lse,
            aiter_output,
            exp_sums,
            max_logits,
            lengths,
            tertiary_output,
            tertiary_lse,
            kv_group_size=groups,
            exact_group_size=union_group_size,
            scale=scale,
            sink_k=sink_k,
            sink_v=sink_v,
        )
        torch.testing.assert_close(
            fused_merged, reference_merged, rtol=2e-2, atol=2e-2
        )
        print("AITER-stat fused merge parity: passed", flush=True)

        (
            _,
            stage1_exp_sums,
            stage1_max_logits,
            stage1_lengths,
            _,
            stage1_buffers,
        ) = gqa_union_aiter_attention(
            q,
            leaf_k,
            leaf_v,
            top_slots,
            offsets,
            packed,
            kv_group_size=groups,
            union_group_size=union_group_size,
            scale=scale,
            buffers=aiter_buffers,
            stage1_only=True,
        )
        stage1_merged = merge_attention_branches_with_aiter_stats(
            q,
            primary_output,
            primary_lse,
            aiter_output,
            stage1_exp_sums,
            stage1_max_logits,
            stage1_lengths,
            tertiary_output,
            tertiary_lse,
            exact_partial_out=stage1_buffers["partial_output"],
            kv_group_size=groups,
            exact_group_size=union_group_size,
            scale=scale,
            sink_k=sink_k,
            sink_v=sink_v,
        )
        torch.testing.assert_close(
            stage1_merged, reference_merged, rtol=2e-2, atol=2e-2
        )
        print("AITER stage-one fused reduction parity: passed", flush=True)

        group_count = 5
        primary_group_output = torch.randn(
            1,
            query_heads,
            group_count,
            head_dim,
            device=device,
            dtype=torch.float32,
        )
        primary_group_lse = torch.randn(
            1,
            query_heads,
            group_count,
            device=device,
            dtype=torch.float32,
        )
        group_weight = torch.softmax(primary_group_lse, dim=-1)
        reduced_primary_output = torch.sum(
            group_weight[..., None] * primary_group_output, dim=2
        ).unsqueeze(2).to(dtype)
        reduced_primary_lse = torch.logsumexp(
            primary_group_lse, dim=-1
        ).unsqueeze(2)
        reference_group_merged = merge_attention_branches_with_aiter_stats(
            q,
            reduced_primary_output,
            reduced_primary_lse,
            aiter_output,
            exp_sums,
            max_logits,
            lengths,
            tertiary_output,
            tertiary_lse,
            kv_group_size=groups,
            exact_group_size=union_group_size,
            scale=scale,
            sink_k=sink_k,
            sink_v=sink_v,
        )
        fused_group_merged = merge_attention_branches_with_aiter_stats(
            q,
            primary_output,
            primary_lse,
            aiter_output,
            exp_sums,
            max_logits,
            lengths,
            tertiary_output,
            tertiary_lse,
            kv_group_size=groups,
            exact_group_size=union_group_size,
            scale=scale,
            sink_k=sink_k,
            sink_v=sink_v,
            primary_group_out=primary_group_output,
            primary_group_lse=primary_group_lse,
            active_primary_groups=group_count,
        )
        torch.testing.assert_close(
            fused_group_merged, reference_group_merged, rtol=2e-2, atol=2e-2
        )
        print("AITER-stat fused primary-group reduction parity: passed", flush=True)

        state_count = torch.randint(
            1,
            9,
            (1, kv_heads, parity_centroids, 1),
            device=device,
            dtype=torch.int32,
        )
        state_mean_k = torch.randn(
            1,
            kv_heads,
            parity_centroids,
            head_dim,
            device=device,
            dtype=dtype,
        )
        state_mean_v = torch.randn_like(state_mean_k)
        state_k = state_mean_k * state_count.to(dtype)
        state_v = state_mean_v * state_count.to(dtype)
        primary_output = torch.empty_like(q)
        primary_lse = torch.empty_like(reference_lse)
        for query_head in range(query_heads):
            kv_head = query_head // groups
            count = state_count[0, kv_head, :, 0].float()
            scores = (
                q[0, query_head, 0].float()
                @ state_mean_k[0, kv_head].float().T
                * scale
                + count.log()
            )
            probability = torch.softmax(scores, dim=-1)
            primary_output[0, query_head, 0] = (
                probability.to(dtype) @ state_mean_v[0, kv_head]
            )
            primary_lse[0, query_head, 0] = torch.logsumexp(scores, dim=-1)
        corrected_output, corrected_lse = remove_state_slots_from_attention(
            q,
            state_k,
            state_v,
            state_count,
            aiter_routes,
            primary_output.clone(),
            primary_lse.clone(),
            kv_group_size=groups,
            route_group_size=union_group_size,
            scale=scale,
            gqa_aware=head_dim <= 256,
        )
        normalized_output, normalized_lse = remove_state_slots_from_attention(
            q,
            state_mean_k,
            state_mean_v,
            state_count.float().log(),
            aiter_routes,
            primary_output.clone(),
            primary_lse.clone(),
            kv_group_size=groups,
            route_group_size=union_group_size,
            scale=scale,
            gqa_aware=head_dim <= 256,
            state_is_normalized=True,
        )
        torch.testing.assert_close(
            normalized_output, corrected_output, rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            # The legacy path reconstructs means after a BF16 sum/count
            # round-trip; the normalized cache keeps the original BF16 mean.
            normalized_lse,
            corrected_lse,
            rtol=1e-3,
            atol=1e-3,
        )
        reference_corrected = merge_attention_branches_with_aiter_stats(
            q,
            corrected_output,
            corrected_lse,
            aiter_output,
            exp_sums,
            max_logits,
            lengths,
            tertiary_output,
            tertiary_lse,
            kv_group_size=groups,
            exact_group_size=union_group_size,
            scale=scale,
            sink_k=sink_k,
            sink_v=sink_v,
        )
        fused_corrected = merge_attention_branches_with_aiter_stats(
            q,
            primary_output,
            primary_lse,
            aiter_output,
            exp_sums,
            max_logits,
            lengths,
            tertiary_output,
            tertiary_lse,
            kv_group_size=groups,
            exact_group_size=union_group_size,
            scale=scale,
            sink_k=sink_k,
            sink_v=sink_v,
            state_k=state_k,
            state_v=state_v,
            counts=state_count,
            union_top_slots=aiter_routes,
        )
        torch.testing.assert_close(
            fused_corrected, reference_corrected, rtol=2e-2, atol=2e-2
        )
        normalized_corrected = merge_attention_branches_with_aiter_stats(
            q,
            primary_output,
            primary_lse,
            aiter_output,
            exp_sums,
            max_logits,
            lengths,
            tertiary_output,
            tertiary_lse,
            kv_group_size=groups,
            exact_group_size=union_group_size,
            scale=scale,
            sink_k=sink_k,
            sink_v=sink_v,
            state_k=state_mean_k,
            state_v=state_mean_v,
            counts=state_count.float().log(),
            union_top_slots=aiter_routes,
            state_is_normalized=True,
        )
        torch.testing.assert_close(
            normalized_corrected, fused_corrected, rtol=2e-2, atol=2e-2
        )
        print("AITER-stat fused correction parity: passed", flush=True)

        if args.capture_empty_aiter:
            empty_slots = torch.full_like(top_slots, -1)
            empty_offsets = torch.zeros_like(offsets)
            empty_result = gqa_union_aiter_attention(
                q,
                leaf_k,
                leaf_v,
                empty_slots,
                empty_offsets,
                packed,
                kv_group_size=groups,
                union_group_size=union_group_size,
                scale=scale,
                buffers=aiter_buffers,
            )
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                empty_result = gqa_union_aiter_attention(
                    q,
                    leaf_k,
                    leaf_v,
                    empty_slots,
                    empty_offsets,
                    packed,
                    kv_group_size=groups,
                    union_group_size=union_group_size,
                    scale=scale,
                    buffers=aiter_buffers,
                )
            graph.replay()
            torch.cuda.synchronize()
            if torch.count_nonzero(empty_result[3]).item():
                raise AssertionError("captured empty AITER rows lost their true length")
            print("AITER empty-row graph capture: passed", flush=True)

    results: list[dict[str, float | int]] = []
    for batch in args.batch_sizes:
        for context in args.contexts:
            centroids = math.ceil(16 * math.sqrt(context))
            q = torch.randn(
                batch, query_heads, 1, head_dim, device=device, dtype=dtype
            )
            leaf_k = torch.randn(
                batch, kv_heads, context, head_dim, device=device, dtype=dtype
            )
            leaf_v = torch.randn_like(leaf_k)
            offsets, packed = _metadata(
                batch,
                kv_heads,
                context,
                centroids,
                device,
                hot_slots=groups * route_count,
                hot_slot_leaves=args.hot_slot_leaves,
            )
            # Use disjoint routes within a GQA group to benchmark its maximum union.
            top_slots = torch.empty(
                batch,
                query_heads,
                1,
                route_count,
                dtype=torch.int64,
                device=device,
            )
            for query_head in range(query_heads):
                group_head = query_head % groups
                top_slots[:, query_head, 0] = (
                    torch.arange(route_count, device=device)
                    + group_head * route_count
                )
            buffers = None

            def run():
                nonlocal buffers
                output, lse, union_routes, buffers = gqa_union_indexed_attention(
                    q,
                    leaf_k,
                    leaf_v,
                    top_slots,
                    offsets,
                    packed,
                    kv_group_size=groups,
                    union_group_size=union_group_size,
                    scale=scale,
                    buffers=buffers,
                    max_slot_leaves=args.max_slot_leaves,
                    block_n=block_n,
                    num_warps=args.num_warps,
                )
                return output, lse, union_routes

            run()
            elapsed_ms = _elapsed_ms(run, args.warmup, args.iterations)
            union_lengths = buffers["lengths"]
            row: dict[str, float | int] = {
                "batch": batch,
                "context": context,
                "centroids": centroids,
                "max_union_centroids": groups * route_count,
                "mean_union_leaves": float(union_lengths.float().mean()),
                "max_union_leaves": int(union_lengths.max()),
                "elapsed_ms": elapsed_ms,
            }
            results.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    payload = {"device": torch.cuda.get_device_name(), "results": results}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
