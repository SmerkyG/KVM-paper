#!/usr/bin/env python3
"""Check top-8 GQA-union decode against the ordinary two-tier path."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from model.kernels.paged_leaf_attention import (
    fused_decode_paged_lod_attention,
    materialize_page1_coarse_means,
    materialize_page1_fixed_indices,
    new_fused_decode_buffers,
)


def elapsed_ms(call, repeats: int) -> float:
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        call()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end)) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--gqa", type=int, default=16)
    parser.add_argument("--head-dim", type=int, choices=(128, 256), default=128)
    parser.add_argument("--state-len", type=int, default=256)
    parser.add_argument(
        "--route-group-size", type=int, choices=(8, 16, 32, 64), default=None
    )
    parser.add_argument(
        "--route-segment-tiles", type=int, choices=(1, 2, 3, 4), default=None
    )
    parser.add_argument("--route-num-warps", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument(
        "--route-reduce-num-warps", type=int, choices=(1, 2, 4, 8), default=2
    )
    parser.add_argument("--splits", type=int, choices=(8, 16, 32), default=8)
    parser.add_argument(
        "--union-final",
        choices=("unified", "subtract"),
        default="unified",
    )
    parser.add_argument("--equal-leaves", action="store_true")
    parser.add_argument("--leaves-per-slot", type=int, default=0)
    parser.add_argument(
        "--profile-kernels",
        action="store_true",
        help="Profile repeated complete calls and report device time by kernel.",
    )
    parser.add_argument("--cuda-graph-timing", action="store_true")
    parser.add_argument("--max-open-leaves", type=int, default=0)
    parser.add_argument("--open-count", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--mass-fraction", type=float, default=None)
    parser.add_argument("--predicted-mass", action="store_true")
    parser.add_argument("--hip-union", action="store_true")
    parser.add_argument("--centroid-major-hip", action="store_true")
    parser.add_argument("--unified-arena", action="store_true")
    parser.add_argument("--group64-padded", action="store_true")
    parser.add_argument("--staged-fixed-aiter", action="store_true")
    parser.add_argument("--fixed-mask-aiter", action="store_true")
    parser.add_argument(
        "--fixed-mask-block-n", type=int, choices=(16, 64, 128), default=64
    )
    parser.add_argument(
        "--fixed-mask-segments",
        type=int,
        choices=(32, 64, 128, 256, 512),
        default=128,
    )
    parser.add_argument("--fixed-mask-adaptive-segments", action="store_true")
    parser.add_argument(
        "--fixed-mask-reduce-block-d",
        type=int,
        choices=(0, 16, 32, 64, 128),
        default=0,
    )
    parser.add_argument("--fixed-mask-direct-routes", action="store_true")
    parser.add_argument("--inject-oversized-centroid", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(7)
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16
    batch = args.batch_size
    cache_batch = max(3, batch)
    kv_heads, gqa, head_dim = args.kv_heads, args.gqa, args.head_dim
    if kv_heads < 1 or not 1 < gqa <= 16:
        raise ValueError("positive KV heads and a GQA group in [2, 16] required")
    query_heads = kv_heads * gqa
    state_len = args.state_len
    page_size = 16
    max_leaves = 31
    leaf_capacity = state_len * max_leaves
    page_capacity = state_len * math.ceil(max_leaves / page_size) + state_len
    local_limit, local_capacity = 512, 576
    active_local = 173
    splits = args.splits
    segmented_route = gqa == 16 and head_dim == 128
    route_group_size = (
        args.route_group_size
        if args.route_group_size is not None
        else (64 if segmented_route else 32)
    )
    route_segment_tiles = (
        args.route_segment_tiles
        if args.route_segment_tiles is not None
        else (2 if segmented_route else 1)
    )

    q = torch.randn(
        batch, query_heads, 1, head_dim, device=device, dtype=dtype
    )
    cache_indices = torch.roll(
        torch.arange(batch, device=device, dtype=torch.int64), shifts=1
    )
    if not 0 <= args.leaves_per_slot <= max_leaves:
        raise ValueError("--leaves-per-slot must be between zero and 31")
    counts_i32 = (
        torch.full(
            (cache_batch, kv_heads, state_len),
            args.leaves_per_slot or 16,
            device=device,
            dtype=torch.int32,
        )
        if args.equal_leaves or args.leaves_per_slot
        else torch.randint(
            1,
            max_leaves + 1,
            (cache_batch, kv_heads, state_len),
            device=device,
            dtype=torch.int32,
        )
    )
    if args.inject_oversized_centroid:
        if args.max_open_leaves <= 0:
            raise ValueError(
                "--inject-oversized-centroid requires --max-open-leaves"
            )
        for row in cache_indices.tolist():
            counts_i32[row, :, 0] = args.max_open_leaves
    counts = counts_i32[..., None].float()
    mean_k = torch.randn(
        cache_batch, kv_heads, state_len, head_dim, device=device, dtype=dtype
    )
    mean_v = torch.randn_like(mean_k)
    if args.inject_oversized_centroid:
        # Make slot zero overwhelmingly attractive before the count guard. The
        # verification below proves it remains coarse instead of entering any
        # head's top-eight route set.
        for batch_row, cache_row in enumerate(cache_indices.tolist()):
            for kv_head in range(kv_heads):
                mean_k[cache_row, kv_head, 0] = (
                    q[batch_row, kv_head * gqa, 0] * 128.0
                )
    state_k = (mean_k.float() * counts).to(dtype)
    state_v = (mean_v.float() * counts).to(dtype)

    leaf_k = torch.zeros(
        cache_batch, kv_heads, leaf_capacity, head_dim, device=device, dtype=dtype
    )
    leaf_v = torch.zeros_like(leaf_k)
    page_indices = torch.full(
        (cache_batch, kv_heads, page_capacity, page_size),
        -1,
        device=device,
        dtype=torch.int32,
    )
    # Paged-directory mode: a root contains a directory row ID, whose 64
    # entries hold physical page IDs. This also tests underfull final pages.
    slot_pages = torch.full(
        (cache_batch, kv_heads, state_len, 1),
        -1,
        device=device,
        dtype=torch.int32,
    )
    overflow_page_keys = torch.full(
        (cache_batch, kv_heads, 1), -1, device=device, dtype=torch.int32
    )
    overflow_page_values = torch.full(
        (cache_batch, kv_heads, page_capacity, 64),
        -1,
        device=device,
        dtype=torch.int32,
    )
    overflow_used = torch.zeros((), device=device, dtype=torch.int32)
    slot_lengths = counts_i32.clone()
    if args.equal_leaves:
        slots = torch.arange(state_len, device=device, dtype=torch.int32)
        slot_pages[..., 0] = slots
        overflow_page_values[..., :state_len, 0] = slots
        page_indices[..., :state_len, :] = torch.arange(
            state_len * page_size, device=device, dtype=torch.int32
        ).reshape(state_len, page_size)
        leaf_k[..., : state_len * page_size, :] = torch.randn(
            cache_batch,
            kv_heads,
            state_len * page_size,
            head_dim,
            device=device,
            dtype=dtype,
        )
        leaf_v[..., : state_len * page_size, :] = torch.randn(
            cache_batch,
            kv_heads,
            state_len * page_size,
            head_dim,
            device=device,
            dtype=dtype,
        )
    else:
        for cache_row in range(cache_batch):
            for kv_head in range(kv_heads):
                next_leaf = 0
                next_page = 0
                for slot in range(state_len):
                    count = int(counts_i32[cache_row, kv_head, slot].item())
                    slot_pages[cache_row, kv_head, slot, 0] = slot
                    for page_ordinal in range(math.ceil(count / page_size)):
                        page = next_page
                        next_page += 1
                        overflow_page_values[
                            cache_row, kv_head, slot, page_ordinal
                        ] = page
                        begin = page_ordinal * page_size
                        stop = min(begin + page_size, count)
                        width = stop - begin
                        leaves = torch.arange(
                            next_leaf,
                            next_leaf + width,
                            device=device,
                            dtype=torch.int32,
                        )
                        page_indices[
                            cache_row, kv_head, page, :width
                        ] = leaves
                        leaf_k[
                            cache_row, kv_head, next_leaf : next_leaf + width
                        ] = torch.randn(
                            width, head_dim, device=device, dtype=dtype
                        )
                        leaf_v[
                            cache_row, kv_head, next_leaf : next_leaf + width
                        ] = torch.randn(
                            width, head_dim, device=device, dtype=dtype
                        )
                        next_leaf += width

    local_k = torch.randn(
        cache_batch, kv_heads, local_capacity, head_dim, device=device, dtype=dtype
    )
    local_v = torch.randn_like(local_k)
    local_lens = torch.full(
        (cache_batch,), active_local, device=device, dtype=torch.int32
    )
    arena_k = None
    arena_v = None
    arena_bias = None
    arena_leaf_offset = 0
    kv_rows = cache_batch * kv_heads
    arena_local_offset = kv_rows * leaf_capacity
    arena_sink_offset = arena_local_offset + kv_rows * local_capacity
    arena_coarse_offset = arena_sink_offset
    arena_padding_index = -1
    fixed_indices = None
    fixed_leaf_owners = None
    fixed_slot_offsets = None
    fixed_lengths = None
    if args.unified_arena:
        if not args.hip_union:
            raise ValueError("--unified-arena requires --hip-union")
        arena_capacity = (
            arena_coarse_offset
            + kv_rows * state_len
            + int(args.group64_padded)
        )
        arena_k = torch.empty(
            arena_capacity,
            head_dim,
            device=device,
            dtype=dtype,
        )
        arena_v = torch.empty_like(arena_k)
        arena_bias = torch.zeros(
            arena_capacity,
            device=device,
            dtype=torch.float16,
        )
        arena_k[
            arena_leaf_offset : arena_leaf_offset + kv_rows * leaf_capacity
        ].view_as(leaf_k).copy_(leaf_k)
        arena_v[
            arena_leaf_offset : arena_leaf_offset + kv_rows * leaf_capacity
        ].view_as(leaf_v).copy_(leaf_v)
        arena_k[
            arena_local_offset : arena_local_offset + kv_rows * local_capacity
        ].view_as(local_k).copy_(local_k)
        arena_v[
            arena_local_offset : arena_local_offset + kv_rows * local_capacity
        ].view_as(local_v).copy_(local_v)
        arena_coarse_k = arena_k[
            arena_coarse_offset : arena_coarse_offset + kv_rows * state_len
        ].view_as(mean_k)
        arena_coarse_v = arena_v[
            arena_coarse_offset : arena_coarse_offset + kv_rows * state_len
        ].view_as(mean_v)
        arena_coarse_bias = arena_bias[
            arena_coarse_offset : arena_coarse_offset + kv_rows * state_len
        ].view(cache_batch, kv_heads, state_len)
        materialize_page1_coarse_means(
            state_k,
            state_v,
            counts,
            arena_coarse_k,
            arena_coarse_v,
            arena_coarse_bias,
        )
        if args.group64_padded:
            arena_padding_index = arena_capacity - 1
            arena_k[arena_padding_index].zero_()
            arena_v[arena_padding_index].zero_()
            arena_bias[arena_padding_index] = -float("inf")
        if args.fixed_mask_aiter:
            if args.staged_fixed_aiter:
                raise ValueError(
                    "--fixed-mask-aiter and --staged-fixed-aiter are exclusive"
                )
            fixed_capacity = leaf_capacity + local_limit + state_len
            fixed_indices = torch.empty(
                cache_batch,
                kv_heads,
                fixed_capacity,
                device=device,
                dtype=torch.int32,
            )
            fixed_leaf_owners = torch.empty(
                cache_batch,
                kv_heads,
                leaf_capacity,
                device=device,
                dtype=torch.int32,
            )
            fixed_slot_offsets = torch.empty(
                cache_batch,
                kv_heads,
                state_len + 1,
                device=device,
                dtype=torch.int32,
            )
            fixed_lengths = torch.empty(
                cache_batch,
                kv_heads,
                device=device,
                dtype=torch.int32,
            )
            materialize_page1_fixed_indices(
                page_indices,
                slot_pages,
                overflow_page_keys,
                overflow_page_values,
                overflow_used,
                slot_lengths,
                fixed_indices,
                fixed_leaf_owners,
                fixed_slot_offsets,
                fixed_lengths,
                row_offset=0,
                arena_leaf_offset=arena_leaf_offset,
                arena_local_offset=arena_local_offset,
                arena_sink_offset=arena_sink_offset,
                arena_coarse_offset=arena_coarse_offset,
                local_capacity=local_capacity,
                local_limit=local_limit,
                sink_capacity=0,
                sink_len=0,
                hash_probes=-1,
            )

    common = dict(
        state_len=state_len,
        local_len=local_limit,
        cache_indices=cache_indices,
        local_lens=local_lens,
        kv_group_size=gqa,
        scale=head_dim**-0.5,
        hash_probes=-1,
        block_n=16,
        num_warps=2,
        waves_per_eu=1,
        split_kv=splits,
        use_dot=False,
        fuse_state_route=True,
        route_group_size=route_group_size,
        route_segment_tiles=route_segment_tiles,
        route_num_warps=args.route_num_warps,
        route_reduce_num_warps=args.route_reduce_num_warps,
        route_parallel_reduce=segmented_route,
        route_use_dot=True,
        route_gqa_grouped=True,
        max_leaf_tokens=args.max_open_leaves or None,
        open_count=args.open_count,
        fuse_final_reduce=False,
        flat_page_indices=page_indices,
    )

    baseline_buffers = new_fused_decode_buffers(
        q,
        splits=splits,
        state_capacity=state_len,
        route_group_size=route_group_size,
        route_segment_tiles=route_segment_tiles,
    )
    union_buffers = new_fused_decode_buffers(
        q,
        splits=splits,
        state_capacity=state_len,
        route_group_size=route_group_size,
        route_segment_tiles=route_segment_tiles,
        gqa_union_mass_fraction=args.mass_fraction,
        gqa_union_predicted_mass=args.predicted_mass,
        gqa_union_kv_heads=kv_heads,
        gqa_union_index_capacity=(
            leaf_capacity + local_limit + 1 + (state_len if args.unified_arena else 0)
        ),
        gqa_union_hip=args.hip_union,
        gqa_union_fixed_mask=args.fixed_mask_aiter,
        gqa_union_fixed_mask_tile_size=args.fixed_mask_block_n,
        gqa_union_fixed_mask_segments=args.fixed_mask_segments,
    )
    triton_union_buffers = (
        new_fused_decode_buffers(
            q,
            splits=splits,
            state_capacity=state_len,
            route_group_size=route_group_size,
            route_segment_tiles=route_segment_tiles,
            gqa_union_mass_fraction=args.mass_fraction,
            gqa_union_kv_heads=kv_heads,
            gqa_union_index_capacity=leaf_capacity + local_limit + 1,
        )
        if (
            args.hip_union
            and not args.predicted_mass
            and (gqa & (gqa - 1)) == 0
        )
        else None
    )

    previous_total_lse = torch.full(
        (cache_batch, query_heads),
        float("inf"),
        dtype=torch.float32,
        device=device,
    )

    def run(buffers, union: bool, timing_events=None, hip=None):
        use_hip = args.hip_union if hip is None else hip
        return fused_decode_paged_lod_attention(
            q,
            state_k,
            state_v,
            counts,
            local_k,
            local_v,
            leaf_k,
            leaf_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            None,
            buffers=buffers,
            gqa_union_decode=union,
            gqa_union_unified=args.union_final == "unified",
            gqa_union_mass_fraction=(args.mass_fraction if union else None),
            gqa_union_predicted_mass=(args.predicted_mass if union else False),
            gqa_union_hip=(use_hip if union else False),
            route_centroid_major_hip=(
                args.centroid_major_hip if union else False
            ),
            gqa_union_group64_padded=(args.group64_padded if union else False),
            gqa_union_staged_fixed_aiter=(
                args.staged_fixed_aiter if union else False
            ),
            gqa_union_fixed_mask_aiter=(
                args.fixed_mask_aiter if union else False
            ),
            gqa_union_fixed_mask_tile_size=args.fixed_mask_block_n,
            gqa_union_fixed_mask_adaptive_segments=(
                args.fixed_mask_adaptive_segments if union else False
            ),
            gqa_union_fixed_mask_reduce_block_d=(
                args.fixed_mask_reduce_block_d if union else 0
            ),
            gqa_union_fixed_mask_direct_routes=(
                args.fixed_mask_direct_routes if union else False
            ),
            gqa_union_page1_k=(arena_k if union and args.unified_arena else None),
            gqa_union_page1_v=(arena_v if union and args.unified_arena else None),
            gqa_union_page1_bias=(
                arena_bias if union and args.unified_arena else None
            ),
            gqa_union_page1_leaf_offset=arena_leaf_offset,
            gqa_union_page1_local_offset=arena_local_offset,
            gqa_union_page1_sink_offset=arena_sink_offset,
            gqa_union_page1_coarse_offset=arena_coarse_offset,
            gqa_union_page1_padding_index=arena_padding_index,
            gqa_union_fixed_indices=(fixed_indices if union else None),
            gqa_union_fixed_leaf_owners=(
                fixed_leaf_owners if union else None
            ),
            gqa_union_fixed_slot_offsets=(
                fixed_slot_offsets if union else None
            ),
            gqa_union_fixed_lengths=(fixed_lengths if union else None),
            gqa_union_previous_total_lse=(
                previous_total_lse if union and args.predicted_mass else None
            ),
            timing_events=timing_events,
            **common,
        )

    baseline = run(baseline_buffers, False).clone()
    triton_union = (
        run(triton_union_buffers, True, hip=False).clone()
        if triton_union_buffers is not None
        else None
    )
    union = run(union_buffers, True).clone()
    torch.cuda.synchronize()
    initial_union_counts = union_buffers["gqa_union_counts"][
        : batch * kv_heads
    ].clone()
    initial_union_epochs = union_buffers["gqa_union_epochs"][
        : batch * kv_heads
    ].clone()
    # The fixed predicted path prepares its queue for the next token in the
    # final reducer. Stamps describing the output just produced therefore
    # belong to the immediately preceding epoch.
    output_route_epochs = (
        initial_union_epochs - 1
        if args.predicted_mass and args.fixed_mask_aiter
        else initial_union_epochs
    )
    initial_seen_stamps = union_buffers["gqa_union_seen_stamps"][
        : batch * kv_heads
    ].clone()
    oversized_route_excluded = None
    if args.inject_oversized_centroid:
        if args.mass_fraction is None:
            oversized_route_excluded = bool(
                (union_buffers["route_top_slots"] != 0).all().item()
            )
        else:
            oversized_route_excluded = all(
                int(union_buffers["gqa_union_seen_stamps"][sequence, 0].item())
                != int(output_route_epochs[sequence].item())
                for sequence in range(batch * kv_heads)
            )
        if not oversized_route_excluded:
            raise AssertionError("an oversized centroid entered the top-eight routes")
    absolute = (baseline.float() - union.float()).abs()
    hip_absolute = (
        (triton_union.float() - union.float()).abs()
        if triton_union is not None
        else None
    )
    if args.cuda_graph_timing:
        torch.cuda.synchronize()
        baseline_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(baseline_graph):
            run(baseline_buffers, False)
        union_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(union_graph):
            run(union_buffers, True)
        baseline_ms = elapsed_ms(baseline_graph.replay, args.repeats)
        union_ms = elapsed_ms(union_graph.replay, args.repeats)
    else:
        baseline_ms = elapsed_ms(
            lambda: run(baseline_buffers, False), args.repeats
        )
        union_ms = elapsed_ms(lambda: run(union_buffers, True), args.repeats)
    manual_union_error = None
    if args.equal_leaves and args.union_final == "unified":
        # Validate the changed semantics directly: every head attends the same
        # GQA union, opened summaries disappear, and unopened summaries retain
        # their log(count) mass bias.
        references = []
        actuals = []
        cache_row = int(cache_indices[0].item())
        for kv_head in range(kv_heads):
            sequence = kv_head
            epoch = union_buffers["gqa_union_epochs"][sequence]
            opened = torch.nonzero(
                union_buffers["gqa_union_seen_stamps"][sequence] == epoch,
                as_tuple=False,
            ).flatten()
            closed = torch.ones(state_len, device=device, dtype=torch.bool)
            closed[opened] = False
            closed_slots = torch.arange(state_len, device=device)[closed]
            state_count = counts[cache_row, kv_head, closed_slots, 0]
            coarse_k = (
                state_k[cache_row, kv_head, closed_slots].float()
                / state_count[:, None]
            ).to(dtype)
            coarse_v = (
                state_v[cache_row, kv_head, closed_slots].float()
                / state_count[:, None]
            ).to(dtype)
            exact_indices = (
                opened[:, None] * page_size
                + torch.arange(page_size, device=device)[None, :]
            ).reshape(-1)
            exact_k = leaf_k[cache_row, kv_head, exact_indices]
            exact_v = leaf_v[cache_row, kv_head, exact_indices]
            keys = torch.cat(
                (coarse_k, exact_k, local_k[cache_row, kv_head, :active_local]),
                dim=0,
            )
            values = torch.cat(
                (coarse_v, exact_v, local_v[cache_row, kv_head, :active_local]),
                dim=0,
            ).float()
            bias = torch.cat(
                (
                    state_count.log(),
                    torch.zeros(
                        exact_k.size(0) + active_local,
                        device=device,
                        dtype=torch.float32,
                    ),
                )
            )
            for lane in range(gqa):
                query_head = kv_head * gqa + lane
                scores = (
                    torch.mv(keys.float(), q[0, query_head, 0].float())
                    * (head_dim**-0.5)
                    + bias
                )
                references.append(torch.softmax(scores, dim=0) @ values)
                actuals.append(union[0, query_head, 0].float())
        manual_absolute = (
            torch.stack(references) - torch.stack(actuals)
        ).abs()
        manual_union_error = {
            "max_abs": float(manual_absolute.max().item()),
            "mean_abs": float(manual_absolute.mean().item()),
        }
    manual_fixed_mask_error = None
    if args.fixed_mask_aiter:
        if not all(
            isinstance(tensor, torch.Tensor)
            for tensor in (
                fixed_indices,
                fixed_leaf_owners,
                fixed_lengths,
                arena_k,
                arena_v,
                arena_bias,
            )
        ):
            raise AssertionError("fixed-mask reference is missing its persistent arena")
        references = []
        actuals = []
        leaf_begin = local_limit + state_len
        for batch_row in range(batch):
            cache_row = int(cache_indices[batch_row].item())
            for kv_head in range(kv_heads):
                sequence = batch_row * kv_heads + kv_head
                physical_length = int(fixed_lengths[cache_row, kv_head].item())
                physical = fixed_indices[
                    cache_row, kv_head, :physical_length
                ].long()
                opened = (
                    initial_seen_stamps[sequence]
                    == output_route_epochs[sequence]
                )
                active = torch.zeros(
                    physical_length, device=device, dtype=torch.bool
                )
                active[:active_local] = True
                active[local_limit : local_limit + state_len] = ~opened
                leaf_count = physical_length - leaf_begin
                owners = fixed_leaf_owners[
                    cache_row, kv_head, :leaf_count
                ].long()
                active[leaf_begin:] = opened[owners]
                physical = physical[active]
                keys = arena_k[physical].float()
                values = arena_v[physical].float()
                bias = arena_bias[physical].float()
                for lane in range(gqa):
                    query_head = kv_head * gqa + lane
                    scores = (
                        torch.mv(keys, q[batch_row, query_head, 0].float())
                        * (head_dim**-0.5)
                        + bias
                    )
                    references.append(torch.softmax(scores, dim=0) @ values)
                    actuals.append(union[batch_row, query_head, 0].float())
        manual_fixed_absolute = (
            torch.stack(references) - torch.stack(actuals)
        ).abs()
        manual_fixed_mask_error = {
            "max_abs": float(manual_fixed_absolute.max().item()),
            "mean_abs": float(manual_fixed_absolute.mean().item()),
        }
    phase_profiles = {}
    for name, buffers, enabled in (
        ("baseline", baseline_buffers, False),
        ("gqa_union", union_buffers, True),
    ):
        events = {}
        run(buffers, enabled, events)
        torch.cuda.synchronize()
        phase_profiles[name] = {
            phase: 1000.0
            * sum(float(begin.elapsed_time(end)) for begin, end in pairs)
            / len(pairs)
            for phase, pairs in events.items()
            if pairs and "_b" not in phase
        }
    kernel_profile = None
    if args.profile_kernels:
        # Profiling complete calls preserves the real stream dependencies while
        # avoiding the timestamp-event perturbation of these very short stages.
        # Report time per decode call so the entries can be compared directly
        # with timing_ms.gqa_union.
        profile_repeats = min(100, max(20, args.repeats))
        for _ in range(5):
            run(union_buffers, True)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
        ) as profiler:
            for _ in range(profile_repeats):
                run(union_buffers, True)
            torch.cuda.synchronize()
        rows = []
        for event in profiler.key_averages():
            device_total = float(getattr(event, "device_time_total", 0.0))
            if device_total <= 0.0:
                device_total = float(getattr(event, "cuda_time_total", 0.0))
            if device_total <= 0.0:
                continue
            rows.append(
                {
                    "name": event.key,
                    "device_us_per_call": device_total / profile_repeats,
                    "launches_per_call": float(event.count) / profile_repeats,
                }
            )
        rows.sort(key=lambda row: row["device_us_per_call"], reverse=True)
        kernel_profile = {
            "repeats": profile_repeats,
            "kernels": rows,
            "summed_device_us_per_call": sum(
                row["device_us_per_call"] for row in rows
            ),
        }
    result = {
        "geometry": {
            "batch": batch,
            "kv_heads": kv_heads,
            "gqa": gqa,
            "head_dim": head_dim,
            "state_len": state_len,
            "local_limit": local_limit,
            "active_local": active_local,
            "leaf_count_mean": float(counts.float().mean().item()),
            "splits": splits,
            "union_final": args.union_final,
            "mass_fraction": args.mass_fraction,
            "hip_union": args.hip_union,
            "unified_arena": args.unified_arena,
            "group64_padded": args.group64_padded,
            "staged_fixed_aiter": args.staged_fixed_aiter,
            "fixed_mask_aiter": args.fixed_mask_aiter,
            "fixed_mask_block_n": args.fixed_mask_block_n,
            "fixed_mask_segments": args.fixed_mask_segments,
            "fixed_mask_adaptive_segments": (
                args.fixed_mask_adaptive_segments
            ),
            "fixed_mask_reduce_block_d": args.fixed_mask_reduce_block_d,
            "fixed_mask_direct_routes": args.fixed_mask_direct_routes,
            "route_group_size": route_group_size,
            "route_segment_tiles": route_segment_tiles,
            "route_num_warps": args.route_num_warps,
            "route_reduce_num_warps": args.route_reduce_num_warps,
            "cuda_graph_timing": args.cuda_graph_timing,
        },
        "correctness": {
            "max_abs": float(absolute.max().item()),
            "mean_abs": float(absolute.mean().item()),
            "hip_vs_triton_max_abs": (
                float(hip_absolute.max().item())
                if hip_absolute is not None
                else None
            ),
            "hip_vs_triton_mean_abs": (
                float(hip_absolute.mean().item())
                if hip_absolute is not None
                else None
            ),
            "manual_shared_union": manual_union_error,
            "manual_fixed_mask": manual_fixed_mask_error,
            "oversized_route_excluded": oversized_route_excluded,
        },
        "timing_ms": {
            "baseline_two_tier": baseline_ms,
            "gqa_union": union_ms,
            "speedup": baseline_ms / union_ms,
        },
        "phase_microseconds": phase_profiles,
        "kernel_profile": kernel_profile,
        "union": {
            "initial_counts": initial_union_counts.cpu().tolist(),
            "initial_epochs": initial_union_epochs.cpu().tolist(),
            "epochs": union_buffers["gqa_union_epochs"][: batch * kv_heads]
            .cpu()
            .tolist(),
            "route_valid_counts": (
                union_buffers["route_top_slots"] >= 0
            ).sum(dim=-1).cpu().tolist(),
            "mean_centroids": float(
                union_buffers["gqa_union_counts"][: batch * kv_heads]
                .float()
                .mean()
                .item()
            ),
            "mean_tokens_including_local": float(
                (
                    union_buffers["gqa_union_hip_context_lens"]
                    if args.unified_arena
                    else union_buffers["gqa_union_token_counts"]
                )[: batch * kv_heads]
                .float()
                .mean()
                .item()
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
