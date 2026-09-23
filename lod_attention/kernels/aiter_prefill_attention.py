"""AITER-backed top-four/top-eight routing and coarse attention for prefill."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import torch
import triton
import triton.language as tl

from ._paged_common import _pack_route_score_index, _unpack_route_score_index


def _workspace_tensor(
    buffers: dict[str, torch.Tensor] | None,
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a shared flat workspace view, growing it only when necessary."""

    elements = math.prod(shape)
    if buffers is None:
        return torch.empty(shape, dtype=dtype, device=device)
    storage = buffers.get(name)
    if (
        storage is None
        or storage.dtype != dtype
        or storage.device != device
        or int(storage.numel()) < elements
    ):
        storage = torch.empty(elements, dtype=dtype, device=device)
        buffers[name] = storage
    return storage[:elements].view(shape)


@dataclass(frozen=True)
class AiterPrefillCoarse:
    """Unmodified centroid attention retained for exact final correction."""

    output_0: torch.Tensor
    lse_0: torch.Tensor
    output_1: torch.Tensor
    lse_1: torch.Tensor
    mean_k: torch.Tensor
    mean_v: torch.Tensor
    counts: torch.Tensor
    has_second_partition: bool


@triton.jit(do_not_specialize=["STATE_LEN"])
def _prepare_aiter_state_kernel(
    state_k,
    state_v,
    counts,
    key_norm_sums,
    mean_k,
    mean_v,
    active_counts,
    log_count_bias,
    STATE_LEN,
    DISPATCH_STATE_LEN,
    STATE_CAPACITY: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    BLOCK_G: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_KEY_NORM_SUMS: tl.constexpr,
):
    """Materialize contiguous means and count bias in one launch."""

    row = tl.program_id(0).to(tl.int64)
    dimension = tl.arange(0, BLOCK_D)
    batch_kv = row // DISPATCH_STATE_LEN
    slot = row - batch_kv * DISPATCH_STATE_LEN
    batch = batch_kv // KV_HEADS
    kv_head = batch_kv - batch * KV_HEADS
    source_row = batch_kv * STATE_CAPACITY + slot
    valid_slot = slot < STATE_LEN
    count = tl.maximum(
        tl.load(counts + source_row, mask=valid_slot, other=1.0), 1.0
    ).to(tl.float32)
    valid_dimension = dimension < HEAD_DIM
    state_offset = source_row * HEAD_DIM + dimension
    output_offset = row * HEAD_DIM + dimension
    key = tl.load(
        state_k + state_offset,
        mask=valid_slot & valid_dimension,
        other=0.0,
    )
    value = tl.load(
        state_v + state_offset,
        mask=valid_slot & valid_dimension,
        other=0.0,
    )
    tl.store(mean_k + output_offset, key / count, mask=valid_dimension)
    tl.store(mean_v + output_offset, value / count, mask=valid_dimension)
    mass = count
    if HAS_KEY_NORM_SUMS:
        radial_sum = tl.load(key_norm_sums + source_row, mask=valid_slot, other=0.0)
        squared_key_sum = tl.sum(
            tl.where(valid_dimension, key.to(tl.float32) * key.to(tl.float32), 0.0),
            axis=0,
        )
        centroid_rms = tl.sqrt(squared_key_sum / HEAD_DIM) / count
        mass = count * tl.maximum(
            (radial_sum / count) / tl.maximum(centroid_rms, 1.0e-12), 1.0
        )
    tl.store(active_counts + row, mass)
    # Triton requires ``tl.arange`` bounds to be powers of two. Qwen3.8 has
    # six query heads per K/V head, so pad the lane vector and mask its two
    # inactive entries rather than specializing the calculation to GQA=8.
    group = tl.arange(0, BLOCK_G)
    valid_group = group < KV_GROUP_SIZE
    query_head = kv_head * KV_GROUP_SIZE + group
    bias_offset = (
        (batch * KV_HEADS * KV_GROUP_SIZE + query_head) * DISPATCH_STATE_LEN
        + slot
    )
    tl.store(
        log_count_bias + bias_offset,
        tl.where(valid_slot, tl.log(mass), -float("inf")),
        mask=valid_group,
    )


@lru_cache(maxsize=3)
def _specialized_route_mha_fwd(
    route_count: int, normalize_route_query: bool
) -> Callable[..., tuple[torch.Tensor, ...]]:
    """Build raw-query or eight-route AITER probe specializations."""
    from aiter.jit.core import compile_ops, get_args_of_build
    from aiter.ops.mha import cmdGenFunc_mha_fwd

    if route_count not in (4, 8):
        raise ValueError("AITER routing supports four or eight routes")

    def route_build_args(*args: object, **kwargs: object) -> dict[str, object]:
        q = args[0] if args else kwargs["q"]
        if not isinstance(q, torch.Tensor):
            raise TypeError("AITER routing requires a query tensor")
        if not normalize_route_query and int(q.size(-1)) != 256:
            raise ValueError("raw AITER routing is specialized for Qwen D=256")
        generated = cmdGenFunc_mha_fwd(*args, **kwargs)
        suffix = (
            "_lod_route_raw"
            if route_count == 4
            else "_lod_route8" + ("" if normalize_route_query else "_raw")
        )
        generated["md_name"] = f"{generated['md_name']}{suffix}"
        if not normalize_route_query:
            generated["blob_gen_cmd"] = [
                command.replace(" --output_dir", " --optdim 256 --output_dir")
                for command in generated["blob_gen_cmd"]
            ]
        base_flags = list(get_args_of_build("module_mha_fwd")["flags_extra_hip"])
        generated["flags_extra_hip"] = [
            flag
            for flag in base_flags
            if not flag.startswith(
                ("-DCK_TILE_FMHA_ROUTE_QUERY_NORMALIZE=", "-DCK_TILE_FMHA_ROUTE_TOPK=")
            )
        ] + [
            f"-DCK_TILE_FMHA_ROUTE_QUERY_NORMALIZE={int(normalize_route_query)}",
            f"-DCK_TILE_FMHA_ROUTE_TOPK={route_count}",
        ]
        return generated

    def specialized_route_mha_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float,
        softmax_scale: float,
        is_causal: bool,
        window_size_left: int,
        window_size_right: int,
        sink_size: int,
        return_softmax_lse: bool,
        return_dropout_randval: bool,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        q_descale: Optional[torch.Tensor] = None,
        k_descale: Optional[torch.Tensor] = None,
        v_descale: Optional[torch.Tensor] = None,
        sink_ptr: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    # AITER registers compiled calls by the Python function name. A shared
    # name would alias top-four and top-eight raw-query variants in one process.
    specialized_route_mha_fwd.__name__ = (
        f"lod_route_mha_fwd_{route_count}_{int(normalize_route_query)}"
    )
    return compile_ops(
        "module_mha_fwd",
        fc_name="mha_fwd",
        gen_func=route_build_args,
    )(specialized_route_mha_fwd)


@triton.jit(
    do_not_specialize=["QUERY_LEN", "ACTIVE_BLOCKS", "STATE_LEN"],
    do_not_specialize_on_alignment=["QUERY_LEN", "ACTIVE_BLOCKS", "STATE_LEN"],
)
def _reduce_route_candidates_kernel(
    candidates,
    slot_lengths,
    output,
    head_counts,
    route_offsets,
    QUERY_LEN,
    ACTIVE_BLOCKS,
    STATE_LEN,
    SLOT_BATCH_STRIDE: tl.constexpr,
    SLOT_HEAD_STRIDE: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    MAX_OPEN_LEAF_TOKENS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    CANDIDATE_BLOCK: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    HIERARCHICAL_TOP8: tl.constexpr,
):
    """Reduce per-tile winners to the exact global top-k."""
    batch_head = tl.program_id(0).to(tl.int64)
    query = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    valid_query = query < QUERY_LEN
    if HIERARCHICAL_TOP8:
        # A native key tile emits its candidates in descending score order.
        # A tile whose maximum is not among the global top eight tile maxima
        # cannot contain a global top-eight item. Select those eight tiles,
        # then inspect only their 8x8 candidates. This is the same exact
        # two-stage reduction used by decode, but keeps FP32 route scores.
        block_lane = tl.arange(0, CANDIDATE_BLOCK // ROUTE_COUNT)
        valid_block = block_lane < ACTIVE_BLOCKS
        first_base = (
            (batch_head * ACTIVE_BLOCKS + block_lane) * (2 * ROUTE_COUNT)
        ) * QUERY_LEN + query[:, None]
        first_scores = tl.load(
            candidates + first_base,
            mask=valid_query[:, None] & valid_block[None, :],
            other=-float("inf"),
        ).to(tl.float32)
        packed_blocks = _pack_route_score_index(
            first_scores, block_lane[None, :]
        )
        best_blocks = tl.topk(packed_blocks, ROUTE_COUNT, dim=1)
        _, selected_blocks = _unpack_route_score_index(best_blocks)
        local_rank_lane = tl.arange(0, ROUTE_COUNT)
        block = tl.reshape(
            selected_blocks[:, :, None]
            + local_rank_lane[None, None, :] * 0,
            (BLOCK_M, ROUTE_COUNT * ROUTE_COUNT),
        )
        rank = tl.arange(0, ROUTE_COUNT * ROUTE_COUNT) % ROUTE_COUNT
        valid_candidate = block < ACTIVE_BLOCKS
        base = (
            ((batch_head * ACTIVE_BLOCKS + block) * (2 * ROUTE_COUNT) + rank)
            * QUERY_LEN
            + query[:, None]
        )
        scores = tl.load(
            candidates + base,
            mask=valid_query[:, None] & valid_candidate,
            other=-float("inf"),
        ).to(tl.float32)
        index_values = tl.load(
            candidates + base + ROUTE_COUNT * QUERY_LEN,
            mask=valid_query[:, None] & valid_candidate,
            other=-1.0,
        )
        candidate = tl.arange(0, ROUTE_COUNT * ROUTE_COUNT)
    else:
        candidate = tl.arange(0, CANDIDATE_BLOCK)
        block = candidate // ROUTE_COUNT
        rank = candidate - block * ROUTE_COUNT
        valid_candidate = block < ACTIVE_BLOCKS
        base = (
            ((batch_head * ACTIVE_BLOCKS + block) * (2 * ROUTE_COUNT) + rank)
            * QUERY_LEN
            + query[:, None]
        )
        scores = tl.load(
            candidates + base,
            mask=valid_query[:, None] & valid_candidate[None, :],
            other=-float("inf"),
        ).to(tl.float32)
        index_values = tl.load(
            candidates + base + ROUTE_COUNT * QUERY_LEN,
            mask=valid_query[:, None] & valid_candidate[None, :],
            other=-1.0,
        )
    valid_index = (index_values >= 0) & (index_values < STATE_LEN)
    indices = tl.where(valid_index, index_values, 0.0).to(tl.int64)
    scores = tl.where(valid_index, scores, -float("inf"))
    if MAX_OPEN_LEAF_TOKENS:
        batch = batch_head // QUERY_HEADS
        kv_head = (batch_head % QUERY_HEADS) // KV_GROUP_SIZE
        lengths = tl.load(
            slot_lengths
            + batch * SLOT_BATCH_STRIDE
            + kv_head * SLOT_HEAD_STRIDE
            + indices,
            mask=valid_query[:, None] & valid_candidate[None, :] & valid_index,
            other=0,
        )
        scores = tl.where(lengths <= MAX_OPEN_LEAF_TOKENS, scores, -float("inf"))
    route_rank = tl.arange(0, ROUTE_COUNT)
    if HIERARCHICAL_TOP8:
        packed = _pack_route_score_index(scores, indices)
        best = tl.topk(packed, ROUTE_COUNT, dim=1)
        _, selected_slots = _unpack_route_score_index(best)
    else:
        selected_slots = tl.full((BLOCK_M, ROUTE_COUNT), -1, tl.int64)
        for output_rank in tl.static_range(0, ROUTE_COUNT):
            selected_candidate = tl.argmax(scores, axis=1)
            selected_index = tl.sum(
                tl.where(
                    candidate[None, :] == selected_candidate[:, None],
                    indices,
                    0,
                ),
                axis=1,
            )
            if MAX_OPEN_LEAF_TOKENS:
                selected_index = tl.where(
                    tl.max(scores, axis=1) > -float("inf"), selected_index, -1
                )
            selected_slots = tl.where(
                route_rank[None, :] == output_rank,
                selected_index[:, None],
                selected_slots,
            )
            scores = tl.where(
                candidate[None, :] == selected_candidate[:, None],
                -float("inf"),
                scores,
            )

    # Expert grouping expects the lowest-scoring boundary route last and the
    # other routes ordered by slot ID.
    output_base = (batch_head * QUERY_LEN + query) * ROUTE_COUNT
    boundary_slot = tl.max(
        tl.where(route_rank[None, :] == ROUTE_COUNT - 1, selected_slots, -1), axis=1
    )
    remaining_slots = tl.where(
        route_rank[None, :] < ROUTE_COUNT - 1,
        selected_slots,
        0x7FFFFFFFFFFFFFFF,
    )
    ordered_slots = tl.full((BLOCK_M, ROUTE_COUNT), -1, tl.int64)
    for output_rank in tl.static_range(0, ROUTE_COUNT - 1):
        output_slot = tl.min(remaining_slots, axis=1)
        output_slot = tl.where(output_slot == 0x7FFFFFFFFFFFFFFF, -1, output_slot)
        ordered_slots = tl.where(
            route_rank[None, :] == output_rank,
            output_slot[:, None],
            ordered_slots,
        )
        tl.store(output + output_base + output_rank, output_slot, mask=valid_query)
        remaining_slots = tl.where(
            remaining_slots == output_slot[:, None],
            0x7FFFFFFFFFFFFFFF,
            remaining_slots,
        )
    tl.store(
        output + output_base + ROUTE_COUNT - 1, boundary_slot, mask=valid_query
    )
    ordered_slots = tl.where(
        route_rank[None, :] == ROUTE_COUNT - 1,
        boundary_slot[:, None],
        ordered_slots,
    )
    for rank in tl.static_range(0, ROUTE_COUNT):
        selected_slot = tl.sum(
            tl.where(route_rank[None, :] == rank, ordered_slots, 0), axis=1
        )
        valid_slot = valid_query & (selected_slot >= 0) & (selected_slot < STATE_LEN)
        local_offset = tl.atomic_add(
            head_counts + batch_head * STATE_LEN + selected_slot,
            1,
            mask=valid_slot,
            sem="relaxed",
        )
        tl.store(
            route_offsets + output_base + rank,
            local_offset,
            mask=valid_slot,
        )


def _reduce_route_candidates(
    candidates: torch.Tensor,
    *,
    slot_lengths: torch.Tensor | None = None,
    max_open_leaf_tokens: int | None = None,
    route_count: int,
    state_len: int,
    head_dim: int,
    buffers: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not candidates.is_cuda or not candidates.is_contiguous():
        raise ValueError("AITER route candidates must be contiguous on the GPU")
    if candidates.ndim != 5 or int(candidates.size(3)) != 2 * route_count:
        raise RuntimeError(
            f"active AITER returned route candidates shaped {tuple(candidates.shape)}, "
            f"expected [batch, heads, blocks, {2 * route_count}, queries]; "
            "apply integrations/vllm_lod/patches/"
            "aiter-mha-prefill-route4.patch and rebuild AITER"
        )
    batch, query_heads, active_blocks, _, query_len = candidates.shape
    output = _workspace_tensor(
        buffers,
        "route_slots",
        (batch, query_heads, query_len, route_count),
        dtype=torch.long,
        device=candidates.device,
    )
    head_counts = _workspace_tensor(
        buffers,
        "route_head_counts",
        (batch * query_heads * state_len,),
        dtype=torch.int32,
        device=candidates.device,
    )
    head_counts.zero_()
    route_offsets = _workspace_tensor(
        buffers,
        "route_offsets",
        tuple(output.shape),
        dtype=torch.int32,
        device=candidates.device,
    )
    candidate_block = triton.next_power_of_2(active_blocks * route_count)
    hierarchical_top8 = (
        route_count == 8
        and max_open_leaf_tokens is None
        and active_blocks * route_count > 128
    )
    if head_dim <= 128:
        block_m = 16
        num_warps = 2
    else:
        block_m = 32 if candidate_block <= 128 else 16
        num_warps = 4
    _reduce_route_candidates_kernel[
        (batch * query_heads, triton.cdiv(query_len, block_m))
    ](
        candidates,
        slot_lengths if slot_lengths is not None else candidates,
        output,
        head_counts,
        route_offsets,
        query_len,
        active_blocks,
        state_len,
        SLOT_BATCH_STRIDE=slot_lengths.stride(0) if slot_lengths is not None else 0,
        SLOT_HEAD_STRIDE=slot_lengths.stride(1) if slot_lengths is not None else 0,
        QUERY_HEADS=query_heads,
        KV_GROUP_SIZE=(
            query_heads // int(slot_lengths.size(1)) if slot_lengths is not None else 1
        ),
        MAX_OPEN_LEAF_TOKENS=max_open_leaf_tokens or 0,
        BLOCK_M=block_m,
        CANDIDATE_BLOCK=candidate_block,
        ROUTE_COUNT=route_count,
        HIERARCHICAL_TOP8=hierarchical_top8,
        num_warps=num_warps,
    )
    return output, head_counts, route_offsets


@triton.jit(
    do_not_specialize=[
        "QUERY_LEN",
        "ACTIVE_BLOCKS_0",
        "ACTIVE_BLOCKS_1",
        "SECOND_INDEX_OFFSET",
        "STATE_LEN",
    ],
    do_not_specialize_on_alignment=[
        "QUERY_LEN",
        "ACTIVE_BLOCKS_0",
        "ACTIVE_BLOCKS_1",
        "SECOND_INDEX_OFFSET",
        "STATE_LEN",
    ],
)
def _reduce_split_route_candidates_kernel(
    candidates_0,
    candidates_1,
    slot_lengths,
    output,
    head_counts,
    route_offsets,
    QUERY_LEN,
    ACTIVE_BLOCKS_0,
    ACTIVE_BLOCKS_1,
    SECOND_INDEX_OFFSET,
    STATE_LEN,
    SLOT_BATCH_STRIDE: tl.constexpr,
    SLOT_HEAD_STRIDE: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    MAX_OPEN_LEAF_TOKENS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    CANDIDATE_BLOCK: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    HIERARCHICAL_TOP8: tl.constexpr,
):
    """Reduce compact candidates from two exact attention partitions."""
    batch_head = tl.program_id(0).to(tl.int64)
    query = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    first_candidates = ACTIVE_BLOCKS_0 * ROUTE_COUNT
    valid_query = query < QUERY_LEN
    if HIERARCHICAL_TOP8:
        block_lane = tl.arange(0, CANDIDATE_BLOCK // ROUTE_COUNT)
        first_block = block_lane < ACTIVE_BLOCKS_0
        local_block = tl.where(
            first_block, block_lane, block_lane - ACTIVE_BLOCKS_0
        )
        valid_first_block = first_block & (local_block < ACTIVE_BLOCKS_0)
        valid_second_block = (~first_block) & (local_block < ACTIVE_BLOCKS_1)
        first_base_0 = (
            (batch_head * ACTIVE_BLOCKS_0 + local_block) * (2 * ROUTE_COUNT)
        ) * QUERY_LEN + query[:, None]
        first_base_1 = (
            (batch_head * ACTIVE_BLOCKS_1 + local_block) * (2 * ROUTE_COUNT)
        ) * QUERY_LEN + query[:, None]
        first_score_0 = tl.load(
            candidates_0 + first_base_0,
            mask=valid_query[:, None] & valid_first_block[None, :],
            other=-float("inf"),
        ).to(tl.float32)
        first_score_1 = tl.load(
            candidates_1 + first_base_1,
            mask=valid_query[:, None] & valid_second_block[None, :],
            other=-float("inf"),
        ).to(tl.float32)
        first_scores = tl.where(first_block[None, :], first_score_0, first_score_1)
        packed_blocks = _pack_route_score_index(
            first_scores, block_lane[None, :]
        )
        best_blocks = tl.topk(packed_blocks, ROUTE_COUNT, dim=1)
        _, selected_blocks = _unpack_route_score_index(best_blocks)
        local_rank_lane = tl.arange(0, ROUTE_COUNT)
        global_block = tl.reshape(
            selected_blocks[:, :, None]
            + local_rank_lane[None, None, :] * 0,
            (BLOCK_M, ROUTE_COUNT * ROUTE_COUNT),
        )
        rank = tl.arange(0, ROUTE_COUNT * ROUTE_COUNT) % ROUTE_COUNT
        from_first = global_block < ACTIVE_BLOCKS_0
        block = tl.where(
            from_first, global_block, global_block - ACTIVE_BLOCKS_0
        )
        valid_first = from_first & (block < ACTIVE_BLOCKS_0)
        valid_second = (~from_first) & (block < ACTIVE_BLOCKS_1)
        candidate = tl.arange(0, ROUTE_COUNT * ROUTE_COUNT)
    else:
        candidate = tl.arange(0, CANDIDATE_BLOCK)
        from_first = candidate < first_candidates
        local_candidate = tl.where(
            from_first, candidate, candidate - first_candidates
        )
        block = local_candidate // ROUTE_COUNT
        rank = local_candidate - block * ROUTE_COUNT
        valid_first = from_first & (block < ACTIVE_BLOCKS_0)
        valid_second = (~from_first) & (block < ACTIVE_BLOCKS_1)

    base_0 = (
        ((batch_head * ACTIVE_BLOCKS_0 + block) * (2 * ROUTE_COUNT) + rank) * QUERY_LEN
        + query[:, None]
    )
    base_1 = (
        ((batch_head * ACTIVE_BLOCKS_1 + block) * (2 * ROUTE_COUNT) + rank) * QUERY_LEN
        + query[:, None]
    )
    score_0 = tl.load(
        candidates_0 + base_0,
        mask=valid_query[:, None] & valid_first,
        other=-float("inf"),
    ).to(tl.float32)
    score_1 = tl.load(
        candidates_1 + base_1,
        mask=valid_query[:, None] & valid_second,
        other=-float("inf"),
    ).to(tl.float32)
    index_0 = tl.load(
        candidates_0 + base_0 + ROUTE_COUNT * QUERY_LEN,
        mask=valid_query[:, None] & valid_first,
        other=-1.0,
    )
    index_1 = tl.load(
        candidates_1 + base_1 + ROUTE_COUNT * QUERY_LEN,
        mask=valid_query[:, None] & valid_second,
        other=-1.0,
    )
    index_values = tl.where(from_first, index_0, index_1)
    valid_index = index_values >= 0
    indices = tl.where(valid_index, index_values, 0.0).to(tl.int64)
    indices += tl.where(from_first, 0, SECOND_INDEX_OFFSET)
    valid_index = valid_index & (indices < STATE_LEN)
    scores = tl.where(
        valid_index,
        tl.where(from_first, score_0, score_1),
        -float("inf"),
    )
    if MAX_OPEN_LEAF_TOKENS:
        batch = batch_head // QUERY_HEADS
        kv_head = (batch_head % QUERY_HEADS) // KV_GROUP_SIZE
        lengths = tl.load(
            slot_lengths
            + batch * SLOT_BATCH_STRIDE
            + kv_head * SLOT_HEAD_STRIDE
            + indices,
            mask=valid_query[:, None] & valid_index,
            other=0,
        )
        scores = tl.where(lengths <= MAX_OPEN_LEAF_TOKENS, scores, -float("inf"))

    route_rank = tl.arange(0, ROUTE_COUNT)
    if HIERARCHICAL_TOP8:
        packed = _pack_route_score_index(scores, indices)
        best = tl.topk(packed, ROUTE_COUNT, dim=1)
        _, selected_slots = _unpack_route_score_index(best)
    else:
        selected_slots = tl.full((BLOCK_M, ROUTE_COUNT), -1, tl.int64)
        for output_rank in tl.static_range(0, ROUTE_COUNT):
            selected_candidate = tl.argmax(scores, axis=1)
            selected_index = tl.sum(
                tl.where(
                    candidate[None, :] == selected_candidate[:, None],
                    indices,
                    0,
                ),
                axis=1,
            )
            if MAX_OPEN_LEAF_TOKENS:
                selected_index = tl.where(
                    tl.max(scores, axis=1) > -float("inf"), selected_index, -1
                )
            selected_slots = tl.where(
                route_rank[None, :] == output_rank,
                selected_index[:, None],
                selected_slots,
            )
            scores = tl.where(
                candidate[None, :] == selected_candidate[:, None],
                -float("inf"),
                scores,
            )

    output_base = (batch_head * QUERY_LEN + query) * ROUTE_COUNT
    boundary_slot = tl.max(
        tl.where(route_rank[None, :] == ROUTE_COUNT - 1, selected_slots, -1), axis=1
    )
    remaining_slots = tl.where(
        route_rank[None, :] < ROUTE_COUNT - 1,
        selected_slots,
        0x7FFFFFFFFFFFFFFF,
    )
    ordered_slots = tl.full((BLOCK_M, ROUTE_COUNT), -1, tl.int64)
    for output_rank in tl.static_range(0, ROUTE_COUNT - 1):
        output_slot = tl.min(remaining_slots, axis=1)
        output_slot = tl.where(output_slot == 0x7FFFFFFFFFFFFFFF, -1, output_slot)
        ordered_slots = tl.where(
            route_rank[None, :] == output_rank,
            output_slot[:, None],
            ordered_slots,
        )
        tl.store(output + output_base + output_rank, output_slot, mask=valid_query)
        remaining_slots = tl.where(
            remaining_slots == output_slot[:, None],
            0x7FFFFFFFFFFFFFFF,
            remaining_slots,
        )
    tl.store(
        output + output_base + ROUTE_COUNT - 1, boundary_slot, mask=valid_query
    )
    ordered_slots = tl.where(
        route_rank[None, :] == ROUTE_COUNT - 1,
        boundary_slot[:, None],
        ordered_slots,
    )
    for rank in tl.static_range(0, ROUTE_COUNT):
        selected_slot = tl.sum(
            tl.where(route_rank[None, :] == rank, ordered_slots, 0), axis=1
        )
        valid_slot = valid_query & (selected_slot >= 0) & (selected_slot < STATE_LEN)
        local_offset = tl.atomic_add(
            head_counts + batch_head * STATE_LEN + selected_slot,
            1,
            mask=valid_slot,
            sem="relaxed",
        )
        tl.store(
            route_offsets + output_base + rank,
            local_offset,
            mask=valid_slot,
        )


def _reduce_split_route_candidates(
    candidates_0: torch.Tensor,
    candidates_1: torch.Tensor,
    *,
    slot_lengths: torch.Tensor | None = None,
    max_open_leaf_tokens: int | None = None,
    route_count: int,
    second_index_offset: int,
    state_len: int,
    head_dim: int,
    buffers: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not candidates_0.is_cuda or not candidates_1.is_cuda:
        raise ValueError("AITER route candidates must be on the GPU")
    if not candidates_0.is_contiguous() or not candidates_1.is_contiguous():
        raise ValueError("AITER route candidates must be contiguous")
    if candidates_0.ndim != 5 or candidates_1.ndim != 5:
        raise RuntimeError("split AITER route candidates have the wrong rank")
    if (
        int(candidates_0.size(3)) != 2 * route_count
        or int(candidates_1.size(3)) != 2 * route_count
    ):
        raise RuntimeError("active AITER is missing compact route candidates")
    if tuple(candidates_0.shape[:2]) != tuple(candidates_1.shape[:2]):
        raise ValueError("split AITER route candidate heads differ")
    if int(candidates_0.size(4)) != int(candidates_1.size(4)):
        raise ValueError("split AITER route candidate query lengths differ")
    batch, query_heads, blocks_0, _, query_len = candidates_0.shape
    blocks_1 = int(candidates_1.size(2))
    output = _workspace_tensor(
        buffers,
        "route_slots",
        (batch, query_heads, query_len, route_count),
        dtype=torch.long,
        device=candidates_0.device,
    )
    head_counts = _workspace_tensor(
        buffers,
        "route_head_counts",
        (batch * query_heads * state_len,),
        dtype=torch.int32,
        device=candidates_0.device,
    )
    head_counts.zero_()
    route_offsets = _workspace_tensor(
        buffers,
        "route_offsets",
        tuple(output.shape),
        dtype=torch.int32,
        device=candidates_0.device,
    )
    candidate_block = triton.next_power_of_2(
        (blocks_0 + blocks_1) * route_count
    )
    hierarchical_top8 = (
        route_count == 8
        and max_open_leaf_tokens is None
        and (blocks_0 + blocks_1) * route_count > 128
    )
    if head_dim <= 128:
        block_m = 16
        num_warps = 2
    else:
        block_m = 32 if candidate_block <= 128 else 16
        num_warps = 4
    _reduce_split_route_candidates_kernel[
        (batch * query_heads, triton.cdiv(query_len, block_m))
    ](
        candidates_0,
        candidates_1,
        slot_lengths if slot_lengths is not None else candidates_0,
        output,
        head_counts,
        route_offsets,
        query_len,
        blocks_0,
        blocks_1,
        second_index_offset,
        state_len,
        SLOT_BATCH_STRIDE=slot_lengths.stride(0) if slot_lengths is not None else 0,
        SLOT_HEAD_STRIDE=slot_lengths.stride(1) if slot_lengths is not None else 0,
        QUERY_HEADS=query_heads,
        KV_GROUP_SIZE=(
            query_heads // int(slot_lengths.size(1)) if slot_lengths is not None else 1
        ),
        MAX_OPEN_LEAF_TOKENS=max_open_leaf_tokens or 0,
        BLOCK_M=block_m,
        CANDIDATE_BLOCK=candidate_block,
        ROUTE_COUNT=route_count,
        HIERARCHICAL_TOP8=hierarchical_top8,
        num_warps=num_warps,
    )
    return output, head_counts, route_offsets


@triton.jit(do_not_specialize=["QUERY_LEN", "STATE_LEN"])
def _select_routes_by_mass_coverage_kernel(
    q,
    mean_k,
    masses,
    coarse_lse_0,
    coarse_lse_1,
    local_lse,
    sink_k,
    slots,
    head_counts,
    route_offsets,
    QUERY_LEN,
    STATE_LEN,
    MEAN_K_STRIDE,
    LOCAL_LSE_BATCH_STRIDE: tl.constexpr,
    LOCAL_LSE_HEAD_STRIDE: tl.constexpr,
    LOCAL_LSE_TOKEN_STRIDE: tl.constexpr,
    SINK_K_BATCH_STRIDE: tl.constexpr,
    SINK_K_HEAD_STRIDE: tl.constexpr,
    SINK_K_TOKEN_STRIDE: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    SINK_LEN: tl.constexpr,
    HAS_SECOND_PARTITION: tl.constexpr,
    SCALE: tl.constexpr,
    COVERAGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Refine up to eight regions until local+sink+refined mass reaches p."""
    batch_head = tl.program_id(0)
    query = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    dim = tl.arange(0, BLOCK_D)
    rank_vector = tl.arange(0, ROUTE_COUNT)
    valid_query = query < QUERY_LEN
    batch = batch_head // QUERY_HEADS
    query_head = batch_head % QUERY_HEADS
    kv_head = query_head // KV_GROUP_SIZE
    query_row = batch_head * QUERY_LEN + query
    query_value = tl.load(
        q + query_row[:, None] * HEAD_DIM + dim[None, :],
        mask=valid_query[:, None] & (dim[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)

    remote_lse = tl.load(
        coarse_lse_0 + query_row, mask=valid_query, other=0.0
    ).to(tl.float32)
    if HAS_SECOND_PARTITION:
        second_lse = tl.load(
            coarse_lse_1 + query_row, mask=valid_query, other=-float("inf")
        ).to(tl.float32)
        maximum = tl.maximum(remote_lse, second_lse)
        remote_lse = maximum + tl.log(
            tl.exp(remote_lse - maximum) + tl.exp(second_lse - maximum)
        )
    local_score = tl.load(
        local_lse
        + batch * LOCAL_LSE_BATCH_STRIDE
        + query_head * LOCAL_LSE_HEAD_STRIDE
        + query * LOCAL_LSE_TOKEN_STRIDE,
        mask=valid_query,
        other=-float("inf"),
    ).to(tl.float32)
    sink_lse = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    for sink_index in tl.static_range(0, SINK_LEN):
        sink_key = tl.load(
            sink_k
            + batch * SINK_K_BATCH_STRIDE
            + kv_head * SINK_K_HEAD_STRIDE
            + sink_index * SINK_K_TOKEN_STRIDE
            + dim,
            mask=dim < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        sink_score = tl.sum(query_value * sink_key[None, :], axis=1) * SCALE
        maximum = tl.maximum(sink_lse, sink_score)
        sink_lse = maximum + tl.log(
            tl.exp(sink_lse - maximum) + tl.exp(sink_score - maximum)
        )
    maximum = tl.maximum(remote_lse, local_score)
    total_lse = maximum + tl.log(
        tl.exp(remote_lse - maximum) + tl.exp(local_score - maximum)
    )
    maximum = tl.maximum(total_lse, sink_lse)
    total_lse = maximum + tl.log(
        tl.exp(total_lse - maximum) + tl.exp(sink_lse - maximum)
    )
    covered = tl.exp(local_score - total_lse) + tl.exp(sink_lse - total_lse)

    logits = tl.full((BLOCK_M, ROUTE_COUNT), -float("inf"), tl.float32)
    for rank in tl.static_range(0, ROUTE_COUNT):
        slot = tl.load(
            slots + query_row * ROUTE_COUNT + rank,
            mask=valid_query,
            other=-1,
        ).to(tl.int32)
        valid_slot = valid_query & (slot >= 0) & (slot < STATE_LEN)
        safe_slot = tl.maximum(slot, 0)
        key_row = (
            (batch * KV_HEADS + kv_head) * MEAN_K_STRIDE + safe_slot
        ) * HEAD_DIM
        key = tl.load(
            mean_k + key_row[:, None] + dim[None, :],
            mask=valid_slot[:, None] & (dim[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        mass = tl.load(
            masses + (batch * KV_HEADS + kv_head) * MEAN_K_STRIDE + safe_slot,
            mask=valid_slot,
            other=1.0,
        ).to(tl.float32)
        logit = tl.sum(query_value * key, axis=1) * SCALE + tl.log(mass)
        logits = tl.where(
            rank_vector[None, :] == rank,
            tl.where(valid_slot, logit, -float("inf"))[:, None],
            logits,
        )
    remaining = logits
    selected = tl.full((BLOCK_M, ROUTE_COUNT), False, tl.int1)
    for _ in tl.static_range(0, ROUTE_COUNT):
        best_rank = tl.argmax(remaining, axis=1)
        best_logit = tl.max(remaining, axis=1)
        open_region = valid_query & (covered < COVERAGE) & (
            best_logit > -float("inf")
        )
        selected |= (rank_vector[None, :] == best_rank[:, None]) & open_region[:, None]
        covered += tl.where(open_region, tl.exp(best_logit - total_lse), 0.0)
        remaining = tl.where(
            rank_vector[None, :] == best_rank[:, None],
            -float("inf"),
            remaining,
        )
    for rank in tl.static_range(0, ROUTE_COUNT):
        slot = tl.load(
            slots + query_row * ROUTE_COUNT + rank,
            mask=valid_query,
            other=-1,
        ).to(tl.int32)
        keep_rank = tl.sum(
            tl.where(rank_vector[None, :] == rank, selected.to(tl.int32), 0), axis=1
        ) > 0
        keep = valid_query & (slot >= 0) & (slot < STATE_LEN) & keep_rank
        tl.store(
            slots + query_row * ROUTE_COUNT + rank,
            tl.where(keep, slot, -1),
            mask=valid_query,
        )
        local_offset = tl.atomic_add(
            head_counts + batch_head * STATE_LEN + tl.maximum(slot, 0),
            1,
            mask=keep,
            sem="relaxed",
        )
        tl.store(
            route_offsets + query_row * ROUTE_COUNT + rank,
            local_offset,
            mask=keep,
        )


@triton.jit(
    do_not_specialize=["QUERY_LEN", "STATE_LEN"],
    do_not_specialize_on_alignment=["QUERY_LEN", "STATE_LEN"],
)
def _remove_route4_from_coarse_kernel(
    q,
    mean_k,
    mean_v,
    counts,
    slots,
    attention_out_0,
    attention_lse_0,
    attention_out_1,
    attention_lse_1,
    output,
    QUERY_LEN,
    STATE_LEN,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_SECOND_PARTITION: tl.constexpr,
):
    """Remove the four refined centroids from AITER's coarse partition."""
    batch_head = tl.program_id(0).to(tl.int64)
    query = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    batch = batch_head // QUERY_HEADS
    query_head = batch_head - batch * QUERY_HEADS
    kv_head = query_head // KV_GROUP_SIZE
    dim = tl.arange(0, BLOCK_D)
    valid_query = query < QUERY_LEN
    valid_dim = dim < HEAD_DIM
    query_row = (batch * QUERY_HEADS + query_head) * QUERY_LEN + query
    query_value = tl.load(
        q + query_row[:, None] * HEAD_DIM + dim[None, :],
        mask=valid_query[:, None] & valid_dim[None, :],
        other=0.0,
    )
    lse_0 = tl.load(
        attention_lse_0 + query_row,
        mask=valid_query,
        other=0.0,
    ).to(tl.float32)
    aiter_row = (batch * QUERY_LEN + query) * QUERY_HEADS + query_head
    remainder_0 = tl.load(
        attention_out_0 + aiter_row[:, None] * HEAD_DIM + dim[None, :],
        mask=valid_query[:, None] & valid_dim[None, :],
        other=0.0,
    ).to(tl.float32)
    if HAS_SECOND_PARTITION:
        lse_1 = tl.load(
            attention_lse_1 + query_row,
            mask=valid_query,
            other=0.0,
        ).to(tl.float32)
        remainder_1 = tl.load(
            attention_out_1 + aiter_row[:, None] * HEAD_DIM + dim[None, :],
            mask=valid_query[:, None] & valid_dim[None, :],
            other=0.0,
        ).to(tl.float32)
        full_lse = tl.maximum(lse_0, lse_1)
        partition_mass_0 = tl.exp(lse_0 - full_lse)
        partition_mass_1 = tl.exp(lse_1 - full_lse)
        partition_mass = partition_mass_0 + partition_mass_1
        full_lse += tl.log(partition_mass)
        remainder = (
            partition_mass_0[:, None] * remainder_0
            + partition_mass_1[:, None] * remainder_1
        ) / partition_mass[:, None]
    else:
        full_lse = lse_0
        remainder = remainder_0
    selected_mass = tl.zeros((BLOCK_M,), tl.float32)
    selected_value = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    for rank in tl.static_range(0, 4):
        slot = tl.load(
            slots + query_row * 4 + rank,
            mask=valid_query,
            other=-1,
        ).to(tl.int64)
        valid_slot = valid_query & (slot >= 0) & (slot < STATE_LEN)
        safe_slot = tl.where(valid_slot, slot, 0)
        state_row = (
            (batch * KV_HEADS + kv_head) * STATE_LEN + safe_slot
        ) * HEAD_DIM
        count = tl.load(
            counts + (batch * KV_HEADS + kv_head) * STATE_LEN + safe_slot,
            mask=valid_slot,
            other=1.0,
        ).to(tl.float32)
        key = tl.load(
            mean_k + state_row[:, None] + dim[None, :],
            mask=valid_slot[:, None] & valid_dim[None, :],
            other=0.0,
        )
        value = tl.load(
            mean_v + state_row[:, None] + dim[None, :],
            mask=valid_slot[:, None] & valid_dim[None, :],
            other=0.0,
        )
        score = tl.sum(query_value * key, axis=1) * SCALE + tl.log(count)
        mass = tl.where(valid_slot, tl.exp(score - full_lse), 0.0)
        selected_mass += mass
        selected_value += mass[:, None] * value

    remaining_mass = tl.maximum(1.0 - selected_mass, 1.0e-7)
    tl.store(
        output + query_row[:, None] * HEAD_DIM + dim[None, :],
        (remainder - selected_value) / remaining_mass[:, None],
        mask=valid_query[:, None] & valid_dim[None, :],
    )
    tl.store(
        attention_lse_0 + query_row,
        full_lse + tl.log(remaining_mass),
        mask=valid_query,
    )


@triton.jit(
    do_not_specialize=["QUERY_LEN", "STATE_LEN"],
    do_not_specialize_on_alignment=["QUERY_LEN", "STATE_LEN"],
)
def _merge_route_refinement_kernel(
    q,
    sink_k,
    sink_v,
    mean_k,
    mean_v,
    counts,
    slots,
    coarse_out_0,
    coarse_lse_0,
    coarse_out_1,
    coarse_lse_1,
    route_out,
    route_lse,
    local_out,
    local_lse,
    output,
    Q_BATCH_STRIDE,
    Q_HEAD_STRIDE,
    Q_TOKEN_STRIDE,
    SINK_K_BATCH_STRIDE,
    SINK_K_HEAD_STRIDE,
    SINK_K_TOKEN_STRIDE,
    SINK_V_BATCH_STRIDE,
    SINK_V_HEAD_STRIDE,
    SINK_V_TOKEN_STRIDE,
    LOCAL_BATCH_STRIDE,
    LOCAL_HEAD_STRIDE,
    LOCAL_TOKEN_STRIDE,
    LOCAL_LSE_BATCH_STRIDE,
    LOCAL_LSE_HEAD_STRIDE,
    LOCAL_LSE_TOKEN_STRIDE,
    OUTPUT_BATCH_STRIDE,
    OUTPUT_HEAD_STRIDE,
    OUTPUT_TOKEN_STRIDE,
    QUERY_LEN,
    STATE_LEN,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SINK_LEN: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_SECOND_PARTITION: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
):
    """Replace routed coarse centroids with their leaves and merge local/sink."""
    batch_head = tl.program_id(0).to(tl.int64)
    query = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    batch = batch_head // QUERY_HEADS
    query_head = batch_head - batch * QUERY_HEADS
    kv_head = query_head // KV_GROUP_SIZE
    dim = tl.arange(0, BLOCK_D)
    valid_query = query < QUERY_LEN
    valid_dim = dim < HEAD_DIM
    query_row = (batch * QUERY_HEADS + query_head) * QUERY_LEN + query
    query_value = tl.load(
        q
        + batch * Q_BATCH_STRIDE
        + query_head * Q_HEAD_STRIDE
        + query[:, None] * Q_TOKEN_STRIDE
        + dim[None, :],
        mask=valid_query[:, None] & valid_dim[None, :],
        other=0.0,
    ).to(tl.float32)

    coarse_score_0 = tl.load(
        coarse_lse_0 + query_row, mask=valid_query, other=-float("inf")
    ).to(tl.float32)
    coarse_row = (batch * QUERY_LEN + query) * QUERY_HEADS + query_head
    coarse_value_0 = tl.load(
        coarse_out_0 + coarse_row[:, None] * HEAD_DIM + dim[None, :],
        mask=valid_query[:, None] & valid_dim[None, :],
        other=0.0,
    ).to(tl.float32)
    full_lse = coarse_score_0
    coarse_value = coarse_value_0
    if HAS_SECOND_PARTITION:
        coarse_score_1 = tl.load(
            coarse_lse_1 + query_row, mask=valid_query, other=-float("inf")
        ).to(tl.float32)
        coarse_value_1 = tl.load(
            coarse_out_1 + coarse_row[:, None] * HEAD_DIM + dim[None, :],
            mask=valid_query[:, None] & valid_dim[None, :],
            other=0.0,
        ).to(tl.float32)
        full_lse = tl.maximum(coarse_score_0, coarse_score_1)
        partition_weight_0 = tl.exp(coarse_score_0 - full_lse)
        partition_weight_1 = tl.exp(coarse_score_1 - full_lse)
        partition_weight = partition_weight_0 + partition_weight_1
        coarse_value = (
            partition_weight_0[:, None] * coarse_value_0
            + partition_weight_1[:, None] * coarse_value_1
        ) / partition_weight[:, None]
        full_lse += tl.log(partition_weight)

    selected_mass = tl.zeros((BLOCK_M,), tl.float32)
    selected_value = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    for rank in tl.static_range(0, ROUTE_COUNT):
        slot = tl.load(
            slots + query_row * ROUTE_COUNT + rank,
            mask=valid_query,
            other=-1,
        ).to(tl.int64)
        valid_slot = valid_query & (slot >= 0) & (slot < STATE_LEN)
        safe_slot = tl.where(valid_slot, slot, 0)
        state_row = (
            (batch * KV_HEADS + kv_head) * STATE_LEN + safe_slot
        ) * HEAD_DIM
        count = tl.load(
            counts + (batch * KV_HEADS + kv_head) * STATE_LEN + safe_slot,
            mask=valid_slot,
            other=1.0,
        ).to(tl.float32)
        key = tl.load(
            mean_k + state_row[:, None] + dim[None, :],
            mask=valid_slot[:, None] & valid_dim[None, :],
            other=0.0,
        ).to(tl.float32)
        value = tl.load(
            mean_v + state_row[:, None] + dim[None, :],
            mask=valid_slot[:, None] & valid_dim[None, :],
            other=0.0,
        ).to(tl.float32)
        score = tl.sum(query_value * key, axis=1) * SCALE + tl.log(count)
        weight = tl.where(valid_slot, tl.exp(score - full_lse), 0.0)
        selected_mass += weight
        selected_value += weight[:, None] * value
    remaining_mass = tl.maximum(1.0 - selected_mass, 1.0e-7)
    coarse_value = (coarse_value - selected_value) / remaining_mass[:, None]
    coarse_lse = full_lse + tl.log(remaining_mass)

    local_score = tl.load(
        local_lse
        + batch * LOCAL_LSE_BATCH_STRIDE
        + query_head * LOCAL_LSE_HEAD_STRIDE
        + query * LOCAL_LSE_TOKEN_STRIDE,
        mask=valid_query,
        other=-float("inf"),
    ).to(tl.float32)
    route_rank = tl.arange(0, ROUTE_COUNT)
    route_scores = tl.load(
        route_lse + query_row[:, None] * ROUTE_COUNT + route_rank[None, :],
        mask=valid_query[:, None] & (
            tl.load(
                slots + query_row[:, None] * ROUTE_COUNT + route_rank[None, :],
                mask=valid_query[:, None],
                other=-1,
            ) >= 0
        ),
        other=-float("inf"),
    ).to(tl.float32)
    maximum = tl.maximum(coarse_lse, local_score)
    maximum = tl.maximum(maximum, tl.max(route_scores, axis=1))
    for sink_index in tl.static_range(0, SINK_LEN):
        sink_key = tl.load(
            sink_k
            + batch * SINK_K_BATCH_STRIDE
            + kv_head * SINK_K_HEAD_STRIDE
            + sink_index * SINK_K_TOKEN_STRIDE
            + dim,
            mask=valid_dim,
            other=0.0,
        ).to(tl.float32)
        sink_score = tl.sum(query_value * sink_key[None, :], axis=1) * SCALE
        maximum = tl.maximum(maximum, sink_score)

    coarse_weight = tl.exp(coarse_lse - maximum)
    local_weight = tl.exp(local_score - maximum)
    denominator = coarse_weight + local_weight
    numerator = coarse_weight[:, None] * coarse_value
    local_value = tl.load(
        local_out
        + batch * LOCAL_BATCH_STRIDE
        + query_head * LOCAL_HEAD_STRIDE
        + query[:, None] * LOCAL_TOKEN_STRIDE
        + dim[None, :],
        mask=valid_query[:, None] & valid_dim[None, :],
        other=0.0,
    ).to(tl.float32)
    numerator += local_weight[:, None] * local_value
    for rank in tl.static_range(0, ROUTE_COUNT):
        refined_slot = tl.load(
            slots + query_row * ROUTE_COUNT + rank,
            mask=valid_query,
            other=-1,
        )
        refined = valid_query & (refined_slot >= 0)
        route_score = tl.load(
            route_lse + query_row * ROUTE_COUNT + rank,
            mask=refined,
            other=-float("inf"),
        ).to(tl.float32)
        route_weight = tl.exp(route_score - maximum)
        route_value = tl.load(
            route_out
            + (query_row * ROUTE_COUNT + rank)[:, None] * HEAD_DIM
            + dim[None, :],
            mask=refined[:, None] & valid_dim[None, :],
            other=0.0,
        ).to(tl.float32)
        denominator += route_weight
        numerator += route_weight[:, None] * route_value
    for sink_index in tl.static_range(0, SINK_LEN):
        sink_key = tl.load(
            sink_k
            + batch * SINK_K_BATCH_STRIDE
            + kv_head * SINK_K_HEAD_STRIDE
            + sink_index * SINK_K_TOKEN_STRIDE
            + dim,
            mask=valid_dim,
            other=0.0,
        ).to(tl.float32)
        sink_value = tl.load(
            sink_v
            + batch * SINK_V_BATCH_STRIDE
            + kv_head * SINK_V_HEAD_STRIDE
            + sink_index * SINK_V_TOKEN_STRIDE
            + dim,
            mask=valid_dim,
            other=0.0,
        ).to(tl.float32)
        sink_score = tl.sum(query_value * sink_key[None, :], axis=1) * SCALE
        sink_weight = tl.exp(sink_score - maximum)
        denominator += sink_weight
        numerator += sink_weight[:, None] * sink_value[None, :]

    tl.store(
        output
        + batch * OUTPUT_BATCH_STRIDE
        + query_head * OUTPUT_HEAD_STRIDE
        + query[:, None] * OUTPUT_TOKEN_STRIDE
        + dim[None, :],
        numerator / denominator[:, None],
        mask=valid_query[:, None] & valid_dim[None, :],
    )


def merge_aiter_prefill_refinement(
    q: torch.Tensor,
    sink_k: torch.Tensor,
    sink_v: torch.Tensor,
    coarse: AiterPrefillCoarse,
    slots: torch.Tensor,
    route_out: torch.Tensor,
    route_lse: torch.Tensor,
    local_out: torch.Tensor,
    local_lse: torch.Tensor,
    *,
    kv_group_size: int,
    scale: float,
    output_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the exact coarse-to-leaf replacement in one final GPU pass."""
    batch, query_heads, query_len, head_dim = q.shape
    kv_heads = int(coarse.mean_k.size(1))
    state_len = int(coarse.mean_k.size(2))
    route_count = int(slots.size(-1))
    if route_count not in (4, 8):
        raise ValueError("AITER refinement requires four or eight routes")
    expected = (batch, query_heads, query_len, head_dim)
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("AITER refinement has incompatible GQA geometry")
    if tuple(local_out.shape) != expected or tuple(local_lse.shape) != expected[:-1]:
        raise ValueError("AITER refinement local branch has incompatible geometry")
    if tuple(route_out.shape) != (*expected[:-1], route_count, head_dim):
        raise ValueError("AITER refinement route output has incompatible geometry")
    if tuple(route_lse.shape) != (*expected[:-1], route_count):
        raise ValueError("AITER refinement route LSE has incompatible geometry")
    if tuple(slots.shape) != (*expected[:-1], route_count):
        raise ValueError("AITER refinement routes have incompatible geometry")
    output = torch.empty_like(q) if output_buffer is None else output_buffer
    if tuple(output.shape) != expected or output.stride(-1) != 1:
        raise ValueError("AITER refinement output buffer has incompatible geometry")
    if head_dim <= 128:
        block_m = 32
        num_warps = 8
    else:
        block_m = 16
        num_warps = 8
    _merge_route_refinement_kernel[
        (batch * query_heads, triton.cdiv(query_len, block_m))
    ](
        q,
        sink_k,
        sink_v,
        coarse.mean_k,
        coarse.mean_v,
        coarse.counts,
        slots,
        coarse.output_0,
        coarse.lse_0,
        coarse.output_1,
        coarse.lse_1,
        route_out,
        route_lse,
        local_out,
        local_lse,
        output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        sink_k.stride(0),
        sink_k.stride(1),
        sink_k.stride(2),
        sink_v.stride(0),
        sink_v.stride(1),
        sink_v.stride(2),
        local_out.stride(0),
        local_out.stride(1),
        local_out.stride(2),
        local_lse.stride(0),
        local_lse.stride(1),
        local_lse.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        query_len,
        state_len,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        SINK_LEN=int(sink_k.size(2)),
        SCALE=float(scale),
        BLOCK_M=block_m,
        HAS_SECOND_PARTITION=coarse.has_second_partition,
        ROUTE_COUNT=route_count,
        num_warps=num_warps,
        waves_per_eu=1,
    )
    return output


def aiter_prefill_route_coarse_attention(
    q: torch.Tensor,
    state_k: torch.Tensor,
    state_v: torch.Tensor,
    counts: torch.Tensor,
    *,
    key_norm_sums: torch.Tensor | None = None,
    route_count: int = 4,
    state_len: int,
    kv_group_size: int,
    scale: float,
    normalize_route_query: bool,
    exact_mass_coverage: float | None = None,
    max_open_leaf_tokens: int | None = None,
    slot_lengths: torch.Tensor | None = None,
    local_lse: torch.Tensor | None = None,
    sink_k: torch.Tensor | None = None,
    buffers: dict[str, torch.Tensor] | None = None,
) -> tuple[
    torch.Tensor, AiterPrefillCoarse, torch.Tensor, torch.Tensor
]:
    """Compute exact top-k routes and retain unmodified coarse attention.

    The patched AITER FMHA emits k winners per native key tile while it
    computes ordinary count-weighted centroid attention. Only those compact
    candidates are reduced, so no query-by-centroid routing tensor is stored.
    """
    if route_count not in (4, 8):
        raise ValueError("AITER prefill requires four or eight routes")
    if exact_mass_coverage is not None:
        if not (0.0 < exact_mass_coverage <= 1.0):
            raise ValueError("exact mass coverage must be in (0, 1]")
        if local_lse is None or sink_k is None:
            raise ValueError("exact mass coverage requires local LSE and sink keys")
    if max_open_leaf_tokens is not None:
        if max_open_leaf_tokens < 1:
            raise ValueError("maximum open leaf tokens must be positive")
        if slot_lengths is None or tuple(slot_lengths.shape[:2]) != (
            q.size(0), state_k.size(1)
        ) or int(slot_lengths.size(2)) < state_len:
            raise ValueError("maximum open leaf tokens requires matching slot lengths")
        if (
            not slot_lengths.is_cuda
            or slot_lengths.dtype != torch.int32
            or slot_lengths.stride(2) != 1
        ):
            raise ValueError("slot lengths must be GPU int32 with contiguous slots")
    if max_open_leaf_tokens is not None and exact_mass_coverage is not None:
        raise ValueError("leaf-size cap and mass coverage cannot be combined")
    tensors = (q, state_k, state_v, counts)
    if key_norm_sums is not None:
        if tuple(key_norm_sums.shape) != tuple(counts.shape):
            raise ValueError("AITER mass correction requires one key norm sum per slot")
        tensors += (key_norm_sums,)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("AITER route/coarse prefill requires GPU tensors")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("AITER route/coarse prefill requires contiguous tensors")
    batch, query_heads, query_len, head_dim = q.shape
    kv_heads = int(state_k.size(1))
    if query_len <= 1:
        raise ValueError("AITER route/coarse prefill requires multiple queries")
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("AITER route/coarse prefill has incompatible GQA geometry")
    if head_dim > 256 or int(state_v.size(-1)) != head_dim:
        raise ValueError("AITER route/coarse prefill requires equal heads up to 256")
    if tuple(state_k.shape[:2]) != (batch, kv_heads) or tuple(
        state_k.shape[-1:]
    ) != (head_dim,):
        raise ValueError("AITER route/coarse prefill received the wrong state keys")
    if tuple(state_v.shape[:3]) != tuple(counts.shape[:3]):
        raise ValueError("AITER route/coarse state/count geometry differs")
    if tuple(state_v.shape[:2]) != (batch, kv_heads):
        raise ValueError("AITER route/coarse state heads differ")
    if state_len > int(state_v.size(2)):
        raise ValueError("AITER route/coarse state exceeds its storage")
    if exact_mass_coverage is not None:
        if tuple(local_lse.shape) != (batch, query_heads, query_len):
            raise ValueError("local LSE has incompatible route geometry")
        if (
            tuple(sink_k.shape[:2]) != (batch, kv_heads)
            or int(sink_k.size(-1)) != head_dim
            or not sink_k.is_cuda
        ):
            raise ValueError("sink keys have incompatible route geometry")

    # CK's MI325X prefill dispatch has severe cliffs when K is not a complete
    # native 128-key tile. Padding with zero K/V and -inf bias is
    # mathematically inert. Round only to that native tile: padding K=2896 all
    # the way to 4096 adds 39% needless remote work, while K=2944 selects the
    # same fast CK path. Above 4096, retain the established two-call partition
    # because CK's single-call dispatch changes to a slower kernel.
    def padded_partition(length: int) -> int:
        return ((length + 127) // 128) * 128

    if state_len <= 4096:
        dispatch_state_len = padded_partition(state_len)
        split_at = 0
    elif state_len <= 8192:
        dispatch_state_len = 4096 + padded_partition(state_len - 4096)
        split_at = 4096
    else:
        dispatch_state_len = state_len
        split_at = 0

    mean_shape = (batch, kv_heads, dispatch_state_len, head_dim)
    mean_k = _workspace_tensor(
        buffers,
        "coarse_mean_k",
        mean_shape,
        dtype=state_k.dtype,
        device=state_k.device,
    )
    mean_v = _workspace_tensor(
        buffers,
        "coarse_mean_v",
        mean_shape,
        dtype=state_v.dtype,
        device=state_v.device,
    )
    active_counts = _workspace_tensor(
        buffers,
        "coarse_counts",
        (batch, kv_heads, dispatch_state_len, 1),
        dtype=counts.dtype,
        device=counts.device,
    )
    log_count_bias = _workspace_tensor(
        buffers,
        "coarse_log_count_bias",
        (batch, query_heads, 1, dispatch_state_len),
        dtype=q.dtype,
        device=q.device,
    )
    _prepare_aiter_state_kernel[(batch * kv_heads * dispatch_state_len,)](
        state_k,
        state_v,
        counts,
        key_norm_sums if key_norm_sums is not None else counts,
        mean_k,
        mean_v,
        active_counts,
        log_count_bias,
        state_len,
        dispatch_state_len,
        STATE_CAPACITY=int(state_k.size(2)),
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        BLOCK_G=triton.next_power_of_2(kv_group_size),
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        HAS_KEY_NORM_SUMS=key_norm_sums is not None,
        num_warps=4,
    )
    q_aiter = q.permute(0, 2, 1, 3)
    k_aiter = mean_k.permute(0, 2, 1, 3)
    v_aiter = mean_v.permute(0, 2, 1, 3)

    original_dlopen_flags = sys.getdlopenflags()
    deepbind = getattr(os, "RTLD_DEEPBIND", 0)
    if deepbind:
        sys.setdlopenflags(original_dlopen_flags | deepbind)
    try:
        from aiter.ops.mha import mha_fwd

        route_mha_fwd = (
            mha_fwd
            if route_count == 4 and normalize_route_query
            else _specialized_route_mha_fwd(route_count, normalize_route_query)
        )

        def run_partition(
            begin: int,
            end: int,
            output_name: str,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            attention_buffer = _workspace_tensor(
                buffers,
                output_name,
                (batch, query_len, query_heads, head_dim),
                dtype=q.dtype,
                device=q.device,
            )
            attention_out, attention_lse, candidates, _ = route_mha_fwd(
                q_aiter,
                k_aiter[:, begin:end],
                v_aiter[:, begin:end],
                0.0,
                float(scale),
                False,
                -1,
                -1,
                0,
                True,
                True,
                None,
                None,
                attention_buffer,
                log_count_bias[..., begin:end],
                None,
                None,
                None,
                None,
                None,
            )
            return attention_out, attention_lse, candidates

        if split_at:
            attention_out_0, attention_lse_0, candidates_0 = run_partition(
                0, split_at, "coarse_output_0"
            )
            attention_out_1, attention_lse_1, candidates_1 = run_partition(
                split_at, dispatch_state_len, "coarse_output_1"
            )
        else:
            attention_out_0, attention_lse_0, candidates_0 = run_partition(
                0, dispatch_state_len, "coarse_output_0"
            )
            attention_out_1 = attention_out_0
            attention_lse_1 = attention_lse_0
    finally:
        sys.setdlopenflags(original_dlopen_flags)

    if split_at:
        top_slots, route_head_counts, route_offsets = _reduce_split_route_candidates(
            candidates_0,
            candidates_1,
            slot_lengths=slot_lengths,
            max_open_leaf_tokens=max_open_leaf_tokens,
            route_count=route_count,
            second_index_offset=split_at,
            state_len=state_len,
            head_dim=head_dim,
            buffers=buffers,
        )
    else:
        top_slots, route_head_counts, route_offsets = _reduce_route_candidates(
            candidates_0,
            slot_lengths=slot_lengths,
            max_open_leaf_tokens=max_open_leaf_tokens,
            route_count=route_count,
            state_len=state_len,
            head_dim=head_dim,
            buffers=buffers,
        )
    if exact_mass_coverage is not None:
        # Compare coarse region mass against the full denominator; local and
        # sink are already exact and count toward the coverage target.
        route_head_counts.zero_()
        _select_routes_by_mass_coverage_kernel[
            (batch * query_heads, triton.cdiv(query_len, 16))
        ](
            q,
            mean_k,
            active_counts,
            attention_lse_0,
            attention_lse_1,
            local_lse,
            sink_k,
            top_slots,
            route_head_counts,
            route_offsets,
            query_len,
            state_len,
            dispatch_state_len,
            LOCAL_LSE_BATCH_STRIDE=local_lse.stride(0),
            LOCAL_LSE_HEAD_STRIDE=local_lse.stride(1),
            LOCAL_LSE_TOKEN_STRIDE=local_lse.stride(2),
            SINK_K_BATCH_STRIDE=sink_k.stride(0),
            SINK_K_HEAD_STRIDE=sink_k.stride(1),
            SINK_K_TOKEN_STRIDE=sink_k.stride(2),
            QUERY_HEADS=query_heads,
            KV_HEADS=kv_heads,
            KV_GROUP_SIZE=kv_group_size,
            HEAD_DIM=head_dim,
            ROUTE_COUNT=route_count,
            SINK_LEN=int(sink_k.size(2)),
            HAS_SECOND_PARTITION=split_at != 0,
            SCALE=float(scale),
            COVERAGE=float(exact_mass_coverage),
            BLOCK_M=16,
            BLOCK_D=triton.next_power_of_2(head_dim),
            num_warps=8 if head_dim > 128 else 4,
        )
    coarse = AiterPrefillCoarse(
        output_0=attention_out_0,
        lse_0=attention_lse_0,
        output_1=attention_out_1,
        lse_1=attention_lse_1,
        mean_k=mean_k,
        mean_v=mean_v,
        counts=active_counts,
        has_second_partition=split_at != 0,
    )
    return top_slots, coarse, route_head_counts, route_offsets


__all__ = [
    "AiterPrefillCoarse",
    "aiter_prefill_route_coarse_attention",
    "merge_aiter_prefill_refinement",
]
