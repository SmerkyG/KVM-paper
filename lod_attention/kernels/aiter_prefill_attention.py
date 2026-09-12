"""AITER-backed top-four routing and coarse attention for prefill."""

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
    mean_k,
    mean_v,
    active_counts,
    log_count_bias,
    STATE_LEN,
    DISPATCH_STATE_LEN,
    STATE_CAPACITY: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
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
    tl.store(active_counts + row, count)
    group = tl.arange(0, KV_GROUP_SIZE)
    query_head = kv_head * KV_GROUP_SIZE + group
    bias_offset = (
        (batch * KV_HEADS * KV_GROUP_SIZE + query_head) * DISPATCH_STATE_LEN
        + slot
    )
    tl.store(
        log_count_bias + bias_offset,
        tl.where(valid_slot, tl.log(count), -float("inf")),
    )


@lru_cache(maxsize=1)
def _raw_route_mha_fwd() -> Callable[..., tuple[torch.Tensor, ...]]:
    """Build the AITER specialization whose route probe uses raw queries."""
    from aiter.jit.core import compile_ops, get_args_of_build
    from aiter.ops.mha import cmdGenFunc_mha_fwd

    def raw_route_build_args(*args: object, **kwargs: object) -> dict[str, object]:
        q = args[0] if args else kwargs["q"]
        if not isinstance(q, torch.Tensor) or int(q.size(-1)) != 256:
            raise ValueError("raw AITER routing is specialized for Qwen D=256")
        generated = cmdGenFunc_mha_fwd(*args, **kwargs)
        generated["md_name"] = f"{generated['md_name']}_lod_route_raw"
        generated["blob_gen_cmd"] = [
            command.replace(" --output_dir", " --optdim 256 --output_dir")
            for command in generated["blob_gen_cmd"]
        ]
        base_flags = list(get_args_of_build("module_mha_fwd")["flags_extra_hip"])
        generated["flags_extra_hip"] = [
            flag
            for flag in base_flags
            if not flag.startswith("-DCK_TILE_FMHA_ROUTE_QUERY_NORMALIZE=")
        ] + ["-DCK_TILE_FMHA_ROUTE_QUERY_NORMALIZE=0"]
        return generated

    @compile_ops(
        "module_mha_fwd",
        fc_name="mha_fwd",
        gen_func=raw_route_build_args,
    )
    def raw_route_mha_fwd(
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

    return raw_route_mha_fwd


@triton.jit(
    do_not_specialize=["QUERY_LEN", "ACTIVE_BLOCKS", "STATE_LEN"],
    do_not_specialize_on_alignment=["QUERY_LEN", "ACTIVE_BLOCKS", "STATE_LEN"],
)
def _reduce_route4_candidates_kernel(
    candidates,
    output,
    head_counts,
    route_offsets,
    QUERY_LEN,
    ACTIVE_BLOCKS,
    STATE_LEN,
    BLOCK_M: tl.constexpr,
    CANDIDATE_BLOCK: tl.constexpr,
):
    """Reduce four winners from each native AITER key tile to global top-four."""
    batch_head = tl.program_id(0).to(tl.int64)
    query = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    candidate = tl.arange(0, CANDIDATE_BLOCK)
    block = candidate // 4
    rank = candidate - block * 4
    valid_query = query < QUERY_LEN
    valid_candidate = block < ACTIVE_BLOCKS
    base = (
        ((batch_head * ACTIVE_BLOCKS + block) * 8 + rank) * QUERY_LEN
        + query[:, None]
    )
    scores = tl.load(
        candidates + base,
        mask=valid_query[:, None] & valid_candidate[None, :],
        other=-float("inf"),
    ).to(tl.float32)
    index_values = tl.load(
        candidates + base + 4 * QUERY_LEN,
        mask=valid_query[:, None] & valid_candidate[None, :],
        other=-1.0,
    )
    valid_index = index_values >= 0
    indices = tl.where(valid_index, index_values, 0.0).to(tl.int64)
    scores = tl.where(valid_index, scores, -float("inf"))
    route_rank = tl.arange(0, 4)
    selected_slots = tl.full((BLOCK_M, 4), -1, tl.int64)
    for output_rank in tl.static_range(0, 4):
        selected_candidate = tl.argmax(scores, axis=1)
        selected_index = tl.sum(
            tl.where(
                candidate[None, :] == selected_candidate[:, None],
                indices,
                0,
            ),
            axis=1,
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
    # other three routes ordered by slot ID.
    output_base = (batch_head * QUERY_LEN + query) * 4
    boundary_slot = tl.max(
        tl.where(route_rank[None, :] == 3, selected_slots, -1), axis=1
    )
    remaining_slots = tl.where(
        route_rank[None, :] < 3,
        selected_slots,
        0x7FFFFFFFFFFFFFFF,
    )
    ordered_slots = tl.full((BLOCK_M, 4), -1, tl.int64)
    for output_rank in tl.static_range(0, 3):
        output_slot = tl.min(remaining_slots, axis=1)
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
    tl.store(output + output_base + 3, boundary_slot, mask=valid_query)
    ordered_slots = tl.where(
        route_rank[None, :] == 3,
        boundary_slot[:, None],
        ordered_slots,
    )
    for rank in tl.static_range(0, 4):
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


def _reduce_route4_candidates(
    candidates: torch.Tensor,
    *,
    state_len: int,
    head_dim: int,
    buffers: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not candidates.is_cuda or not candidates.is_contiguous():
        raise ValueError("AITER route candidates must be contiguous on the GPU")
    if candidates.ndim != 5 or int(candidates.size(3)) != 8:
        raise RuntimeError(
            "active AITER is missing compact top-four LoD route candidates; "
            "apply integrations/vllm_lod/patches/"
            "aiter-mha-prefill-route4.patch and rebuild AITER"
        )
    batch, query_heads, active_blocks, _, query_len = candidates.shape
    output = _workspace_tensor(
        buffers,
        "route_slots",
        (batch, query_heads, query_len, 4),
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
    candidate_block = triton.next_power_of_2(active_blocks * 4)
    if head_dim <= 128:
        block_m = 16
        num_warps = 2
    else:
        block_m = 32 if candidate_block <= 128 else 16
        num_warps = 4
    _reduce_route4_candidates_kernel[
        (batch * query_heads, triton.cdiv(query_len, block_m))
    ](
        candidates,
        output,
        head_counts,
        route_offsets,
        query_len,
        active_blocks,
        state_len,
        BLOCK_M=block_m,
        CANDIDATE_BLOCK=candidate_block,
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
def _reduce_split_route4_candidates_kernel(
    candidates_0,
    candidates_1,
    output,
    head_counts,
    route_offsets,
    QUERY_LEN,
    ACTIVE_BLOCKS_0,
    ACTIVE_BLOCKS_1,
    SECOND_INDEX_OFFSET,
    STATE_LEN,
    BLOCK_M: tl.constexpr,
    CANDIDATE_BLOCK: tl.constexpr,
):
    """Reduce compact candidates from two exact attention partitions."""
    batch_head = tl.program_id(0).to(tl.int64)
    query = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    candidate = tl.arange(0, CANDIDATE_BLOCK)
    first_candidates = ACTIVE_BLOCKS_0 * 4
    from_first = candidate < first_candidates
    local_candidate = tl.where(from_first, candidate, candidate - first_candidates)
    block = local_candidate // 4
    rank = local_candidate - block * 4
    valid_query = query < QUERY_LEN
    valid_first = from_first & (block < ACTIVE_BLOCKS_0)
    valid_second = (~from_first) & (block < ACTIVE_BLOCKS_1)

    base_0 = (
        ((batch_head * ACTIVE_BLOCKS_0 + block) * 8 + rank) * QUERY_LEN
        + query[:, None]
    )
    base_1 = (
        ((batch_head * ACTIVE_BLOCKS_1 + block) * 8 + rank) * QUERY_LEN
        + query[:, None]
    )
    score_0 = tl.load(
        candidates_0 + base_0,
        mask=valid_query[:, None] & valid_first[None, :],
        other=-float("inf"),
    ).to(tl.float32)
    score_1 = tl.load(
        candidates_1 + base_1,
        mask=valid_query[:, None] & valid_second[None, :],
        other=-float("inf"),
    ).to(tl.float32)
    index_0 = tl.load(
        candidates_0 + base_0 + 4 * QUERY_LEN,
        mask=valid_query[:, None] & valid_first[None, :],
        other=-1.0,
    )
    index_1 = tl.load(
        candidates_1 + base_1 + 4 * QUERY_LEN,
        mask=valid_query[:, None] & valid_second[None, :],
        other=-1.0,
    )
    index_values = tl.where(from_first[None, :], index_0, index_1)
    valid_index = index_values >= 0
    indices = tl.where(valid_index, index_values, 0.0).to(tl.int64)
    indices += tl.where(from_first[None, :], 0, SECOND_INDEX_OFFSET)
    scores = tl.where(
        valid_index,
        tl.where(from_first[None, :], score_0, score_1),
        -float("inf"),
    )

    route_rank = tl.arange(0, 4)
    selected_slots = tl.full((BLOCK_M, 4), -1, tl.int64)
    for output_rank in tl.static_range(0, 4):
        selected_candidate = tl.argmax(scores, axis=1)
        selected_index = tl.sum(
            tl.where(
                candidate[None, :] == selected_candidate[:, None],
                indices,
                0,
            ),
            axis=1,
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

    output_base = (batch_head * QUERY_LEN + query) * 4
    boundary_slot = tl.max(
        tl.where(route_rank[None, :] == 3, selected_slots, -1), axis=1
    )
    remaining_slots = tl.where(
        route_rank[None, :] < 3,
        selected_slots,
        0x7FFFFFFFFFFFFFFF,
    )
    ordered_slots = tl.full((BLOCK_M, 4), -1, tl.int64)
    for output_rank in tl.static_range(0, 3):
        output_slot = tl.min(remaining_slots, axis=1)
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
    tl.store(output + output_base + 3, boundary_slot, mask=valid_query)
    ordered_slots = tl.where(
        route_rank[None, :] == 3,
        boundary_slot[:, None],
        ordered_slots,
    )
    for rank in tl.static_range(0, 4):
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


def _reduce_split_route4_candidates(
    candidates_0: torch.Tensor,
    candidates_1: torch.Tensor,
    *,
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
    if int(candidates_0.size(3)) != 8 or int(candidates_1.size(3)) != 8:
        raise RuntimeError("active AITER is missing compact top-four candidates")
    if tuple(candidates_0.shape[:2]) != tuple(candidates_1.shape[:2]):
        raise ValueError("split AITER route candidate heads differ")
    if int(candidates_0.size(4)) != int(candidates_1.size(4)):
        raise ValueError("split AITER route candidate query lengths differ")
    batch, query_heads, blocks_0, _, query_len = candidates_0.shape
    blocks_1 = int(candidates_1.size(2))
    output = _workspace_tensor(
        buffers,
        "route_slots",
        (batch, query_heads, query_len, 4),
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
    candidate_block = triton.next_power_of_2((blocks_0 + blocks_1) * 4)
    if head_dim <= 128:
        block_m = 16
        num_warps = 2
    else:
        block_m = 32 if candidate_block <= 128 else 16
        num_warps = 4
    _reduce_split_route4_candidates_kernel[
        (batch * query_heads, triton.cdiv(query_len, block_m))
    ](
        candidates_0,
        candidates_1,
        output,
        head_counts,
        route_offsets,
        query_len,
        blocks_0,
        blocks_1,
        second_index_offset,
        state_len,
        BLOCK_M=block_m,
        CANDIDATE_BLOCK=candidate_block,
        num_warps=num_warps,
    )
    return output, head_counts, route_offsets


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
def _merge_route4_refinement_kernel(
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
):
    """Replace four coarse centroids with their leaves and merge local/sink."""
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
    route_rank = tl.arange(0, 4)
    route_scores = tl.load(
        route_lse + query_row[:, None] * 4 + route_rank[None, :],
        mask=valid_query[:, None],
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
    for rank in tl.static_range(0, 4):
        route_score = tl.load(
            route_lse + query_row * 4 + rank,
            mask=valid_query,
            other=-float("inf"),
        ).to(tl.float32)
        route_weight = tl.exp(route_score - maximum)
        route_value = tl.load(
            route_out
            + (query_row * 4 + rank)[:, None] * HEAD_DIM
            + dim[None, :],
            mask=valid_query[:, None] & valid_dim[None, :],
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
    expected = (batch, query_heads, query_len, head_dim)
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("AITER refinement has incompatible GQA geometry")
    if tuple(local_out.shape) != expected or tuple(local_lse.shape) != expected[:-1]:
        raise ValueError("AITER refinement local branch has incompatible geometry")
    if tuple(route_out.shape) != (*expected[:-1], 4, head_dim):
        raise ValueError("AITER refinement route output has incompatible geometry")
    if tuple(route_lse.shape) != (*expected[:-1], 4):
        raise ValueError("AITER refinement route LSE has incompatible geometry")
    if tuple(slots.shape) != (*expected[:-1], 4):
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
    _merge_route4_refinement_kernel[
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
    state_len: int,
    kv_group_size: int,
    scale: float,
    normalize_route_query: bool,
    buffers: dict[str, torch.Tensor] | None = None,
) -> tuple[
    torch.Tensor, AiterPrefillCoarse, torch.Tensor, torch.Tensor
]:
    """Compute top-four routes and retain unmodified coarse attention.

    The patched AITER FMHA emits four winners per native key tile while it
    computes ordinary count-weighted centroid attention. Only those compact
    candidates are reduced, so no query-by-centroid routing tensor is stored.
    """
    tensors = (q, state_k, state_v, counts)
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
        mean_k,
        mean_v,
        active_counts,
        log_count_bias,
        state_len,
        dispatch_state_len,
        STATE_CAPACITY=int(state_k.size(2)),
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
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

        route_mha_fwd = mha_fwd if normalize_route_query else _raw_route_mha_fwd()

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
        top_slots, route_head_counts, route_offsets = _reduce_split_route4_candidates(
            candidates_0,
            candidates_1,
            second_index_offset=split_at,
            state_len=state_len,
            head_dim=head_dim,
            buffers=buffers,
        )
    else:
        top_slots, route_head_counts, route_offsets = _reduce_route4_candidates(
            candidates_0,
            state_len=state_len,
            head_dim=head_dim,
            buffers=buffers,
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
