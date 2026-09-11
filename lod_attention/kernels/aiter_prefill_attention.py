"""AITER-backed top-four routing and coarse attention for prefill."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Callable, Optional

import torch
import triton
import triton.language as tl


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
    do_not_specialize=["QUERY_LEN", "ACTIVE_BLOCKS"],
    do_not_specialize_on_alignment=["QUERY_LEN", "ACTIVE_BLOCKS"],
)
def _reduce_route4_candidates_kernel(
    candidates,
    output,
    QUERY_LEN,
    ACTIVE_BLOCKS,
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
    for output_rank in tl.static_range(0, 3):
        output_slot = tl.min(remaining_slots, axis=1)
        tl.store(output + output_base + output_rank, output_slot, mask=valid_query)
        remaining_slots = tl.where(
            remaining_slots == output_slot[:, None],
            0x7FFFFFFFFFFFFFFF,
            remaining_slots,
        )
    tl.store(output + output_base + 3, boundary_slot, mask=valid_query)


def _reduce_route4_candidates(candidates: torch.Tensor) -> torch.Tensor:
    if not candidates.is_cuda or not candidates.is_contiguous():
        raise ValueError("AITER route candidates must be contiguous on the GPU")
    if candidates.ndim != 5 or int(candidates.size(3)) != 8:
        raise RuntimeError(
            "active AITER is missing compact top-four LoD route candidates; "
            "apply integrations/vllm_lod/patches/"
            "aiter-mha-prefill-route4.patch and rebuild AITER"
        )
    batch, query_heads, active_blocks, _, query_len = candidates.shape
    output = torch.empty(
        batch,
        query_heads,
        query_len,
        4,
        dtype=torch.long,
        device=candidates.device,
    )
    block_m = 8
    candidate_block = triton.next_power_of_2(active_blocks * 4)
    _reduce_route4_candidates_kernel[
        (batch * query_heads, triton.cdiv(query_len, block_m))
    ](
        candidates,
        output,
        query_len,
        active_blocks,
        BLOCK_M=block_m,
        CANDIDATE_BLOCK=candidate_block,
        num_warps=4,
    )
    return output


@triton.jit(
    do_not_specialize=[
        "QUERY_LEN",
        "ACTIVE_BLOCKS_0",
        "ACTIVE_BLOCKS_1",
        "SECOND_INDEX_OFFSET",
    ],
    do_not_specialize_on_alignment=[
        "QUERY_LEN",
        "ACTIVE_BLOCKS_0",
        "ACTIVE_BLOCKS_1",
        "SECOND_INDEX_OFFSET",
    ],
)
def _reduce_split_route4_candidates_kernel(
    candidates_0,
    candidates_1,
    output,
    QUERY_LEN,
    ACTIVE_BLOCKS_0,
    ACTIVE_BLOCKS_1,
    SECOND_INDEX_OFFSET,
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
    for output_rank in tl.static_range(0, 3):
        output_slot = tl.min(remaining_slots, axis=1)
        tl.store(output + output_base + output_rank, output_slot, mask=valid_query)
        remaining_slots = tl.where(
            remaining_slots == output_slot[:, None],
            0x7FFFFFFFFFFFFFFF,
            remaining_slots,
        )
    tl.store(output + output_base + 3, boundary_slot, mask=valid_query)


def _reduce_split_route4_candidates(
    candidates_0: torch.Tensor,
    candidates_1: torch.Tensor,
    *,
    second_index_offset: int,
) -> torch.Tensor:
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
    output = torch.empty(
        batch,
        query_heads,
        query_len,
        4,
        dtype=torch.long,
        device=candidates_0.device,
    )
    block_m = 8
    candidate_block = triton.next_power_of_2((blocks_0 + blocks_1) * 4)
    _reduce_split_route4_candidates_kernel[
        (batch * query_heads, triton.cdiv(query_len, block_m))
    ](
        candidates_0,
        candidates_1,
        output,
        query_len,
        blocks_0,
        blocks_1,
        second_index_offset,
        BLOCK_M=block_m,
        CANDIDATE_BLOCK=candidate_block,
        num_warps=4,
    )
    return output


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


def aiter_prefill_route_coarse_attention(
    q: torch.Tensor,
    mean_k: torch.Tensor,
    state_v: torch.Tensor,
    counts: torch.Tensor,
    *,
    state_len: int,
    kv_group_size: int,
    scale: float,
    normalize_route_query: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute top-four routes and their exact coarse-state complement.

    The patched AITER FMHA emits four winners per native key tile while it
    computes ordinary count-weighted centroid attention. Only those compact
    candidates are reduced, so no query-by-centroid routing tensor is stored.
    """
    tensors = (q, mean_k, state_v, counts)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("AITER route/coarse prefill requires GPU tensors")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("AITER route/coarse prefill requires contiguous tensors")
    batch, query_heads, query_len, head_dim = q.shape
    kv_heads = int(mean_k.size(1))
    if query_len <= 1:
        raise ValueError("AITER route/coarse prefill requires multiple queries")
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("AITER route/coarse prefill has incompatible GQA geometry")
    if head_dim > 256 or int(state_v.size(-1)) != head_dim:
        raise ValueError("AITER route/coarse prefill requires equal heads up to 256")
    if tuple(mean_k.shape) != (batch, kv_heads, state_len, head_dim):
        raise ValueError("AITER route/coarse prefill received the wrong mean keys")
    if tuple(state_v.shape[:3]) != tuple(counts.shape[:3]):
        raise ValueError("AITER route/coarse state/count geometry differs")
    if tuple(state_v.shape[:2]) != (batch, kv_heads):
        raise ValueError("AITER route/coarse state heads differ")
    if state_len > int(state_v.size(2)):
        raise ValueError("AITER route/coarse state exceeds its storage")

    active_counts = counts[..., :state_len, :].clamp_min(1.0)
    mean_v = (
        state_v[..., :state_len, :] / active_counts.to(state_v.dtype)
    ).contiguous()
    q_aiter = q.permute(0, 2, 1, 3)
    k_aiter = mean_k.permute(0, 2, 1, 3)
    v_aiter = mean_v.permute(0, 2, 1, 3)
    log_count_bias = (
        active_counts[..., 0]
        .log()
        .to(q.dtype)
        .repeat_interleave(kv_group_size, dim=1)
        .unsqueeze(2)
    )

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
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                None,
                log_count_bias[..., begin:end],
                None,
                None,
                None,
                None,
                None,
            )
            return attention_out, attention_lse, candidates

        split_at = 0
        if 4096 < state_len <= 8192:
            # CK changes to a much slower FMHA dispatch above 4096 keys on
            # MI325X. Two exact partitions plus an LSE merge preserve the
            # calculation while keeping both calls on the fast dispatch. Its
            # exact-4096 specialization is faster than two equal partitions.
            split_at = 4096
            attention_out_0, attention_lse_0, candidates_0 = run_partition(
                0, split_at
            )
            attention_out_1, attention_lse_1, candidates_1 = run_partition(
                split_at, state_len
            )
        else:
            attention_out_0, attention_lse_0, candidates_0 = run_partition(
                0, state_len
            )
            attention_out_1 = attention_out_0
            attention_lse_1 = attention_lse_0
    finally:
        sys.setdlopenflags(original_dlopen_flags)

    if split_at:
        top_slots = _reduce_split_route4_candidates(
            candidates_0,
            candidates_1,
            second_index_offset=split_at,
        )
    else:
        top_slots = _reduce_route4_candidates(candidates_0)
    output = torch.empty_like(q)
    remove_block_m = 8
    _remove_route4_from_coarse_kernel[
        (batch * query_heads, triton.cdiv(query_len, remove_block_m))
    ](
        q,
        mean_k,
        mean_v,
        active_counts,
        top_slots,
        attention_out_0,
        attention_lse_0,
        attention_out_1,
        attention_lse_1,
        output,
        query_len,
        state_len,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        HEAD_DIM=head_dim,
        SCALE=float(scale),
        BLOCK_M=remove_block_m,
        BLOCK_D=triton.next_power_of_2(head_dim),
        HAS_SECOND_PARTITION=split_at != 0,
        num_warps=4,
    )
    return top_slots, output, attention_lse_0


__all__ = ["aiter_prefill_route_coarse_attention"]
