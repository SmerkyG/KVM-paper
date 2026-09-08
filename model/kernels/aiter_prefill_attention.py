"""AITER-backed routing and weighted coarse attention for LOD prefill."""

from __future__ import annotations

import os
import sys

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=["QUERY_LEN", "ACTIVE_BLOCKS"],
    do_not_specialize_on_alignment=["QUERY_LEN", "ACTIVE_BLOCKS"],
)
def _reduce_aiter_route_candidates_kernel(
    candidates,
    output,
    QUERY_LEN,
    ACTIVE_BLOCKS,
    QUERY_HEADS: tl.constexpr,
    BLOCK_CAPACITY: tl.constexpr,
    CANDIDATES_PER_TILE: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    PROTECTED_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    CANDIDATE_BLOCK: tl.constexpr,
):
    batch_head = tl.program_id(0).to(tl.int64)
    query = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    candidate = tl.arange(0, CANDIDATE_BLOCK)
    block = candidate // CANDIDATES_PER_TILE
    rank = candidate - block * CANDIDATES_PER_TILE
    valid_query = query < QUERY_LEN
    valid_candidate = block < ACTIVE_BLOCKS
    base = (
        (
            (batch_head * BLOCK_CAPACITY + block)
            * (2 * CANDIDATES_PER_TILE)
            + rank
        )
        * QUERY_LEN
        + query[:, None]
    )
    scores = tl.load(
        candidates + base,
        mask=valid_query[:, None] & valid_candidate[None, :],
        other=-float("inf"),
    ).to(tl.float32)
    index_values = tl.load(
        candidates + base + CANDIDATES_PER_TILE * QUERY_LEN,
        mask=valid_query[:, None] & valid_candidate[None, :],
        other=-1.0,
    )
    valid_index = index_values >= PROTECTED_LEN
    indices = tl.where(valid_index, index_values, 0.0).to(tl.int64)
    scores = tl.where(valid_index, scores, -float("inf"))
    output_base = (batch_head * QUERY_LEN + query) * ROUTE_COUNT
    route_rank = tl.arange(0, CANDIDATES_PER_TILE)
    selected_slots = tl.full(
        (BLOCK_M, CANDIDATES_PER_TILE), -1, tl.int64
    )
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
    # Preserve the existing ``reorder_like_torch`` contract used by expert
    # grouping: keep the lowest-scoring boundary winner last and sort the
    # preceding selected centroid IDs.
    boundary_slot = tl.max(
        tl.where(
            route_rank[None, :] == ROUTE_COUNT - 1,
            selected_slots,
            -1,
        ),
        axis=1,
    )
    remaining_slots = tl.where(
        route_rank[None, :] < ROUTE_COUNT - 1,
        selected_slots,
        0x7FFFFFFFFFFFFFFF,
    )
    for output_rank in tl.static_range(0, ROUTE_COUNT - 1):
        output_slot = tl.min(remaining_slots, axis=1)
        tl.store(
            output + output_base + output_rank,
            output_slot,
            mask=valid_query,
        )
        remaining_slots = tl.where(
            remaining_slots == output_slot[:, None],
            0x7FFFFFFFFFFFFFFF,
            remaining_slots,
        )
    tl.store(
        output + output_base + ROUTE_COUNT - 1,
        boundary_slot,
        mask=valid_query,
    )


def reduce_aiter_route_candidates(
    candidates: torch.Tensor,
    *,
    active_blocks: int,
    route_count: int,
    protected_len: int = 0,
) -> torch.Tensor:
    """Reduce CK-emitted per-key-tile winners to exact global routes."""
    if not candidates.is_cuda or not candidates.is_contiguous():
        raise ValueError("AITER route candidates must be contiguous on the GPU")
    if candidates.ndim != 5 or int(candidates.size(3)) not in (6, 8):
        raise ValueError("AITER route candidates require [B,H,blocks,6|8,Q]")
    batch, query_heads, block_capacity, _, query_len = candidates.shape
    candidates_per_tile = int(candidates.size(3)) // 2
    if not 0 < active_blocks <= block_capacity:
        raise ValueError("active AITER route blocks exceed their allocation")
    if route_count not in (2, 3, 4):
        raise ValueError("AITER route reduction currently supports top-2/top-3/top-4")
    output = torch.empty(
        batch,
        query_heads,
        query_len,
        route_count,
        dtype=torch.long,
        device=candidates.device,
    )
    block_m = 8
    candidate_block = triton.next_power_of_2(
        active_blocks * candidates_per_tile
    )
    _reduce_aiter_route_candidates_kernel[
        (batch * query_heads, triton.cdiv(query_len, block_m))
    ](
        candidates,
        output,
        query_len,
        active_blocks,
        QUERY_HEADS=query_heads,
        BLOCK_CAPACITY=block_capacity,
        CANDIDATES_PER_TILE=candidates_per_tile,
        ROUTE_COUNT=route_count,
        PROTECTED_LEN=protected_len,
        BLOCK_M=block_m,
        CANDIDATE_BLOCK=candidate_block,
        num_warps=4,
    )
    return output


@triton.jit(
    do_not_specialize=["QUERY_LEN", "STATE_LEN"],
    do_not_specialize_on_alignment=["QUERY_LEN", "STATE_LEN"],
)
def _remove_prefill_routes_kernel(
    q,
    mean_k,
    mean_v,
    counts,
    slots,
    attention_out,
    attention_lse,
    output,
    QUERY_LEN,
    STATE_LEN,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Remove exact-route centroids from an AITER coarse partition."""
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
    full_lse = tl.load(
        attention_lse + query_row,
        mask=valid_query,
        other=0.0,
    ).to(tl.float32)
    aiter_row = (batch * QUERY_LEN + query) * QUERY_HEADS + query_head
    remainder = tl.load(
        attention_out + aiter_row[:, None] * HEAD_DIM + dim[None, :],
        mask=valid_query[:, None] & valid_dim[None, :],
        other=0.0,
    ).to(tl.float32)
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
        attention_lse + query_row,
        full_lse + tl.log(remaining_mass),
        mask=valid_query,
    )


def aiter_prefill_coarse_attention(
    q: torch.Tensor,
    mean_k: torch.Tensor,
    state_v: torch.Tensor,
    counts: torch.Tensor,
    top_slots: torch.Tensor,
    *,
    state_len: int,
    kv_group_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the weighted state remainder with AITER's CK FMHA kernel.

    The accompanying AITER patch permits a broadcast query dimension in a
    per-batch, per-head bias.  That represents ``log(count)`` without
    materializing a query-by-state bias tensor or repeating GQA K/V heads.
    """
    tensors = (q, mean_k, state_v, counts, top_slots)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("AITER coarse prefill requires CUDA tensors")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("AITER coarse prefill requires contiguous tensors")
    batch, query_heads, query_len, head_dim = q.shape
    kv_heads = int(mean_k.size(1))
    if query_len <= 1:
        raise ValueError("AITER coarse prefill requires multiple queries")
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("AITER coarse prefill has incompatible GQA geometry")
    if head_dim > 256 or int(state_v.size(-1)) != head_dim:
        raise ValueError("AITER coarse prefill supports equal heads up to 256")
    if tuple(mean_k.shape) != (batch, kv_heads, state_len, head_dim):
        raise ValueError("AITER coarse prefill received the wrong mean keys")
    if tuple(state_v.shape[:3]) != tuple(counts.shape[:3]):
        raise ValueError("AITER coarse prefill state/count geometry differs")
    if tuple(state_v.shape[:2]) != (batch, kv_heads):
        raise ValueError("AITER coarse prefill state heads differ")
    if state_len > int(state_v.size(2)):
        raise ValueError("AITER coarse prefill state exceeds its storage")
    if tuple(top_slots.shape[:3]) != (batch, query_heads, query_len):
        raise ValueError("AITER coarse prefill routes differ from its queries")
    active_counts = counts[..., :state_len, :].clamp_min(1.0)
    mean_v = (
        state_v[..., :state_len, :] / active_counts.to(state_v.dtype)
    ).contiguous()
    # CK's FMHA interface consumes the tensor strides for the batch, token,
    # and head axes; only the feature axis has to be contiguous.  Preserve
    # these permutations as views.  Materializing q_aiter is especially
    # expensive for prefill because it copies every query head in the chunk.
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
        from aiter.ops.mha import flash_attn_func

        if os.getenv("VLLM_LOD_AITER_COARSE_PER_KV_HEAD") == "1":
            # Stock AITER accepts only one [Q, K] bias shared by the whole
            # launch.  Preserve native GQA reuse while issuing one launch for
            # each independent KV head; this is a useful unpatched-AITER path
            # for measuring whether CK FMHA is worth integrating more deeply.
            attention_out = torch.empty_like(q_aiter)
            attention_lse = torch.empty(
                batch,
                query_heads,
                query_len,
                dtype=torch.float32,
                device=q.device,
            )
            for batch_index in range(batch):
                for kv_head in range(kv_heads):
                    query_begin = kv_head * kv_group_size
                    query_end = query_begin + kv_group_size
                    head_bias = (
                        active_counts[batch_index, kv_head, :, 0]
                        .log()
                        .to(q.dtype)
                        .unsqueeze(0)
                        .expand(query_len, -1)
                    )
                    head_out, head_lse = flash_attn_func(
                        q_aiter[
                            batch_index : batch_index + 1,
                            :,
                            query_begin:query_end,
                            :,
                        ],
                        k_aiter[
                            batch_index : batch_index + 1,
                            :,
                            kv_head : kv_head + 1,
                            :,
                        ],
                        v_aiter[
                            batch_index : batch_index + 1,
                            :,
                            kv_head : kv_head + 1,
                            :,
                        ],
                        softmax_scale=scale,
                        causal=False,
                        bias=head_bias,
                        return_lse=True,
                    )
                    attention_out[
                        batch_index : batch_index + 1,
                        :,
                        query_begin:query_end,
                        :,
                    ].copy_(head_out)
                    attention_lse[
                        batch_index : batch_index + 1,
                        query_begin:query_end,
                        :,
                    ].copy_(head_lse)
        else:
            try:
                attention_out, attention_lse = flash_attn_func(
                    q_aiter,
                    k_aiter,
                    v_aiter,
                    softmax_scale=scale,
                    causal=False,
                    bias=log_count_bias,
                    return_lse=True,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "AITER coarse prefill rejected per-head count bias; apply "
                    "integrations/vllm_lod/patches/"
                    "aiter-mha-per-head-bias.patch to the active AITER checkout"
                ) from exc
    finally:
        sys.setdlopenflags(original_dlopen_flags)

    output = torch.empty_like(q)
    route_count = int(top_slots.size(-1))
    if route_count == 0:
        output.copy_(attention_out.permute(0, 2, 1, 3))
        return output, attention_lse
    remove_block_m = 8
    _remove_prefill_routes_kernel[
        (batch * query_heads, triton.cdiv(query_len, remove_block_m))
    ](
        q,
        mean_k,
        mean_v,
        active_counts,
        top_slots,
        attention_out,
        attention_lse,
        output,
        query_len,
        state_len,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        HEAD_DIM=head_dim,
        ROUTE_COUNT=route_count,
        SCALE=float(scale),
        BLOCK_M=remove_block_m,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return output, attention_lse


def aiter_prefill_route_coarse_attention(
    q: torch.Tensor,
    mean_k: torch.Tensor,
    state_v: torch.Tensor,
    counts: torch.Tensor,
    *,
    state_len: int,
    kv_group_size: int,
    scale: float,
    route_count: int,
    protected_len: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select exact top routes while computing the weighted coarse field.

    The LOD AITER patch emits each native state tile's four best routing
    scores and indices while CK already has its QK tile resident.  Reducing
    those compact candidates gives the exact global top-2/top-3/top-4 without
    ever materializing the query-by-state routing matrix.
    """
    tensors = (q, mean_k, state_v, counts)
    if protected_len:
        raise ValueError(
            "combined AITER route/coarse attention cannot exclude a routing "
            "slot while retaining its coarse attention mass; use the "
            "separate exact sink branch"
        )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("AITER route/coarse prefill requires CUDA tensors")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("AITER route/coarse prefill requires contiguous tensors")
    batch, query_heads, query_len, head_dim = q.shape
    kv_heads = int(mean_k.size(1))
    if query_len <= 1:
        raise ValueError("AITER route/coarse prefill requires multiple queries")
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("AITER route/coarse prefill has incompatible GQA geometry")
    if route_count not in (2, 3, 4):
        raise ValueError("AITER route/coarse prefill supports top-2/top-3/top-4")
    if head_dim > 256 or int(state_v.size(-1)) != head_dim:
        raise ValueError("AITER route/coarse prefill supports equal heads up to 256")
    if tuple(mean_k.shape) != (batch, kv_heads, state_len, head_dim):
        raise ValueError("AITER route/coarse prefill received the wrong mean keys")
    if tuple(state_v.shape[:3]) != tuple(counts.shape[:3]):
        raise ValueError("AITER route/coarse prefill state/count geometry differs")
    if tuple(state_v.shape[:2]) != (batch, kv_heads):
        raise ValueError("AITER route/coarse prefill state heads differ")
    if state_len > int(state_v.size(2)):
        raise ValueError("AITER route/coarse prefill state exceeds its storage")

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

        attention_out, attention_lse, candidates, _ = mha_fwd(
            q_aiter,
            k_aiter,
            v_aiter,
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
            log_count_bias,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    finally:
        sys.setdlopenflags(original_dlopen_flags)

    if candidates.ndim != 5 or int(candidates.size(3)) not in (6, 8):
        raise RuntimeError(
            "active AITER is missing compact LOD route candidates; apply the "
            "LOD route/coarse patch before enabling this path"
        )
    top_slots = reduce_aiter_route_candidates(
        candidates,
        active_blocks=int(candidates.size(2)),
        route_count=route_count,
        protected_len=protected_len,
    )
    output = torch.empty_like(q)
    remove_block_m = 8
    _remove_prefill_routes_kernel[
        (batch * query_heads, triton.cdiv(query_len, remove_block_m))
    ](
        q,
        mean_k,
        mean_v,
        active_counts,
        top_slots,
        attention_out,
        attention_lse,
        output,
        query_len,
        state_len,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        HEAD_DIM=head_dim,
        ROUTE_COUNT=route_count,
        SCALE=float(scale),
        BLOCK_M=remove_block_m,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return top_slots, output, attention_lse


def aiter_prefill_route_attention(
    q: torch.Tensor,
    mean_k: torch.Tensor,
    counts: torch.Tensor,
    *,
    state_len: int,
    kv_group_size: int,
    scale: float,
    route_count: int,
    protected_len: int = 0,
) -> torch.Tensor:
    """Select exact top routes without materializing query-by-state scores.

    The patched CK kernel emits four winners from each native key tile and
    skips its softmax/PV stages.  A small Triton reduction then finds the exact
    global top-2/top-3/top-4.  Coarse and leaf attention remain separate so
    the caller can overlap them after routing completes.
    """
    tensors = (q, mean_k, counts)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("AITER prefill routing requires CUDA tensors")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("AITER prefill routing requires contiguous tensors")
    batch, query_heads, query_len, head_dim = q.shape
    kv_heads = int(mean_k.size(1))
    if query_len <= 1:
        raise ValueError("AITER prefill routing requires multiple queries")
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("AITER prefill routing has incompatible GQA geometry")
    if route_count not in (2, 3, 4):
        raise ValueError("AITER prefill routing supports top-2/top-3/top-4")
    if head_dim > 256:
        raise ValueError("AITER prefill routing supports heads up to 256")
    if tuple(mean_k.shape) != (batch, kv_heads, state_len, head_dim):
        raise ValueError("AITER prefill routing received the wrong mean keys")
    if (
        tuple(counts.shape[:2]) != (batch, kv_heads)
        or int(counts.size(2)) < state_len
    ):
        raise ValueError("AITER prefill routing received the wrong counts")

    active_counts = counts[..., :state_len, :].clamp_min(1.0)
    q_aiter = q.permute(0, 2, 1, 3)
    k_aiter = mean_k.permute(0, 2, 1, 3)
    log_count_bias = (
        active_counts[..., 0]
        .log()
        .to(q.dtype)
        .repeat_interleave(kv_group_size, dim=1)
        .unsqueeze(2)
    )
    if protected_len:
        # Exclude protected slots before CK retains only four candidates per
        # native key tile.  Masking them only in the global reducer could
        # discard a tile candidate and invalidate exact global top-k. Use the
        # finite dtype minimum because route-only CK recovers the raw dot
        # product by subtracting this bias; ``-inf - -inf`` would be NaN.
        log_count_bias[..., :protected_len] = torch.finfo(q.dtype).min

    original_dlopen_flags = sys.getdlopenflags()
    deepbind = getattr(os, "RTLD_DEEPBIND", 0)
    if deepbind:
        sys.setdlopenflags(original_dlopen_flags | deepbind)
    try:
        from aiter.ops.mha import mha_fwd

        _, _, candidates, _ = mha_fwd(
            q_aiter,
            k_aiter,
            # Route-only CK never reads V; a valid same-geometry view keeps
            # the public AITER interface unchanged and avoids a value copy.
            k_aiter,
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
            log_count_bias,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    finally:
        sys.setdlopenflags(original_dlopen_flags)

    if candidates.ndim != 5 or int(candidates.size(3)) not in (6, 8):
        raise RuntimeError(
            "active AITER is missing compact LOD route candidates; apply the "
            "LOD route-only patch before enabling this path"
        )
    expected_block_capacity = (state_len + 63) // 64
    if int(candidates.size(2)) != expected_block_capacity:
        raise RuntimeError(
            "AITER route candidate ABI capacity drifted: expected storage for "
            f"ceil({state_len}/64)={expected_block_capacity} key tiles, got "
            f"{int(candidates.size(2))}"
        )
    return reduce_aiter_route_candidates(
        candidates,
        # CK reserves for its smallest supported N tile. Larger selected tiles
        # leave the remaining candidates at -inf, which this reduction masks.
        active_blocks=int(candidates.size(2)),
        route_count=route_count,
        protected_len=protected_len,
    )


__all__ = [
    "aiter_prefill_coarse_attention",
    "aiter_prefill_route_attention",
    "aiter_prefill_route_coarse_attention",
    "reduce_aiter_route_candidates",
]
