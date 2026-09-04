"""AITER-backed weighted coarse attention for LOD prefill."""

from __future__ import annotations

import os
import sys

import torch
import triton
import triton.language as tl


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
    BLOCK_D: tl.constexpr,
):
    """Remove exact-route centroids from an AITER coarse partition."""
    row = tl.program_id(0).to(tl.int64)
    query = row % QUERY_LEN
    batch_head = row // QUERY_LEN
    batch = batch_head // QUERY_HEADS
    query_head = batch_head - batch * QUERY_HEADS
    kv_head = query_head // KV_GROUP_SIZE
    dim = tl.arange(0, BLOCK_D)
    valid_dim = dim < HEAD_DIM
    query_row = (batch * QUERY_HEADS + query_head) * QUERY_LEN + query
    query_value = tl.load(
        q + query_row * HEAD_DIM + dim,
        mask=valid_dim,
        other=0.0,
    )
    full_lse = tl.load(attention_lse + row).to(tl.float32)
    aiter_row = (batch * QUERY_LEN + query) * QUERY_HEADS + query_head
    remainder = tl.load(
        attention_out + aiter_row * HEAD_DIM + dim,
        mask=valid_dim,
        other=0.0,
    ).to(tl.float32)
    selected_mass = tl.zeros((), tl.float32)
    selected_value = tl.zeros((BLOCK_D,), tl.float32)
    for rank in tl.static_range(0, ROUTE_COUNT):
        slot = tl.load(slots + row * ROUTE_COUNT + rank).to(tl.int64)
        valid_slot = (slot >= 0) & (slot < STATE_LEN)
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
            mean_k + state_row + dim,
            mask=valid_slot & valid_dim,
            other=0.0,
        )
        value = tl.load(
            mean_v + state_row + dim,
            mask=valid_slot & valid_dim,
            other=0.0,
        )
        score = tl.sum(query_value * key, axis=0) * SCALE + tl.log(count)
        mass = tl.where(valid_slot, tl.exp(score - full_lse), 0.0)
        selected_mass += mass
        selected_value += mass * value
    remaining_mass = tl.maximum(1.0 - selected_mass, 1.0e-7)
    tl.store(
        output + row * HEAD_DIM + dim,
        (remainder - selected_value) / remaining_mass,
        mask=valid_dim,
    )
    tl.store(
        attention_lse + row,
        full_lse + tl.log(remaining_mass),
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
    _remove_prefill_routes_kernel[(batch * query_heads * query_len,)](
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
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return output, attention_lse


__all__ = ["aiter_prefill_coarse_attention"]
