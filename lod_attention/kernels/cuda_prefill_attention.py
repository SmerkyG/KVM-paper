"""SM90 local FlashAttention and fused count-corrected prefill routing.

The routing producer keeps only eight winners per query in registers while
streaming centroid tiles. It never writes a query-by-centroid score matrix.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._paged_common import _pack_route_score_index, _unpack_route_score_index
from .aiter_prefill_attention import (
    AiterPrefillCoarse,
    _prepare_aiter_state_kernel,
    _workspace_tensor,
)


@triton.jit
def _cuda_route_coarse_kernel(
    Q,
    K,
    V,
    Counts,
    Out,
    LSE,
    Slots,
    HeadCounts,
    Offsets,
    QLEN: tl.constexpr,
    SLEN: tl.constexpr,
    QHEADS: tl.constexpr,
    KVHEADS: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    NORMALIZE: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    bh = tl.program_id(0)
    batch = bh // QHEADS
    head = bh % QHEADS
    kv = batch * KVHEADS + head // GROUP
    row = tl.program_id(1) * M + tl.arange(0, M)
    dim = tl.arange(0, D)
    col = tl.arange(0, N)
    query = tl.load(
        Q + (bh * QLEN + row[:, None]) * D + dim[None, :],
        mask=row[:, None] < QLEN,
        other=0,
    )
    query_float = query.to(tl.float32)
    inverse_rms = tl.rsqrt(
        tl.maximum(tl.sum(query_float * query_float, 1) / D, 1.0e-12)
    )
    maximum = tl.full((M,), -float("inf"), tl.float32)
    denominator = tl.zeros((M,), tl.float32)
    numerator = tl.zeros((M, D), tl.float32)
    winners = tl.full((M, 8), -9223372036854775807, tl.int64)
    for begin in range(tl.cdiv(SLEN, N)):
        index = begin * N + col
        keys = tl.load(
            K + (kv * SLEN + index[None, :]) * D + dim[:, None],
            mask=index[None, :] < SLEN,
            other=0,
        )
        values = tl.load(
            V + (kv * SLEN + index[:, None]) * D + dim[None, :],
            mask=index[:, None] < SLEN,
            other=0,
        )
        count = tl.load(Counts + kv * SLEN + index, mask=index < SLEN, other=1).to(
            tl.float32
        )
        dot = tl.dot(query, keys) * SCALE
        log_count = tl.log(count)
        score = tl.where(index[None, :] < SLEN, dot + log_count[None, :], -float("inf"))
        new_max = tl.maximum(maximum, tl.max(score, 1))
        correction = tl.exp(maximum - new_max)
        probability = tl.exp(score - new_max[:, None])
        numerator = numerator * correction[:, None] + tl.dot(
            probability.to(values.dtype), values
        )
        denominator = denominator * correction + tl.sum(probability, 1)
        maximum = new_max
        route_score = dot
        if NORMALIZE:
            route_score = route_score * inverse_rms[:, None]
        route_score = tl.where(
            index[None, :] < SLEN, route_score + log_count[None, :], -float("inf")
        )
        packed = _pack_route_score_index(route_score, index[None, :])
        tile_winners = tl.topk(packed, 8, dim=1)
        candidates = tl.reshape(tl.join(winners, tile_winners), (M, 16))
        winners = tl.topk(candidates, 8, dim=1)
    _, selected = _unpack_route_score_index(winners)
    rank = tl.arange(0, 8)
    boundary = tl.sum(tl.where(rank[None, :] == 7, selected, 0), 1)
    ordered = tl.sort(
        tl.where(rank[None, :] < 7, selected, 0x7FFFFFFF), descending=False, dim=1
    )
    ordered = tl.where(rank[None, :] == 7, boundary[:, None], ordered)
    for i in tl.static_range(8):
        slot = tl.sum(tl.where(rank[None, :] == i, ordered, 0), 1)
        position = (bh * QLEN + row) * 8 + i
        local_offset = tl.atomic_add(
            HeadCounts + bh * SLEN + slot, 1, mask=row < QLEN, sem="relaxed"
        )
        tl.store(Slots + position, slot, mask=row < QLEN)
        tl.store(Offsets + position, local_offset, mask=row < QLEN)
    out_row = (batch * QLEN + row) * QHEADS + head
    tl.store(
        Out + out_row[:, None] * D + dim[None, :],
        numerator / denominator[:, None],
        mask=row[:, None] < QLEN,
    )
    tl.store(LSE + bh * QLEN + row, maximum + tl.log(denominator), mask=row < QLEN)


def cuda_prefill_route_coarse_attention(
    q,
    state_k,
    state_v,
    counts,
    *,
    state_len,
    kv_group_size,
    scale,
    normalize_route_query,
    max_open_leaf_tokens=None,
    slot_lengths=None,
    buffers=None,
    block_m=64,
    block_n=64,
    num_warps=4,
):
    """Return exact top-eight routes, unmodified coarse attention, and packing."""
    if torch.version.hip or not q.is_cuda:
        raise ValueError("CUDA prefill routing requires an NVIDIA GPU")
    if max_open_leaf_tokens is not None or slot_lengths is not None:
        raise ValueError("Apply centroid caps after selecting the original top eight")
    batch, heads, queries, dimension = q.shape
    kv_heads = state_k.size(1)
    if dimension != 128 or heads != kv_heads * kv_group_size or state_len < 8:
        raise ValueError("CUDA prefill supports D128 GQA with at least eight centroids")
    if not all(t.is_contiguous() for t in (q, state_k, state_v, counts)):
        raise ValueError("CUDA prefill routing requires contiguous tensors")

    def workspace(name, shape, dtype):
        return _workspace_tensor(buffers, name, shape, dtype=dtype, device=q.device)

    mean_k = workspace(
        "coarse_mean_k", (batch, kv_heads, state_len, dimension), state_k.dtype
    )
    mean_v = workspace(
        "coarse_mean_v", (batch, kv_heads, state_len, dimension), state_v.dtype
    )
    active_counts = workspace(
        "coarse_counts", (batch, kv_heads, state_len, 1), counts.dtype
    )
    bias = workspace("coarse_log_count_bias", (batch, heads, 1, state_len), q.dtype)
    _prepare_aiter_state_kernel[(batch * kv_heads * state_len,)](
        state_k,
        state_v,
        counts,
        mean_k,
        mean_v,
        active_counts,
        bias,
        state_len,
        state_len,
        STATE_CAPACITY=state_k.size(2),
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        BLOCK_G=triton.next_power_of_2(kv_group_size),
        HEAD_DIM=dimension,
        BLOCK_D=dimension,
        num_warps=4,
    )
    output = workspace("coarse_output_0", (batch, queries, heads, dimension), q.dtype)
    lse = workspace("coarse_lse_0", (batch, heads, queries), torch.float32)
    routes = workspace("route_slots", (batch, heads, queries, 8), torch.int64)
    head_counts = workspace(
        "route_head_counts", (batch * heads * state_len,), torch.int32
    )
    offsets = workspace("route_offsets", tuple(routes.shape), torch.int32)
    head_counts.zero_()
    _cuda_route_coarse_kernel[(batch * heads, triton.cdiv(queries, block_m))](
        q,
        mean_k,
        mean_v,
        active_counts,
        output,
        lse,
        routes,
        head_counts,
        offsets,
        QLEN=queries,
        SLEN=state_len,
        QHEADS=heads,
        KVHEADS=kv_heads,
        GROUP=kv_group_size,
        D=dimension,
        SCALE=scale,
        NORMALIZE=normalize_route_query,
        M=block_m,
        N=block_n,
        num_warps=num_warps,
        num_stages=2,
    )
    coarse = AiterPrefillCoarse(
        output, lse, output, lse, mean_k, mean_v, active_counts, False
    )
    return routes, coarse, head_counts, offsets


def cuda_local_attention(q, k, v, *, scale, output_buffer=None, return_lse=True):
    """Native GQA FlashAttention-3 with bottom-right causal suffix alignment."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    batch, heads, queries, dimension = q.shape
    kv_heads = k.size(1)
    keys = k.size(2)
    q_flat = q.permute(0, 2, 1, 3).reshape(batch * queries, heads, dimension)
    k_flat = k.permute(0, 2, 1, 3).reshape(batch * keys, kv_heads, dimension)
    v_flat = v.permute(0, 2, 1, 3).reshape(batch * keys, kv_heads, dimension)
    cuq = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * queries
    cuk = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * keys
    out = (
        None
        if output_buffer is None
        else output_buffer.permute(0, 2, 1, 3).reshape(
            batch * queries, heads, dimension
        )
    )
    result = flash_attn_varlen_func(
        q_flat,
        k_flat,
        v_flat,
        max_seqlen_q=queries,
        cu_seqlens_q=cuq,
        max_seqlen_k=keys,
        cu_seqlens_k=cuk,
        softmax_scale=scale,
        causal=True,
        return_softmax_lse=return_lse,
        out=out,
        fa_version=3,
    )
    if return_lse:
        output, lse = result
        lse = lse.reshape(heads, batch, queries).permute(1, 0, 2)
    else:
        output = result
        lse = torch.empty(0, device=q.device, dtype=torch.float32)
    output = output.reshape(batch, queries, heads, dimension).permute(0, 2, 1, 3)
    if output_buffer is not None and output.data_ptr() != output_buffer.data_ptr():
        output_buffer.copy_(output)
        output = output_buffer
    return output, lse


@triton.jit
def _selected_decode_logits_kernel(
    Q,
    K,
    Counts,
    CacheIndices,
    Slots,
    Scores,
    QHEADS: tl.constexpr,
    KVHEADS: tl.constexpr,
    GROUP: tl.constexpr,
    CAPACITY: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // QHEADS
    head = row % QHEADS
    cache = tl.load(CacheIndices + batch)
    kv = cache * KVHEADS + head // GROUP
    dim = tl.arange(0, D)
    rank = tl.arange(0, 8)
    slot = tl.load(Slots + row * 8 + rank)
    valid = (slot >= 0) & (slot < CAPACITY)
    count = tl.maximum(tl.load(Counts + kv * CAPACITY + slot, mask=valid, other=1), 1.0)
    query = tl.load(Q + row * D + dim).to(tl.float32)
    key = tl.load(
        K + (kv * CAPACITY + slot[:, None]) * D + dim[None, :],
        mask=valid[:, None],
        other=0,
    )
    mean = (key.to(tl.float32) / count[:, None]).to(key.dtype).to(tl.float32)
    logits = tl.sum(mean * query[None, :], 1) * SCALE + tl.log(count)
    tl.store(Scores + row * 8 + rank, tl.where(valid, logits, -float("inf")))


def restore_decode_attention_logits(
    q, state_k, counts, cache_indices, slots, scores, *, kv_group_size, scale
):
    """Keep routing normalization out of exact coarse-mass replacement.

    The grouped producer ranks with q/RMS(q) but computes its coarse branch
    with q. Recompute just eight native logits per head for the final frontier
    merge; normalized routing scores must never enter attention's softmax.
    """
    _selected_decode_logits_kernel[(q.size(0) * q.size(1),)](
        q,
        state_k,
        counts,
        cache_indices,
        slots,
        scores,
        QHEADS=q.size(1),
        KVHEADS=state_k.size(1),
        GROUP=kv_group_size,
        CAPACITY=state_k.size(2),
        D=q.size(-1),
        SCALE=scale,
        num_warps=4,
    )
