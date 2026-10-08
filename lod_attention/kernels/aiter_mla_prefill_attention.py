"""Source-derived AITER MLA route/coarse prefill for absorbed MLA.

The direct-key work mapping follows AITER's MLA prefill kernel: a program
processes one logical query token and a tile of query heads. For NoPE MLA a
program instead tiles query positions for one head, reusing each loaded key
across those positions. Both score the latent and any direct-key parts
separately and accumulate only the latent value. LoD adds centroid
count mass and retains the exact global top-eight centroid IDs while streaming
the same score tiles used by coarse attention.

The first implementation target is Kimi K3's absorbed-MQA geometry
(``K=576``, ``V=512``, GQA=96).  ``include_direct_key=False`` deliberately
ignores the final 64 key channels and exists only as a timing probe; release
execution always enables the direct-key contribution.
"""

from __future__ import annotations

import functools
import math
import os
import sys

import torch
import triton
import triton.language as tl

from ._paged_common import _pack_route_score_index, _unpack_route_score_index
from .aiter_prefill_attention import (
    AiterPrefillCoarse,
    _gather_selected_route_scores,
    _reduce_route_candidates,
    _reduce_split_route_candidates,
    _specialized_kimi_coarse_mha_fwd,
    _specialized_route_mha_fwd,
)
@functools.lru_cache(maxsize=1)
def _kimi_local_mla_prefill_op():
    """Return AITER's supported absorbed-MLA prefill entry point.

    The operator is already a torch-free ctypes wrapper, so loading a second
    copy under a LoD-specific module name buys no isolation.  More importantly,
    only ``module_mla_asm`` has an AITER build recipe.  A private alias happened
    to work when a manually built DSO was present, but could not be rebuilt on a
    clean machine (and made all other ranks fail after waiting on the build
    lock).  Reusing the canonical wrapper keeps the same assembly kernel while
    making initialization deterministic.
    """
    from aiter.ops.attention import mla_prefill_asm_fwd

    return mla_prefill_asm_fwd


def _workspace_tensor(
    buffers: dict[str, torch.Tensor] | None,
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
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


def _cached_flat_weight(
    buffers: dict[str, torch.Tensor] | None,
    name: str,
    source: torch.Tensor,
    permutation: tuple[int, ...],
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Cache a fixed inference-weight layout in its layer-local workspace."""
    if buffers is not None and os.environ.get("LOD_KIMI_CACHE_PROJECTION_WEIGHTS") == "1":
        # Serving shares scratch across layers. A single name only caches the
        # most recently executed layer, so every following chunk rebuilds all
        # layouts. Retain small immutable layouts per source instead; source
        # references below prevent allocator-address recycling. Keep stage
        # names distinct so local-stream creation is not read by coarse work
        # before that stream has completed.
        name = f"{name}_{source.data_ptr()}"
    source_name = f"{name}_source"
    cached = None if buffers is None else buffers.get(name)
    cached_source = None if buffers is None else buffers.get(source_name)
    if (
        isinstance(cached, torch.Tensor)
        and isinstance(cached_source, torch.Tensor)
        and cached_source.data_ptr() == source.data_ptr()
        and tuple(cached_source.shape) == tuple(source.shape)
        and tuple(cached_source.stride()) == tuple(source.stride())
        and cached.dtype == source.dtype
        and cached.device == source.device
        and tuple(cached.shape) == shape
    ):
        return cached
    cached = source.permute(*permutation).reshape(shape).contiguous()
    if buffers is None:
        return cached
    buffers[name] = cached
    # Retain the source tensor itself so a recycled allocator address cannot
    # accidentally validate a layout prepared for an earlier weight.
    buffers[source_name] = source
    return cached


@functools.lru_cache(maxsize=16)
def _kimi_prefill_attention_streams(
    device_index: int,
) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
    """Keep the coarse and route streams alive for repeated prefill chunks."""
    with torch.cuda.device(device_index):
        return torch.cuda.Stream(), torch.cuda.Stream()


def project_kimi_head_values(
    values: torch.Tensor,
    w_uv: torch.Tensor,
) -> torch.Tensor:
    """Project per-head latent values, retaining any route axis."""
    if values.ndim not in (4, 5) or int(values.size(-1)) != 512:
        raise ValueError("MLA per-head projection expects [...,512] values")
    batch, heads = int(values.size(0)), int(values.size(1))
    value_dim = int(w_uv.size(-1))
    if tuple(w_uv.shape[:2]) != (heads, 512) or value_dim not in (128, 256):
        raise ValueError("MLA W_UV must be [H,512,128/256]")
    middle_shape = tuple(int(size) for size in values.shape[2:-1])
    head_major = values.permute(
        1, 0, *range(2, values.ndim)
    ).contiguous().view(heads, -1, 512)
    projected = torch.bmm(head_major, w_uv)
    return (
        projected.view(heads, batch, *middle_shape, value_dim)
        .permute(1, 0, *range(2, values.ndim))
        .contiguous()
    )


def project_kimi_shared_values(
    values: torch.Tensor,
    w_uv: torch.Tensor,
) -> torch.Tensor:
    """Project one shared latent-value head into every query head."""
    if values.ndim != 4 or int(values.size(1)) != 1 or int(values.size(-1)) != 512:
        raise ValueError("MLA shared projection expects [B,1,T,512]")
    heads = int(w_uv.size(0))
    value_dim = int(w_uv.size(-1))
    if int(w_uv.size(1)) != 512 or value_dim not in (128, 256):
        raise ValueError("MLA W_UV must be [H,512,128/256]")
    batch, tokens = int(values.size(0)), int(values.size(2))
    latent = values[:, 0].reshape(batch * tokens, 512)
    flat_weight = w_uv.permute(1, 0, 2).reshape(512, heads * value_dim)
    return (
        torch.mm(latent, flat_weight)
        .view(batch, tokens, heads, value_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


@triton.jit
def _pack_kimi_expanded_keys_kernel(
    projected_nope,
    direct_source,
    expanded_key,
    rows,
    heads: tl.constexpr,
    source_batch_stride: tl.constexpr,
    source_token_stride: tl.constexpr,
    projected_row_stride: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    dimension = tl.arange(0, BLOCK_D)
    token_row = row // heads
    head = row - token_row * heads
    batch = token_row // rows
    token = token_row - batch * rows
    no_pe = tl.load(
        projected_nope + row * projected_row_stride + dimension,
        mask=dimension < 128,
        other=0.0,
    )
    direct = tl.load(
        direct_source
        + batch * source_batch_stride
        + token * source_token_stride
        + 512
        + (dimension - 128),
        mask=(dimension >= 128) & (dimension < 192),
        other=0.0,
    )
    tl.store(
        expanded_key + row * 192 + dimension,
        tl.where(dimension < 128, no_pe, direct),
        mask=dimension < 192,
    )


def absorb_kimi_prefill_query(
    expanded_q: torch.Tensor,
    w_uk_t: torch.Tensor,
    *,
    buffers: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Map a batched D192 Kimi query chunk into compact D576 MLA space."""
    if expanded_q.ndim != 4 or int(expanded_q.size(-1)) != 192:
        raise ValueError("Kimi query absorption expects [B,H,T,192]")
    batch, heads, tokens = map(int, expanded_q.shape[:3])
    if tuple(w_uk_t.shape) != (heads, 128, 512):
        raise ValueError("Kimi query absorption has incompatible W_UK_T")
    latent = _workspace_tensor(
        buffers,
        "kimi_absorbed_query_latent",
        (batch, heads, tokens, 512),
        dtype=expanded_q.dtype,
        device=expanded_q.device,
    )
    for batch_index in range(batch):
        torch.bmm(
            expanded_q[batch_index, ..., :128],
            w_uk_t,
            out=latent[batch_index],
        )
    absorbed = _workspace_tensor(
        buffers,
        "kimi_absorbed_query",
        (batch, heads, tokens, 576),
        dtype=expanded_q.dtype,
        device=expanded_q.device,
    )
    absorbed[..., :512].copy_(latent)
    absorbed[..., 512:].copy_(expanded_q[..., 128:])
    if buffers is not None:
        # ``absorbed`` owns the copied latent coordinates.  Retaining the
        # D512 staging GEMM output would add another ~192 MiB for K3's TP8,
        # Q=16K production shape without serving later attention work.
        buffers.pop("kimi_absorbed_query_latent", None)
    return absorbed


def expand_kimi_leaf_kv(
    key: torch.Tensor,
    w_uk_t: torch.Tensor,
    w_uv: torch.Tensor,
    *,
    buffers: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand a compact Kimi leaf shard into token-major D192/V128 views.

    The projection GEMMs naturally produce ``[B,T,H,D]``.  The paged leaf
    consumer accepts explicit head and token strides, so return permuted
    ``[B,H,T,D]`` views instead of retaining a second, head-major copy of both
    tensors.  This matters at long context: the old transpose doubled an
    already sizeable temporary working set.
    """
    if key.ndim != 4 or int(key.size(1)) != 1 or int(key.size(-1)) != 576:
        raise ValueError("Kimi leaf expansion expects [B,1,T,576]")
    batch, tokens = int(key.size(0)), int(key.size(2))
    heads = int(w_uk_t.size(0))
    if tuple(w_uk_t.shape) != (heads, 128, 512):
        raise ValueError("Kimi leaf W_UK_T has incompatible geometry")
    if tuple(w_uv.shape) != (heads, 512, 128):
        raise ValueError("Kimi leaf W_UV has incompatible geometry")
    latent = key[:, 0, :, :512].reshape(batch * tokens, 512)
    if os.environ.get("LOD_KIMI_FUSED_LEAF_KV") == "1":
        # Both projections consume the same latent rows. Interleaving their
        # columns makes one GEMM emit [K_nope,V] per head; V remains a strided
        # view, accepted directly by the existing leaf-attention consumer.
        name = f"kimi_leaf_combined_weight_{w_uk_t.data_ptr()}_{w_uv.data_ptr()}_{heads}"
        weight = None if buffers is None else buffers.get(name)
        if weight is None:
            weight = torch.cat((w_uk_t.transpose(1, 2), w_uv), dim=-1)
            weight = weight.permute(1, 0, 2).reshape(512, heads * 256).contiguous()
            if buffers is not None:
                buffers[name] = weight
                # Keep source storage alive, matching _cached_flat_weight's
                # immutable-inference-weight contract and pointer safety.
                buffers[name + "_uk_source"] = w_uk_t
                buffers[name + "_uv_source"] = w_uv
        projected = _workspace_tensor(
            buffers, "kimi_leaf_fused_kv", (batch, tokens, heads, 256),
            dtype=key.dtype, device=key.device,
        )
        torch.mm(latent, weight, out=projected.view(batch * tokens, heads * 256))
        expanded_key = _workspace_tensor(
            buffers, "kimi_leaf_expanded_k_token_major", (batch, tokens, heads, 192),
            dtype=key.dtype, device=key.device,
        )
        _pack_kimi_expanded_keys_kernel[(batch * tokens * heads,)](
            projected, key, expanded_key, tokens, heads=heads,
            source_batch_stride=int(key.stride(0)),
            source_token_stride=int(key.stride(2)),
            projected_row_stride=256, BLOCK_D=256, num_warps=4,
        )
        return (expanded_key.permute(0, 2, 1, 3),
                projected[..., 128:].permute(0, 2, 1, 3))
    expanded_k_token_major = _workspace_tensor(
        buffers,
        "kimi_leaf_expanded_k_token_major",
        (batch, tokens, heads, 192),
        dtype=key.dtype,
        device=key.device,
    )
    expanded_k_nope_token_major = _workspace_tensor(
        buffers,
        "kimi_leaf_expanded_k_nope_token_major",
        (batch, tokens, heads, 128),
        dtype=key.dtype,
        device=key.device,
    )
    torch.mm(
        latent,
        (_cached_flat_weight(buffers, "kimi_leaf_flat_uk", w_uk_t, (2, 0, 1), (512, heads * 128))
         if os.environ.get("LOD_KIMI_CACHE_PROJECTION_WEIGHTS") == "1"
         else w_uk_t.permute(2, 0, 1).reshape(512, heads * 128)),
        out=expanded_k_nope_token_major.reshape(batch * tokens, heads * 128),
    )
    expanded_v_token_major = _workspace_tensor(
        buffers,
        "kimi_leaf_expanded_v_token_major",
        (batch, tokens, heads, 128),
        dtype=key.dtype,
        device=key.device,
    )
    torch.mm(
        latent,
        (_cached_flat_weight(buffers, "kimi_leaf_flat_uv", w_uv, (1, 0, 2), (512, heads * 128))
         if os.environ.get("LOD_KIMI_CACHE_PROJECTION_WEIGHTS") == "1"
         else w_uv.permute(1, 0, 2).reshape(512, heads * 128)),
        out=expanded_v_token_major.reshape(batch * tokens, heads * 128),
    )
    # The D128 prefix of a D192 head is not flattenable across multiple heads:
    # adjacent heads are separated by the 64 direct-key channels.  Writing the
    # GEMM through ``expanded_k[..., :128].reshape(...)`` therefore targets a
    # temporary copy when H>1.  Materialize the contiguous D128 result above,
    # then copy it into the expanded-key prefix explicitly.
    _pack_kimi_expanded_keys_kernel[(batch * tokens * heads,)](
        expanded_k_nope_token_major,
        key,
        expanded_k_token_major,
        tokens,
        heads=heads,
        source_batch_stride=int(key.stride(0)),
        source_token_stride=int(key.stride(2)),
        projected_row_stride=128,
        BLOCK_D=256,
        num_warps=4,
    )
    return (
        expanded_k_token_major.permute(0, 2, 1, 3),
        expanded_v_token_major.permute(0, 2, 1, 3),
    )


def expand_kimi_sink_kv(
    key: torch.Tensor,
    w_uk_t: torch.Tensor,
    w_uv: torch.Tensor,
    *,
    buffers: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project only the separate sink, never the shape-only query carrier.

    The real D192 query scores this per-head key directly. Dedicated tiny
    workspaces avoid overwriting expanded leaf prefixes or materializing a
    Q*H*576 absorbed-query buffer just to score one protected token.
    """
    if key.ndim != 4 or key.size(1) != 1 or key.size(-1) != 576:
        raise ValueError("Kimi sink projection expects [B,1,S,576]")
    batch, tokens = int(key.size(0)), int(key.size(2))
    heads = int(w_uk_t.size(0))
    if tuple(w_uk_t.shape) != (heads, 128, 512) or tuple(w_uv.shape) != (heads, 512, 128):
        raise ValueError("Kimi sink projection has incompatible weights")
    latent = key[:, 0, :, :512].reshape(batch * tokens, 512)
    nope = _workspace_tensor(buffers, "kimi_sink_projected_nope",
                             (batch, tokens, heads, 128), dtype=key.dtype, device=key.device)
    expanded = _workspace_tensor(buffers, "kimi_sink_projected_k",
                                 (batch, tokens, heads, 192), dtype=key.dtype, device=key.device)
    value = _workspace_tensor(buffers, "kimi_sink_projected_v",
                              (batch, tokens, heads, 128), dtype=key.dtype, device=key.device)
    key_weight = _cached_flat_weight(buffers, "kimi_flat_w_uk_t", w_uk_t,
                                    (2, 0, 1), (512, heads * 128))
    value_weight = _cached_flat_weight(buffers, "kimi_flat_w_uv", w_uv,
                                      (1, 0, 2), (512, heads * 128))
    torch.mm(latent, key_weight, out=nope.view(batch * tokens, heads * 128))
    torch.mm(latent, value_weight, out=value.view(batch * tokens, heads * 128))
    _pack_kimi_expanded_keys_kernel[(batch * tokens * heads,)](
        nope, key, expanded, tokens, heads=heads,
        source_batch_stride=int(key.stride(0)), source_token_stride=int(key.stride(2)),
        projected_row_stride=128, BLOCK_D=256, num_warps=4,
    )
    return expanded.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3)


@triton.jit(do_not_specialize=["STATE_LEN"])
def _prepare_mla_state_kernel(
    state_k,
    state_v,
    counts,
    mean_k,
    mean_v,
    active_counts,
    STATE_LEN,
    STATE_CAPACITY: tl.constexpr,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    KEY_BLOCK: tl.constexpr,
    VALUE_BLOCK: tl.constexpr,
):
    """Materialize contiguous centroid means without duplicating GQA heads."""
    row = tl.program_id(0).to(tl.int64)
    slot = row % STATE_LEN
    batch_kv = row // STATE_LEN
    source_row = batch_kv * STATE_CAPACITY + slot
    count = tl.maximum(tl.load(counts + source_row), 1.0).to(tl.float32)

    key_dim = tl.arange(0, KEY_BLOCK)
    key = tl.load(
        state_k + source_row * KEY_DIM + key_dim,
        mask=key_dim < KEY_DIM,
        other=0.0,
    )
    tl.store(
        mean_k + row * KEY_DIM + key_dim,
        key / count,
        mask=key_dim < KEY_DIM,
    )

    value_dim = tl.arange(0, VALUE_BLOCK)
    value = tl.load(
        state_v + source_row * VALUE_DIM + value_dim,
        mask=value_dim < VALUE_DIM,
        other=0.0,
    )
    tl.store(
        mean_v + row * VALUE_DIM + value_dim,
        value / count,
        mask=value_dim < VALUE_DIM,
    )
    tl.store(active_counts + row, count)


@triton.jit(do_not_specialize=["STATE_LEN"])
def _prepare_expanded_mla_state_kernel(
    state_k,
    state_v,
    counts,
    mean_k,
    mean_v,
    active_counts,
    log_counts,
    STATE_LEN,
    DISPATCH_STATE_LEN,
    KEY_BATCH_STRIDE: tl.constexpr,
    KEY_TOKEN_STRIDE: tl.constexpr,
    VALUE_BATCH_STRIDE: tl.constexpr,
    VALUE_TOKEN_STRIDE: tl.constexpr,
    COUNT_BATCH_STRIDE: tl.constexpr,
    COUNT_TOKEN_STRIDE: tl.constexpr,
    LATENT_DIM: tl.constexpr,
    DIRECT_DIM: tl.constexpr,
    LATENT_BLOCK: tl.constexpr,
    DIRECT_BLOCK: tl.constexpr,
):
    """Prepare padded Kimi latent means and exact centroid log-masses."""
    row = tl.program_id(0).to(tl.int64)
    batch = row // DISPATCH_STATE_LEN
    slot = row - batch * DISPATCH_STATE_LEN
    valid_slot = slot < STATE_LEN
    key_offset = batch * KEY_BATCH_STRIDE + slot * KEY_TOKEN_STRIDE
    value_offset = batch * VALUE_BATCH_STRIDE + slot * VALUE_TOKEN_STRIDE
    count_offset = batch * COUNT_BATCH_STRIDE + slot * COUNT_TOKEN_STRIDE
    count = tl.maximum(
        tl.load(counts + count_offset, mask=valid_slot, other=1.0), 1.0
    ).to(tl.float32)

    latent = tl.arange(0, LATENT_BLOCK)
    valid_latent = latent < LATENT_DIM
    key_latent = tl.load(
        state_k + key_offset + latent,
        mask=valid_slot & valid_latent,
        other=0.0,
    )
    value = tl.load(
        state_v + value_offset + latent,
        mask=valid_slot & valid_latent,
        other=0.0,
    )
    output_row = batch * DISPATCH_STATE_LEN + slot
    tl.store(
        mean_k + output_row * (LATENT_DIM + DIRECT_DIM) + latent,
        key_latent / count,
        mask=valid_latent,
    )
    tl.store(
        mean_v + output_row * LATENT_DIM + latent,
        value / count,
        mask=valid_latent,
    )

    direct = tl.arange(0, DIRECT_BLOCK)
    valid_direct = direct < DIRECT_DIM
    key_direct = tl.load(
        state_k
        + key_offset
        + LATENT_DIM
        + direct,
        mask=valid_slot & valid_direct,
        other=0.0,
    )
    tl.store(
        mean_k
        + output_row * (LATENT_DIM + DIRECT_DIM)
        + LATENT_DIM
        + direct,
        key_direct / count,
        mask=valid_direct,
    )
    tl.store(active_counts + output_row, count)

    tl.store(
        log_counts + output_row,
        tl.where(valid_slot, tl.log(count), -float("inf")),
    )


@triton.jit(
    do_not_specialize=["QUERY_LEN", "STATE_LEN"],
    do_not_specialize_on_alignment=["QUERY_LEN", "STATE_LEN"],
)
def _mla_route_coarse_prefill_kernel(
    q,
    mean_k,
    mean_v,
    counts,
    output,
    output_lse,
    top_slots,
    selected_route_scores,
    Q_BATCH_STRIDE,
    Q_HEAD_STRIDE,
    Q_TOKEN_STRIDE,
    QUERY_LEN,
    STATE_LEN,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    LATENT_DIM: tl.constexpr,
    DIRECT_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_BLOCKS: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    SCALE: tl.constexpr,
    NORMALIZE_ROUTE_QUERY: tl.constexpr,
    INCLUDE_DIRECT_KEY: tl.constexpr,
    QUERY_TILED: tl.constexpr = False,
    SHARED_LATENT: tl.constexpr = False,
):
    """Stream count-corrected MLA attention and exact top-eight routes."""
    head_lane = tl.arange(0, BLOCK_M)
    if QUERY_TILED:
        batch_head = tl.program_id(0).to(tl.int64)
        batch = batch_head // QUERY_HEADS
        head = batch_head % QUERY_HEADS
        kv_head = head // KV_GROUP_SIZE
        batch_kv = batch * KV_HEADS + kv_head
        query = tl.program_id(1) * BLOCK_M + head_lane
        query_head = tl.full((BLOCK_M,), head, tl.int32)
        valid_head = query_head < QUERY_HEADS
    else:
        batch_kv = tl.program_id(0).to(tl.int64)
        batch = batch_kv // KV_HEADS
        kv_head = batch_kv - batch * KV_HEADS
        query = tl.full((BLOCK_M,), tl.program_id(1), tl.int32)
        head_block = tl.program_id(2)
        query_head = kv_head * KV_GROUP_SIZE + head_block * BLOCK_M + head_lane
        valid_head = query_head < (kv_head + 1) * KV_GROUP_SIZE
        valid_head &= query_head < QUERY_HEADS
    valid_query = query < QUERY_LEN
    valid_row = valid_head & valid_query

    latent_dim = tl.arange(0, LATENT_DIM)
    q_base = (
        batch * Q_BATCH_STRIDE
        + query_head[:, None] * Q_HEAD_STRIDE
        + query[:, None] * Q_TOKEN_STRIDE
    )
    q_latent = tl.load(
        q + q_base + latent_dim[None, :],
        mask=valid_row[:, None],
        other=0.0,
    )
    if INCLUDE_DIRECT_KEY:
        direct_dim = tl.arange(0, DIRECT_DIM)
        q_direct = tl.load(
            q + q_base + LATENT_DIM + direct_dim[None, :],
            mask=valid_row[:, None],
            other=0.0,
        )

    # The route normalizer is a scalar per head/token.  Attention itself uses
    # the unnormalized query, exactly as in the common LoD implementation.
    if NORMALIZE_ROUTE_QUERY:
        square_sum = tl.sum(q_latent.to(tl.float32) * q_latent.to(tl.float32), axis=1)
        route_dimensions: tl.constexpr = LATENT_DIM
        if INCLUDE_DIRECT_KEY:
            square_sum += tl.sum(
                q_direct.to(tl.float32) * q_direct.to(tl.float32), axis=1
            )
            route_dimensions = LATENT_DIM + DIRECT_DIM
        query_rms = tl.sqrt(tl.maximum(square_sum / route_dimensions, 1.0e-12))
    else:
        query_rms = tl.full((BLOCK_M,), 1.0, tl.float32)

    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, VALUE_DIM), tl.float32)
    best_packed = tl.full(
        (BLOCK_M, ROUTE_COUNT), -9223372036854775807, tl.int64
    )
    slot_lane = tl.arange(0, BLOCK_N)
    value_dim = tl.arange(0, VALUE_DIM)

    for slot_begin in tl.range(0, STATE_LEN, BLOCK_N, num_stages=1):
        slot = slot_begin + slot_lane
        valid_slot = slot < STATE_LEN
        state_row = batch_kv * STATE_LEN + slot
        key_latent = tl.load(
            mean_k + state_row[:, None] * (LATENT_DIM + DIRECT_DIM) + latent_dim[None, :],
            mask=valid_slot[:, None],
            other=0.0,
        )
        raw_score = tl.dot(q_latent, key_latent.trans(1, 0))
        if INCLUDE_DIRECT_KEY:
            key_direct = tl.load(
                mean_k
                + state_row[None, :] * (LATENT_DIM + DIRECT_DIM)
                + LATENT_DIM
                + direct_dim[:, None],
                mask=valid_slot[None, :],
                other=0.0,
            )
            raw_score += tl.dot(q_direct, key_direct)

        count = tl.load(counts + state_row, mask=valid_slot, other=1.0).to(tl.float32)
        log_count = tl.log(count)
        score = raw_score.to(tl.float32) * SCALE + log_count[None, :]
        valid = valid_row[:, None] & valid_slot[None, :]
        score = tl.where(valid, score, -float("inf"))

        # Routing may normalize q, but coarse attention always retains the
        # model's original score.  This is algebraically identical to the
        # common query-normalized selector.
        # Multiplication by the positive query RMS preserves the normalized
        # ranking while avoiding a division for every score:
        #   rms * (scale * dot / rms + log(count))
        #     = scale * dot + rms * log(count).
        route_score = raw_score.to(tl.float32) * SCALE
        route_score += query_rms[:, None] * log_count[None, :]
        route_score = tl.where(valid, route_score, -float("inf"))
        packed = _pack_route_score_index(route_score, slot[None, :])
        tile_best = tl.topk(packed, ROUTE_COUNT, dim=1)
        best_packed = tl.topk(
            tl.interleave(best_packed, tile_best), ROUTE_COUNT, dim=1
        )

        block_maximum = tl.max(score, axis=1)
        new_maximum = tl.maximum(maximum, block_maximum)
        correction = tl.exp(maximum - new_maximum)
        probability = tl.exp(score - new_maximum[:, None])
        probability = tl.where(valid, probability, 0.0)
        denominator = denominator * correction + tl.sum(probability, axis=1)
        accumulator *= correction[:, None]
        if SHARED_LATENT:
            value = key_latent
        else:
            value = tl.load(
                mean_v + state_row[:, None] * VALUE_DIM + value_dim[None, :],
                mask=valid_slot[:, None], other=0.0,
            )
        accumulator += tl.dot(probability.to(value.dtype), value)
        maximum = new_maximum

    reciprocal = 1.0 / denominator[:, None]
    result = accumulator * reciprocal
    output_row = (batch * QUERY_LEN + query) * QUERY_HEADS + query_head
    tl.store(
        output + output_row[:, None] * VALUE_DIM + value_dim[None, :],
        result,
        mask=valid_row[:, None],
    )
    lse = maximum + tl.log(denominator)
    lse_row = (batch * QUERY_HEADS + query_head) * QUERY_LEN + query
    tl.store(output_lse + lse_row, lse, mask=valid_row)

    selected_scores, selected_slots = _unpack_route_score_index(best_packed)
    rank = tl.arange(0, ROUTE_COUNT)
    tl.store(
        top_slots + lse_row[:, None] * ROUTE_COUNT + rank[None, :],
        selected_slots,
        mask=valid_row[:, None],
    )
    tl.store(
        selected_route_scores + lse_row[:, None] * ROUTE_COUNT + rank[None, :],
        selected_scores,
        mask=valid_row[:, None],
    )


def aiter_mla_prefill_route_coarse_attention(
    q: torch.Tensor,
    state_k: torch.Tensor,
    state_v: torch.Tensor,
    counts: torch.Tensor,
    *,
    state_len: int,
    kv_group_size: int,
    scale: float,
    normalize_route_query: bool,
    include_direct_key: bool = True,
    buffers: dict[str, torch.Tensor] | None = None,
    query_tiled: bool | None = None,
    block_m: int | None = None, block_n: int | None = None,
    num_warps: int | None = None,
    gluon_layout: bool | None = None,
    early_exit: bool = True,
) -> tuple[
    torch.Tensor,
    AiterPrefillCoarse,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Run fused asymmetric MLA route/coarse prefill.

    The returned ``None`` route metadata asks the existing expert consumer to
    construct its compact expert lists.  Keeping this first version separate
    makes the fused kernel's timing attributable; route-list fusion can be
    added after the MLA math is validated.
    """
    tensors = (q, state_k, state_v, counts)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("AITER MLA route/coarse prefill requires GPU tensors")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("AITER MLA route/coarse prefill requires contiguous tensors")
    batch, query_heads, query_len, key_dim = q.shape
    kv_heads = int(state_k.size(1))
    value_dim = int(state_v.size(-1))
    direct_dim = key_dim - value_dim
    if query_len <= 1:
        raise ValueError("AITER MLA route/coarse prefill requires multiple queries")
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("AITER MLA route/coarse prefill has incompatible GQA")
    if direct_dim not in (0, 64) or value_dim <= 0 or key_dim > 576:
        raise ValueError(
            "AITER MLA route/coarse prefill requires K/V=(L, L) or (L+64, L) "
            "with key width at most 576"
        )
    if tuple(state_k.shape[:2]) != (batch, kv_heads) or int(state_k.size(-1)) != key_dim:
        raise ValueError("AITER MLA route/coarse prefill received the wrong state keys")
    if tuple(state_v.shape[:2]) != (batch, kv_heads):
        raise ValueError("AITER MLA route/coarse prefill received the wrong state values")
    if tuple(state_v.shape[:3]) != tuple(counts.shape[:3]):
        raise ValueError("AITER MLA route/coarse state/count geometry differs")
    if state_len < 8 or state_len > int(state_k.size(2)):
        raise ValueError("AITER MLA route/coarse state length is invalid")

    mean_k = _workspace_tensor(
        buffers,
        "mla_coarse_mean_k",
        (batch, kv_heads, state_len, key_dim),
        dtype=state_k.dtype,
        device=state_k.device,
    )
    mean_v = _workspace_tensor(
        buffers,
        "mla_coarse_mean_v",
        (batch, kv_heads, state_len, value_dim),
        dtype=state_v.dtype,
        device=state_v.device,
    )
    active_counts = _workspace_tensor(
        buffers,
        "mla_coarse_counts",
        (batch, kv_heads, state_len, 1),
        dtype=counts.dtype,
        device=counts.device,
    )
    _prepare_mla_state_kernel[(batch * kv_heads * state_len,)](
        state_k,
        state_v,
        counts,
        mean_k,
        mean_v,
        active_counts,
        state_len,
        STATE_CAPACITY=int(state_k.size(2)),
        KEY_DIM=key_dim,
        VALUE_DIM=value_dim,
        KEY_BLOCK=triton.next_power_of_2(key_dim),
        VALUE_BLOCK=triton.next_power_of_2(value_dim),
        num_warps=4,
    )

    output = _workspace_tensor(
        buffers,
        "mla_coarse_output",
        (batch, query_len, query_heads, value_dim),
        dtype=q.dtype,
        device=q.device,
    )
    output_lse = _workspace_tensor(
        buffers,
        "mla_coarse_lse",
        (batch, query_heads, query_len),
        dtype=torch.float32,
        device=q.device,
    )
    top_slots = _workspace_tensor(
        buffers,
        "route_slots",
        (batch, query_heads, query_len, 8),
        dtype=torch.long,
        device=q.device,
    )
    selected_route_scores = _workspace_tensor(
        buffers,
        "selected_route_scores",
        (batch, query_heads, query_len, 8),
        dtype=torch.float32,
        device=q.device,
    )
    if query_tiled is None:
        query_tiled = direct_dim == 0 and value_dim == 512
    shared_latent = (direct_dim == 0 and state_k.data_ptr() == state_v.data_ptr()
                    and state_k.stride() == state_v.stride())
    if gluon_layout is None:
        gluon_layout = bool(torch.version.hip and query_tiled and shared_latent and value_dim == 512)
    block_m = (64 if query_tiled else 32) if block_m is None else block_m
    block_n = (64 if gluon_layout else 32 if query_tiled else 16) if block_n is None else block_n
    num_warps = (4 if query_tiled else 8) if num_warps is None else num_warps
    head_blocks = triton.cdiv(kv_group_size, block_m)
    grid = ((batch * query_heads, triton.cdiv(query_len, block_m), 1)
            if query_tiled else (batch * kv_heads, query_len, head_blocks))
    if gluon_layout:
        if not query_tiled or not shared_latent or value_dim != 512 or num_warps != 4:
            raise ValueError("Gluon route/coarse requires query-tiled shared L512 and four waves")
        from .latent_route_coarse import latent_route_coarse_gluon
        latent_route_coarse_gluon[grid[:2]](
            q, mean_k, active_counts, output, output_lse, top_slots, selected_route_scores,
            q.stride(0), q.stride(1), q.stride(2), query_len, state_len,
            QUERY_HEADS=query_heads, KV_HEADS=kv_heads, KV_GROUP_SIZE=kv_group_size,
            SCALE=float(scale), NORMALIZE_ROUTE_QUERY=normalize_route_query,
            BLOCK_M=block_m, BLOCK_N=block_n, EARLY_EXIT=early_exit,
            num_warps=4, num_stages=1)
    else:
        _mla_route_coarse_prefill_kernel[grid](
            q, mean_k, mean_v, active_counts, output, output_lse,
            top_slots, selected_route_scores,
            q.stride(0), q.stride(1), q.stride(2), query_len, state_len,
            QUERY_HEADS=query_heads, KV_HEADS=kv_heads, KV_GROUP_SIZE=kv_group_size,
            LATENT_DIM=value_dim, DIRECT_DIM=direct_dim, VALUE_DIM=value_dim,
            BLOCK_M=block_m, BLOCK_N=block_n, HEAD_BLOCKS=head_blocks,
            ROUTE_COUNT=8, SCALE=float(scale), NORMALIZE_ROUTE_QUERY=normalize_route_query,
            INCLUDE_DIRECT_KEY=include_direct_key and direct_dim > 0,
            QUERY_TILED=query_tiled, SHARED_LATENT=shared_latent,
            num_warps=num_warps, waves_per_eu=1,
        )
    coarse = AiterPrefillCoarse(
        output_0=output,
        lse_0=output_lse,
        output_1=output,
        lse_1=output_lse,
        mean_k=mean_k,
        mean_v=mean_v,
        counts=active_counts,
        has_second_partition=False,
        selected_route_scores=selected_route_scores,
    )
    return top_slots, coarse, None, None


def aiter_kimi_expanded_prefill_route_coarse_attention(
    q_expanded: torch.Tensor,
    state_k: torch.Tensor,
    state_v: torch.Tensor,
    counts: torch.Tensor,
    w_uk_t: torch.Tensor,
    w_uv: torch.Tensor,
    *,
    state_len: int,
    scale: float,
    normalize_route_query: bool,
    slot_lengths: torch.Tensor | None = None,
    max_open_leaf_tokens: int | None = None,
    buffers: dict[str, torch.Tensor] | None = None,
    cache_immutable_weights: bool = True,
) -> tuple[torch.Tensor, AiterPrefillCoarse, torch.Tensor, torch.Tensor]:
    """Run Kimi route/coarse attention with count-augmented absorbed Q/K.

    Persistent state and exact leaves remain in Kimi's compact 512+64 latent
    representation.  Only the O(sqrt(N)) centroid means are expanded through
    ``W_UK`` and ``W_UV``.  Applying the linear value projection before the
    softmax-weighted merge is exact and lets route/coarse use one V=128 pass.
    """
    if normalize_route_query:
        raise ValueError(
            "expanded Kimi routing requires the model's normalized-query policy"
        )
    tensors = (q_expanded, state_k, state_v, counts, w_uk_t, w_uv)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("expanded Kimi prefill requires GPU tensors")
    if q_expanded.ndim != 4 or int(q_expanded.size(-1)) != 192:
        raise ValueError("expanded Kimi query must be [B,H,Q,192]")
    batch, query_heads, query_len, _ = q_expanded.shape
    if query_len <= 1:
        raise ValueError("expanded Kimi prefill requires multiple queries")
    if tuple(state_k.shape[:2]) != (batch, 1) or int(state_k.size(-1)) != 576:
        raise ValueError("expanded Kimi state keys must be [B,1,S,576]")
    if tuple(state_v.shape[:2]) != (batch, 1) or int(state_v.size(-1)) != 512:
        raise ValueError("expanded Kimi state values must be [B,1,S,512]")
    if tuple(counts.shape[:3]) != tuple(state_v.shape[:3]):
        raise ValueError("expanded Kimi state/count geometry differs")
    if any(tensor.stride(-1) != 1 for tensor in (state_k, state_v, counts)):
        raise ValueError("expanded Kimi state channels must have unit stride")
    if tuple(w_uk_t.shape) != (query_heads, 128, 512):
        raise ValueError(
            "expanded Kimi W_UK_T must be [query_heads,128,512], got "
            f"{tuple(w_uk_t.shape)}"
        )
    if tuple(w_uv.shape) != (query_heads, 512, 128):
        raise ValueError(
            "expanded Kimi W_UV must be [query_heads,512,128], got "
            f"{tuple(w_uv.shape)}"
        )
    if state_len < 8 or state_len > int(state_k.size(2)):
        raise ValueError("expanded Kimi state length is invalid")
    if (slot_lengths is None) != (max_open_leaf_tokens is None):
        raise ValueError(
            "expanded Kimi selected-route cap requires lengths and a limit"
        )
    if slot_lengths is not None and (
        tuple(slot_lengths.shape[:2]) != (batch, 1)
        or int(slot_lengths.size(2)) < state_len
    ):
        raise ValueError("expanded Kimi slot lengths have incompatible geometry")

    # Opt-in capacity bound, not a different routing rule. Reuse expanded K/V
    # projection storage across head groups; preserve every head's own routes,
    # coarse output/LSE and projected means needed for exact replacement.
    head_group = int(os.environ.get("LOD_KIMI_COARSE_PREFILL_HEAD_GROUP", "0"))
    if head_group < 0 or (0 < head_group < query_heads and query_heads % head_group):
        raise ValueError("coarse prefill head group must be zero or a divisor of query heads")
    if 0 < head_group < query_heads:
        output = _workspace_tensor(buffers, "kimi_grouped_coarse_output",
            (batch, query_len, query_heads, 128), dtype=q_expanded.dtype, device=q_expanded.device)
        lse = _workspace_tensor(buffers, "kimi_grouped_coarse_lse", (batch, query_heads, query_len),
            dtype=torch.float32, device=q_expanded.device)
        slots = _workspace_tensor(buffers, "kimi_grouped_coarse_slots",
            (batch, query_heads, query_len, 8), dtype=torch.long, device=q_expanded.device)
        scores = _workspace_tensor(buffers, "kimi_grouped_coarse_scores", slots.shape,
            dtype=torch.float32, device=q_expanded.device)
        padded_states = ((state_len + 127) // 128) * 128
        values = _workspace_tensor(buffers, "kimi_grouped_coarse_values",
            (batch, query_heads, padded_states, 128), dtype=q_expanded.dtype, device=q_expanded.device)
        foreground = torch.cuda.current_stream(q_expanded.device)
        for begin in range(0, query_heads, head_group):
            end = begin + head_group
            group_slots, coarse, _, _ = aiter_kimi_expanded_prefill_route_coarse_attention(
                q_expanded[:, begin:end], state_k, state_v, counts,
                w_uk_t[begin:end], w_uv[begin:end], state_len=state_len,
                scale=scale, normalize_route_query=normalize_route_query,
                slot_lengths=slot_lengths, max_open_leaf_tokens=max_open_leaf_tokens,
                buffers=buffers, cache_immutable_weights=cache_immutable_weights)
            if coarse.ready_stream is not None:
                foreground.wait_stream(coarse.ready_stream)
            if coarse.has_second_partition or coarse.selected_route_scores is None:
                raise AssertionError("grouped Kimi coarse attention requires one partition and route scores")
            output[:, :, begin:end].copy_(coarse.output_0)
            lse[:, begin:end].copy_(coarse.lse_0)
            values[:, begin:end].copy_(coarse.mean_v)
            slots[:, begin:end].copy_(group_slots)
            scores[:, begin:end].copy_(coarse.selected_route_scores)
        coarse = AiterPrefillCoarse(output_0=output, lse_0=lse, output_1=output, lse_1=lse,
            mean_k=coarse.mean_k, mean_v=values, counts=coarse.counts,
            has_second_partition=False, ready_stream=foreground, selected_route_scores=scores)
        # The normal consumer already reconstructs expert counts when omitted.
        return slots, coarse, None, None

    def padded_partition(length: int) -> int:
        return ((length + 127) // 128) * 128

    # Keep one coarse partition. Two concurrent half-state calls improve the
    # 2K-state fixture slightly but regress its 4K-state path after accounting
    # for the second candidate reduction and final LSE merge.
    dispatch_state_len = padded_partition(state_len)
    split_at = 0
    profile = os.environ.get("LOD_KIMI_PROFILE_PREFILL") == "1"
    prepare_begin = torch.cuda.Event(enable_timing=True) if profile else None
    prepare_end = torch.cuda.Event(enable_timing=True) if profile else None
    key_weight_end = torch.cuda.Event(enable_timing=True) if profile else None
    key_mm_end = torch.cuda.Event(enable_timing=True) if profile else None
    key_pack_end = torch.cuda.Event(enable_timing=True) if profile else None
    value_weight_end = torch.cuda.Event(enable_timing=True) if profile else None
    expand_end = torch.cuda.Event(enable_timing=True) if profile else None
    route_q_end = torch.cuda.Event(enable_timing=True) if profile else None
    attention_end = torch.cuda.Event(enable_timing=True) if profile else None
    reduce_end = torch.cuda.Event(enable_timing=True) if profile else None
    if prepare_begin is not None:
        prepare_begin.record()

    mean_k = _workspace_tensor(
        buffers,
        "kimi_coarse_mean_k",
        (batch, 1, dispatch_state_len, 576),
        # Persistent sums may be FP32 (e.g. exchanged DCP summaries). Divide
        # before rounding, then project in the model's Q/weight dtype.
        dtype=q_expanded.dtype,
        device=state_k.device,
    )
    mean_v = _workspace_tensor(
        buffers,
        "kimi_coarse_mean_v",
        (batch, 1, dispatch_state_len, 512),
        dtype=q_expanded.dtype,
        device=state_v.device,
    )
    active_counts = _workspace_tensor(
        buffers,
        "kimi_coarse_counts",
        (batch, 1, dispatch_state_len, 1),
        dtype=counts.dtype,
        device=counts.device,
    )
    log_counts = _workspace_tensor(
        buffers,
        "kimi_coarse_log_counts",
        (batch, dispatch_state_len),
        dtype=q_expanded.dtype,
        device=q_expanded.device,
    )
    _prepare_expanded_mla_state_kernel[(batch * dispatch_state_len,)](
        state_k,
        state_v,
        counts,
        mean_k,
        mean_v,
        active_counts,
        log_counts,
        state_len,
        dispatch_state_len,
        KEY_BATCH_STRIDE=state_k.stride(0),
        KEY_TOKEN_STRIDE=state_k.stride(2),
        VALUE_BATCH_STRIDE=state_v.stride(0),
        VALUE_TOKEN_STRIDE=state_v.stride(2),
        COUNT_BATCH_STRIDE=counts.stride(0),
        COUNT_TOKEN_STRIDE=counts.stride(2),
        LATENT_DIM=512,
        DIRECT_DIM=64,
        LATENT_BLOCK=512,
        DIRECT_BLOCK=64,
        num_warps=8,
    )
    if prepare_end is not None:
        prepare_end.record()

    # Route and attend in the model's natural 128+64 expanded-key geometry.
    # Emit eight winners from every native score tile already consumed by
    # coarse attention, then reduce those compact candidates to the exact
    # global top eight. This is faster than maintaining a global ordered list
    # inside the long-running CK attention workgroup and changes no routes.
    fused_route_coarse = True
    route_dim = 192
    attention_dim = route_dim
    expanded_k = _workspace_tensor(
        buffers,
        "kimi_expanded_coarse_k",
        (batch, dispatch_state_len, query_heads, route_dim),
        dtype=q_expanded.dtype,
        device=q_expanded.device,
    )
    expanded_k_nope = _workspace_tensor(
        buffers,
        "kimi_expanded_coarse_k_nope",
        (batch, dispatch_state_len, query_heads, 128),
        dtype=q_expanded.dtype,
        device=q_expanded.device,
    )
    flat_weight = _cached_flat_weight(
        buffers if cache_immutable_weights else None,
        "kimi_flat_w_uk_t",
        w_uk_t,
        (2, 0, 1),
        (512, query_heads * 128),
    )
    if key_weight_end is not None:
        key_weight_end.record()
    torch.mm(
        mean_v[:, 0].reshape(batch * dispatch_state_len, 512),
        flat_weight,
        out=expanded_k_nope.reshape(batch * dispatch_state_len, query_heads * 128),
    )
    if key_mm_end is not None:
        key_mm_end.record()
    expanded_k[..., :128].copy_(expanded_k_nope)
    expanded_k[..., 128:192].copy_(
        mean_k[:, 0, :, None, 512:].expand(-1, -1, query_heads, -1)
    )
    if key_pack_end is not None:
        key_pack_end.record()
    expanded_v = _workspace_tensor(
        buffers,
        "kimi_expanded_coarse_v",
        (batch, dispatch_state_len, query_heads, 128),
        dtype=q_expanded.dtype,
        device=state_v.device,
    )
    flat_value_weight = _cached_flat_weight(
        buffers if cache_immutable_weights else None,
        "kimi_flat_w_uv",
        w_uv,
        (1, 0, 2),
        (512, query_heads * 128),
    )
    if value_weight_end is not None:
        value_weight_end.record()
    torch.mm(
        mean_v[:, 0].reshape(batch * dispatch_state_len, 512),
        flat_value_weight,
        out=expanded_v.reshape(batch * dispatch_state_len, query_heads * 128),
    )
    projected_mean_v = expanded_v.permute(0, 2, 1, 3)
    if expand_end is not None:
        expand_end.record()
    # CK consumes explicit tensor strides, so the model's contiguous BHQD
    # query can be exposed as a zero-copy BQHD view.  Materializing this view
    # used to copy 72 MiB per K3 layer for a 16K/12-head TP8 chunk.
    route_q = q_expanded.permute(0, 2, 1, 3)
    if route_q_end is not None:
        route_q_end.record()
    count_bias = log_counts[:, None, None, :].expand(
        batch, query_heads, 1, dispatch_state_len
    )
    coarse_q = route_q
    coarse_k = expanded_k

    original_dlopen_flags = sys.getdlopenflags()
    deepbind = getattr(os, "RTLD_DEEPBIND", 0)
    if deepbind:
        sys.setdlopenflags(original_dlopen_flags | deepbind)
    try:
        coarse_mha_fwd = _specialized_kimi_coarse_mha_fwd(
            attention_dim,
            async_bias=True,
            fused_route=fused_route_coarse,
            tile_max_probe=os.environ.get("LOD_KIMI_TILE_REFINE") == "1",
            query_tile=int(os.environ.get("LOD_KIMI_COARSE_QUERY_TILE", "128")),
            key_step=int(os.environ.get("LOD_KIMI_COARSE_KEY_STEP", "32")),
        )
        subtile_mode = os.environ.get("LOD_KIMI_SUBTILE64", "0")
        subtile_probe = (subtile_mode in ("1", "score", "reuse")
                         and os.environ.get("LOD_KIMI_TILE_REFINE") == "1")
        if subtile_probe:
            from benchmarks.kimi_k3_subtile_route import serving_subtile_factory

            coarse_mha_fwd = serving_subtile_factory(
                score_only=subtile_mode in ("score", "reuse"),
                reuse_max=subtile_mode == "reuse",
                query_tile=int(os.environ.get("LOD_KIMI_COARSE_QUERY_TILE", "128")))
        route_mha_fwd = _specialized_route_mha_fwd(False, route_dim)

        def run_coarse_partition(
            begin: int,
            end: int,
            output_name: str,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
            attention_out = _workspace_tensor(
                buffers,
                output_name,
                (batch, query_len, query_heads, 128),
                dtype=q_expanded.dtype,
                device=q_expanded.device,
            )
            partition_k = coarse_k[:, begin:end]
            partition_v = expanded_v[:, begin:end]
            coarse_result = coarse_mha_fwd(
                coarse_q,
                partition_k,
                partition_v,
                0.0,  # dropout_p
                float(scale),  # softmax_scale
                False,  # all remote centroids precede every query row
                -1,
                -1,
                0,
                True,  # return_softmax_lse
                fused_route_coarse,
                None,
                None,
                attention_out,
                count_bias[..., begin:end],
                None,
                None,
                None,
                None,
                None,
            )
            return (
                attention_out,
                coarse_result[1],
                coarse_result[2] if fused_route_coarse else None,
            )

        def run_route_partition(
            begin: int,
            end: int,
        ) -> torch.Tensor:
            # The patched route specialization recognizes K==V as a QK-only
            # pass: it emits compact per-tile winners and skips value loads,
            # softmax/PV, LSE storage, and output storage.  The AITER wrapper
            # still requires an output-shaped tensor, so reuse the read-only
            # route query rather than retaining a dead D=192 workspace for
            # every MLA layer.
            route_result = route_mha_fwd(
                route_q,
                coarse_k[:, begin:end],
                coarse_k[:, begin:end],
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
                route_q,
                count_bias[..., begin:end],
                None,
                None,
                None,
                None,
                None,
            )
            return route_result[2]

        device_index = q_expanded.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        coarse_stream, route_stream = _kimi_prefill_attention_streams(device_index)
        foreground_stream = torch.cuda.current_stream(q_expanded.device)
        coarse_stream.wait_stream(foreground_stream)
        if not fused_route_coarse:
            route_stream.wait_stream(foreground_stream)

        with torch.cuda.stream(coarse_stream):
            if split_at:
                output_0, lse_0, fused_candidates_0 = run_coarse_partition(
                    0, split_at, "kimi_coarse_output_0"
                )
                output_1, lse_1, fused_candidates_1 = run_coarse_partition(
                    split_at, dispatch_state_len, "kimi_coarse_output_1"
                )
            else:
                output_0, lse_0, fused_candidates_0 = run_coarse_partition(
                    0, dispatch_state_len, "kimi_coarse_output_0"
                )
                output_1, lse_1 = output_0, lse_0
                fused_candidates_1 = None

        if fused_route_coarse:
            if fused_candidates_0 is None:
                raise RuntimeError("fused Kimi coarse attention omitted routes")
            candidates_0 = fused_candidates_0
            if split_at:
                if fused_candidates_1 is None:
                    raise RuntimeError("fused Kimi coarse partition omitted routes")
                candidates_1 = fused_candidates_1
            # Candidate reduction must follow the coarse score traversal that
            # produced the candidates.  Reusing this stream also gives the
            # caller one completion dependency for coarse output and routes.
            route_stream = coarse_stream
        else:
            with torch.cuda.stream(route_stream):
                if split_at:
                    candidates_0 = run_route_partition(0, split_at)
                    candidates_1 = run_route_partition(split_at, dispatch_state_len)
                else:
                    candidates_0 = run_route_partition(0, dispatch_state_len)
    finally:
        sys.setdlopenflags(original_dlopen_flags)
    if attention_end is not None:
        attention_end.record(route_stream)

    with torch.cuda.stream(route_stream):
        if os.environ.get("LOD_KIMI_TILE_REFINE") == "1":
            if split_at:
                raise RuntimeError("experimental tile refinement requires one coarse partition")
            from .kimi_route_tile_refine import refine_kimi_centroid_tiles

            candidates_0 = refine_kimi_centroid_tiles(
                candidates_0, q_expanded.contiguous(), expanded_k, log_counts,
                state_len=state_len, scale=scale, buffers=buffers,
                tile_n=64 if subtile_probe else 128,
            )
        if split_at:
            top_slots, route_head_counts, route_offsets = (
                _reduce_split_route_candidates(
                    candidates_0,
                    candidates_1,
                    second_index_offset=split_at,
                    state_len=state_len,
                    head_dim=route_dim,
                    buffers=buffers,
                )
            )
            selected_route_scores = _gather_selected_route_scores(
                candidates_0,
                top_slots,
                state_len=state_len,
                candidates_1=candidates_1,
                second_index_offset=split_at,
                buffers=buffers,
            )
        else:
            candidate_reducer = _reduce_route_candidates
            if os.environ.get("LOD_KIMI_KWAY_REDUCE") == "1":
                from .kimi_route_candidate_merge import merge_sorted_kimi_candidates

                # Eight selected tile lists are already sorted. The ordinary
                # global reducer remains the fallback for any other layout.
                if candidates_0.size(2) == 8:
                    def candidate_reducer(data, **kwargs):
                        return merge_sorted_kimi_candidates(
                            data, state_len=kwargs["state_len"],
                            slot_lengths=kwargs["slot_lengths"],
                            max_open_leaf_tokens=kwargs["max_open_leaf_tokens"],
                            buffers=kwargs["buffers"],
                        )
            (
                top_slots,
                route_head_counts,
                route_offsets,
                selected_route_scores,
            ) = candidate_reducer(
                candidates_0,
                slot_lengths=slot_lengths,
                max_open_leaf_tokens=max_open_leaf_tokens,
                close_selected_above_limit=max_open_leaf_tokens is not None,
                state_len=state_len,
                head_dim=route_dim,
                # Per-route atomics here serialize the stream that also owns
                # coarse attention.  Kimi's leaf consumer reconstructs the
                # same counts after that dependency, which lets the local
                # exact branch continue overlapping the coarse path.  A
                # matched 24-layer fixture improves by about 3% at 32--64K.
                emit_metadata=False,
                buffers=buffers,
            )
        # The caller waits on this one stream before opening leaves. Make that
        # dependency cover both independently scheduled attention branches.
        if route_stream is not coarse_stream:
            route_stream.wait_stream(coarse_stream)
    if reduce_end is not None and prepare_begin is not None:
        reduce_end.record(route_stream)
        reduce_end.synchronize()
        print(
            "KIMI_ROUTE_PHASES "
            f"prepare={prepare_begin.elapsed_time(prepare_end):.3f}ms "
            f"key_weight={prepare_end.elapsed_time(key_weight_end):.3f}ms "
            f"key_mm={key_weight_end.elapsed_time(key_mm_end):.3f}ms "
            f"key_pack={key_mm_end.elapsed_time(key_pack_end):.3f}ms "
            f"value_weight={key_pack_end.elapsed_time(value_weight_end):.3f}ms "
            f"value_mm={value_weight_end.elapsed_time(expand_end):.3f}ms "
            f"route_q={expand_end.elapsed_time(route_q_end):.3f}ms "
            f"attention={route_q_end.elapsed_time(attention_end):.3f}ms "
            f"reduce={attention_end.elapsed_time(reduce_end):.3f}ms",
            flush=True,
        )
    coarse = AiterPrefillCoarse(
        output_0=output_0,
        lse_0=lse_0,
        output_1=output_1,
        lse_1=lse_1,
        mean_k=mean_k,
        mean_v=projected_mean_v,
        counts=active_counts,
        has_second_partition=split_at != 0,
        ready_stream=route_stream,
        selected_route_scores=selected_route_scores,
    )
    return top_slots, coarse, route_head_counts, route_offsets


def aiter_kimi_local_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    query_offset: int,
    scale: float,
    expanded_q: torch.Tensor | None = None,
    w_uk_t: torch.Tensor | None = None,
    w_uv: torch.Tensor | None = None,
    output_buffer: torch.Tensor | None = None,
    return_lse: bool = True,
    buffers: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Kimi's causal exact field with AITER's absorbed-MLA assembly.

    The gfx942 kernel processes sixteen MQA query heads.  Full K3 has 96
    heads before tensor parallelism, so those heads become independent
    pseudo-sequences sharing the same latent KV field.  A TP rank with fewer
    than sixteen heads is zero padded exactly as the native vLLM backend does.
    """
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("Kimi local MLA requires rank-four Q/K")
    batch, query_heads, supplied_query_len, key_dim = q.shape
    key_len = int(k.size(2))
    query_len = key_len - int(query_offset)
    if key_dim != 576 or int(k.size(1)) != 1 or int(k.size(-1)) != 576:
        raise ValueError("Kimi local MLA requires Q/K=(576, 576) and one KV head")
    if supplied_query_len == key_len:
        actual_q = q[..., query_offset:, :]
    elif supplied_query_len == query_len:
        actual_q = q
    else:
        raise ValueError("Kimi local MLA query length is incompatible with K")
    if query_len <= 0:
        empty = q.new_empty(batch, query_heads, 0, 512)
        return empty, torch.empty(
            batch, query_heads, 0, dtype=torch.float32, device=q.device
        )

    # Prefill can stay entirely in the model's compute-friendly expanded
    # geometry.  Because W_UV is linear, merging attention branches after this
    # projection is exactly equivalent to merging 512-d latent values and then
    # projecting once.  This path obtains output and LSE together and avoids a
    # second absorbed-MLA QK pass.
    if w_uv is not None:
        if expanded_q is None or w_uk_t is None:
            raise ValueError("projected Kimi prefill requires expanded Q/W_UK_T")
        if tuple(expanded_q.shape) != (batch, query_heads, query_len, 192):
            raise ValueError("Kimi expanded local query has incompatible geometry")
        if tuple(w_uk_t.shape) != (query_heads, 128, 512):
            raise ValueError("Kimi local W_UK_T has incompatible geometry")
        if tuple(w_uv.shape) != (query_heads, 512, 128):
            raise ValueError("Kimi local W_UV has incompatible geometry")

        # Capacity-only opt-in: the fine-leaf head-group limit does not
        # bound these exact local projections. Serial groups keep all heads
        # and keys, but reuse a much smaller K/V workspace. Copies finish on
        # this stream before the next group overwrites that workspace.
        head_group = int(os.environ.get("LOD_KIMI_LOCAL_PREFILL_HEAD_GROUP", "0"))
        if head_group < 0 or (0 < head_group < query_heads and query_heads % head_group):
            raise ValueError("local prefill head group must be zero or a divisor of query heads")
        if 0 < head_group < query_heads:
            shape = (batch, query_heads, query_len, 128)
            if output_buffer is not None and tuple(output_buffer.shape) != shape:
                raise ValueError("projected Kimi local output buffer has incompatible shape")
            output = output_buffer if output_buffer is not None else _workspace_tensor(
                buffers, "kimi_grouped_local_output", shape, dtype=q.dtype, device=q.device)
            lse = (_workspace_tensor(buffers, "kimi_grouped_local_lse", shape[:3],
                                    dtype=torch.float32, device=q.device) if return_lse
                   else torch.empty(0, dtype=torch.float32, device=q.device))
            for begin in range(0, query_heads, head_group):
                end = begin + head_group
                _, group_lse = aiter_kimi_local_prefill_attention(
                    q[:, begin:end], k, query_offset=query_offset, scale=scale,
                    expanded_q=expanded_q[:, begin:end], w_uk_t=w_uk_t[begin:end],
                    w_uv=w_uv[begin:end], output_buffer=output[:, begin:end],
                    return_lse=return_lse, buffers=buffers)
                if return_lse:
                    lse[:, begin:end].copy_(group_lse)
            return output, lse

        expanded_k = _workspace_tensor(
            buffers,
            "kimi_local_expanded_k",
            (batch, key_len, query_heads, 192),
            dtype=q.dtype,
            device=q.device,
        )
        expanded_k_nope = _workspace_tensor(
            buffers,
            "kimi_local_expanded_k_nope",
            (batch, key_len, query_heads, 128),
            dtype=q.dtype,
            device=q.device,
        )
        latent = k[:, 0, :, :512].reshape(batch * key_len, 512)
        flat_key_weight = (
            _cached_flat_weight(buffers, "kimi_local_flat_uk", w_uk_t, (2, 0, 1), (512, query_heads * 128))
            if os.environ.get("LOD_KIMI_CACHE_PROJECTION_WEIGHTS") == "1"
            else w_uk_t.permute(2, 0, 1).reshape(512, query_heads * 128)
        )
        torch.mm(
            latent,
            flat_key_weight,
            out=expanded_k_nope.reshape(batch * key_len, query_heads * 128),
        )
        expanded_k[..., :128].copy_(expanded_k_nope)
        expanded_k[..., 128:].copy_(
            k[:, 0, :, None, 512:].expand(-1, -1, query_heads, -1)
        )
        expanded_v = _workspace_tensor(
            buffers,
            "kimi_local_expanded_v",
            (batch, key_len, query_heads, 128),
            dtype=q.dtype,
            device=q.device,
        )
        flat_value_weight = (
            _cached_flat_weight(buffers, "kimi_local_flat_uv", w_uv, (1, 0, 2), (512, query_heads * 128))
            if os.environ.get("LOD_KIMI_CACHE_PROJECTION_WEIGHTS") == "1"
            else w_uv.permute(1, 0, 2).reshape(512, query_heads * 128)
        )
        torch.mm(
            latent,
            flat_value_weight,
            out=expanded_v.reshape(
                batch * key_len, query_heads * 128
            ),
        )
        native_local = os.environ.get("LOD_KIMI_NATIVE_LOCAL_PREFILL") == "1"
        attention_out = None if native_local else _workspace_tensor(
            buffers, "kimi_local_attention_out",
            (batch, query_len, query_heads, 128), dtype=q.dtype, device=q.device,
        )
        # Dqk=192/Dv=128 is supported by both this CK specialization and the
        # image's native AITER v3 dispatcher, including LSE. Test the latter
        # only for the ordinary local field: coarse still needs our bias and
        # routing extension. Neither path pads V or recovers LSE separately.
        original_dlopen_flags = sys.getdlopenflags()
        deepbind = getattr(os, "RTLD_DEEPBIND", 0)
        if deepbind:
            sys.setdlopenflags(original_dlopen_flags | deepbind)
        try:
            if native_local:
                from aiter import flash_attn_func

                result = flash_attn_func(
                    expanded_q.permute(0, 2, 1, 3), expanded_k, expanded_v,
                    softmax_scale=float(scale), causal=True, return_lse=return_lse,
                )
                attention_out = result[0] if return_lse else result
            else:
                result = _specialized_kimi_coarse_mha_fwd()(
                    expanded_q.permute(0, 2, 1, 3),
                    expanded_k, expanded_v,
                    0.0, float(scale), True, -1, -1, 0, return_lse, False,
                    None, None, attention_out,
                    None, None, None, None, None, None,
                )
        finally:
            if deepbind:
                sys.setdlopenflags(original_dlopen_flags)
        projected = attention_out.permute(0, 2, 1, 3)
        if output_buffer is None:
            # The MLA refinement kernel consumes explicit strides, so retain
            # AITER's token-major storage as a head-major view instead of
            # copying the complete local output solely to change layout.
            output = projected
        else:
            if tuple(output_buffer.shape) != tuple(projected.shape):
                raise ValueError(
                    "projected Kimi local output buffer has incompatible shape"
                )
            output_buffer.copy_(projected)
            output = output_buffer
        if return_lse:
            lse = result[1]
        else:
            lse = torch.empty(0, dtype=torch.float32, device=q.device)
        return output, lse

    padded_heads = max(16, ((query_heads + 15) // 16) * 16)
    head_groups = padded_heads // 16
    sequence_count = batch * head_groups
    folded_q = _workspace_tensor(
        buffers,
        "kimi_local_folded_q",
        (sequence_count * query_len, 16, 576),
        dtype=q.dtype,
        device=q.device,
    )
    padded_q = _workspace_tensor(
        buffers,
        "kimi_local_padded_q",
        (batch, query_len, padded_heads, 576),
        dtype=q.dtype,
        device=q.device,
    )
    padded_q.zero_()
    padded_q[:, :, :query_heads].copy_(actual_q.permute(0, 2, 1, 3))
    folded_q.copy_(
        padded_q.view(batch, query_len, head_groups, 16, 576)
        .permute(0, 2, 1, 3, 4)
        .reshape_as(folded_q)
    )

    folded_output = _workspace_tensor(
        buffers,
        "kimi_local_folded_output",
        (sequence_count * query_len, 16, 512),
        dtype=q.dtype,
        device=q.device,
    )
    qo_indptr = _workspace_tensor(
        buffers,
        "kimi_local_qo_indptr",
        (sequence_count + 1,),
        dtype=torch.int32,
        device=q.device,
    )
    kv_indptr = _workspace_tensor(
        buffers,
        "kimi_local_kv_indptr",
        (sequence_count + 1,),
        dtype=torch.int32,
        device=q.device,
    )
    kv_indices = _workspace_tensor(
        buffers,
        "kimi_local_kv_indices",
        (sequence_count * key_len,),
        dtype=torch.int32,
        device=q.device,
    )
    kv_last_page_lens = _workspace_tensor(
        buffers,
        "kimi_local_kv_last_page_lens",
        (sequence_count,),
        dtype=torch.int32,
        device=q.device,
    )
    torch.arange(
        sequence_count + 1, device=q.device, dtype=torch.int32, out=qo_indptr
    )
    qo_indptr.mul_(query_len)
    torch.arange(
        sequence_count + 1, device=q.device, dtype=torch.int32, out=kv_indptr
    )
    kv_indptr.mul_(key_len)
    base_indices = torch.arange(key_len, device=q.device, dtype=torch.int32)
    for batch_index in range(batch):
        begin = batch_index * head_groups * key_len
        end = begin + head_groups * key_len
        kv_indices[begin:end].copy_(
            (base_indices + batch_index * key_len).repeat(head_groups)
        )
    kv_last_page_lens.fill_(1)
    kv_buffer = k.permute(0, 2, 1, 3).reshape(batch * key_len, 1, 1, 576)

    folded_lse = _workspace_tensor(
        buffers,
        "kimi_local_folded_lse",
        (sequence_count * query_len, 1, 16, 1),
        dtype=torch.float32,
        device=q.device,
    )
    profile = os.environ.get("LOD_KIMI_PROFILE_PREFILL") == "1"
    native_begin = torch.cuda.Event(enable_timing=True) if profile else None
    native_end = torch.cuda.Event(enable_timing=True) if profile else None
    expand_end = torch.cuda.Event(enable_timing=True) if profile else None
    flash_end = torch.cuda.Event(enable_timing=True) if profile else None
    if native_begin is not None:
        native_begin.record()
    _kimi_local_mla_prefill_op()(
        folded_q,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        query_len,
        float(scale),
        folded_output.view(sequence_count * query_len, 1, 16, 512),
        folded_lse,
    )
    if native_end is not None:
        native_end.record()

    unfolded = (
        folded_output.view(batch, head_groups, query_len, 16, 512)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, query_len, padded_heads, 512)[:, :, :query_heads]
        .permute(0, 2, 1, 3)
    )
    if output_buffer is None:
        output = unfolded.contiguous()
    else:
        if tuple(output_buffer.shape) != tuple(unfolded.shape):
            raise ValueError("Kimi local MLA output buffer has incompatible shape")
        output_buffer.copy_(unfolded)
        output = output_buffer
    if return_lse:
        # The gfx942 absorbed-MLA assembly intentionally omits LSE when its
        # only KV split writes the final output directly.  Recover the exact
        # normalizer with the compute-friendly D=192 AITER path.  This still
        # avoids the much more expensive generic D=576/V=512 local attention:
        # the absorbed kernel supplies the value output, while this call uses
        # only a 128-wide disposable value to obtain LSE.
        if expanded_q is None or w_uk_t is None:
            raise ValueError("Kimi local MLA LSE requires expanded Q and W_UK_T")
        if tuple(expanded_q.shape) != (batch, query_heads, query_len, 192):
            raise ValueError("Kimi expanded local query has incompatible geometry")
        if tuple(w_uk_t.shape) != (query_heads, 128, 512):
            raise ValueError("Kimi local W_UK_T has incompatible geometry")
        expanded_k = _workspace_tensor(
            buffers,
            "kimi_local_expanded_k",
            (batch, key_len, query_heads, 192),
            dtype=q.dtype,
            device=q.device,
        )
        expanded_k_nope = _workspace_tensor(
            buffers,
            "kimi_local_expanded_k_nope",
            (batch, key_len, query_heads, 128),
            dtype=q.dtype,
            device=q.device,
        )
        flat_weight = w_uk_t.permute(2, 0, 1).reshape(512, query_heads * 128)
        torch.mm(
            k[:, 0, :, :512].reshape(batch * key_len, 512),
            flat_weight,
            out=expanded_k_nope.reshape(batch * key_len, query_heads * 128),
        )
        expanded_k[..., :128].copy_(expanded_k_nope)
        expanded_k[..., 128:].copy_(
            k[:, 0, :, None, 512:].expand(-1, -1, query_heads, -1)
        )
        if expand_end is not None:
            expand_end.record()
        # Use the same long-prefill FlashAttention implementation selected by
        # vLLM for native K3.  The direct CK entry point rejects some long
        # Dqk=192/Dv=128 causal shapes, whereas the public varlen dispatcher
        # handles them.  Values do not affect LSE, so reusing expanded K as a
        # disposable 192-wide V avoids another expanded-cache allocation.
        from flash_attn import flash_attn_varlen_func

        cu_q = _workspace_tensor(
            buffers,
            "kimi_local_cu_q",
            (batch + 1,),
            dtype=torch.int32,
            device=q.device,
        )
        cu_k = _workspace_tensor(
            buffers,
            "kimi_local_cu_k",
            (batch + 1,),
            dtype=torch.int32,
            device=q.device,
        )
        torch.arange(batch + 1, device=q.device, dtype=torch.int32, out=cu_q)
        cu_q.mul_(query_len)
        torch.arange(batch + 1, device=q.device, dtype=torch.int32, out=cu_k)
        cu_k.mul_(key_len)
        flash_result = flash_attn_varlen_func(
            expanded_q.permute(0, 2, 1, 3).reshape(
                batch * query_len, query_heads, 192
            ),
            expanded_k.reshape(batch * key_len, query_heads, 192),
            expanded_k.reshape(batch * key_len, query_heads, 192),
            cu_q,
            cu_k,
            query_len,
            key_len,
            dropout_p=0.0,
            softmax_scale=float(scale),
            causal=True,
            return_attn_probs=True,
        )
        raw_lse = flash_result[1]
        lse = raw_lse.view(query_heads, batch, query_len).permute(1, 0, 2)
        if flash_end is not None:
            flash_end.record()
    else:
        lse = torch.empty(0, dtype=torch.float32, device=q.device)
    if native_begin is not None and native_end is not None:
        torch.cuda.synchronize(q.device)
        parts = [f"native={native_begin.elapsed_time(native_end):.3f}ms"]
        if return_lse and expand_end is not None and flash_end is not None:
            parts.extend(
                (
                    f"expand={native_end.elapsed_time(expand_end):.3f}ms",
                    f"lse={expand_end.elapsed_time(flash_end):.3f}ms",
                )
            )
        print("KIMI_LOCAL_PHASES " + " ".join(parts), flush=True)
    return output, lse


@triton.jit(
    do_not_specialize=["QUERY_LEN", "STATE_LEN"],
    do_not_specialize_on_alignment=["QUERY_LEN", "STATE_LEN"],
)
def _merge_mla_route_refinement_kernel(
    q,
    sink_k,
    sink_v,
    mean_k,
    mean_v,
    counts,
    slots,
    selected_route_scores,
    coarse_out,
    coarse_lse,
    route_out,
    route_lse,
    local_out,
    local_lse,
    output,
    output_lse,
    Q_BATCH_STRIDE,
    Q_HEAD_STRIDE,
    Q_TOKEN_STRIDE,
    SINK_K_BATCH_STRIDE,
    SINK_K_HEAD_STRIDE,
    SINK_K_TOKEN_STRIDE,
    SINK_V_BATCH_STRIDE,
    SINK_V_HEAD_STRIDE,
    SINK_V_TOKEN_STRIDE,
    MEAN_V_BATCH_STRIDE,
    MEAN_V_HEAD_STRIDE,
    MEAN_V_TOKEN_STRIDE,
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
    LATENT_DIM: tl.constexpr,
    DIRECT_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    SINK_LEN: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    PROJECTED_VALUES: tl.constexpr,
    SINK_KEY_GROUP_SIZE: tl.constexpr,
    AGGREGATED_ROUTES: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    """Exact coarse replacement for asymmetric absorbed-MLA K/V widths."""
    batch_head = tl.program_id(0).to(tl.int64)
    query = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    batch = batch_head // QUERY_HEADS
    query_head = batch_head - batch * QUERY_HEADS
    kv_head = query_head // KV_GROUP_SIZE
    sink_key_head = query_head // SINK_KEY_GROUP_SIZE
    valid_query = query < QUERY_LEN
    latent_dim = tl.arange(0, LATENT_DIM)
    direct_dim = tl.arange(0, max(1, DIRECT_DIM))
    value_dim = tl.arange(0, VALUE_DIM)
    query_base = (
        batch * Q_BATCH_STRIDE
        + query_head * Q_HEAD_STRIDE
        + query[:, None] * Q_TOKEN_STRIDE
    )
    query_latent = tl.load(
        q + query_base + latent_dim[None, :],
        mask=valid_query[:, None],
        other=0.0,
    )
    query_direct = tl.load(
        q + query_base + LATENT_DIM + direct_dim[None, :],
        mask=valid_query[:, None] & (direct_dim[None, :] < DIRECT_DIM),
        other=0.0,
    )

    query_row = (batch * QUERY_HEADS + query_head) * QUERY_LEN + query
    coarse_score = tl.load(
        coarse_lse + query_row, mask=valid_query, other=-float("inf")
    ).to(tl.float32)
    coarse_row = (batch * QUERY_LEN + query) * QUERY_HEADS + query_head
    coarse_value = tl.load(
        coarse_out + coarse_row[:, None] * VALUE_DIM + value_dim[None, :],
        mask=valid_query[:, None],
        other=0.0,
    ).to(tl.float32)

    selected_mass = tl.zeros((BLOCK_M,), tl.float32)
    selected_value = tl.zeros((BLOCK_M, VALUE_DIM), tl.float32)
    for rank in tl.static_range(0, ROUTE_COUNT):
        slot = tl.load(
            slots + query_row * ROUTE_COUNT + rank,
            mask=valid_query,
            other=-1,
        ).to(tl.int64)
        valid_slot = valid_query & (slot >= 0) & (slot < STATE_LEN)
        safe_slot = tl.where(valid_slot, slot, 0)
        state_index = (batch * KV_HEADS + kv_head) * STATE_LEN + safe_slot
        value = tl.load(
            mean_v
            + batch * MEAN_V_BATCH_STRIDE
            + (query_head if PROJECTED_VALUES else kv_head) * MEAN_V_HEAD_STRIDE
            + safe_slot[:, None] * MEAN_V_TOKEN_STRIDE
            + value_dim[None, :],
            mask=valid_slot[:, None],
            other=0.0,
        )
        score = tl.load(
            selected_route_scores + query_row * ROUTE_COUNT + rank,
            mask=valid_slot,
            other=-float("inf"),
        ).to(tl.float32)
        weight = tl.where(valid_slot, tl.exp(score - coarse_score), 0.0)
        selected_mass += weight
        selected_value += weight[:, None] * value

    remaining_mass = tl.maximum(1.0 - selected_mass, 1.0e-7)
    coarse_value = (coarse_value - selected_value) / remaining_mass[:, None]
    residual_coarse_lse = coarse_score + tl.log(remaining_mass)

    local_score = tl.load(
        local_lse
        + batch * LOCAL_LSE_BATCH_STRIDE
        + query_head * LOCAL_LSE_HEAD_STRIDE
        + query * LOCAL_LSE_TOKEN_STRIDE,
        mask=valid_query,
        other=-float("inf"),
    ).to(tl.float32)
    if AGGREGATED_ROUTES:
        refined_score = tl.load(
            route_lse + query_row,
            mask=valid_query,
            other=-float("inf"),
        ).to(tl.float32)
    else:
        rank_lane = tl.arange(0, ROUTE_COUNT)
        route_scores = tl.load(
            route_lse + query_row[:, None] * ROUTE_COUNT + rank_lane[None, :],
            mask=valid_query[:, None],
            other=-float("inf"),
        ).to(tl.float32)
    maximum = tl.maximum(residual_coarse_lse, local_score)
    if AGGREGATED_ROUTES:
        maximum = tl.maximum(maximum, refined_score)
    else:
        maximum = tl.maximum(maximum, tl.max(route_scores, axis=1))
    for sink_index in tl.static_range(0, SINK_LEN):
        sink_key_latent = tl.load(
            sink_k
            + batch * SINK_K_BATCH_STRIDE
            + sink_key_head * SINK_K_HEAD_STRIDE
            + sink_index * SINK_K_TOKEN_STRIDE
            + latent_dim
        )
        sink_key_direct = tl.load(
            sink_k
            + batch * SINK_K_BATCH_STRIDE
            + sink_key_head * SINK_K_HEAD_STRIDE
            + sink_index * SINK_K_TOKEN_STRIDE
            + LATENT_DIM
            + direct_dim,
            mask=direct_dim < DIRECT_DIM, other=0.0,
        )
        sink_score = tl.sum(query_latent.to(tl.float32) * sink_key_latent[None, :].to(tl.float32), axis=1)
        sink_score += tl.sum(query_direct.to(tl.float32) * sink_key_direct[None, :].to(tl.float32), axis=1)
        maximum = tl.maximum(maximum, sink_score * SCALE)

    coarse_weight = tl.exp(residual_coarse_lse - maximum)
    local_weight = tl.exp(local_score - maximum)
    denominator = coarse_weight + local_weight
    numerator = coarse_weight[:, None] * coarse_value
    local_value = tl.load(
        local_out
        + batch * LOCAL_BATCH_STRIDE
        + query_head * LOCAL_HEAD_STRIDE
        + query[:, None] * LOCAL_TOKEN_STRIDE
        + value_dim[None, :],
        mask=valid_query[:, None],
        other=0.0,
    ).to(tl.float32)
    numerator += local_weight[:, None] * local_value
    if AGGREGATED_ROUTES:
        weight = tl.exp(refined_score - maximum)
        value = tl.load(
            route_out
            + query_row[:, None] * VALUE_DIM
            + value_dim[None, :],
            mask=valid_query[:, None],
            other=0.0,
        ).to(tl.float32)
        denominator += weight
        numerator += weight[:, None] * value
    else:
        for rank in tl.static_range(0, ROUTE_COUNT):
            slot = tl.load(
                slots + query_row * ROUTE_COUNT + rank,
                mask=valid_query,
                other=-1,
            )
            refined = valid_query & (slot >= 0)
            score = tl.load(
                route_lse + query_row * ROUTE_COUNT + rank,
                mask=refined,
                other=-float("inf"),
            ).to(tl.float32)
            weight = tl.exp(score - maximum)
            value = tl.load(
                route_out
                + (query_row * ROUTE_COUNT + rank)[:, None] * VALUE_DIM
                + value_dim[None, :],
                mask=refined[:, None],
                other=0.0,
            ).to(tl.float32)
            denominator += weight
            numerator += weight[:, None] * value
    for sink_index in tl.static_range(0, SINK_LEN):
        sink_key_latent = tl.load(
            sink_k
            + batch * SINK_K_BATCH_STRIDE
            + sink_key_head * SINK_K_HEAD_STRIDE
            + sink_index * SINK_K_TOKEN_STRIDE
            + latent_dim
        )
        sink_key_direct = tl.load(
            sink_k
            + batch * SINK_K_BATCH_STRIDE
            + sink_key_head * SINK_K_HEAD_STRIDE
            + sink_index * SINK_K_TOKEN_STRIDE
            + LATENT_DIM
            + direct_dim,
            mask=direct_dim < DIRECT_DIM, other=0.0,
        )
        sink_value = tl.load(
            sink_v
            + batch * SINK_V_BATCH_STRIDE
            + (query_head if PROJECTED_VALUES else kv_head) * SINK_V_HEAD_STRIDE
            + sink_index * SINK_V_TOKEN_STRIDE
            + value_dim
        )
        sink_score = tl.sum(query_latent.to(tl.float32) * sink_key_latent[None, :].to(tl.float32), axis=1)
        sink_score += tl.sum(query_direct.to(tl.float32) * sink_key_direct[None, :].to(tl.float32), axis=1)
        weight = tl.exp(sink_score * SCALE - maximum)
        denominator += weight
        numerator += weight[:, None] * sink_value[None, :]

    tl.store(
        output
        + batch * OUTPUT_BATCH_STRIDE
        + query_head * OUTPUT_HEAD_STRIDE
        + query[:, None] * OUTPUT_TOKEN_STRIDE
        + value_dim[None, :],
        numerator / denominator[:, None],
        mask=valid_query[:, None],
    )
    if RETURN_LSE:
        tl.store(
            output_lse + query_row,
            maximum + tl.log(denominator),
            mask=valid_query,
        )


def merge_aiter_mla_prefill_refinement(
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
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Merge asymmetric MLA coarse, refined, local, and sink branches."""
    batch, query_heads, query_len, key_dim = q.shape
    kv_heads = int(coarse.mean_k.size(1))
    state_len = int(coarse.mean_k.size(2))
    value_dim = int(coarse.mean_v.size(-1))
    projected_values = int(coarse.mean_v.size(1)) == query_heads
    latent_dim = (key_dim - (0 if key_dim in (256, 512) else 64)
                  if projected_values else value_dim)
    direct_dim = key_dim - latent_dim
    route_count = int(slots.size(-1))
    expected_prefix = (batch, query_heads, query_len)
    if direct_dim not in (0, 64) or latent_dim <= 0 or key_dim > 576:
        raise ValueError(
            "AITER MLA refinement requires Kimi K/V=(L+64,L) or "
            "latent512 or projected256 NoPE keys with optional projected values"
        )
    expected_value_heads = query_heads if projected_values else kv_heads
    if (
        tuple(coarse.mean_v.shape[:2]) != (batch, expected_value_heads)
        or int(coarse.mean_v.size(2)) < state_len
    ):
        raise ValueError("AITER MLA refinement centroid values have wrong geometry")
    if query_heads != kv_heads * kv_group_size or route_count != 8:
        raise ValueError("AITER MLA refinement has incompatible GQA/routes")
    sink_heads = int(sink_k.size(1))
    if (sink_k.ndim != 4 or sink_k.size(0) != batch or sink_k.size(-1) != key_dim
            or sink_heads not in (kv_heads, query_heads)
            or tuple(sink_v.shape) != (batch, expected_value_heads, sink_k.size(2), value_dim)):
        raise ValueError("AITER MLA refinement sink geometry is incompatible")
    if tuple(local_out.shape) != (*expected_prefix, value_dim):
        raise ValueError("AITER MLA refinement local output has wrong geometry")
    if tuple(local_lse.shape) != expected_prefix:
        raise ValueError("AITER MLA refinement local LSE has wrong geometry")
    aggregated_routes = tuple(route_out.shape) == (*expected_prefix, value_dim)
    if not aggregated_routes and tuple(route_out.shape) != (
        *expected_prefix,
        route_count,
        value_dim,
    ):
        raise ValueError("AITER MLA refinement route output has wrong geometry")
    expected_route_lse = (
        expected_prefix if aggregated_routes else (*expected_prefix, route_count)
    )
    if tuple(route_lse.shape) != expected_route_lse:
        raise ValueError("AITER MLA refinement route LSE has wrong geometry")
    if tuple(slots.shape) != (*expected_prefix, route_count):
        raise ValueError("AITER MLA refinement slots have wrong geometry")
    if coarse.has_second_partition:
        raise ValueError("AITER MLA refinement does not use split coarse partitions")
    if coarse.selected_route_scores is None or tuple(
        coarse.selected_route_scores.shape
    ) != tuple(slots.shape):
        raise ValueError("AITER MLA refinement is missing selected route scores")
    expected_output = (*expected_prefix, value_dim)
    output = (
        q.new_empty(expected_output) if output_buffer is None else output_buffer
    )
    if tuple(output.shape) != expected_output or output.stride(-1) != 1:
        raise ValueError("AITER MLA refinement output buffer has wrong geometry")
    output_lse = (
        torch.empty(expected_prefix, device=q.device, dtype=torch.float32)
        if return_lse else None
    )

    block_m = 16
    _merge_mla_route_refinement_kernel[
        (batch * query_heads, triton.cdiv(query_len, block_m))
    ](
        q,
        sink_k,
        sink_v,
        coarse.mean_k,
        coarse.mean_v,
        coarse.counts,
        slots,
        coarse.selected_route_scores,
        coarse.output_0,
        coarse.lse_0,
        route_out,
        route_lse,
        local_out,
        local_lse,
        output,
        output_lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        sink_k.stride(0),
        sink_k.stride(1),
        sink_k.stride(2),
        sink_v.stride(0),
        sink_v.stride(1),
        sink_v.stride(2),
        coarse.mean_v.stride(0),
        coarse.mean_v.stride(1),
        coarse.mean_v.stride(2),
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
        LATENT_DIM=latent_dim,
        DIRECT_DIM=direct_dim,
        VALUE_DIM=value_dim,
        SINK_LEN=int(sink_k.size(2)),
        SCALE=float(scale),
        BLOCK_M=block_m,
        ROUTE_COUNT=route_count,
        PROJECTED_VALUES=projected_values,
        SINK_KEY_GROUP_SIZE=query_heads // sink_heads,
        AGGREGATED_ROUTES=aggregated_routes,
        RETURN_LSE=return_lse,
        num_warps=8,
        waves_per_eu=1,
    )
    return (output, output_lse) if return_lse else output


__all__ = [
    "aiter_kimi_expanded_prefill_route_coarse_attention",
    "aiter_kimi_local_prefill_attention",
    "aiter_mla_prefill_route_coarse_attention",
    "merge_aiter_mla_prefill_refinement",
]
