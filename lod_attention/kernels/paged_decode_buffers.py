"""Fixed-address decode metadata, union construction, and scratch buffers."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._paged_common import _lookup_page_id


@triton.jit
def _advance_decode_cache_lengths_kernel(
    cache_indices,
    local_lens,
    num_rows,
    increment: tl.constexpr,
):
    row = tl.program_id(0)
    if row < num_rows:
        cache_row = tl.load(cache_indices + row).to(tl.int64)
        length = tl.load(local_lens + cache_row)
        tl.store(local_lens + cache_row, length + increment)


def advance_decode_cache_lengths(
    cache_indices: torch.Tensor,
    local_lens: torch.Tensor,
    *,
    increment: int = 1,
) -> None:
    """Advance fixed-pool local lengths after a fused decode launch.

    ``cache_indices`` must be unique.  Keeping this as a separate launch makes
    every attention program observe the same pre-append length and remains
    safe to capture and replay in a CUDA graph.
    """
    if cache_indices.ndim != 1 or local_lens.ndim != 1:
        raise ValueError("decode cache indices and lengths must be vectors")
    if cache_indices.device != local_lens.device:
        raise ValueError("decode cache indices and lengths must share a device")
    if increment <= 0:
        raise ValueError("decode cache length increment must be positive")
    rows = int(cache_indices.numel())
    if rows:
        _advance_decode_cache_lengths_kernel[(rows,)](
            cache_indices,
            local_lens,
            rows,
            increment=increment,
            num_warps=1,
        )


@triton.jit
def _prepare_speculative_decode_kv_kernel(
    cache_indices,
    local_lens,
    new_k,
    new_v,
    local_k,
    local_v,
    expanded_cache_indices,
    expanded_local_lens,
    NEW_K_BATCH_STRIDE,
    NEW_K_HEAD_STRIDE,
    NEW_V_BATCH_STRIDE,
    NEW_V_HEAD_STRIDE,
    LOCAL_K_BATCH_STRIDE,
    LOCAL_K_HEAD_STRIDE,
    LOCAL_K_TOKEN_STRIDE,
    LOCAL_V_BATCH_STRIDE,
    LOCAL_V_HEAD_STRIDE,
    LOCAL_V_TOKEN_STRIDE,
    ROWS: tl.constexpr,
    STEPS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Stage all proposal K/V before parallel speculative attention.

    The second target query must see the first proposal key.  Writing every
    proposal key in a separate launch provides the required device-wide
    barrier while the per-position logical lengths preserve causality.
    """
    sequence = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)
    logical_batch = sequence // KV_HEADS
    kv_head = sequence - logical_batch * KV_HEADS
    step = logical_batch // ROWS
    request = logical_batch - step * ROWS
    cache_batch = tl.load(cache_indices + request).to(tl.int64)
    base_length = tl.load(local_lens + cache_batch).to(tl.int32)
    dimension = block * BLOCK_D + tl.arange(0, BLOCK_D)
    valid = dimension < HEAD_DIM
    key = tl.load(
        new_k
        + logical_batch * NEW_K_BATCH_STRIDE
        + kv_head * NEW_K_HEAD_STRIDE
        + dimension,
        mask=valid,
        other=0.0,
    )
    value = tl.load(
        new_v
        + logical_batch * NEW_V_BATCH_STRIDE
        + kv_head * NEW_V_HEAD_STRIDE
        + dimension,
        mask=valid,
        other=0.0,
    )
    tl.store(
        local_k
        + cache_batch * LOCAL_K_BATCH_STRIDE
        + kv_head * LOCAL_K_HEAD_STRIDE
        + (base_length + step) * LOCAL_K_TOKEN_STRIDE
        + dimension,
        key,
        mask=valid,
    )
    tl.store(
        local_v
        + cache_batch * LOCAL_V_BATCH_STRIDE
        + kv_head * LOCAL_V_HEAD_STRIDE
        + (base_length + step) * LOCAL_V_TOKEN_STRIDE
        + dimension,
        value,
        mask=valid,
    )
    if kv_head == 0 and block == 0:
        tl.store(expanded_cache_indices + logical_batch, cache_batch)
        tl.store(expanded_local_lens + logical_batch, base_length + step)


def prepare_speculative_decode_kv(
    cache_indices: torch.Tensor,
    local_lens: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    local_k: torch.Tensor,
    local_v: torch.Tensor,
    expanded_cache_indices: torch.Tensor,
    expanded_local_lens: torch.Tensor,
    *,
    rows: int,
    steps: int,
) -> None:
    """Prepare step-major proposal rows for one parallel LOD verifier call."""
    total_rows, kv_heads, query_len, head_dim = new_k.shape
    if query_len != 1 or tuple(new_v.shape[:3]) != (total_rows, kv_heads, 1):
        raise ValueError("speculative decode K/V must contain one token per row")
    if total_rows != rows * steps or rows < 1 or steps < 2:
        raise ValueError("speculative decode row geometry is inconsistent")
    if tuple(cache_indices.shape) != (rows,):
        raise ValueError("speculative decode needs one physical cache row per request")
    if tuple(expanded_cache_indices.shape) != (total_rows,) or tuple(
        expanded_local_lens.shape
    ) != (total_rows,):
        raise ValueError("speculative decode metadata staging has the wrong shape")
    if int(local_k.size(-1)) != head_dim or int(local_v.size(-1)) != int(
        new_v.size(-1)
    ):
        raise ValueError("speculative decode local K/V dimensions do not match")
    block_d = min(256, triton.next_power_of_2(head_dim))
    _prepare_speculative_decode_kv_kernel[
        (total_rows * kv_heads, triton.cdiv(head_dim, block_d))
    ](
        cache_indices,
        local_lens,
        new_k,
        new_v,
        local_k,
        local_v,
        expanded_cache_indices,
        expanded_local_lens,
        new_k.stride(0),
        new_k.stride(1),
        new_v.stride(0),
        new_v.stride(1),
        local_k.stride(0),
        local_k.stride(1),
        local_k.stride(2),
        local_v.stride(0),
        local_v.stride(1),
        local_v.stride(2),
        ROWS=rows,
        STEPS=steps,
        KV_HEADS=kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        num_warps=1,
    )


@triton.jit
def _decode_topk_gqa_union_kernel(
    top_slots,
    cache_indices,
    local_lens,
    state_lens,
    seen_stamps,
    sequence_epochs,
    union_counts,
    union_token_counts,
    union_slots,
    context_lens,
    new_k,
    new_v,
    arena_k,
    arena_v,
    state_len,
    TOP_BATCH_STRIDE,
    TOP_HEAD_STRIDE,
    NEW_K_BATCH_STRIDE,
    NEW_K_HEAD_STRIDE,
    NEW_V_BATCH_STRIDE,
    NEW_V_HEAD_STRIDE,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    CANDIDATE_BLOCK: tl.constexpr,
    LOCAL_LIMIT: tl.constexpr,
    LOCAL_OFFSET: tl.constexpr,
    LOCAL_CAPACITY: tl.constexpr,
    SINK_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    INCLUDE_NEW: tl.constexpr,
    PREPARE_IMPLICIT_LOD: tl.constexpr,
    USE_STATE_LENS: tl.constexpr,
):
    """Build the unique top-k centroid list for one decode GQA group."""
    sequence = tl.program_id(0).to(tl.int64)
    batch = sequence // KV_HEADS
    kv_head = sequence - batch * KV_HEADS
    candidate = tl.arange(0, CANDIDATE_BLOCK)
    epoch = tl.load(sequence_epochs + sequence).to(tl.int32) + 1
    tl.store(sequence_epochs + sequence, epoch)
    candidate_valid = candidate < KV_GROUP_SIZE * ROUTE_COUNT
    query_lane = candidate // ROUTE_COUNT
    route = candidate - query_lane * ROUTE_COUNT
    query_head = kv_head * KV_GROUP_SIZE + query_lane
    slot = tl.load(
        top_slots + batch * TOP_BATCH_STRIDE + query_head * TOP_HEAD_STRIDE + route,
        mask=candidate_valid,
        other=-1,
    ).to(tl.int32)
    valid = candidate_valid & (slot >= 0) & (slot < state_len)
    safe_slot = tl.where(valid, slot, 0)
    stamp_pointer = seen_stamps + sequence * STATE_CAPACITY + safe_slot
    epoch_vector = tl.full((CANDIDATE_BLOCK,), 0, tl.int32) + epoch
    observed_epoch = tl.load(stamp_pointer, mask=valid, other=epoch_vector)
    old_epoch = tl.atomic_cas(
        stamp_pointer,
        tl.where(valid, observed_epoch, -1),
        epoch_vector,
        sem="relaxed",
    )
    unique = valid & (old_epoch != epoch_vector)
    unique_integer = unique.to(tl.int32)
    destination = tl.cumsum(unique_integer, axis=0) - 1
    tl.store(
        union_counts + sequence + candidate * 0,
        tl.sum(unique_integer, axis=0),
        mask=candidate == 0,
    )
    tl.store(
        union_token_counts + sequence + candidate * 0,
        0,
        mask=candidate == 0,
    )
    tl.store(
        union_slots + sequence * (KV_GROUP_SIZE * ROUTE_COUNT) + destination,
        slot,
        mask=unique,
    )
    if PREPARE_IMPLICIT_LOD:
        cache_batch = tl.load(cache_indices + batch).to(tl.int64)
        active_local = tl.minimum(
            tl.load(local_lens + cache_batch).to(tl.int32), LOCAL_LIMIT
        )
        if USE_STATE_LENS:
            active_state = tl.minimum(
                tl.load(state_lens + cache_batch).to(tl.int32), state_len
            )
        else:
            active_state = state_len
        tl.store(
            context_lens + sequence,
            active_local + INCLUDE_NEW + SINK_LEN + active_state,
        )
        if INCLUDE_NEW:
            dimension = tl.arange(0, HEAD_DIM)
            current_key = tl.load(
                new_k
                + batch * NEW_K_BATCH_STRIDE
                + kv_head * NEW_K_HEAD_STRIDE
                + dimension
            )
            current_value = tl.load(
                new_v
                + batch * NEW_V_BATCH_STRIDE
                + kv_head * NEW_V_HEAD_STRIDE
                + dimension
            )
            storage = (
                LOCAL_OFFSET
                + (cache_batch * KV_HEADS + kv_head) * LOCAL_CAPACITY
                + active_local
            ) * HEAD_DIM + dimension
            tl.store(arena_k + storage, current_key)
            tl.store(arena_v + storage, current_value)


@triton.jit
def _expand_decode_topk_gqa_union_kernel(
    cache_indices,
    local_lens,
    page_indices,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    slot_lengths,
    union_counts,
    union_token_counts,
    union_slots,
    token_indices,
    hip_block_table,
    hip_context_lens,
    fixed_indices,
    fixed_slot_offsets,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    LOCAL_LIMIT: tl.constexpr,
    INDEX_CAPACITY: tl.constexpr,
    UNION_CAPACITY: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INCLUDE_NEW: tl.constexpr,
    HIP_EXACT: tl.constexpr,
    HIP_UNIFIED_ARENA: tl.constexpr,
    ARENA_LEAF_OFFSET: tl.constexpr,
    IMPLICIT_LOD: tl.constexpr = False,
    PERSISTENT_SLOT_LEAVES: tl.constexpr = False,
    FIXED_CAPACITY: tl.constexpr = 1,
    FIXED_LEAF_BEGIN: tl.constexpr = 0,
):
    """Expand a centroid union into one shared leaf/local index list."""
    sequence = tl.program_id(0).to(tl.int64)
    work = tl.program_id(1).to(tl.int64)
    batch = sequence // KV_HEADS
    kv_head = sequence - batch * KV_HEADS
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    kv_row = cache_batch * KV_HEADS + kv_head
    token_offset = tl.arange(0, BLOCK_K)

    if work == UNION_CAPACITY:
        if not HIP_UNIFIED_ARENA:
            active_local = tl.minimum(
                tl.load(local_lens + cache_batch).to(tl.int32), LOCAL_LIMIT
            )
            local_and_new = active_local + INCLUDE_NEW
            destination = tl.atomic_add(
                union_token_counts + sequence, local_and_new, sem="relaxed"
            ).to(tl.int32)
            for begin in tl.range(0, LOCAL_LIMIT, BLOCK_K, num_stages=1):
                token = begin + token_offset
                valid = token < active_local
                tl.store(
                    token_indices + sequence * INDEX_CAPACITY + destination + token,
                    -1 - token,
                    mask=valid,
                )
            if INCLUDE_NEW:
                tl.store(
                    token_indices
                    + sequence * INDEX_CAPACITY
                    + destination
                    + active_local,
                    -2147483648,
                )
    else:
        selected_count = tl.load(union_counts + sequence).to(tl.int32)
        valid_rank = work < selected_count
        slot = tl.load(
            union_slots + sequence * UNION_CAPACITY + work,
            mask=valid_rank,
            other=0,
        ).to(tl.int32)
        leaf_count = tl.load(
            slot_lengths + kv_row * STATE_CAPACITY + slot,
            mask=valid_rank,
            other=0,
        ).to(tl.int32)
        if HIP_EXACT:
            if HIP_UNIFIED_ARENA:
                # Preserve the exact-leaf prefix length in a buffer that the
                # following fixed-suffix kernel never modifies. This avoids a
                # snapshot/copy kernel and lets every suffix program derive
                # its output offset without communicating with its peers.
                destination = tl.atomic_add(
                    union_token_counts + sequence,
                    leaf_count,
                    mask=valid_rank,
                    sem="relaxed",
                ).to(tl.int32)
                if IMPLICIT_LOD:
                    tl.atomic_add(
                        hip_context_lens + sequence,
                        leaf_count,
                        mask=valid_rank,
                        sem="relaxed",
                    )
            else:
                destination = tl.atomic_add(
                    hip_context_lens + sequence,
                    leaf_count,
                    mask=valid_rank,
                    sem="relaxed",
                ).to(tl.int32)
        else:
            destination = tl.atomic_add(
                union_token_counts + sequence,
                leaf_count,
                mask=valid_rank,
                sem="relaxed",
            ).to(tl.int32)

        for begin in tl.range(0, leaf_count, BLOCK_K, num_stages=1):
            logical_token = begin + token_offset
            valid = valid_rank & (logical_token < leaf_count)
            if PERSISTENT_SLOT_LEAVES:
                slot_begin = tl.load(
                    fixed_slot_offsets + kv_row * (STATE_CAPACITY + 1) + slot,
                    mask=valid_rank,
                    other=0,
                ).to(tl.int32)
                physical_leaf = tl.load(
                    fixed_indices
                    + kv_row * FIXED_CAPACITY
                    + FIXED_LEAF_BEGIN
                    + slot_begin
                    + logical_token,
                    mask=valid,
                    other=0,
                ).to(tl.int32)
                leaf_valid = valid
            else:
                page_ordinal = logical_token // PAGE_SIZE
                within_page = logical_token % PAGE_SIZE
                page_id = _lookup_page_id(
                    slot_pages,
                    overflow_page_keys,
                    overflow_page_values,
                    overflow_used,
                    kv_row,
                    slot,
                    page_ordinal,
                    valid,
                    STATE_CAPACITY,
                    INLINE_PAGES_PER_SLOT,
                    PAGE_CAPACITY,
                    HASH_CAPACITY,
                    HASH_PROBES,
                ).to(tl.int64)
                page_valid = valid & (page_id >= 0) & (page_id < PAGE_CAPACITY)
                safe_page = tl.where(page_valid, page_id, 0)
                physical_token = (
                    kv_row * PAGE_CAPACITY + safe_page
                ) * PAGE_SIZE + within_page
                leaf_index = tl.load(
                    page_indices + physical_token, mask=page_valid, other=0
                ).to(tl.int32)
                leaf_valid = (
                    page_valid & (leaf_index >= 0) & (leaf_index < LEAF_CAPACITY)
                )
                physical_leaf = ARENA_LEAF_OFFSET + kv_row * LEAF_CAPACITY + leaf_index
            output_offset = sequence * INDEX_CAPACITY + destination + logical_token
            if HIP_EXACT:
                if not HIP_UNIFIED_ARENA:
                    physical_leaf = kv_row * LEAF_CAPACITY + leaf_index
                tl.store(
                    hip_block_table + output_offset,
                    tl.where(leaf_valid, physical_leaf, 0),
                    mask=valid,
                )
            else:
                tl.store(
                    token_indices + output_offset,
                    tl.where(leaf_valid, leaf_index, -2147483647),
                    mask=valid,
                )


@triton.jit
def _append_decode_gqa_union_arena_entries_kernel(
    cache_indices,
    local_lens,
    state_lens,
    counts,
    seen_stamps,
    sequence_epochs,
    new_k,
    new_v,
    arena_k,
    arena_v,
    arena_bias,
    block_table,
    context_lens,
    exact_context_lens,
    COUNT_BATCH_STRIDE,
    COUNT_HEAD_STRIDE,
    COUNT_TOKEN_STRIDE,
    NEW_K_BATCH_STRIDE,
    NEW_K_HEAD_STRIDE,
    NEW_V_BATCH_STRIDE,
    NEW_V_HEAD_STRIDE,
    KV_HEADS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    INDEX_CAPACITY: tl.constexpr,
    LOCAL_OFFSET: tl.constexpr,
    SINK_OFFSET: tl.constexpr,
    COARSE_OFFSET: tl.constexpr,
    LOCAL_CAPACITY: tl.constexpr,
    SINK_CAPACITY: tl.constexpr,
    LOCAL_LIMIT: tl.constexpr,
    SINK_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INCLUDE_NEW: tl.constexpr,
    MASK_OPENED: tl.constexpr = True,
    USE_EXACT_PREFIX: tl.constexpr = True,
    USE_STATE_LENS: tl.constexpr = False,
    INCLUDE_COARSE: tl.constexpr = True,
):
    """Append local/sink plus the live, bias-masked centroid prefix."""
    sequence = tl.program_id(0).to(tl.int64)
    work = tl.program_id(1).to(tl.int64)
    batch = sequence // KV_HEADS
    kv_head = sequence - batch * KV_HEADS
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    kv_row = cache_batch * KV_HEADS + kv_head
    if USE_STATE_LENS:
        active_state_len = tl.minimum(
            tl.load(state_lens + cache_batch).to(tl.int32), STATE_LEN
        )
    else:
        active_state_len = STATE_LEN
    local_base = LOCAL_OFFSET + kv_row * LOCAL_CAPACITY
    sink_base = SINK_OFFSET + kv_row * SINK_CAPACITY
    coarse_base = COARSE_OFFSET + kv_row * STATE_CAPACITY
    if USE_EXACT_PREFIX:
        exact_count = tl.load(exact_context_lens + sequence).to(tl.int32)
    else:
        exact_count = 0
    active_local = tl.minimum(
        tl.load(local_lens + cache_batch).to(tl.int32), LOCAL_LIMIT
    )
    local_and_new = active_local + INCLUDE_NEW
    coarse_destination = exact_count + local_and_new + SINK_LEN
    token = tl.arange(0, BLOCK_K)

    if work == 0:
        destination = exact_count
        tl.store(
            context_lens + sequence,
            coarse_destination + (active_state_len if INCLUDE_COARSE else 0),
        )
        for begin in tl.range(0, LOCAL_LIMIT, BLOCK_K, num_stages=1):
            local_token = begin + token
            valid = local_token < active_local
            tl.store(
                block_table + sequence * INDEX_CAPACITY + destination + local_token,
                local_base + local_token,
                mask=valid,
            )
        if INCLUDE_NEW:
            dimension = tl.arange(0, HEAD_DIM)
            current_key = tl.load(
                new_k
                + batch * NEW_K_BATCH_STRIDE
                + kv_head * NEW_K_HEAD_STRIDE
                + dimension
            )
            current_value = tl.load(
                new_v
                + batch * NEW_V_BATCH_STRIDE
                + kv_head * NEW_V_HEAD_STRIDE
                + dimension
            )
            local_storage = (local_base + active_local) * HEAD_DIM + dimension
            tl.store(arena_k + local_storage, current_key)
            tl.store(arena_v + local_storage, current_value)
            tl.store(
                block_table + sequence * INDEX_CAPACITY + destination + active_local,
                local_base + active_local,
            )
        for begin in tl.range(0, SINK_LEN, BLOCK_K, num_stages=1):
            sink_token = begin + token
            valid = sink_token < SINK_LEN
            tl.store(
                block_table
                + sequence * INDEX_CAPACITY
                + destination
                + local_and_new
                + sink_token,
                sink_base + sink_token,
                mask=valid,
            )
    elif INCLUDE_COARSE:
        if (work - 1) * BLOCK_K >= active_state_len:
            return
        slot = (work - 1) * BLOCK_K + token
        in_state = slot < active_state_len
        count = tl.load(
            counts
            + cache_batch * COUNT_BATCH_STRIDE
            + kv_head * COUNT_HEAD_STRIDE
            + slot * COUNT_TOKEN_STRIDE,
            mask=in_state,
            other=0.0,
        ).to(tl.float32)
        epoch = tl.load(sequence_epochs + sequence).to(tl.int32)
        if MASK_OPENED:
            opened = (
                tl.load(
                    seen_stamps + sequence * STATE_CAPACITY + slot,
                    mask=in_state,
                    other=epoch,
                ).to(tl.int32)
                == epoch
            )
        else:
            opened = tl.zeros((BLOCK_K,), tl.int1)
        active = in_state & (count > 0.0) & ~opened
        tl.store(
            block_table + sequence * INDEX_CAPACITY + coarse_destination + slot,
            coarse_base + slot,
            mask=in_state,
        )
        tl.store(
            arena_bias + coarse_base + slot,
            tl.where(active, tl.log(count), -float("inf")),
            mask=in_state,
        )


def new_fused_decode_buffers(
    q: torch.Tensor,
    *,
    splits: int,
    exact_kv_heads: int | None = None,
    exact_segments: int = 64,
    state_capacity: int | None = None,
    route_group_size: int = 64,
    route_segment_tiles: int = 1,
    gqa_route_splits: int | None = None,
    materialized_state_route: bool = False,
    gqa_union_kv_heads: int | None = None,
    gqa_union_index_capacity: int | None = None,
    gqa_union_hip: bool = False,
    gqa_union_fixed_mask: bool = False,
    gqa_union_fixed_mask_tile_size: int = 64,
    gqa_union_fixed_mask_segments: int = 128,
) -> dict[str, torch.Tensor]:
    batch, query_heads, _, value_dim = q.shape
    if gqa_route_splits is not None and gqa_route_splits not in {4, 8, 16, 32}:
        raise ValueError("GQA cooperative route splits must be 4, 8, 16, or 32")
    buffers = {
        "cache_indices": torch.arange(batch, dtype=torch.long, device=q.device),
        "local_lens": torch.empty(batch, dtype=torch.int32, device=q.device),
        "partial_out": torch.empty(
            batch,
            query_heads,
            splits,
            value_dim,
            dtype=torch.float32,
            device=q.device,
        ),
        "partial_lse": torch.empty(
            batch,
            query_heads,
            splits,
            dtype=torch.float32,
            device=q.device,
        ),
        "output": torch.empty_like(q),
    }
    if exact_kv_heads is not None:
        if exact_kv_heads <= 0 or query_heads % exact_kv_heads:
            raise ValueError("exact decode has inconsistent query/KV heads")
        if exact_segments not in (64, 128, 256, 512):
            raise ValueError("exact decode segments must be 64, 128, 256, or 512")
        exact_group_size = query_heads // exact_kv_heads
        exact_sequences = batch * exact_kv_heads
        buffers.update(
            exact_context_lens=torch.empty(
                exact_sequences, dtype=torch.int32, device=q.device
            ),
            exact_segment_out=torch.empty(
                exact_sequences,
                exact_group_size,
                exact_segments,
                value_dim,
                dtype=torch.float32,
                device=q.device,
            ),
            exact_segment_max=torch.empty(
                exact_sequences,
                exact_group_size,
                exact_segments,
                dtype=torch.float32,
                device=q.device,
            ),
            exact_segment_exp_sum=torch.empty(
                exact_sequences,
                exact_group_size,
                exact_segments,
                dtype=torch.float32,
                device=q.device,
            ),
        )
    if gqa_route_splits is not None:
        buffers.update(
            gqa_local_partial_out=torch.empty(
                batch,
                query_heads,
                32,
                value_dim,
                dtype=torch.float32,
                device=q.device,
            ),
            gqa_local_partial_lse=torch.empty(
                batch,
                query_heads,
                32,
                dtype=torch.float32,
                device=q.device,
            ),
            gqa_route_partial_out=torch.empty(
                batch,
                query_heads,
                8,
                gqa_route_splits,
                value_dim,
                dtype=torch.float32,
                device=q.device,
            ),
            gqa_route_partial_lse=torch.empty(
                batch,
                query_heads,
                8,
                gqa_route_splits,
                dtype=torch.float32,
                device=q.device,
            ),
            route_local_out=torch.empty(
                batch,
                query_heads,
                value_dim,
                dtype=torch.float32,
                device=q.device,
            ),
            route_local_lse=torch.empty(
                batch,
                query_heads,
                dtype=torch.float32,
                device=q.device,
            ),
        )
    if state_capacity is not None:
        if route_segment_tiles not in {1, 2, 3, 4}:
            raise ValueError("route segment tiles must be 1, 2, 3, or 4")
        max_groups = triton.cdiv(state_capacity, route_group_size * route_segment_tiles)
        buffers.update(
            route_candidate_scores=torch.empty(
                batch,
                query_heads,
                max_groups,
                8,
                dtype=torch.float32,
                device=q.device,
            ),
            route_candidate_indices=torch.empty(
                batch,
                query_heads,
                max_groups,
                8,
                dtype=torch.long,
                device=q.device,
            ),
            route_group_out=torch.empty(
                batch,
                query_heads,
                max_groups,
                value_dim,
                dtype=torch.float32,
                device=q.device,
            ),
            route_group_lse=torch.empty(
                batch,
                query_heads,
                max_groups,
                dtype=torch.float32,
                device=q.device,
            ),
            route_top_slots=torch.empty(
                batch,
                query_heads,
                1,
                8,
                dtype=torch.long,
                device=q.device,
            ),
            route_top_scores=torch.empty(
                batch,
                query_heads,
                1,
                8,
                dtype=torch.float32,
                device=q.device,
            ),
            coarse_out=torch.empty(
                batch,
                query_heads,
                value_dim,
                dtype=torch.float32,
                device=q.device,
            ),
            coarse_lse=torch.empty(
                batch,
                query_heads,
                dtype=torch.float32,
                device=q.device,
            ),
            route_local_out=torch.empty(
                batch,
                query_heads,
                value_dim,
                dtype=torch.float32,
                device=q.device,
            ),
            route_local_lse=torch.empty(
                batch,
                query_heads,
                dtype=torch.float32,
                device=q.device,
            ),
            completion=torch.zeros(
                batch,
                query_heads,
                dtype=torch.int32,
                device=q.device,
            ),
        )
        if materialized_state_route:
            buffers["route_state_scores"] = torch.empty(
                batch,
                query_heads,
                1,
                state_capacity,
                dtype=torch.float32,
                device=q.device,
            )
        if gqa_union_kv_heads is not None:
            if gqa_union_kv_heads <= 0 or query_heads % gqa_union_kv_heads:
                raise ValueError("GQA-union decode has inconsistent query/KV heads")
            gqa_union_group_size = query_heads // gqa_union_kv_heads
            if not 1 < gqa_union_group_size <= 16:
                raise ValueError("GQA-union decode requires a GQA group in [2, 16]")
            if gqa_union_index_capacity is None or gqa_union_index_capacity <= 0:
                raise ValueError("GQA-union decode requires an index capacity")
            sequences = batch * gqa_union_kv_heads
            union_slot_capacity = gqa_union_group_size * 8
            buffers.update(
                gqa_union_seen_stamps=torch.zeros(
                    sequences,
                    state_capacity,
                    dtype=torch.int32,
                    device=q.device,
                ),
                gqa_union_epochs=torch.ones(
                    sequences, dtype=torch.int32, device=q.device
                ),
                gqa_union_counts=torch.zeros(
                    sequences, dtype=torch.int32, device=q.device
                ),
                gqa_union_token_counts=torch.zeros(
                    sequences, dtype=torch.int32, device=q.device
                ),
                gqa_union_slots=torch.empty(
                    sequences,
                    union_slot_capacity,
                    dtype=torch.int32,
                    device=q.device,
                ),
                gqa_union_destinations=torch.empty(
                    sequences,
                    union_slot_capacity,
                    dtype=torch.int32,
                    device=q.device,
                ),
                gqa_union_token_indices=torch.empty(
                    sequences,
                    gqa_union_index_capacity,
                    dtype=torch.int32,
                    device=q.device,
                ),
            )
            if gqa_union_hip:
                if gqa_union_fixed_mask_tile_size not in (16, 64, 128):
                    raise ValueError("unsupported fixed-mask attention tile size")
                if gqa_union_fixed_mask_segments not in (
                    8,
                    16,
                    32,
                    64,
                    128,
                    256,
                    512,
                ):
                    raise ValueError("unsupported fixed-mask segment count")
                unified_segments = (
                    gqa_union_fixed_mask_segments if gqa_union_fixed_mask else 32
                )
                if unified_segments not in (8, 16, 32, 64, 128, 256, 512):
                    raise ValueError(
                        "unified segments must be 8, 16, 32, 64, 128, 256, or 512"
                    )
                buffers.update(
                    gqa_union_hip_block_table=torch.empty(
                        sequences,
                        gqa_union_index_capacity,
                        dtype=torch.int32,
                        device=q.device,
                    ),
                    gqa_union_hip_context_lens=torch.empty(
                        sequences, dtype=torch.int32, device=q.device
                    ),
                    gqa_union_hip_launch_lens=torch.empty(
                        sequences, dtype=torch.int32, device=q.device
                    ),
                    gqa_union_hip_cu_q=torch.arange(
                        sequences + 1, dtype=torch.int32, device=q.device
                    ),
                    gqa_union_hip_out=torch.empty(
                        sequences,
                        gqa_union_group_size,
                        value_dim,
                        dtype=torch.float32,
                        device=q.device,
                    ),
                    gqa_union_hip_lse=torch.empty(
                        sequences,
                        gqa_union_group_size,
                        dtype=torch.float32,
                        device=q.device,
                    ),
                    gqa_union_hip_segment_out=torch.empty(
                        sequences,
                        gqa_union_group_size,
                        unified_segments,
                        value_dim,
                        dtype=torch.float32,
                        device=q.device,
                    ),
                    gqa_union_hip_exp_sums=torch.empty(
                        sequences,
                        gqa_union_group_size,
                        unified_segments,
                        dtype=torch.float32,
                        device=q.device,
                    ),
                    gqa_union_hip_max_logits=torch.empty(
                        sequences,
                        gqa_union_group_size,
                        unified_segments,
                        dtype=torch.float32,
                        device=q.device,
                    ),
                )
                if gqa_union_fixed_mask:
                    fixed_mask_tiles = triton.cdiv(
                        gqa_union_index_capacity,
                        gqa_union_fixed_mask_tile_size,
                    )
                    buffers.update(
                        # Prefix entries begin enabled; leaf bytes are ignored
                        # unless their block flag is set, and every newly
                        # selected leaf block is cleared before being filled.
                        # Prefix/coarse lanes are enabled by the per-token
                        # preparation kernel. Leaf lanes begin at zero and are
                        # changed only by sparse route activation/reset.
                        gqa_union_fixed_active_mask=torch.zeros(
                            sequences,
                            gqa_union_index_capacity,
                            dtype=torch.uint8,
                            device=q.device,
                        ),
                        gqa_union_fixed_active_blocks=torch.zeros(
                            sequences,
                            fixed_mask_tiles,
                            dtype=torch.uint8,
                            device=q.device,
                        ),
                        gqa_union_fixed_previous_counts=torch.zeros(
                            sequences,
                            dtype=torch.int32,
                            device=q.device,
                        ),
                        gqa_union_fixed_previous_slots=torch.zeros(
                            sequences,
                            union_slot_capacity,
                            dtype=torch.int32,
                            device=q.device,
                        ),
                        gqa_union_fixed_previous_cache_rows=torch.full(
                            (sequences,),
                            -1,
                            dtype=torch.int32,
                            device=q.device,
                        ),
                        gqa_union_fixed_execution_geometry=torch.zeros(
                            4,
                            dtype=torch.int32,
                            device=q.device,
                        ),
                    )
    return buffers
