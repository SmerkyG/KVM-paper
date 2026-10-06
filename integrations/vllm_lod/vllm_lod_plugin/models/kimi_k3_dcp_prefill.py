"""Experimental owner-sharded centroid construction and DCP prefill for K3.

Each rank archives only its interleaved sequence slice, retaining the original
whole-sequence centroid budget. All query heads visit this local state and
independently open eight regions. Only output/LSE are combined across ranks;
there is no global routing top-k. The current causal chunk stays head-sharded
and exact, disjoint from every rank's remote cache.

This first prototype supports aligned 16K initial-prefill scheduler chunks and
two-tier BF16. It is opt-in, not a production-policy change.

The shared-summary variant instead divides the original centroid budget among
ranks, replicates their summaries after construction, and selects eight total
regions on the query-head owner. Only exact leaf refinement is distributed.
"""

from __future__ import annotations

from typing import Any

import torch

from lod_attention._engines import KernelLODCache
from lod_attention.kernels.aiter_mla_prefill_attention import (
    aiter_kimi_expanded_prefill_route_coarse_attention,
    aiter_kimi_local_prefill_attention,
    expand_kimi_leaf_kv,
    merge_aiter_mla_prefill_refinement,
)
from lod_attention.kernels.aiter_prefill_attention import (
    _specialized_kimi_coarse_mha_fwd,
)
from .kimi_k3_sharded_prefill import (
    archive_workspace,
    combine_prefill_partials,
    gather_prefill,
    projection_scope,
)


def merge_partitions(
    first: torch.Tensor, first_lse: torch.Tensor,
    second: torch.Tensor, second_lse: torch.Tensor,
) -> torch.Tensor:
    """Combine disjoint normalized value fields; LSE uses natural logs."""
    total = torch.logaddexp(first_lse, second_lse)
    return (
        first.float() * torch.exp(first_lse - total)[..., None]
        + second.float() * torch.exp(second_lse - total)[..., None]
    ).to(first.dtype)


def _exact_history(
    query: torch.Tensor, key: torch.Tensor,
    w_uk_t: torch.Tensor, w_uv: torch.Tensor, scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Noncausal attention to the rank's small old exact tail and owned sink."""
    expanded_k, expanded_v = expand_kimi_leaf_kv(key, w_uk_t, w_uv)
    result = _specialized_kimi_coarse_mha_fwd()(
        query.permute(0, 2, 1, 3),
        expanded_k.permute(0, 2, 1, 3),
        expanded_v.permute(0, 2, 1, 3),
        0.0, scale, False, -1, -1, 0, True, False,
        None, None, None, None, None, None, None, None, None,
    )
    return result[0].permute(0, 2, 1, 3), result[1]


def _remote_attention(
    pool: Any, cache: KernelLODCache, query: torch.Tensor,
    w_uk_t: torch.Tensor, w_uv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One rank's complete coarse-replacement result and its LSE."""
    engine, state = pool.engine, cache.state
    page = state["page_cache"]
    tail = state["recent_k"][..., :int(state["recent_len"]), :]
    if pool._dcp_owns_position(0):
        tail = torch.cat((state["sink_k"], tail), dim=2)
    tail_out, tail_lse = _exact_history(
        query, tail, w_uk_t, w_uv, float(engine.scaling)
    )
    slots, coarse, _, _ = aiter_kimi_expanded_prefill_route_coarse_attention(
        query, state["state_k"].contiguous(), state["state_v"].contiguous(),
        state["counts"].contiguous(),
        w_uk_t, w_uv, state_len=int(state["state_len"]),
        scale=float(engine.scaling), normalize_route_query=False,
        slot_lengths=page["slot_lengths"],
        max_open_leaf_tokens=engine.max_open_centroid_leaves,
        buffers=getattr(engine, "_lod_prefill_attention_buffers", None),
    )
    if coarse.ready_stream is not None:
        torch.cuda.current_stream(query.device).wait_stream(coarse.ready_stream)
    # A stride-zero shape carrier: fine attention uses the actual D192 query,
    # and with SINK_LEN=0 the refinement kernel never consumes its D576 data.
    carrier = page["leaf_k"][..., :1, :].expand(
        1, query.size(1), query.size(2), 576
    )
    engine._lod_kimi_expanded_prefill_chunk = query
    engine._lod_kimi_w_uk_t = w_uk_t
    engine._lod_kimi_w_uv = w_uv
    try:
        route_out, route_lse = engine._paged_leaf_attention(
            carrier, slots, page, active_slots=int(state["state_len"]),
            reduce_routes=False,
        )
    finally:
        for name in ("_lod_kimi_expanded_prefill_chunk", "_lod_kimi_w_uk_t",
                     "_lod_kimi_w_uv"):
            delattr(engine, name)
    empty_k = state["state_k"][..., :0, :]
    empty_v = coarse.mean_v[..., :0, :]
    return merge_aiter_mla_prefill_refinement(
        carrier, empty_k, empty_v, coarse, slots, route_out, route_lse,
        tail_out, tail_lse, kv_group_size=query.size(1),
        scale=float(engine.scaling), return_lse=True,
    )


def _advance_cache(pool: Any, slot: int, key: torch.Tensor, *, total: int) -> None:
    """Archive only this rank's new tokens at a GLOBAL 16K boundary."""
    engine = pool.engine
    local_total = pool._dcp_local_length(total)
    global_coverage = pool._dcp_global_decode_coverage(total)
    local_coverage = pool._dcp_local_length(global_coverage)
    with pool._dcp_local_state_schedule():
        if not pool.ready[slot]:
            old_sink_len = engine.sink_len
            # Only the owner of global position zero protects a sink. Others
            # must archive their first owned token, not invent extra sinks.
            engine.sink_len = old_sink_len if pool._dcp_owns_position(0) else 0
            engine._lod_prefill_cache_capacity = pool.persistent_request_capacity
            try:
                converted = engine.build_cache_from_bf16(
                    key, key[..., :512], final_cache_coverage=local_coverage
                )
                pool._install_dcp_converted_row(slot, converted, global_length=total)
            finally:
                engine.sink_len = old_sink_len
                del engine._lod_prefill_cache_capacity
            return

        cache = pool._row_cache(slot)
        state, page = cache.state, cache.state["page_cache"]
        old_tail = state["recent_k"][..., :int(state["recent_len"]), :]
        working = torch.cat((old_tail, key), dim=2)
        advance = local_coverage - int(state["coverage"])
        overflow = working[..., :advance, :]
        # The pool's V is a prefix alias of K. The generic sum updater writes
        # both independently, so do not let its two writes overlap. Its V sum
        # equals the updated K prefix; only K remains persistent afterwards.
        update_v = state["state_v"].contiguous()
        state_k, state_v, counts, state_len, owners, remap = engine._update_state(
            state["state_k"], update_v, state["counts"],
            state.get("key_norm_sums"), overflow, overflow[..., :512],
            state_len=int(state["state_len"]), ctx_len=local_total,
            available_context=local_coverage, state_capacity=pool.state_capacity,
            scheduled_state_len=int(state["scheduled_state_len"]),
        )
        if remap is not None:
            raise AssertionError("independent DCP pages cannot remap centroids")
        if state_k.data_ptr() != state["state_k"].data_ptr():
            raise AssertionError("independent DCP update changed fixed pool storage")
        engine._append_page_cache(page, overflow, overflow[..., :512], owners)
        remaining = working[..., advance:, :]
        tail_len = int(remaining.size(2))
        if tail_len > pool.local_capacity:
            raise AssertionError("independent DCP exact tail exceeded fixed capacity")
        state["recent_k"][..., :tail_len, :].copy_(remaining)
        # K and V share their latent prefix; never write it twice.
        pool.metadata[slot].update(
            state_len=state_len, scheduled_state_len=state_len,
            coverage=local_coverage, total_len=local_total, recent_len=tail_len,
            leaf_count=int(page["leaf_count"]),
            overflow_safe_until=int(page["overflow_safe_until"]),
            dcp_global_total_len=total, dcp_global_coverage=global_coverage,
        )
        pool.state_lens[slot].fill_(state_len)
        pool.local_lens[slot].fill_(tail_len)
        pool.leaf_lens[slot].fill_(int(page["leaf_count"]))
        pool.dcp_global_lens[slot].fill_(total)
        pool._refresh_unified_page1_coarse((slot,))


def local_dcp_prefill(
    layer: Any, pool: Any, query: torch.Tensor, record: torch.Tensor,
) -> torch.Tensor:
    """Projected [tokens, local heads, 128] output for aligned initial prefill."""
    from vllm.v1.attention.ops.dcp import cp_lse_ag_out_rs

    if pool.dcp_interleave_size != 1 or record.size(-1) != 576:
        raise NotImplementedError("local DCP prototype needs full K3 and unit interleave")
    plan = pool.direct_prefill_plan
    chunk = int(pool._dcp_global_lengths["prefill_chunk_len"])
    if any(end - begin != chunk or previous % chunk for _, begin, end, previous in plan):
        raise NotImplementedError("local DCP prototype needs aligned 16K scheduler chunks")
    # Materialize only once; these fixed linear maps do not change between
    # requests. Gathered Q is temporary and is included in the measured path.
    if getattr(layer, "_lod_dcp_w_uv", None) is None:
        layer._lod_dcp_w_uv = pool.dcp_group.all_gather(layer.W_UV.contiguous(), dim=0)
    w_uk_t = layer.W_UK_T_dcp_qrep
    if w_uk_t is None:
        # Native vLLM materializes this lazily for decode. This prefill path
        # needs it earlier, but still gathers the immutable weight only once.
        w_uk_t = pool.dcp_group.all_gather(layer.W_UK_T.contiguous(), dim=0)
        layer.W_UK_T_dcp_qrep = w_uk_t
    result = query.new_empty(query.size(0), query.size(1), 128)
    for slot, begin, end, previous in plan:
        pool.wait_deferred_prefill((slot,))
        row_q = query[begin:end].permute(1, 0, 2).unsqueeze(0)
        row_k = record[begin:end].permute(1, 0, 2).unsqueeze(0)
        local_out, local_lse = aiter_kimi_local_prefill_attention(
            row_k.expand(1, query.size(1), chunk, 576), row_k,
            query_offset=0, scale=float(pool.engine.scaling),
            expanded_q=row_q, w_uk_t=layer.W_UK_T, w_uv=layer.W_UV,
        )
        if previous:
            if not pool.ready[slot] or int(pool.metadata[slot]["dcp_global_total_len"]) != previous:
                raise AssertionError("independent DCP history is missing or out of order")
            gathered = pool.dcp_group.all_gather(query[begin:end].contiguous(), dim=1)
            all_q = gathered.permute(1, 0, 2).unsqueeze(0)
            remote, remote_lse = _remote_attention(
                pool, pool._row_cache(slot), all_q, w_uk_t, layer._lod_dcp_w_uv
            )
            remote, remote_lse = cp_lse_ag_out_rs(
                remote[0].permute(1, 0, 2).contiguous(),
                remote_lse[0].transpose(0, 1).contiguous(), pool.dcp_group,
                return_lse=True, is_lse_base_on_e=True,
            )
            merged = merge_partitions(
                local_out, local_lse, remote.transpose(0, 1).unsqueeze(0),
                remote_lse.transpose(0, 1).unsqueeze(0),
            )
        else:
            merged = local_out
        result[begin:end].copy_(merged[0].permute(1, 0, 2))
        owned_offset = (pool.dcp_rank - previous) % pool.dcp_world_size
        owned = row_k[..., owned_offset::pool.dcp_world_size, :].contiguous()
        # Synchronize construction in this first prototype. This includes all
        # final cache work in first-token latency and avoids hiding any cost.
        _advance_cache(pool, slot, owned, total=previous + chunk)
    return result


def owned_routes(slots: torch.Tensor, *, rank: int, local_states: int) -> torch.Tensor:
    """Convert replicated rank-major centroid IDs into this rank's leaf IDs."""
    first = rank * local_states
    owned = slots.ge(first) & slots.lt(first + local_states)
    return torch.where(owned, slots - first, -1).to(torch.int32)


def _shared_mixed_decode(layer: Any, pool: Any, query: torch.Tensor,
                         record: torch.Tensor, result: torch.Tensor, plan: tuple) -> None:
    """Use ordinary global DCP decode for one-token rows beside a prefill."""
    from .kimi_k3 import absorb_query
    from lod_attention.kernels.aiter_mla_prefill_attention import project_kimi_head_values
    from vllm.v1.attention.ops.dcp import cp_lse_ag_out_rs

    if tuple(item[0] for item in plan) != pool.active_decode_rows[:len(plan)]:
        raise AssertionError("mixed DCP decode disagrees with its active-row map")
    positions = torch.tensor([item[1] for item in plan], device=query.device, dtype=torch.long)
    raw_q = gather_prefill(pool.dcp_group, query.index_select(0, positions), dim=1)
    q = absorb_query(raw_q, layer.W_UK_T_dcp_qrep, nope_dim=int(layer.qk_nope_head_dim))
    key = record.index_select(0, positions)
    partial = query.new_empty(len(plan), q.size(1), 512)
    partial, lse = pool.decode_dcp(q, key, key[..., :512], partial)
    attention = cp_lse_ag_out_rs(partial, lse, pool.dcp_group, is_lse_base_on_e=True)
    projected = project_kimi_head_values(attention.unsqueeze(2), layer.W_UV).squeeze(2)
    result.index_copy_(0, positions, projected)
    for slot, _, _, previous in plan:
        total = previous + 1
        local = pool._dcp_local_length(total)
        pool.metadata[slot].update(
            total_len=local, recent_len=local - int(pool.metadata[slot]["coverage"]),
            dcp_global_total_len=total,
        )
        pool.dcp_global_lens[slot].fill_(total)


def _refresh_shared_summaries(pool: Any, slot: int, record: torch.Tensor, total: int) -> None:
    """One small centroid exchange per GLOBAL 16K construction boundary."""
    state = pool._row_cache(slot).state
    local_states = int(state["state_len"])
    with pool._dcp_local_state_schedule():
        expected = pool.engine._desired_state_len(
            pool._dcp_local_length(total), int(state["coverage"]), 0
        )
    # Aligned equal-sized rank slices have the same scheduled count. Prove
    # that here before calling an equal-shape collective, rather than padding
    # inactive centroid slots and accidentally assigning them softmax mass.
    if local_states != expected:
        raise AssertionError("shared DCP centroid count differs from its aligned schedule")
    lengths = state["page_cache"]["slot_lengths"][..., :local_states]
    packed = torch.cat((
        state["state_k"][..., :local_states, :],
        state["counts"][..., :local_states, :],
        lengths[..., None].float(),
    ), dim=-1).contiguous()
    buffers = getattr(pool.engine, "_lod_prefill_attention_buffers", None)
    gathered = gather_prefill(
        pool.dcp_group, packed, dim=2,
        buffer=archive_workspace(
            buffers, "shared_centroid_exchange", (pool.dcp_world_size, *packed.shape),
            token_axis=3, dtype=packed.dtype, device=packed.device,
        ),
    )
    keys = gathered[..., :576].contiguous()
    old = pool.dcp_prefill_summaries.get(slot)
    coverage = pool._dcp_global_decode_coverage(total)
    tail_len = total - coverage
    pool.dcp_prefill_summaries[slot] = {
        "state_k": keys,
        # The CK preparation kernel requires packed V, not a strided prefix
        # view into D576 K. This is a small replicated centroid workspace.
        "state_v": keys[..., :512].contiguous(),
        "counts": gathered[..., 576:577].contiguous(),
        "slot_lengths": gathered[..., 577].to(torch.int32).contiguous(),
        "local_states": local_states,
        "total": total,
        "tail_k": record[..., -tail_len:, :].clone() if tail_len else record[..., :0, :],
        "sink_k": old["sink_k"] if old is not None else record[..., :1, :].clone(),
    }


def _shared_remote_attention(
    layer: Any, pool: Any, slot: int, query: torch.Tensor,
    local_out: torch.Tensor, local_lse: torch.Tensor,
) -> torch.Tensor:
    """Head-sharded coarse/routing; owner-sharded exact leaf refinement."""
    engine = pool.engine
    buffers = getattr(engine, "_lod_prefill_attention_buffers", None)
    shared = pool.dcp_prefill_summaries[slot]
    query_heads, query_len = int(query.size(1)), int(query.size(2))
    local_states = int(shared["local_states"])
    total_states = local_states * pool.dcp_world_size
    slots, coarse, _, _ = aiter_kimi_expanded_prefill_route_coarse_attention(
        query.contiguous(), shared["state_k"], shared["state_v"], shared["counts"],
        layer.W_UK_T, layer.W_UV, state_len=total_states,
        scale=float(engine.scaling), normalize_route_query=False,
        slot_lengths=shared["slot_lengths"],
        max_open_leaf_tokens=engine.max_open_centroid_leaves,
        buffers=buffers,
    )
    if coarse.ready_stream is not None:
        torch.cuda.current_stream(query.device).wait_stream(coarse.ready_stream)
    selected_lengths = torch.gather(
        shared["slot_lengths"].unsqueeze(2).expand(1, query_heads, query_len, total_states),
        -1, slots.clamp_min(0).long(),
    )
    # This does not rerank or change an approximation: singleton summaries
    # already equal their exact leaves. Keep their existing coarse term.
    slots = torch.where(slots.ge(0) & selected_lengths.gt(1), slots, -1)
    slot_buffer = archive_workspace(
        buffers, "sharded_prefill_slots", (pool.dcp_world_size, *slots.shape),
        token_axis=3, dtype=slots.dtype, device=slots.device,
    )
    all_slots = gather_prefill(pool.dcp_group, slots, dim=1, buffer=slot_buffer)
    local_slots = owned_routes(
        all_slots, rank=pool.dcp_rank, local_states=local_states
    )
    all_q = gather_prefill(
        pool.dcp_group, query, dim=1,
        buffer=archive_workspace(
            buffers, "sharded_prefill_query", (pool.dcp_world_size, *query.shape),
            token_axis=3, dtype=query.dtype, device=query.device,
        ),
    )
    cache = pool._row_cache(slot).state
    carrier = cache["page_cache"]["leaf_k"][..., :1, :].expand(
        1, all_q.size(1), query_len, 576
    )
    with projection_scope(engine, all_q, layer.W_UK_T_dcp_qrep, layer._lod_dcp_w_uv,
                          head_group_limit=query_heads):
        fine, fine_lse = engine._paged_leaf_attention(
            carrier, local_slots, cache["page_cache"],
            active_slots=local_states, reduce_routes=True,
        )
    fine, fine_lse = combine_prefill_partials(
        pool, fine, fine_lse, heads=query_heads, buffers=buffers,
    )
    # The local field was enqueued before routing on its own stream. Its
    # scratch stays live until the final merge, before synchronous update.
    torch.cuda.current_stream(query.device).wait_stream(engine._lod_prefill_local_stream)
    own_carrier = carrier[:, :query_heads]
    return merge_aiter_mla_prefill_refinement(
        own_carrier, carrier[:, :1, :0], coarse.mean_v[..., :0, :],
        coarse, slots, fine, fine_lse, local_out, local_lse,
        kv_group_size=query_heads, scale=float(engine.scaling),
    )


def shared_dcp_prefill(layer: Any, pool: Any, query: torch.Tensor, record: torch.Tensor) -> torch.Tensor:
    """Replicated centroid summaries, original total budget, eight global routes."""
    if pool.dcp_interleave_size != 1 or record.size(-1) != 576:
        raise NotImplementedError("shared DCP prototype needs full K3 and unit interleave")
    chunk = int(pool._dcp_global_lengths["prefill_chunk_len"])
    mixed = tuple(item for item in pool.direct_prefill_plan if item[2] - item[1] == 1 and item[3] > 0)
    plan = tuple(item for item in pool.direct_prefill_plan if item not in mixed)
    if any(end - begin != chunk or previous % chunk for _, begin, end, previous in plan):
        raise NotImplementedError("shared DCP prototype needs aligned 16K scheduler chunks")
    if getattr(layer, "_lod_dcp_w_uv", None) is None:
        layer._lod_dcp_w_uv = gather_prefill(pool.dcp_group, layer.W_UV, dim=0)
    if layer.W_UK_T_dcp_qrep is None:
        layer.W_UK_T_dcp_qrep = gather_prefill(pool.dcp_group, layer.W_UK_T, dim=0)
    engine = pool.engine
    buffers = getattr(engine, "_lod_prefill_attention_buffers", None)
    foreground = torch.cuda.current_stream(query.device)
    local_stream = getattr(engine, "_lod_prefill_local_stream", None)
    if local_stream is None:
        local_stream = torch.cuda.Stream(device=query.device)
        engine._lod_prefill_local_stream = local_stream
    result = query.new_empty(query.size(0), query.size(1), 128)
    if mixed:
        _shared_mixed_decode(layer, pool, query, record, result, mixed)
    for slot, begin, end, previous in plan:
        pool.wait_deferred_prefill((slot,))
        row_q = query[begin:end].permute(1, 0, 2).unsqueeze(0)
        row_k = record[begin:end].permute(1, 0, 2).unsqueeze(0)
        shared = pool.dcp_prefill_summaries.get(slot)
        if previous:
            if shared is None or int(shared["total"]) != previous:
                raise AssertionError("shared DCP history is missing or out of order")
            exact_k = torch.cat((shared["sink_k"], shared["tail_k"], row_k), dim=2)
        else:
            exact_k = row_k
        local_stream.wait_stream(foreground)
        with torch.cuda.stream(local_stream):
            local_out, local_lse = aiter_kimi_local_prefill_attention(
                row_k.expand(1, query.size(1), chunk, 576), exact_k,
                query_offset=exact_k.size(2) - chunk, scale=float(engine.scaling),
                expanded_q=row_q, w_uk_t=layer.W_UK_T, w_uv=layer.W_UV,
                buffers=buffers,
            )
        merged = (_shared_remote_attention(layer, pool, slot, row_q, local_out, local_lse)
                  if previous else local_out)
        foreground.wait_stream(local_stream)
        result[begin:end].copy_(merged[0].permute(1, 0, 2))
        owned_offset = (pool.dcp_rank - previous) % pool.dcp_world_size
        owned = row_k[..., owned_offset::pool.dcp_world_size, :].contiguous()
        _advance_cache(pool, slot, owned, total=previous + chunk)
        _refresh_shared_summaries(pool, slot, row_k, previous + chunk)
    return result
