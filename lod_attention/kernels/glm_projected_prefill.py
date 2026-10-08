"""NoPE MLA: project centroid means, then fuse routing with coarse PV.

Persistent K=V records stay latent512. Coarse/local attention uses per-head
K256/V256. Default refinement projects its latent output after reduction;
the experimental alternative projects only the selected leaf union before
attention. Expanded leaves are never kept in the persistent cache.
"""

from __future__ import annotations

import os
import sys

import torch
import triton
import triton.language as tl

from .._mla_projection import combined_kv_weight
from .aiter_prefill_attention import (
    AiterPrefillCoarse, _workspace_tensor, _specialized_route_mha_fwd,
    _reduce_route_candidates,
)
from .aiter_mla_prefill_attention import (
    project_kimi_head_values, project_kimi_shared_values,
    merge_aiter_mla_prefill_refinement,
)


@triton.jit(do_not_specialize=["state_len", "padded_len"])
def _prepare_shared_centroids(state, counts, means, log_counts, active_counts,
                              state_len, padded_len,
                              SB: tl.constexpr, ST: tl.constexpr,
                              CB: tl.constexpr, CT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    batch = row // padded_len
    slot = row % padded_len
    channel = tl.arange(0, 512)
    count = tl.load(counts + batch * CB + slot * CT,
                    mask=slot < state_len, other=0).to(tl.float32)
    active = (slot < state_len) & (count > 0)
    value = tl.load(state + batch * SB + slot * ST + channel,
                    mask=active, other=0).to(tl.float32)
    tl.store(means + row * 512 + channel, value / tl.maximum(count, 1.0))
    tl.store(log_counts + row, tl.where(active, tl.log(tl.maximum(count, 1.0)), -float("inf")))
    tl.store(active_counts + row, tl.where(active, count, 0))


def projected_route_coarse(q, state, counts, w_uk_t, w_uv, *, state_len,
                           scale, slot_lengths, max_open_leaf_tokens, buffers=None):
    """Exact global top-eight from CK score-tile candidates; cap after ranking."""
    batch, heads, tokens, dim = q.shape
    if dim != 256 or tuple(w_uk_t.shape) != (heads, 256, 512) or tuple(w_uv.shape) != (heads, 512, 256):
        raise ValueError("GLM projected prefill requires latent512/K256/V256")
    if state.ndim != 4 or tuple(state.shape[:2]) != (batch, 1) or state.size(-1) != 512:
        raise ValueError("GLM centroid cache must have one latent512 head")
    if state_len < 8 or state_len > state.size(2) or tuple(counts.shape) != (*state.shape[:3], 1):
        raise ValueError("GLM projected centroid/count geometry differs")
    # Native 128-key padding, never power-of-two padding; zero padded means
    # with -inf bias cannot be attended to or selected as a route.
    padded = triton.cdiv(state_len, 128) * 128
    means = _workspace_tensor(buffers, "glm_coarse_latents", (batch, 1, padded, 512), dtype=q.dtype, device=q.device)
    bias = _workspace_tensor(buffers, "glm_coarse_bias", (batch, padded), dtype=q.dtype, device=q.device)
    active_counts = _workspace_tensor(buffers, "glm_coarse_counts", (batch, 1, padded, 1), dtype=counts.dtype, device=q.device)
    _prepare_shared_centroids[(batch * padded,)](
        state, counts, means, bias, active_counts, state_len, padded,
        SB=state.stride(0), ST=state.stride(2), CB=counts.stride(0), CT=counts.stride(2), num_warps=4)
    packed = _workspace_tensor(buffers, "glm_coarse_projected_kv", (batch, padded, heads, 512), dtype=q.dtype, device=q.device)
    weight = combined_kv_weight(w_uk_t, w_uv, buffers)
    torch.mm(means.view(batch * padded, 512), weight,
             out=packed.view(batch * padded, heads * 512))
    # CK accepts explicit strides, including the interleaved [K,V] layout.
    # No transpose copies, separate K/V GEMMs, or duplicate latent means.
    keys, values = packed[..., :256], packed[..., 256:]
    destination = _workspace_tensor(buffers, "glm_coarse_output", (batch, tokens, heads, 256), dtype=q.dtype, device=q.device)
    original_flags = sys.getdlopenflags()
    try:
        sys.setdlopenflags(original_flags | getattr(os, "RTLD_DEEPBIND", 0))
        attention = _specialized_route_mha_fwd(False)
        output, lse, candidates, _ = attention(
            q.permute(0, 2, 1, 3), keys, values, 0.0, float(scale), False,
            -1, -1, 0, True, True, None, None, destination,
            bias[:, None, None].expand(batch, heads, 1, padded),
            None, None, None, None, None)
    finally:
        sys.setdlopenflags(original_flags)
    selected, _, _, selected_scores = _reduce_route_candidates(
        candidates, slot_lengths=slot_lengths, max_open_leaf_tokens=max_open_leaf_tokens,
        close_selected_above_limit=True, state_len=state_len, head_dim=256,
        emit_metadata=False, buffers=buffers)
    # Only the projected V means participate in replacement. mean_k records
    # the logical shared latent head, retaining the sink/refined GQA geometry.
    coarse = AiterPrefillCoarse(output_0=output, lse_0=lse, output_1=output, lse_1=lse,
        mean_k=means, mean_v=values.permute(0, 2, 1, 3), counts=active_counts,
        has_second_partition=False, selected_route_scores=selected_scores)
    return selected, coarse


def projected_two_level_attention(engine, q, local_k, local_v, state_k, counts,
                                  *, state_len, page_cache, local_branch,
                                  sink_k, sink_v, output_buffer):
    """Use latent or projected exact leaves with the same LSE replacement."""
    if page_cache is None or sink_k is None or sink_v is None:
        raise ValueError("GLM projected prefill requires leaf and separate sink caches")
    if (engine.routing_normalization != "none"
            or getattr(engine, "mla_state_key_normalization", "none") != "none"
            or engine.two_level_topk != 8 or engine.prefill_two_level_topk != 8):
        raise ValueError("projected GLM requires the raw-score top-eight release policy")
    buffers = getattr(engine, "_lod_prefill_attention_buffers", None)
    w_uv = engine._lod_kimi_w_uv
    slots, coarse = projected_route_coarse(
        engine._lod_kimi_expanded_prefill_chunk, state_k, counts,
        engine._lod_kimi_w_uk_t, w_uv, state_len=state_len, scale=engine.scaling,
        slot_lengths=page_cache["slot_lengths"], max_open_leaf_tokens=engine.max_open_centroid_leaves,
        buffers=buffers)
    project_leaves = os.environ.get("LOD_GLM_PROJECTED_LEAVES") == "1"
    if project_leaves:
        engine._lod_glm_projected_leaf_calls = getattr(engine, "_lod_glm_projected_leaf_calls", 0) + 1
        refined, refined_lse = projected_leaf_attention(
            engine, engine._lod_kimi_expanded_prefill_chunk, slots, page_cache,
            active_slots=state_len, buffers=buffers)
    else:
        # Combine regions BEFORE W_UV; eight separate output projections
        # would repeat the same linear work unnecessarily.
        refined, refined_lse = engine._paged_leaf_attention(
            q, slots, page_cache, active_slots=state_len, reduce_routes=True)
        refined = project_kimi_head_values(refined, w_uv)
    if local_branch is None:
        local_branch = engine._prefill_local_attention(
            q, local_k, local_v, query_offset=local_k.size(2) - q.size(2))
    pending_local = getattr(engine, "_lod_prefill_local_stream_pending", None)
    if pending_local is not None:
        torch.cuda.current_stream(q.device).wait_stream(pending_local)
        del engine._lod_prefill_local_stream_pending
    local_out, local_lse = local_branch
    if local_out.size(-1) == 512:
        local_out = project_kimi_head_values(local_out, w_uv)
    projected_sink = project_kimi_shared_values(sink_v, w_uv)
    if project_leaves:
        # The sink is the only key read by the final merge. Project this tiny
        # field instead of absorbing the entire chunk's queries just for it.
        q = engine._lod_kimi_expanded_prefill_chunk
        sink_k = project_kimi_shared_values(sink_k, engine._lod_kimi_w_uk_t.transpose(1, 2))
    return merge_aiter_mla_prefill_refinement(
        q, sink_k, projected_sink, coarse, slots, refined, refined_lse,
        local_out, local_lse, kv_group_size=q.size(1), scale=engine.scaling,
        output_buffer=output_buffer)


def projected_leaf_attention(engine, q, slots, cache, *, active_slots,
                              buffers=None, projection_tile=(64, 128, 4),
                              attention_tile=None, route_counter=None):
    """Project the selected union once, then run ordinary K256/V256 experts."""
    from .glm_compact_leaf_projection import project_compact_glm_leaves
    from .paged_prefill import count_expert_routes, paged_leaf_attention
    if route_counter is None:
        # Long slabs repeatedly hit the same regions. Local sorts avoid
        # contended ordinal atomics; short slabs retain the cheaper counter.
        from .kimi_sorted_route_counts import count_sorted_kimi_routes
        route_counter = count_sorted_kimi_routes if q.size(2) >= 2048 else count_expert_routes
    counts, offsets = route_counter(slots, active_slots=active_slots, buffers=buffers)
    live = int(cache.get("leaf_count", cache["leaf_k"].size(2)))
    k, v, starts = project_compact_glm_leaves(
        cache["leaf_k"][..., :live, :], engine._lod_kimi_w_uk_t, engine._lod_kimi_w_uv,
        cache, counts, active_slots=active_slots, hash_probes=engine._page_lookup_probes(cache),
        buffers=buffers, block_m=projection_tile[0], block_n=projection_tile[1],
        num_warps=projection_tile[2])
    bm, bn, warps = attention_tile or (engine.leaf_block_m, engine.leaf_block_n, engine.leaf_num_warps)
    return paged_leaf_attention(q, k, v, cache["slot_pages"], cache["overflow_page_keys"],
        cache["overflow_page_values"], cache["overflow_used"], cache["slot_lengths"], slots,
        page_indices=cache["page_indices"], compact_leaf_offsets=starts,
        kv_group_size=1, active_slots=active_slots, route_head_counts=counts,
        route_offsets=offsets, scale=engine.scaling, hash_probes=engine._page_lookup_probes(cache),
        block_m=bm, block_n=bn, num_warps=warps, reduce_routes=True, buffers=buffers,
        timing_events=getattr(engine, "_lod_leaf_timing_events", None))


def projected_local_attention(q, latent, w_uk_t, w_uv, *, query_offset,
                              scale, return_lse=True, buffers=None, output_buffer=None):
    """Kimi's projected local idea, without its 64 direct-key channels."""
    from aiter import mha_fwd
    batch, heads, supplied_len, _ = q.shape
    tokens = latent.size(2)
    queries = tokens - query_offset
    if supplied_len == tokens:
        q = q[..., query_offset:, :]
    elif supplied_len != queries:
        raise ValueError("projected local query length disagrees with its causal offset")
    # Exact-front and later local fields can overlap on separate streams.
    # They must not overwrite each other's projection or output workspace.
    prefix = "glm_local" if return_lse else "glm_exact_front"
    packed = _workspace_tensor(buffers, prefix + "_kv", (batch, tokens, heads, 512), dtype=q.dtype, device=q.device)
    torch.mm(latent[:, 0].reshape(batch * tokens, 512),
        combined_kv_weight(w_uk_t, w_uv, buffers),
        out=packed.view(batch * tokens, heads * 512))
    destination = _workspace_tensor(buffers, prefix + "_output", (batch, queries, heads, 256), dtype=q.dtype, device=q.device)
    result = mha_fwd(q.permute(0, 2, 1, 3), packed[..., :256], packed[..., 256:],
        0.0, float(scale), True, -1, -1, 0, return_lse, False,
        None, None, destination, None, None, None, None, None, None)
    output = result[0].permute(0, 2, 1, 3)
    if output_buffer is not None:
        output_buffer.copy_(output)
        output = output_buffer
    lse = result[1] if return_lse else torch.empty(0, device=q.device, dtype=torch.float32)
    return output, lse
