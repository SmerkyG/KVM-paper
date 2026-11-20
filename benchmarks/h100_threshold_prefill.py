"""Correctness-first variable-threshold prefill for the H100 Llama experiment.

This opt-in prototype uses bounded query tiles, PyTorch routing and the
existing grouped paged-leaf consumer. It is not a production-speed kernel.
The local/sink fields and the state update schedule are unchanged.
"""
from __future__ import annotations

import math
import os
import types

import torch


def select_regions(route_scores, sizes, *, rule, multiplier=16.0,
                   entry_bound=16, work_bound=None, max_region_entries=1024, max_regions=None):
    """Return a disjoint selection and remote lengths for B,H,Q,S scores.

    Sizes are B,H,S page-list lengths. Scores are unnormalized proxy logits.
    The integer work guard uses a deterministic prefix in region order.
    """
    if work_bound is None:
        work_bound = entry_bound
    if rule not in ('region', 'entry') or multiplier <= 0 or entry_bound < 1 or work_bound < 1:
        raise ValueError('invalid prefill threshold configuration')
    valid = sizes > 0
    scores = route_scores.float().masked_fill(~valid.unsqueeze(2), -torch.inf)
    log_z = torch.logsumexp(scores, dim=-1, keepdim=True)
    safe_log_z = torch.where(torch.isfinite(log_z), log_z, torch.zeros_like(log_z))
    remote = sizes.to(torch.int64).sum(-1, keepdim=True).unsqueeze(2)
    if rule == 'region':
        num_regions = valid.sum(-1, keepdim=True).unsqueeze(2).clamp_min(1)
        cutoff = safe_log_z + torch.log(torch.as_tensor(multiplier, device=scores.device) / num_regions)
        selected = (scores > cutoff) & valid.unsqueeze(2)
        if max_region_entries is not None:
            selected &= sizes.unsqueeze(2) <= max_region_entries
    else:
        cutoff = safe_log_z - math.log(entry_bound) - 0.5 * remote.clamp_min(1).float().log() + 1.0e-6
        selected = (scores - sizes.clamp_min(1).float().log().unsqueeze(2) > cutoff) & valid.unsqueeze(2)
        cumulative = (selected.to(torch.int64) * sizes.unsqueeze(2)).cumsum(-1)
        selected &= cumulative.square() < work_bound * work_bound * remote
    if max_regions is not None:
        if max_regions < 1:
            raise ValueError('maximum region count must be positive')
        selected &= selected.to(torch.int32).cumsum(-1) <= max_regions
    return selected, remote


def compact_routes(selected):
    """Compact each query's selected indices in region order without sorting."""
    counts = selected.sum(-1)
    maximum = max(1, int(counts.max().item()))
    # Padded routes are masked by the leaf consumer. Bucketing avoids a new
    # Triton compilation for every slight change in the maximum route count.
    width = 1 << (maximum - 1).bit_length()
    slots = torch.full((*selected.shape[:-1], width), -1, device=selected.device, dtype=torch.long)
    indices = selected.nonzero(as_tuple=True)
    ranks = selected.to(torch.int32).cumsum(-1) - 1
    slots[indices[:-1] + (ranks[indices].long(),)] = indices[-1]
    return slots.contiguous()


def threshold_prefill(engine, q, local_k, local_v, state_k, state_v, counts,
                      owners, exact_k, exact_v, *, state_len, state_capacity,
                      page_cache, local_branch, sink_k, sink_v, output_buffer=None,
                      rule, multiplier, entry_bound, work_bound, tile_queries, max_regions=None, **kwargs):
    del local_k, local_v, owners, exact_k, exact_v, state_capacity, kwargs
    from lod_attention.kernels.lod_kernels import merge_attention_branches_with_sink
    if (q.size(0) != 1 or q.size(1) != 32 or q.size(-1) != 128
            or not q.is_cuda or torch.version.hip or local_branch is None
            or page_cache is None or sink_k is None or sink_v is None):
        raise ValueError('experimental prefill requires batch-one H100 Llama with split local attention')
    if bool(page_cache.get('quantization_finalized', False)):
        raise ValueError('experimental prefill currently requires BF16 leaves')
    sizes = engine._repeat_kv(page_cache['slot_lengths'][..., :state_len]).to(torch.int64)
    coarse_counts = engine._repeat_kv(counts[..., :state_len, :]).squeeze(-1).float()
    if not torch.equal(sizes, coarse_counts.to(torch.int64)):
        raise ValueError('prefill state counts and page lengths disagree')
    means_v = engine._repeat_kv(engine._mean(state_v[..., :state_len, :], counts[..., :state_len, :])).float()
    log_counts = coarse_counts.clamp_min(1).log().unsqueeze(2)
    output = torch.empty_like(q) if output_buffer is None else output_buffer
    local_out, local_lse = local_branch
    for begin in range(0, int(q.size(2)), tile_queries):
        end = min(begin + tile_queries, int(q.size(2)))
        query = q[..., begin:end, :].contiguous()
        native = engine._state_route_logits(query, state_k, counts, state_len=state_len).float() * engine.scaling + log_counts
        route = engine._state_routing_logits(query, state_k, counts, state_len=state_len).float() * engine.scaling + log_counts
        selected, remote = select_regions(route, sizes, rule=rule, multiplier=multiplier,
                                          entry_bound=entry_bound, work_bound=work_bound,
                                          max_region_entries=engine.max_open_centroid_leaves,
                                          max_regions=max_regions)
        if rule == 'entry':
            opened = (selected.to(torch.int64) * sizes.unsqueeze(2)).sum(-1, keepdim=True)
            if bool((opened.square() >= work_bound * work_bound * remote).any().item()):
                raise RuntimeError('prefill entry work bound violated')
        if os.environ.get('LOD_H100_PREFILL_AUDIT') == '1':
            if rule != 'entry':
                opened = (selected.to(torch.int64) * sizes.unsqueeze(2)).sum(-1, keepdim=True)
            proxy = torch.softmax(route.float().masked_fill((sizes == 0).unsqueeze(2), -torch.inf), -1)
            bound = work_bound if rule == 'entry' else 16
            fraction = opened.float() / (bound * remote.clamp_min(1).float().sqrt())
            stats = getattr(engine, '_lod_prefill_threshold_stats', None)
            if stats is None:
                stats = {'query_heads': 0, 'entries_sum': torch.zeros((), device=q.device, dtype=torch.int64),
                         'regions_sum': torch.zeros((), device=q.device, dtype=torch.int64),
                         'proxy_mass_sum': torch.zeros((), device=q.device, dtype=torch.float64),
                         'max_bound_fraction': torch.zeros((), device=q.device)}
                engine._lod_prefill_threshold_stats = stats
            stats['query_heads'] += int(selected.numel() // selected.size(-1))
            stats['entries_sum'].add_(opened.sum())
            stats['regions_sum'].add_(selected.sum())
            stats['proxy_mass_sum'].add_(torch.where(selected, proxy, 0).sum(dtype=torch.float64))
            stats['max_bound_fraction'].copy_(torch.maximum(stats['max_bound_fraction'], fraction.max()))
        slots = compact_routes(selected)
        exact_out, exact_lse = engine._paged_leaf_attention(query, slots, page_cache,
                                                            active_slots=state_len, reduce_routes=True)
        has_exact = selected.any(-1)
        exact_out = torch.where(has_exact.unsqueeze(-1), exact_out, torch.zeros_like(exact_out))
        exact_lse = torch.where(has_exact, exact_lse, torch.full_like(exact_lse, -torch.inf))
        coarse_logits = native.masked_fill(selected | (sizes == 0).unsqueeze(2), -torch.inf)
        coarse_lse = torch.logsumexp(coarse_logits, -1)
        safe_lse = torch.where(torch.isfinite(coarse_lse), coarse_lse, torch.zeros_like(coarse_lse))
        weights = torch.exp(coarse_logits - safe_lse.unsqueeze(-1))
        coarse_out = torch.matmul(weights, means_v).to(query.dtype)
        merge_attention_branches_with_sink(
            query, sink_k, sink_v, coarse_out, coarse_lse, exact_out, exact_lse,
            local_out[..., begin:end, :], local_lse[..., begin:end],
            kv_group_size=engine.num_key_value_groups, scale=engine.scaling,
            output_buffer=output[..., begin:end, :])
    return output


def validate_cache_mode(mode, rule):
    if mode == 'two-tier' and rule in ('region', 'entry'):
        return
    if (mode == 'three-tier-bf16' and rule == 'region'
            and os.environ.get('LOD_H100_PREFILL_IMPLEMENTATION') == 'fast'
            and os.environ.get('LOD_H100_FAST_LEAF') == 'native'
            and os.environ.get('LOD_H100_CLOSED_EXACT', 'grouped') == 'grouped'
            and int(os.environ.get('LOD_H100_PREFILL_MAX_REGIONS', '0')) == 0):
        # The grouped leaf kernel reads whole regions through the recursive
        # cache's page directory. Decode retains the same pages and summaries.
        return
    raise ValueError('threshold prefill requires two-tier BF16, or three-tier BF16 '
                     'with uncapped fast per-region routing and native grouped leaves')


def install(engine, *, rule, multiplier=16.0, entry_bound=16, work_bound=None, tile_queries=512):
    """Replace only multi-token prefill attention; preserve the decode path."""
    if work_bound is None:
        work_bound = entry_bound
    if tile_queries <= 0:
        raise ValueError('prefill tile size must be positive')
    original = engine._two_level_attention
    implementation = os.environ.get('LOD_H100_PREFILL_IMPLEMENTATION', 'reference')
    if implementation == 'fused':
        if os.environ.get('LOD_H100_PREFILL_AUDIT') == '1':
            raise ValueError('fused threshold prefill audit is not yet implemented')
        from benchmarks.h100_threshold_prefill_fused import threshold_prefill_fused
        attention = threshold_prefill_fused
    elif implementation == 'fast':
        from benchmarks.h100_threshold_prefill_fast import threshold_prefill_fast
        attention = threshold_prefill_fast
    elif implementation == 'closed':
        from benchmarks.h100_threshold_prefill_closed import threshold_prefill_closed
        attention = threshold_prefill_closed
    elif implementation == 'reference':
        attention = threshold_prefill
    else:
        raise ValueError('unknown threshold prefill implementation')

    def wrapped(self, q, *args, **kwargs):
        if q.size(2) <= 1:
            return original(q, *args, **kwargs)
        return attention(self, q, *args, rule=rule, multiplier=multiplier,
                         entry_bound=entry_bound, work_bound=work_bound,
                         tile_queries=tile_queries, **kwargs)
    engine._two_level_attention = types.MethodType(wrapped, engine)
