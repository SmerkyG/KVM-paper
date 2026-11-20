"""Measure query sharing of selected regions, separately for each query head.

All measurements are diagnostic. A group is confined to one query head and
one contiguous query range. KV reuse factors describe ideal shared-region
work and do not claim measured cache traffic or kernel latency.
"""
from __future__ import annotations

import types
import torch

BINS = ((1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63),
        (64, 127), (128, 255), (256, 511), (512, 1023),
        (1024, 2047), (2048, 4095), (4096, 8191), (8192, 16384))


def route_statistics(slots, lengths, *, state_len, group_size, route_counts=None):
    """Summarize one group using exact integer sums; works on CPU or GPU."""
    if slots.ndim != 4 or slots.size(0) != 1 or lengths.size(0) != 1:
        raise ValueError('expected batch-one slots and lengths')
    heads, queries, width = map(int, slots.shape[1:])
    if heads != int(lengths.size(1)) * group_size:
        raise ValueError('incompatible grouped-query geometry')
    values = slots[0].to(torch.int64)
    valid = (values >= 0) & (values < state_len)
    if route_counts is not None:
        ranks = torch.arange(width, device=slots.device)
        valid &= ranks[None, None, :] < route_counts[:, :queries, None]
    offsets = torch.arange(heads, device=slots.device, dtype=torch.int64)[:, None, None] * state_len
    keys = (values + offsets)[valid]
    histogram = torch.bincount(keys, minlength=heads * state_len).reshape(heads, state_len)
    sizes = lengths[0, :, :state_len].repeat_interleave(group_size, 0).to(torch.int64)
    h = histogram.to(torch.int64)
    active = h > 0
    entry_uses = h * sizes
    base = [heads * queries, heads * queries * (queries - 1) // 2,
            int(queries), h.sum(), active.sum(), (h * h).sum(),
            (h * (h - 1) // 2).sum(), entry_uses.sum(),
            torch.where(active, sizes, 0).sum(),
            torch.where(h > 1, h, 0).sum(),
            torch.where(h > 1, entry_uses, 0).sum(),
            torch.where(h > 1, active, False).sum()]
    bin_values=[]
    for lo, hi in BINS:
        inside = (h >= lo) & (h <= hi)
        bin_values += [inside.sum(), torch.where(inside, h, 0).sum(),
                       torch.where(inside, entry_uses, 0).sum()]
    out = torch.stack([torch.as_tensor(x, dtype=torch.int64, device=slots.device)
                       for x in base + bin_values]).cpu().tolist()
    names = ('query_heads','query_head_pairs','queries_per_group','routes','active_head_regions',
             'route_weighted_group_size_sum','shared_region_query_pairs','opened_entry_uses',
             'unique_region_entries','shared_routes','shared_entry_uses','shared_head_regions')
    result=dict(zip(names,out[:len(names)],strict=True))
    result['bins']={f'{lo}-{hi}': {'groups':out[len(names)+3*i],
                     'routes':out[len(names)+3*i+1], 'entry_uses':out[len(names)+3*i+2]}
                    for i,(lo,hi) in enumerate(BINS)}
    result['state_len']=int(state_len)
    return result


def audit_routes(engine, slots, lengths, *, state_len, group_size, sample_limit, label, route_counts=None):
    n = min(int(slots.size(2)), sample_limit)
    sample = slots[..., :n, :]
    counts = route_counts[:, :n] if route_counts is not None else None
    record = route_statistics(sample, lengths, state_len=state_len, group_size=group_size, route_counts=counts)
    record['label']=label
    audit = getattr(engine, '_lod_prefill_overlap_audit', None)
    if audit is None:
        audit=[]
        engine._lod_prefill_overlap_audit=audit
    audit.append(record)


def install_top8(engine):
    original = engine._paged_leaf_attention
    def replacement(self, q, slots, cache, *, active_slots, reduce_routes=True):
        if q.size(2) > 1 and slots.size(-1) == 8:
            audit_routes(self, slots, cache['slot_lengths'], state_len=active_slots,
                         group_size=self.num_key_value_groups, sample_limit=512, label='512_queries')
            # Actual top-eight grouped dispatch spans the complete prefill chunk.
            audit_routes(self, slots, cache['slot_lengths'], state_len=active_slots,
                         group_size=self.num_key_value_groups, sample_limit=int(q.size(2)), label='full_chunk')
        return original(q, slots, cache, active_slots=active_slots, reduce_routes=reduce_routes)
    engine._paged_leaf_attention = types.MethodType(replacement, engine)
