"""Benchmark-only per-query leaf attention for the existing top-eight router.

The selector and final coarse/local/sink merge remain the production path.
The combined exact branch is placed in the first valid route slot solely to
satisfy the existing per-route merge interface.
"""
from __future__ import annotations

import os
import types
import torch
import triton
import triton.language as tl

from lod_attention.kernels.aiter_prefill_attention import _workspace_tensor
from benchmarks.h100_threshold_prefill_fused import _expand_direct_indices, _indexed_direct_attention


@triton.jit
def _compact_top8(Slots, Lengths, Packed, RouteCounts, EntryCounts,
                  QLEN: tl.constexpr, CAP: tl.constexpr, G: tl.constexpr):
    row = tl.program_id(0)
    kh = row // QLEN // G
    rank = tl.arange(0, 8)
    slot = tl.load(Slots + row * 8 + rank).to(tl.int32)
    valid = (slot >= 0) & (slot < CAP)
    sizes = tl.load(Lengths + kh * CAP + slot, mask=valid, other=0).to(tl.int32)
    valid &= sizes > 0
    dest = tl.cumsum(valid.to(tl.int32), 0) - 1
    tl.store(Packed + row * 8 + dest, slot, mask=valid)
    tl.store(RouteCounts + row, tl.sum(valid.to(tl.int32), 0))
    tl.store(EntryCounts + row, tl.sum(tl.where(valid, sizes, 0), 0))


@triton.jit
def _put_combined_in_first_route(Slots, CombinedOut, CombinedLSE, RouteOut, RouteLSE,
                                 TOTAL_Q: tl.constexpr, TILE_Q: tl.constexpr,
                                 BEGIN: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)
    head = row // TILE_Q
    query = row % TILE_Q
    full_row = head * TOTAL_Q + BEGIN + query
    rank = tl.arange(0, 8)
    dim = tl.arange(0, D)
    slots = tl.load(Slots + row * 8 + rank).to(tl.int32)
    valid = slots >= 0
    first = valid & (tl.cumsum(valid.to(tl.int32), 0) == 1)
    v = tl.load(CombinedOut + row * D + dim).to(tl.float32)
    lse = tl.load(CombinedLSE + row)
    tl.store(RouteOut + (full_row * 8 + rank[:, None]) * D + dim[None, :],
             tl.where(first[:, None], v[None, :], 0.0))
    tl.store(RouteLSE + full_row * 8 + rank, tl.where(first, lse, -float('inf')))


def install(engine, buffers):
    original = engine._paged_leaf_attention

    def replacement(self, q, top_slots, cache, *, active_slots, reduce_routes=True):
        if q.size(2) <= 1 or reduce_routes:
            return original(q, top_slots, cache, active_slots=active_slots, reduce_routes=reduce_routes)
        if q.size(0) != 1 or top_slots.size(-1) != 8 or cache.get('quantization_finalized', False):
            raise RuntimeError('top8 query experiment requires batch-one BF16 eight-route prefill')
        # The unchanged top-8 producer precomputes metadata for grouped dispatch.
        # The per-query backend does not consume it; clear it rather than leaving
        # stale state for the next call.
        for name in ('_lod_prefill_route_head_counts', '_lod_prefill_route_offsets', '_lod_prefill_total_valid_routes'):
            if hasattr(self, name):
                delattr(self, name)
        h, total_q, d, g = int(q.size(1)), int(q.size(2)), int(q.size(-1)), int(self.num_key_value_groups)
        lengths, roots = cache['slot_lengths'], cache['slot_pages']
        leaf_k, leaf_v, pages = cache['leaf_k'], cache['leaf_v'], cache['page_indices']
        directory = cache['overflow_page_values']
        cap, leaf_capacity = int(lengths.size(-1)), int(leaf_k.size(2))
        max_entries = leaf_capacity
        if self.max_open_centroid_leaves is not None:
            max_entries = min(max_entries, 8 * int(self.max_open_centroid_leaves))
        max_entries = max(1, max_entries)
        tile_size = int(os.environ.get('LOD_H100_TOP8_QUERY_TILE', '512'))
        if tile_size <= 0:
            raise ValueError('query tile must be positive')
        def workspace(name, shape, dtype):
            return _workspace_tensor(buffers, 'top8_query_' + name, shape, dtype=dtype, device=q.device)
        route_out = workspace('route_out', (1, h, total_q, 8, d), q.dtype)
        route_lse = workspace('route_lse', (1, h, total_q, 8), torch.float32)
        for begin in range(0, total_q, tile_size):
            end = min(begin + tile_size, total_q)
            n = end - begin
            query = q[..., begin:end, :].contiguous()
            slots = top_slots[..., begin:end, :].contiguous()
            packed = workspace('slots', (1, h, n, 8), torch.int32)
            route_counts = workspace('counts', (h, n), torch.int32)
            entry_counts = workspace('entries', (h, n), torch.int32)
            _compact_top8[(h * n,)](slots, lengths, packed, route_counts, entry_counts, n, cap, g, num_warps=1)
            offsets = workspace('offsets', (h * n + 1,), torch.int64)
            offsets[:1].zero_()
            torch.cumsum(entry_counts.reshape(-1), 0, dtype=torch.int64, out=offsets[1:])
            indices = workspace('indices', (h, n, max_entries), torch.int32)
            _expand_direct_indices[(h * n,)](packed, route_counts, lengths, roots,
                cache['overflow_page_keys'], directory, cache['overflow_used'], pages,
                indices, entry_counts, route_counts, offsets, n, 8, cap, g, int(roots.size(3)),
                int(directory.size(2)), int(pages.size(2)), leaf_capacity,
                self._page_lookup_probes(cache), max_entries, 4, True, True, 256, num_warps=4)
            out = workspace('combined_out', (1, h, n, d), q.dtype)
            lse = workspace('combined_lse', (1, h, n), torch.float32)
            _indexed_direct_attention[(h * n,)](query, indices, entry_counts, offsets,
                leaf_k, leaf_v, out, lse, n, max_entries, g, leaf_capacity, d,
                float(self.scaling), 128, True, num_warps=4)
            if os.environ.get('LOD_H100_TOP8_QUERY_AUDIT') == '1' and begin == 0:
                sample_n = min(16, n)
                sample_q = query[..., :sample_n, :].contiguous()
                sample_slots = slots[..., :sample_n, :].contiguous()
                ref_out, ref_lse = original(sample_q, sample_slots, cache,
                                            active_slots=active_slots, reduce_routes=False)
                valid = sample_slots >= 0
                logmass = torch.where(valid, ref_lse.float(), -float('inf'))
                weights = torch.softmax(logmass, dim=-1)
                weights = torch.where(valid, weights, 0.0)
                ref = (torch.where(valid.unsqueeze(-1), ref_out.float(), 0.0) * weights.unsqueeze(-1)).sum(-2)
                expected_lse = torch.logsumexp(logmass, dim=-1)
                actual = out[..., :sample_n, :].float()
                actual_lse = lse[..., :sample_n]
                has = valid.any(-1)
                err = (ref - actual).abs()
                lse_error = torch.where(has, (expected_lse - actual_lse).abs(), 0.0)
                if not bool(torch.isfinite(err).all().item()) or not bool(torch.isfinite(lse_error).all().item()):
                    raise RuntimeError('top8 per-query crosscheck produced non-finite values')
                max_error, max_lse_error = float(err.max().item()), float(lse_error.max().item())
                if max_error > 0.03 or max_lse_error > 0.03:
                    raise RuntimeError(f'top8 per-query crosscheck failed: output={max_error}, lse={max_lse_error}')
                audit = getattr(self, '_lod_top8_query_audit', None)
                if audit is None:
                    audit = {'query_heads': 0, 'max_abs_output_difference': 0.0,
                             'sum_abs_output_difference': 0.0, 'output_elements': 0,
                             'max_abs_lse_difference': 0.0, 'opened_entries': 0,
                             'max_opened_entries': 0, 'invalid_routes': 0}
                    self._lod_top8_query_audit = audit
                audit['query_heads'] += h * sample_n
                audit['max_abs_output_difference'] = max(audit['max_abs_output_difference'], max_error)
                audit['sum_abs_output_difference'] += float(err.sum().item())
                audit['output_elements'] += err.numel()
                audit['max_abs_lse_difference'] = max(audit['max_abs_lse_difference'], max_lse_error)
                audit['opened_entries'] += int(entry_counts[:, :sample_n].sum().item())
                audit['max_opened_entries'] = max(audit['max_opened_entries'], int(entry_counts[:, :sample_n].max().item()))
                audit['invalid_routes'] += int((sample_slots < 0).sum().item())
            _put_combined_in_first_route[(h * n,)](slots, out, lse, route_out, route_lse,
                                                   total_q, n, begin, d, num_warps=4)
        return route_out, route_lse

    engine._paged_leaf_attention = types.MethodType(replacement, engine)
