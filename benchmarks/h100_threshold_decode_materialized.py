"""Run AMD's four-stage materialized GQA decode pipeline on H100.

Serving cache refreshes maintain the existing BF16-rounded H100 mean arithmetic
and persistent leaf indices. Their costs must be charged in model timings.
"""
import os

import torch

from benchmarks.amd_threshold_frontier import prepare_cache
from benchmarks.amd_threshold_materialized import threshold_attention_materialized


def threshold_attention_h100_materialized(pool, q, k, v, out, multiplier, *, optimized=None, **kwargs):
    if kwargs.get('rule', 'region') != 'region':
        raise ValueError('four-stage H100 candidate supports per-region thresholds only')
    if q.shape[1:] != (1, 32, 128) or pool.dtype != torch.bfloat16 or pool.settings.levels != 2:
        raise ValueError('requires BF16 Llama GQA two-tier')
    if q.shape[0] != pool.state['state_k'].size(0):
        raise ValueError('requires one query per cache row')
    page = pool.state['page_cache']
    fixture = 'unified_page1_k' not in page
    if fixture and q.shape[0] != 1:
        raise ValueError('batched materialized decode requires the serving cache arena')
    if not fixture and page.get('unified_page1_fixed_indices') is None:
        raise RuntimeError('enable materialized implementation before allocating the serving cache')
    cached = getattr(pool, '_threshold_frontier_cache', None)
    if cached is None:
        cached = prepare_cache(pool)
        if fixture:
            from lod_attention.kernels.paged_decode_kernels import materialize_page1_coarse_means
            bias = torch.empty(pool.state['counts'].shape[:-1], device=q.device, dtype=torch.float16)
            materialize_page1_coarse_means(pool.state['state_k'], pool.state['state_v'], pool.state['counts'],
                cached['mean_k'].view_as(pool.state['state_k']), cached['mean_v'].view_as(pool.state['state_v']),
                bias, round_counts_bf16=True)
    if optimized is None:
        optimized = os.environ.get('LOD_H100_TWO_TIER_OPTIMIZED') == '1'
    return threshold_attention_materialized(pool, q, k, v, out, multiplier, optimized=optimized, **kwargs)
