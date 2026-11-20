"""H100 port of AMD's uncapped three-tier threshold decode pipeline.

The routing proxy uses RMS-normalized queries; the active attention frontier
uses native queries, exact selected pages and a disjoint residual. Fixed-page
decode defaults to two pages. On H100, selecting the 0.5 mean-page threshold
uses the tuned adaptive kernel and RMS-normalized page queries by default.
Cached means are refreshed only at the serving pool's existing state-update
boundaries. Those refreshes belong in amortized model timings, outside graphs.
"""
import os
import functools
from itertools import groupby

import torch

from benchmarks.amd_threshold_three_tier import threshold_attention as shared_attention


def use_tuned_adaptive_pages(device):
    page_multiplier = os.environ.get('LOD_H100_PAGE_THRESHOLD')
    return (page_multiplier is not None and float(page_multiplier) == 0.5
            and os.environ.get('LOD_H100_PAGE_THRESHOLD_RULE', 'page') == 'page'
            and os.environ.get('LOD_H100_THRESHOLD_IMPLEMENTATION', 'three-tier') == 'three-tier'
            and os.environ.get('LOD_H100_THREE_TIER_CACHED_MEANS', '1') == '1'
            and os.environ.get('LOD_H100_ADAPTIVE_TUNED_DEFAULTS', '1') == '1'
            and torch.cuda.get_device_capability(device) == (9, 0))


def adaptive_page_routing_options(*, tuned=False):
    return dict(rule=os.environ.get('LOD_H100_PAGE_THRESHOLD_RULE', 'page'),
                query_score=os.environ.get('LOD_H100_PAGE_QUERY', 'normalized' if tuned else 'native'),
                residual=os.environ.get('LOD_H100_PAGE_RESIDUAL', 'region'),
                page_threshold_warps=int(os.environ.get('LOD_H100_PAGE_THRESHOLD_WARPS', '2' if tuned else '4')))


def adaptive_page_options(*, tuned=False):
    flag_default = '1' if tuned else '0'
    return dict(fast_bf16=os.environ.get('LOD_H100_ADAPTIVE_PAGE_FAST', flag_default) == '1',
                exact_pages=int(os.environ.get('LOD_H100_ADAPTIVE_EXACT_PAGES', '2' if tuned else '1')),
                reuse_directory=os.environ.get('LOD_H100_ADAPTIVE_REUSE_DIRECTORY', flag_default) == '1',
                short_scan=int(os.environ.get('LOD_H100_ADAPTIVE_SHORT_SCAN', '8' if tuned else '0')),
                fast_residual_logit=os.environ.get('LOD_H100_ADAPTIVE_RESIDUAL_LOGIT', flag_default) == '1')


def install_cached_means(pool):
    """Attach persistent means to a real serving pool, including update hooks."""
    if getattr(pool, '_h100_three_tier_means', None) is not None:
        return
    if pool.settings.levels != 3 or pool.kv_heads != 8 or pool.head_dim != 128:
        raise ValueError('H100 three-tier cache requires Llama GQA geometry')
    from lod_attention.kernels.paged_decode_kernels import materialize_page1_coarse_means
    state = pool.state
    means = (torch.empty_like(state['state_k']), torch.empty_like(state['state_v']))
    bias = torch.empty(state['counts'].shape[:-1], device=pool.device, dtype=torch.float16)
    original = pool._refresh_unified_page1_coarse

    def refresh(slots):
        original(slots)
        for _, contiguous in groupby(enumerate(sorted(slots)), key=lambda pair: pair[1] - pair[0]):
            rows = [slot for _, slot in contiguous]
            span = slice(rows[0], rows[-1] + 1)
            materialize_page1_coarse_means(
                state['state_k'][span], state['state_v'][span], state['counts'][span],
                means[0][span], means[1][span], bias[span], round_counts_bf16=True)
        pool._h100_three_tier_mean_refreshes += bool(slots)

    pool._h100_three_tier_means = means
    pool._h100_three_tier_mean_refreshes = 0
    pool._refresh_unified_page1_coarse = refresh
    refresh(tuple(range(pool.max_requests)))


def threshold_launch_options(*, tile=64, splits=64, page_block_n=None, tuned_mode=None):
    """Resolve the kernel launch, with explicit benchmark settings taking precedence."""
    page_block = os.environ.get('LOD_H100_THREE_TIER_PAGE_BLOCK')
    local_tile = os.environ.get('LOD_H100_THREE_TIER_LOCAL_TILE')
    tuned = tuned_mode is not None
    enabled = '1' if tuned else '0'
    adaptive = tuned_mode == 'three-tier-adaptive-bf16'
    workers = {'three-tier-int4': 24, 'three-tier-bf16': 16, 'three-tier-adaptive-bf16': 24}.get(tuned_mode, splits)
    fixed_scan = os.environ.get('LOD_H100_FIXED_PAGE_SCAN', '8' if tuned_mode == 'three-tier-bf16' else '')
    return dict(
        tile=int(os.environ.get('LOD_H100_THRESHOLD_TILE', '128')) if tuned else tile,
        workers=int(os.environ.get('LOD_H100_THREE_TIER_WORKERS', str(workers))),
        page_warps=int(os.environ.get('LOD_H100_THREE_TIER_PAGE_WARPS', '1')),
        page_block_n=int(page_block) if page_block is not None else 16 if tuned and not adaptive else page_block_n,
        merge_block=int(os.environ.get('LOD_H100_THRESHOLD_MERGE_BLOCK', '32')),
        pages_per_slot=int(os.environ.get('LOD_H100_THREE_TIER_PAGES', '2')),
        trim_idle_merge=os.environ.get('LOD_H100_THREE_TIER_TRIM_MERGE', enabled) == '1',
        materialize_cutoffs=os.environ.get('LOD_H100_THREE_TIER_MATERIALIZE_CUTOFFS', enabled) == '1',
        gqa_local=os.environ.get('LOD_H100_THREE_TIER_GQA_LOCAL', enabled) == '1',
        fused_pack=os.environ.get('LOD_H100_THREE_TIER_FUSED_PACK', enabled) == '1',
        route_warps=int(os.environ.get('LOD_H100_THREE_TIER_ROUTE_WARPS', '4')),
        local_tile=int(local_tile) if local_tile is not None else 64 if tuned else None,
        factor_int4_page_scales=os.environ.get('LOD_H100_INT4_FACTOR_PAGE_SCALES',
                                               '1' if tuned_mode == 'three-tier-int4' else '0') == '1',
        fixed_page_scan=int(fixed_scan) if fixed_scan else None,
    )


def threshold_attention(pool, q, k, v, out, multiplier, *, tile=64, splits=64, rule='region',
                        page_block_n=None, launch_options=None, **kwargs):
    if (pool.settings.levels != 3 or pool.dtype != torch.bfloat16
            or q.shape[1:] != (1, 32, 128) or pool.kv_heads != 8):
        raise ValueError('H100 three-tier thresholds require single-token BF16 Llama queries')
    if rule != 'region':
        raise ValueError('three-tier H100 port currently supports per-region thresholds')
    implementation = os.environ.get('LOD_H100_THRESHOLD_IMPLEMENTATION', 'three-tier')
    if implementation not in ('three-tier', 'three-tier-reference'):
        raise ValueError('unknown H100 three-tier threshold implementation')
    cached = os.environ.get('LOD_H100_THREE_TIER_CACHED_MEANS', '1') == '1'
    # The uncached arm is a control with identical H100 arithmetic and frontier.
    if implementation == 'three-tier-reference':
        cached = False
    means = getattr(pool, '_h100_three_tier_means', None) if cached else None
    if cached and means is None:
        raise RuntimeError('install three-tier cached means before prefill')
    tuned_adaptive = use_tuned_adaptive_pages(q.device)
    if launch_options is None:
        launch_options = threshold_launch_options(
            tile=tile, splits=splits, page_block_n=page_block_n,
            tuned_mode='three-tier-adaptive-bf16' if tuned_adaptive else None)
    launch = launch_options.copy()
    factor_int4_page_scales = launch.pop('factor_int4_page_scales')
    fixed_page_scan = launch.pop('fixed_page_scan')
    page_attention = None
    page_multiplier = os.environ.get('LOD_H100_PAGE_THRESHOLD')
    if page_multiplier is not None:
        if pool.state['page_cache'].get('leaf_quant_bits', 0):
            raise ValueError('page thresholds currently require BF16 KV')
        from benchmarks.h100_threshold_pages import page_threshold_attention
        workers = launch['workers']
        stats_shape = (q.size(0) * q.size(2), workers, 4)
        stats = getattr(pool, '_h100_page_threshold_stats', None)
        if stats is None or stats.shape != stats_shape:
            stats = torch.empty(stats_shape, device=q.device, dtype=torch.int32)
            pool._h100_page_threshold_stats = stats
        page_attention = functools.partial(page_threshold_attention,
            multiplier=float(page_multiplier), stats_buffer=stats,
            **adaptive_page_routing_options(tuned=tuned_adaptive),
            **adaptive_page_options(tuned=tuned_adaptive))
    elif (fixed_page_scan is not None
          and launch['pages_per_slot'] == 2
          and not pool.state['page_cache'].get('leaf_quant_bits', 0)):
        from benchmarks.h100_threshold_fixed_pages import fixed_two_page_attention
        page_attention = functools.partial(fixed_two_page_attention, small_scan=fixed_page_scan)
    elif pool.state['page_cache'].get('leaf_quant_bits', 0) == 4 and factor_int4_page_scales:
        from lod_attention.kernels.paged_prefill import query_major_indexed_residual_page_attention
        page_attention = functools.partial(query_major_indexed_residual_page_attention, factor_int4_page_scales=True)
    return shared_attention(
        pool, q, k, v, out, multiplier, splits=splits, rule=rule,
        precise=True, cached_means=means, round_counts_bf16=True, page_attention=page_attention, **launch)
