"""Check AMD serving decode with changing GPU-resident cache-row mappings."""
import argparse
import json
import os
from pathlib import Path

import torch

from benchmarks._identity import benchmark_identity
from benchmarks.amd_threshold_decode import selected_region_mask
from benchmarks.amd_threshold_three_tier_check import fixture
from vllm_lod_plugin.pool import VLLMLayerLODPool


def use_two_tier_directory(pool, owned):
    page = pool.state['page_cache']
    page_capacity = page['page_indices'].size(2)
    kv_heads = pool.kv_heads
    roots = torch.full((1, kv_heads, pool.state_capacity, 2), -1, dtype=torch.int32)
    directory = torch.full((1, kv_heads, page_capacity, 64), -1, dtype=torch.int32)
    for head, regions in enumerate(owned):
        for slot, pages in enumerate(regions):
            for ordinal, start in enumerate(range(0, len(pages), 64)):
                chunk = pages[start:start + 64]
                roots[0, head, slot, ordinal] = chunk[0]
                directory[0, head, chunk[0], :len(chunk)] = torch.tensor(chunk)
    page.update(paged_page_directory=True, slot_pages=roots.cuda(), overflow_page_values=directory.cuda())


def check(dimension, group, mode, rule, reference_kernel=False):
    import types

    quantized = mode == 'three-int4'
    fixtures = [fixture(dimension, group, quantized, seed, boost_region=8 + seed % 3)
                for seed in range(10)]
    pools = [record[0] for record in fixtures]
    for index, (pool, _, _, _, owned, *_) in enumerate(fixtures):
        if mode == 'two':
            pool.settings.levels = 2
            use_two_tier_directory(pool, owned)
        pool.settings.kv_bits = 4 if quantized else 0
        pool.query_heads = pool.kv_heads * group
        pool.head_dim = dimension
        pool.max_requests = 1
        pool.decode_calls = 0
        pool.active_indices = torch.zeros(1, device='cuda', dtype=torch.long)
        pool.state_lens.fill_(0 if index == 0 else pool.state_capacity - index)
        pool.local_lens.fill_(41 + index)

    batch = types.SimpleNamespace(**vars(pools[0]))
    batch.max_requests = len(pools)
    batch.active_indices = torch.arange(8, device='cuda', dtype=torch.long)
    batch.state = {
        name: torch.cat([pool.state[name] for pool in pools])
        for name in pools[0].state if name != 'page_cache'
    }
    pages = [pool.state['page_cache'] for pool in pools]
    batch.state['page_cache'] = {
        name: torch.cat([page[name] for page in pages]) if torch.is_tensor(val) and val.ndim else val
        for name, val in pages[0].items()
    }
    batch.state_lens = torch.cat([pool.state_lens for pool in pools])
    batch.local_lens = torch.cat([pool.local_lens for pool in pools])

    if os.environ.get('LOD_AMD_THRESHOLD_CACHED_MEANS') == '1':
        if not reference_kernel:
            raise ValueError('cached means fixture requires the original single-request control')
        from lod_attention.kernels.paged_decode_kernels import materialize_page1_coarse_means

        state = batch.state
        shape = state['state_k'].shape
        offset = 5
        rows = state['state_k'].numel() // dimension
        arena_k = torch.empty((rows + offset, dimension), device='cuda', dtype=torch.bfloat16)
        arena_v = torch.empty_like(arena_k)
        bias = torch.empty(shape[:-1], device='cuda', dtype=torch.float16)
        materialize_page1_coarse_means(state['state_k'], state['state_v'], state['counts'],
                                      arena_k[offset:].view(shape), arena_v[offset:].view(shape), bias)
        state['page_cache'].update(unified_page1_k=arena_k, unified_page1_v=arena_v,
                                   unified_page1_coarse_offset=offset)

    q, k, v = (torch.cat([record[i] for record in fixtures[:8]]).squeeze(1) for i in (1, 2, 3))
    actual, expected = torch.empty_like(q), torch.empty_like(q)
    eight = types.SimpleNamespace(num_actual_tokens=8)
    one = types.SimpleNamespace(num_actual_tokens=1)
    lengths = batch.local_lens.clone()
    VLLMLayerLODPool.decode(batch, q, k, v, eight, actual)
    batch.local_lens.copy_(lengths)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        VLLMLayerLODPool.decode(batch, q, k, v, eight, actual)
    captured_scratch = batch._threshold_scratch
    captured_stats = getattr(batch, '_amd_page_threshold_stats', None)
    # A second launch shape must not release the storage captured above.
    batch.active_indices[0] = 9
    VLLMLayerLODPool.decode(batch, q[:1], k[:1], v[:1], one, actual[:1])
    batch.local_lens.copy_(lengths)
    flags = {}
    if reference_kernel:
        flags = {name: os.environ.pop(name) for name in
                 ('LOD_AMD_THRESHOLD_FIXED_SCAN', 'LOD_AMD_THRESHOLD_CACHED_MEANS')
                 if name in os.environ}
    max_abs = 0.0
    for mapping in ((8, 2, 1, 9, 3, 0, 5, 4), (0, 6, 3, 4, 7, 5, 9, 1), (7, 4, 8, 5, 1, 3, 0, 2)):
        batch.active_indices.copy_(torch.tensor(mapping, device='cuda'))
        for x in (q, k, v):
            x.normal_()
        q[:, 0, 0] = 16
        graph.replay()
        captured = types.SimpleNamespace(_threshold_scratch=captured_scratch)
        masks = selected_region_mask(captured).reshape(8, batch.query_heads, batch.state_capacity)
        for idx, slot in enumerate(mapping):
            row = slice(idx, idx + 1)
            pool = pools[slot]
            VLLMLayerLODPool.decode(pool, q[row], k[row], v[row], one, expected[row])
            torch.testing.assert_close(masks[idx], selected_region_mask(pool), atol=0, rtol=0)
            if mode == 'adaptive':
                stats = captured_stats.view(8, batch.query_heads, -1, 4)
                torch.testing.assert_close(stats[idx].sum(1), pool._amd_page_threshold_stats.sum(1), atol=0, rtol=0)
        torch.testing.assert_close(actual, expected, atol=.008, rtol=.008)
        torch.testing.assert_close(batch.local_lens, torch.cat([pool.local_lens for pool in pools]), atol=0, rtol=0)
        for name in ('recent_k', 'recent_v'):
            # The warmup and capture both wrote the initial append location.
            # Compare exactly the positions that the replays have made live.
            for slot, pool in enumerate(pools):
                begin = 41 + slot
                end = int(pool.local_lens[0])
                torch.testing.assert_close(batch.state[name][slot:slot+1, :, begin:end],
                                           pool.state[name][:, :, begin:end], atol=0, rtol=0)
        max_abs = max(max_abs, float((actual.float() - expected.float()).abs().max()))
    os.environ.update(flags)
    return dict(dimension=dimension, group=group, mode=mode, rule=rule,
                reference_kernel=reference_kernel,
                batch_size=8, pool_size=len(pools), graph_replays=3, max_abs=max_abs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('two', 'three-bf16', 'three-int4', 'adaptive'), required=True)
    parser.add_argument('--rule', choices=('region', 'entry'), default='region')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-kernel', action='store_true',
                        help='Compare batched optimized decode with the original single-request page kernel')
    args = parser.parse_args()
    os.environ['LOD_AMD_THRESHOLD_MULTIPLIER'] = '16'
    os.environ['LOD_AMD_THRESHOLD_RULE'] = args.rule
    if args.mode == 'adaptive':
        os.environ['LOD_AMD_THRESHOLD_PAGE_MULTIPLIER'] = '0.5'
    torch.set_grad_enabled(False)
    records = []
    for dimension, group in ((128, 8), (256, 6)):
        row = check(dimension, group, args.mode, args.rule, args.reference_kernel)
        records.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(dict(stage='complete', records=records,
        hardware=str(torch.cuda.get_device_properties(0)), benchmark_identity=benchmark_identity()), indent=2)+'\n')


if __name__ == '__main__':
    main()
