"""Check independent request addressing in the batched H100 decode ports."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmarks.h100_threshold_decode_materialized import threshold_attention_h100_materialized


def threshold_pool(rows, states, local_lens):
    batch, kv_heads, slots, dim = rows['mean_k'].shape
    local_cap = rows['recent_k'].size(2)
    leaf_cap = rows['leaf_k'].size(2)
    sink_start = batch * kv_heads * (leaf_cap + local_cap)
    local_start = batch * kv_heads * leaf_cap
    mean_start = sink_start + batch * kv_heads

    def arena(suffix):
        order = ('leaf', 'recent', 'sink', 'mean')
        return torch.cat([rows[f'{k}_{suffix}'].reshape(-1, dim) for k in order])

    arena_k, arena_v = arena('k'), arena('v')
    counts = rows['counts']
    starts = torch.nn.functional.pad(counts.squeeze(-1).cumsum(dim=-1, dtype=torch.int32), (1, 0))
    local_limit = 12
    fixed_begin = slots + local_limit + 1
    physical = torch.arange(batch * kv_heads * leaf_cap, dtype=torch.int32, device=arena_k.device)
    fixed = torch.nn.functional.pad(physical.view(batch, kv_heads, leaf_cap), (fixed_begin, 0), value=-1)
    page = {
        'unified_page1_k': arena_k, 'unified_page1_v': arena_v,
        'unified_page1_local_offset': local_start, 'unified_page1_sink_offset': sink_start,
        'unified_page1_coarse_offset': mean_start, 'unified_page1_fixed_indices': fixed,
        'unified_page1_fixed_slot_offsets': starts, 'unified_page1_fixed_leaf_owners': None,
        'leaf_k': rows['leaf_k'], 'slot_lengths': counts.squeeze(-1).to(torch.int32),
    }
    state = {
        'state_k': rows['mean_k'], 'counts': counts,
        'recent_k': arena_k[local_start:sink_start].view(batch, kv_heads, local_cap, dim),
        'recent_v': arena_v[local_start:sink_start].view(batch, kv_heads, local_cap, dim),
        'sink_k': rows['sink_k'], 'page_cache': page,
    }
    return SimpleNamespace(
        state=state, state_lens=states.to(torch.int32), local_lens=local_lens.to(torch.int32),
        state_capacity=slots, kv_heads=kv_heads, decode_local_limit=local_limit, dtype=arena_k.dtype,
        engine=SimpleNamespace(scaling=dim**-0.5, local_len=local_limit),
        settings=SimpleNamespace(levels=2),
    )


def check_threshold(*, compare_reference=False):
    batch, kv_heads, slots, dim = 8, 8, 67, 128
    states = torch.arange(slots-batch+1, slots+1, device='cuda')
    local = torch.arange(7, 7+batch, device='cuda')
    counts = torch.randint(1, 10, (batch, kv_heads, slots, 1), device='cuda')
    counts *= torch.arange(slots, device='cuda')[None, None, :, None] < states[:, None, None, None]
    counts[0].zero_()
    counts[1, :, 0] = 1024
    counts[2, :, 0] = 1025

    def tensor(n):
        return torch.randn(batch, kv_heads, n, dim, device='cuda', dtype=torch.bfloat16)

    rows = {'counts': counts.float()}
    for name, size in (('mean', slots), ('leaf', slots*9+1025), ('recent', 24), ('sink', 1)):
        rows[f'{name}_k'], rows[f'{name}_v'] = tensor(size), tensor(size)
    rows['mean_k'][1:, :, 0, 0] += 16
    pool = threshold_pool(rows, states, local)
    singles = [threshold_pool({k: v[i:i+1] for k, v in rows.items()}, states[i:i+1], local[i:i+1])
               for i in range(batch)]
    q = torch.randn(batch, 1, 32, dim, device='cuda', dtype=torch.bfloat16)
    k, v = tensor(1).transpose(1, 2).contiguous(), tensor(1).transpose(1, 2).contiguous()
    actual, expected = torch.empty_like(q), torch.empty_like(q)

    optimized = os.environ.get('LOD_H100_TWO_TIER_OPTIMIZED') == '1'
    tile = int(os.environ.get('LOD_H100_THRESHOLD_TILE', '64'))
    splits = int(os.environ.get('LOD_H100_THRESHOLD_SPLITS', '8'))

    def decode(pool, q, k, v, out, append, *, reference=False):
        threshold_attention_h100_materialized(
            pool, q, k, v, out, 2.0, tile=64 if reference else tile,
            splits=splits, append=append, optimized=optimized and not reference)

    decode(pool, q, k, v, actual, False)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode(pool, q, k, v, actual, True)
    max_abs = 0.0
    for step in range(3):
        q.normal_(); k.normal_(); v.normal_()
        q[1:, 0, :, 0] = 16
        graph.replay()
        for i, single in enumerate(singles):
            decode(single, q[i:i+1], k[i:i+1], v[i:i+1], expected[i:i+1], True, reference=compare_reference)
            torch.testing.assert_close(pool._threshold_scratch['allowed'].view(batch, 32, slots)[i],
                                       single._threshold_scratch['allowed'], atol=0, rtol=0)
        max_abs = max(max_abs, float((actual.float()-expected.float()).abs().max()))
        tol = .008 if compare_reference else 0
        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
        torch.testing.assert_close(pool.local_lens.long(), local + step + 1)
        allowed = pool._threshold_scratch['allowed'].view(batch, 32, slots)
        assert allowed[1, :, 0].all() and not allowed[2, :, 0].any()
        for i, single in enumerate(singles):
            for field in ('recent_k', 'recent_v'):
                torch.testing.assert_close(pool.state[field][i:i+1], single.state[field], atol=0, rtol=0)
    print('LoD batch graph matches eight independent rows with distinct queries and cache lengths.', flush=True)
    return dict(stage='complete', batch_size=batch, graph_replays=3, max_abs=max_abs,
                reference_checked=compare_reference)


def check_squeezed():
    from benchmarks.retro_squeezed_adapter import RetroSqueezedCache
    from benchmarks.retro_squeezed_batch_adapter import PackedSqueezedBatchCache

    batch, heads, kv_heads, dim, prefix, obs, clusters = 8, 32, 8, 128, 260, 8, 27
    model = SimpleNamespace(batch_size=1, num_gpus=1, num_layers=1, num_heads=heads,
                            num_key_value_heads=kv_heads, head_dim=dim, device_map='cuda:0',
                            input_length=prefix, max_new_length=5, dtype=torch.bfloat16)

    def tensor(*shape):
        return torch.randn(shape, device=model.device_map, dtype=model.dtype)

    index = {'centroids': {0: tensor(1, kv_heads, clusters, dim)},
             'labels': {0: torch.randint(clusters, (1, kv_heads, prefix-obs), device='cuda')},
             'threshold': {0.9: 0.004}, 'observation_window': obs, 'routing_position': 2048}
    prefix_k, prefix_v = tensor(1, kv_heads, prefix, dim), tensor(1, kv_heads, prefix, dim)
    independent = [RetroSqueezedCache(model, index, 0.9) for _ in range(batch)]
    for cache in independent:
        cache.prefill_update_kv_cache(None, prefix_k.transpose(1, 2), prefix_v.transpose(1, 2), 0, 0)
    flat_index = dict(index, centroids={0: index['centroids'][0].repeat(1, batch, 1, 1)},
                      labels={0: index['labels'][0].repeat(1, batch, 1)})
    batch_model = SimpleNamespace(**(vars(model) | {'batch_size': batch}))
    actual = PackedSqueezedBatchCache(batch_model, flat_index, 0.9)
    actual.install_prefix([(prefix_k, prefix_v, None, None)])
    q, route, k, v = tensor(batch, 1, heads, dim), tensor(batch, 1, heads, dim), tensor(batch, 1, kv_heads, dim), tensor(batch, 1, kv_heads, dim)
    actual.current_window_q = route
    actual.decode_attention(q, k, v, 0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = actual.decode_attention(q, k, v, 0)
    for step in range(4):
        for tensor_input in (q, route, k, v):
            tensor_input.normal_()
        actual._set_position(prefix + step)
        graph.replay()
        for i, cache in enumerate(independent):
            cache.current_window_q = route[i:i+1]
            cache._set_position(prefix + step)
            expected = cache.decode_attention(q[i:i+1], k[i:i+1], v[i:i+1], 0)
            torch.testing.assert_close(actual.head_count.view(batch, heads)[i], cache.head_count[0],
                                       atol=0, rtol=0)
            torch.testing.assert_close(out[i:i+1], expected, atol=0.004, rtol=0.004)
    print('Squeezed batch graph matches eight independent controllers on distinct queries and appends.', flush=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threshold-only', action='store_true')
    parser.add_argument('--compare-reference', action='store_true', help='Compare the optimized batch with baseline independent requests')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_grad_enabled(False)
    record = check_threshold(compare_reference=args.compare_reference)
    if not args.threshold_only:
        check_squeezed()
    if args.output:
        args.output.write_text(json.dumps(record, indent=2)+'\n')
