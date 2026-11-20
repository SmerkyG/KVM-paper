"""Time the three-tier decode attention call on real batch-eight Llama caches."""

import argparse
import json
import os
from pathlib import Path
import statistics
import types

import numpy as np
import torch
from config import generate_config
from model_hub import load_model, load_tokenizer

from benchmarks.retro_lod_batch_adapter import attach_lod_batch


def time_attention(invoke):
    invoke()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(8):
            invoke()
    for _ in range(6):
        graph.replay()
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    times = []
    for _ in range(3):
        begin.record()
        for _ in range(24):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(begin.elapsed_time(end) * 1000 / (24 * 8))
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--mode', choices=('three-tier-bf16', 'three-tier-int4'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tune-fixed', action='store_true', help='Tune region and page kernels with fixed two-page refinement')
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    batch, length, tokens = 8, 32768, 1281
    name = 'meta-llama/Llama-3.1-8B-Instruct'
    tokenizer = load_tokenizer(name)
    model = load_model(name, length + tokens, torch.bfloat16, 'cuda:0', tokenizer)
    attach_lod_batch(model, args.mode)
    ids = json.loads(args.manifest.read_text())['contexts'][str(length)]['prompts'][0]['prompt_token_ids']
    inputs = torch.tensor([ids], device='cuda').expand(batch, -1).contiguous()
    model.attention_type = 'LoD'
    model.batch_size, model.prefill_bsz = batch, 1
    model.input_length, model.max_new_length, model.prefill_method = length, tokens, 'full'
    model.init_kv_cache(np.zeros(batch, dtype=int), generate_config(name, length, 'Full_Flash_Attn'))
    token = model.sampling(model.prefill_forward(inputs_ids=inputs))
    model.move()
    cache = model.kv_cache
    original = cache.decode_attention
    saved = {}

    def capture(this, q, k, v, layer):
        pool = this.pools[layer]
        result = original(q, k, v, layer)
        if layer in (0, 15, 31):
            saved[layer] = q.contiguous(), k.contiguous(), v.contiguous(), pool
        return result

    cache.decode_attention = types.MethodType(capture, cache)
    model.decode_forward(inputs_ids=token)
    for _, _, _, pool in saved.values():
        pool.local_lens.sub_(1)
    torch.cuda.synchronize()
    from benchmarks.h100_threshold_three_tier import threshold_attention

    if args.tune_fixed:
        configs = [(False, workers, page_warps, 16, tile, route_warps)
                   for tile, route_warps in ((32,2),(64,2),(64,4),(128,2),(128,4),(256,4))
                   for workers, page_warps in ((16,1),(32,1),(48,1),(32,2))]
    else:
        configs = [(False, workers, warps, block, 64, 4)
                   for block in (32, 16) for warps in (1, 2) for workers in (64, 32, 16, 8)]
        if args.mode == 'three-tier-bf16':
            configs += [(True, workers, warps, 0, 64, 4)
                        for warps in (2, 4) for workers in (32, 16, 8)]
    result = {'stage': 'running', 'batch_size': batch, 'prompt_tokens': length, 'mode': args.mode, 'records': []}
    for layer, (q, k, v, pool) in saved.items():
        print(f'Layer {layer}, default page block {pool.engine.leaf_block_n}', flush=True)
        output = torch.empty_like(q)
        references = {}
        for adaptive, workers, warps, block, tile, route_warps in configs:
            os.environ['LOD_H100_THREE_TIER_WORKERS'] = str(workers)
            os.environ['LOD_H100_THREE_TIER_PAGE_WARPS'] = str(warps)
            os.environ['LOD_H100_PAGE_THRESHOLD_WARPS'] = str(warps)
            os.environ['LOD_H100_PAGE_QUERY'] = 'normalized'
            os.environ['LOD_H100_THREE_TIER_ROUTE_WARPS'] = str(route_warps)
            if adaptive:
                os.environ['LOD_H100_PAGE_THRESHOLD'] = '0.5'
            else:
                os.environ.pop('LOD_H100_PAGE_THRESHOLD', None)
                os.environ['LOD_H100_THREE_TIER_PAGE_BLOCK'] = str(block)

            def invoke():
                threshold_attention(pool, q, k, v, output, 16, tile=tile)
                pool.local_lens.sub_(1)

            microseconds = time_attention(invoke)
            scratch = pool._threshold_scratch
            route_mask = scratch['allowed']
            page_work = pool._h100_page_threshold_stats.sum(1) if adaptive else None
            expected, routed, expected_work = references.setdefault(adaptive, (output.clone(), route_mask.clone(), page_work))
            torch.testing.assert_close(route_mask, routed, rtol=0, atol=0)
            torch.testing.assert_close(output, expected, rtol=.008, atol=.008)
            record = dict(layer=layer, adaptive=adaptive,
                          workers=workers, warps=warps, page_block=block, region_tile=tile, route_warps=route_warps,
                          microseconds=microseconds,
                          max_abs=float((output.float() - expected.float()).abs().max()),
                          mean_opened_regions=float(scratch['count'].float().mean()))
            if adaptive:
                torch.testing.assert_close(page_work, expected_work, rtol=0, atol=0)
                record['mean_opened_entries'] = float(page_work[:, 3].float().mean())
            result['records'].append(record)
            print(json.dumps(record), flush=True)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
    result['stage'] = 'complete'
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
