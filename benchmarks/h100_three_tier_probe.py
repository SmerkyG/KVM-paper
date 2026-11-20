"""Tune the complete three-tier attention call on real Llama queries and KV.

All variants graph the same attention-call scope and local-length rollback.
Fixed-cache timings exclude the model and cache updates; amortized model
timings are required separately. Top-eight uses the serving decoder unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.environ.get('RETROINFER_ROOT', '/home/ubuntu/comparison/RetroInfer'))
from config import generate_config
from model_hub import load_model, load_tokenizer
from benchmarks.retro_lod_adapter import attach_lod


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--length', type=int, required=True)
    p.add_argument('--mode', choices=('three-tier-bf16', 'three-tier-int4'), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_grad_enabled(False)
    os.environ['LOD_H100_THRESHOLD_MULTIPLIER'] = '16'
    os.environ['LOD_H100_THRESHOLD_IMPLEMENTATION'] = 'three-tier'
    name = 'meta-llama/Llama-3.1-8B-Instruct'
    tok = load_tokenizer(name)
    model = load_model(name, a.length + 2048, torch.bfloat16, 'cuda:0', tok)
    attach_lod(model, a.mode)
    ids = json.loads(a.manifest.read_text())['contexts'][str(a.length)]['prompts'][0]['prompt_token_ids']
    inputs = torch.tensor([ids], device='cuda')
    model.attention_type = 'LoD'
    model.batch_size = model.prefill_bsz = 1
    model.input_length = a.length
    model.max_new_length = 2048
    model.prefill_method = 'full'
    model.init_kv_cache(np.zeros(1, dtype=int), generate_config(name, a.length, 'Full_Flash_Attn'))
    token = model.sampling(model.prefill_forward(inputs_ids=inputs))
    model.move()
    cache = model.kv_cache
    original = cache.decode_attention
    saved = {}

    def capture(this, q, k, v, layer):
        result = original(q, k, v, layer)
        if layer in (0, 15, 31):
            saved[layer] = q.clone(), k.clone(), v.clone(), this.pools[layer]
            this.pools[layer].local_lens.sub_(1)
        return result

    cache.decode_attention = types.MethodType(capture, cache)
    model.sampling(model.decode_forward(inputs_ids=token))
    torch.cuda.synchronize()
    from benchmarks.h100_threshold_three_tier import threshold_attention
    records = []
    # The uncached control and every cached candidate select identical regions.
    configs = [('top8', 32, 8, 1, 32), ('three-tier-reference', 64, 64, 1, 32)]
    configs += [('three-tier', tile, workers, 1, 32)
                for tile in (32, 64, 128) for workers in (32, 64, 128)]
    configs += [('three-tier', 64, 64, 2, 32), ('three-tier', 64, 64, 1, 16),
                ('three-tier', 128, 128, 2, 32), ('three-tier', 128, 256, 1, 32)]
    for layer in (0, 15, 31):
        q, k, v, pool = saved[layer]
        out = torch.empty_like(q)
        reference = expected_selection = None
        # Verify the install/catch-up hook against current serving sums.
        divisor = pool.state['counts'].float().to(torch.bfloat16).float().clamp_min(1)
        active = torch.arange(pool.state_capacity, device=q.device)[None, None, :, None] < pool.state_lens
        for means, key in zip(pool._h100_three_tier_means, ('state_k', 'state_v')):
            expected = (pool.state[key].float() / divisor).to(torch.bfloat16)
            if not torch.equal(means.masked_select(active), expected.masked_select(active)):
                raise RuntimeError('cached region means are stale after serving catch-up')
        for implementation, tile, workers, warps, merge in configs:
            os.environ['LOD_H100_THRESHOLD_IMPLEMENTATION'] = implementation
            os.environ['LOD_H100_THREE_TIER_WORKERS'] = str(workers)
            os.environ['LOD_H100_THREE_TIER_PAGE_WARPS'] = str(warps)
            os.environ['LOD_H100_THRESHOLD_MERGE_BLOCK'] = str(merge)

            def invoke():
                if implementation == 'top8':
                    pool.decode(q.reshape(1, 32, 128), k.reshape(1, 8, 128), v.reshape(1, 8, 128),
                                types.SimpleNamespace(num_actual_tokens=1, max_seq_len=a.length+1),
                                out.reshape(1, 32, 128))
                else:
                    threshold_attention(pool, q, k, v, out, 16., tile=tile, splits=workers)
                pool.local_lens.sub_(1)

            invoke()
            torch.cuda.synchronize()
            maximum = mean_regions = mean_region_entries = mean_mass = None
            if implementation != 'top8':
                selection = pool._threshold_scratch['allowed'].bool().clone()
                if reference is None:
                    reference, expected_selection = out.clone(), selection
                if not torch.equal(selection, expected_selection):
                    raise RuntimeError('threshold region selection changed during tuning')
                maximum = float((out.float() - reference.float()).abs().max())
                if not bool(torch.isfinite(out).all()) or maximum > .008:
                    raise RuntimeError('cached pipeline differs from uncached control')
                mean_regions = float(selection.sum(-1).float().mean())
                sizes = pool.state['page_cache']['slot_lengths'][0].repeat_interleave(4, 0)
                mean_region_entries = float((sizes * selection).sum(-1).float().mean())
                mean_mass = float((pool._threshold_scratch['route'].softmax(-1) * selection).sum(-1).mean())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(8):
                    invoke()
            for _ in range(4):
                graph.replay()
            times = []
            for _ in range(3):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(40):
                    graph.replay()
                end.record()
                end.synchronize()
                times.append(begin.elapsed_time(end) * 1000 / 320)
            row = dict(layer=layer, implementation=implementation, tile=tile, workers=workers,
                       page_warps=warps, merge_block=merge, microseconds=statistics.median(times),
                       repetitions_us=times, max_abs_vs_uncached=maximum, selection_mismatches=0,
                       mean_opened_regions=mean_regions, mean_opened_region_entries=mean_region_entries,
                       mean_selected_proxy_mass=mean_mass)
            records.append(row)
            print(json.dumps(row), flush=True)
            a.output.write_text(json.dumps(dict(stage='running', mode=a.mode, length=a.length, records=records), indent=2)+'\n')
            del graph
    a.output.write_text(json.dumps(dict(stage='complete', mode=a.mode, length=a.length, records=records,
        scope='CUDA graph of complete attention call and local-length rollback; excludes model and cache updates'), indent=2)+'\n')


if __name__ == '__main__':
    main()
