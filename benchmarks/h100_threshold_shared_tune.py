"""Graph-time shared-GQA and indexed H100 decode on identical real caches.

Times the entire attention call and identical local-length rollback. Model
layers and cache maintenance are excluded here and measured separately.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.environ.get('RETROINFER_ROOT', '/home/ubuntu/comparison/RetroInfer'))
from config import generate_config
from model_hub import load_model, load_tokenizer
from benchmarks.retro_lod_adapter import attach_lod
from benchmarks.h100_threshold_decode import threshold_attention


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--length', type=int, required=True)
    p.add_argument('--layers', type=int, nargs='+', default=[0, 15, 31])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--four-stage', action='store_true')
    p.add_argument('--rules', nargs='+', choices=('region', 'entry'), default=['region', 'entry'])
    a = p.parse_args()
    torch.set_grad_enabled(False)
    os.environ.pop('LOD_H100_THRESHOLD_MULTIPLIER', None)
    if a.four_stage:
        os.environ['LOD_H100_THRESHOLD_IMPLEMENTATION'] = 'materialized'
    name = 'meta-llama/Llama-3.1-8B-Instruct'
    tok = load_tokenizer(name)
    model = load_model(name, a.length + 1024, torch.bfloat16, 'cuda:0', tok)
    attach_lod(model, 'two-tier')
    ids = json.loads(a.manifest.read_text())['contexts'][str(a.length)]['prompts'][0]['prompt_token_ids']
    inputs = torch.tensor([ids], device='cuda')
    model.attention_type = 'LoD'
    model.batch_size = model.prefill_bsz = 1
    model.input_length = a.length
    model.max_new_length = 1024
    model.prefill_method = 'full'
    model.init_kv_cache(np.zeros(1, dtype=int), generate_config(name, a.length, 'Full_Flash_Attn'))
    token = model.sampling(model.prefill_forward(inputs_ids=inputs))
    model.move()
    cache = model.kv_cache
    original = cache.decode_attention
    saved = {}

    def capture(this, q, k, v, layer):
        result = original(q, k, v, layer)
        if layer in a.layers:
            saved[layer] = q.clone(), k.clone(), v.clone(), this.pools[layer]
            this.pools[layer].local_lens.sub_(1)
        return result

    cache.decode_attention = types.MethodType(capture, cache)
    model.sampling(model.decode_forward(inputs_ids=token))
    torch.cuda.synchronize()
    records = []
    for layer in a.layers:
        q, k, v, pool = saved[layer]
        out = torch.empty_like(q)
        for rule in a.rules:
            expected = encoded_expected = None
            configs = [('indexed', 64, 32, merge) for merge in (0, 16, 32, 64)]
            configs += [('shared', splits, block, merge) for splits in (16, 32) for block in (64, 128) for merge in (16, 32)]
            if a.four_stage:
                configs = [('indexed', 64, 32, 0), ('indexed', 64, 32, 32)]
                if rule == 'region':
                    configs += [('materialized', splits, block, merge) for splits in (16, 32, 64)
                                for block in (64, 128) for merge in (16, 32)]
            for impl, splits, block, merge in configs:
                os.environ['LOD_H100_THRESHOLD_IMPLEMENTATION'] = impl
                os.environ['LOD_H100_SHARED_ENTRY_BLOCK'] = str(block)
                os.environ['LOD_H100_THRESHOLD_MERGE_BLOCK'] = str(merge)
                os.environ['LOD_AMD_THRESHOLD_ENTRY_BLOCK'] = str(block)
                os.environ['LOD_AMD_THRESHOLD_ENTRY_WARPS'] = '4'
                os.environ['LOD_AMD_THRESHOLD_MERGE_BLOCK'] = str(merge or 32)
                os.environ['LOD_AMD_THRESHOLD_PRECISE'] = '1'

                def invoke():
                    threshold_attention(pool, q, k, v, out, 16., tile=64, splits=splits, rule=rule, entry_bound=16)
                    pool.local_lens.sub_(1)

                invoke()
                torch.cuda.synchronize()
                result = out.clone()
                scratch = pool._threshold_scratch
                if scratch.get('mask_only'):
                    selection = scratch['allowed'].bool().clone()
                else:
                    counts = scratch['count'].clone()
                    selected = scratch['selected'].clone()
                    mask = torch.arange(pool.state_capacity, device=q.device)[None, :] < counts[:, None]
                    selection = torch.zeros_like(mask, dtype=torch.int32).scatter_add_(
                        1, torch.where(mask, selected, 0).long(), mask.to(torch.int32)).bool()
                encoded = selection
                counts = selection.sum(-1)
                if expected is None:
                    expected, encoded_expected = result, encoded
                if not torch.equal(encoded, encoded_expected):
                    raise RuntimeError('selected regions differ from indexed')
                delta = (result.float() - expected.float()).abs()
                maximum = float(delta.max().item())
                if not torch.isfinite(result).all() or maximum > .02:
                    raise RuntimeError(f'output differs from indexed: {impl} {rule} {maximum}')
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(16):
                        invoke()
                for _ in range(4):
                    graph.replay()
                torch.cuda.synchronize()
                measurements = []
                for _ in range(3):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(40):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    measurements.append(begin.elapsed_time(end) * 1000 / (40 * 16))
                sizes = pool.state['page_cache']['slot_lengths'][0].repeat_interleave(4, 0)
                entries = sizes * selection
                row = dict(layer=layer, rule=rule, implementation=impl, splits=splits, block=block, merge_block=merge,
                           microseconds=float(np.median(measurements)), repetitions_us=measurements,
                           selection_mismatches=0, max_abs_vs_indexed=maximum,
                           mean_regions=float(counts.float().mean()), mean_entries=float(entries.sum(-1).float().mean()),
                           mean_union_entries=float(scratch['flat_total'].float().mean()) if impl in ('shared', 'materialized') else None)
                records.append(row)
                print(json.dumps(row), flush=True)
                a.output.write_text(json.dumps(dict(stage='running', length=a.length, records=records), indent=2) + '\n')
                del graph
    a.output.write_text(json.dumps(dict(stage='complete', length=a.length, records=records,
        scope='fixed real query/cache; full attention plus local-length rollback; excludes model and cache catch-up'), indent=2) + '\n')


if __name__ == '__main__':
    main()
