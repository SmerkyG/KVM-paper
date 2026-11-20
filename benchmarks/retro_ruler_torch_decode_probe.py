"""RULER examples with optimized LoD prefill and the pure PyTorch decode frontier.

The optimized cache supplies the exact same stored regions and local entries;
the reference computes attention without using the optimized decode output.
"""
import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.environ.get('RETROINFER_ROOT', '/home/ubuntu/comparison/RetroInfer'))
from config import generate_config
from model_hub import load_model, load_tokenizer
from benchmarks.retro_lod_adapter import attach_lod
from benchmarks.retro_ruler_run import save, score


def reference_cache(pool, total_length):
    from lod_attention.pytorch_engine import PytorchLODCache, PytorchLODState
    state = pool.state
    page = state['page_cache']
    nstate = int(pool.state_lens[0].item())
    nleaf = int(pool.leaf_lens[0].item())
    counts = state['counts'][:, :, :nstate, 0]
    if not torch.all(counts.sum(-1) == nleaf):
        raise RuntimeError(f'region counts do not cover physical archive: {nleaf}, {counts.sum(-1)}')
    nheads = pool.kv_heads
    ids = page['slot_pages'][0, :, :nstate, :].long()
    page_owner = torch.full((nheads, pool.page_capacity), -1, dtype=torch.long, device=ids.device)
    page_ordinal = torch.full_like(page_owner, -1)
    h, slot, ordinal = (ids >= 0).nonzero(as_tuple=True)
    page_owner[h, ids[h, slot, ordinal]] = slot
    page_ordinal[h, ids[h, slot, ordinal]] = ordinal
    keys = page['overflow_page_keys'][0].long()
    vals = page['overflow_page_values'][0].long()
    h, x = ((keys >= 0) & (vals >= 0)).nonzero(as_tuple=True)
    page_owner[h, vals[h, x]] = torch.div(keys[h, x], 65536, rounding_mode='floor')
    page_ordinal[h, vals[h, x]] = keys[h, x] % 65536
    indices = page['page_indices'][0].long()
    pcount = page['page_counts'][0]
    valid = (torch.arange(16, device=ids.device)[None, None, :] < pcount[:, :, None]) & (page_owner[:, :, None] >= 0)
    h, p, j = valid.nonzero(as_tuple=True)
    leaf = indices[h, p, j]
    if torch.any((leaf < 0) | (leaf >= nleaf)):
        raise RuntimeError('invalid page leaf index')
    owner = torch.full((nheads, nleaf), -1, dtype=torch.long, device=ids.device)
    owner[h, leaf] = page_owner[h, p]
    if torch.any(owner < 0) or h.numel() != nheads * nleaf:
        raise RuntimeError('page directory does not represent every leaf exactly once')
    actual = torch.zeros_like(counts[0]).scatter_add_(1, owner, torch.ones_like(owner, dtype=counts.dtype))
    if not torch.equal(actual, counts[0]):
        raise RuntimeError('page ownership differs from region populations')
    from lod_attention.kernels.paged_cache import stable_owner_ranks
    ranks = stable_owner_ranks(owner[None])[0]
    if not torch.equal(ranks[h, leaf] // 16, page_ordinal[h, p]) or not torch.equal(ranks[h, leaf] % 16, j):
        raise RuntimeError('reference chronological pages differ from cached page ordering')
    return PytorchLODCache(
        PytorchLODState(state['state_k'][:, :, :nstate, :], state['state_v'][:, :, :nstate, :], counts,
                        torch.zeros_like(counts)),
        owner[None], page['leaf_k'][:, :, :nleaf, :], page['leaf_v'][:, :, :nleaf, :],
        torch.ones(1, nleaf, device=ids.device, dtype=torch.bool), nleaf,
        state['sink_k'], state['sink_v'], total_length)


def cache_native_route(model):
    return model.kv_cache.native_route_four_sink


def install_reference(model, comparison):
    from lod_attention.pytorch_engine import PytorchLODAttention
    reference = PytorchLODAttention(mode='three-tier-bf16',
        normalize_routing_query=not cache_native_route(model))
    cache = model.kv_cache
    cache._reference_states = {}
    cache._reference_checks = []
    original_decode = cache.decode_attention

    def decode(this, q, k, v, layer_idx):
        pool = this.pools[layer_idx]
        # The normal kernel appends the current token to the local KV cache.
        # Its attention output is measured for diagnostics and then discarded.
        optimized = original_decode(q, k, v, layer_idx)
        signature = (int(pool.leaf_lens[0].item()), int(pool.state_lens[0].item()))
        previous = this._reference_states.get(layer_idx)
        if previous is None or previous[0] != signature:
            previous = (signature, reference_cache(pool, this.context))
            this._reference_states[layer_idx] = previous
        ref_cache = previous[1]
        local_len = int(pool.local_lens[0].item())
        q_ref = q.transpose(1, 2)
        ref_out = reference._lod_attention(q_ref, pool.state['recent_k'][:, :, :local_len],
            pool.state['recent_v'][:, :, :local_len], ref_cache,
            scale=1.0 / math.sqrt(pool.head_dim), query_offset=local_len-1,
            full_regions=False).output.transpose(1, 2)
        diff = (optimized.float() - ref_out.float()).abs()
        if comparison and len(this._reference_checks) < 32:
            this._reference_checks.append({'layer': layer_idx, 'max_abs': float(diff.max()),
                                           'mean_abs': float(diff.mean())})
        return ref_out
    cache.decode_attention = types.MethodType(decode, cache)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rows', type=int, nargs='+', default=[46, 59, 78])
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--lod-native-route-four-sink', action='store_true')
    args = p.parse_args()
    if args.lod_native_route_four_sink:
        os.environ['LOD_RETRO_NATIVE_ROUTE_FOUR_SINK'] = '1'
    cohort = json.loads(args.cohort.read_text())
    prior = json.loads(args.prior.read_text())
    originals = {(r['task'], r['source_row']): r for r in prior['records']}
    samples = [next(s for s in cohort['samples'] if s['task'] == 'niah_multikey_3' and s['source_row'] == row) for row in args.rows]
    torch.set_grad_enabled(False)
    name = 'meta-llama/Llama-3.1-8B-Instruct'
    tokenizer = load_tokenizer(name)
    model = load_model(name, 65536, torch.bfloat16, 'cuda:0', tokenizer)
    attach_lod(model, 'three-tier-bf16')
    result = {'stage': 'running', 'model': name, 'task': 'niah_multikey_3',
              'prefill': 'optimized three-tier LoD',
              'decode': 'pure PyTorch reference attention over optimized region/page cache',
              'native_route_four_sink': args.lod_native_route_four_sink,
              'cohort_sha256': hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
              'prior': str(args.prior), 'records': []}
    save(args.output, result)
    for sample in samples:
        started = time.monotonic()
        task, row = sample['task'], sample['source_row']
        original = originals[(task, row)]
        if original['score'] != 0:
            raise RuntimeError(f'original selected example was not incorrect: {row}')
        inputs = tokenizer(sample['prompt'], return_tensors='pt')['input_ids'].to('cuda')
        input_length = inputs.shape[1]
        token_bytes = struct.pack(f'<{input_length}I', *inputs[0].tolist())
        if hashlib.sha256(token_bytes).hexdigest() != original['input_token_sha256']:
            raise RuntimeError('input token IDs differ from original run')
        model.attention_type = 'LoD'
        model.batch_size = model.prefill_bsz = 1
        model.input_length = input_length
        model.max_new_length = args.max_new_tokens
        model.prefill_method = 'full'
        config = generate_config(name, input_length, 'Full_Flash_Attn')
        model.init_kv_cache(np.zeros(1, dtype=int), config)
        token = model.sampling(model.prefill_forward(inputs_ids=inputs))
        if not args.lod_native_route_four_sink and token.item() != original['output_token_ids'][0]:
            raise RuntimeError('sparse prefill first token differs from original run')
        model.move()
        install_reference(model, comparison=True)
        generated = [token.clone()]
        print(f'row={row} prefill verified; beginning reference decode', flush=True)
        for step in range(args.max_new_tokens - 1):
            token = model.sampling(model.decode_forward(inputs_ids=token))
            generated.append(token.clone())
            ids = torch.cat(generated, dim=-1)[0].tolist()
            text = tokenizer.decode(ids, skip_special_tokens=True)
            print(json.dumps({'row': row, 'step': step+1, 'token': int(token.item()),
                              'elapsed_s': round(time.monotonic()-started, 2), 'text': text}), flush=True)
            if token.item() in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')):
                break
        ids = torch.cat(generated, dim=-1)[0].tolist()
        prediction = tokenizer.decode(ids, skip_special_tokens=True)
        result['records'].append({'row': row, 'input_tokens': input_length, 'answers': sample['answers'],
            'prediction': prediction, 'score': score(task, prediction, sample['answers']),
            'token_ids': ids, 'original_prediction': original['prediction'],
            'original_token_ids': original['output_token_ids'],
            'first_step_layer_differences': model.kv_cache._reference_checks,
            'elapsed_s': round(time.monotonic()-started, 2)})
        save(args.output, result)
        del model.kv_cache
        torch.cuda.empty_cache()
    result['stage'] = 'complete'
    save(args.output, result)


if __name__ == '__main__':
    main()
