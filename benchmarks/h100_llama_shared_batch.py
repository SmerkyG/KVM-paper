"""Time batched Llama decoders with dense prefill and a separate LoD arm."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import struct
import time
from types import SimpleNamespace

import numpy as np
import torch

from benchmarks.h100_comparison import configure_prefill_mlp_chunk
from benchmarks.h100_qwen_shared_decode import (
    GPU_SQUEEZED_P80, LLAMA, MULTIPOLE_P90, OBS, calibrate_multipole_index, capture_prefill, load_shared_model,
    read_cases, repeat_decode_arms, run_decode, run_multipole_methods,
)
from benchmarks.retro_lod_batch_adapter import attach_lod_batch
from benchmarks.retro_ruler_run import save
from transformers import AutoTokenizer


METHODS = ['lod', 'dense', 'multipole-512', MULTIPOLE_P90]
MODEL_HOOKS = ('init_kv_cache', 'prefill_attention', 'decode_attention', 'move', 'position_embedd')


def configure_model(model, batch, length, new_tokens):
    model.batch_size = batch
    model.prefill_bsz = 1
    model.input_length = length
    model.max_new_length = new_tokens
    model.prefill_method = 'full'


def release_cache(model, original):
    model.kv_cache = None
    for key, method in original.items():
        setattr(model, key, method)
    gc.collect()
    torch.cuda.empty_cache()


def install_dense(model, layers):
    model.attention_type = 'Full_Flash_Attn'
    model.init_kv_cache(np.zeros(model.batch_size, dtype=int), {})
    cache = model.kv_cache
    if not cache.allocated:
        raise RuntimeError('dense must keep all request KV on the GPU')
    for layer, (k, v, _, _) in enumerate(layers):
        k, v = k.transpose(1, 2).cuda(), v.transpose(1, 2).cuda()
        for request in range(model.batch_size):
            cache.prefill_update_kv_cache(None, k, v, layer, request)
            cache.sync(layer, request)
    if cache.context != model.input_length:
        raise RuntimeError('dense cache context differs from the common prefill')


def reused_index(cache, observations):
    # All requests use the same prefix. Calibrate on its single initial
    # index, then replicate metadata in request-major order for decode.
    heads = cache.request_kv_heads
    state = [(c[:, :heads], i[:, :heads, :cache.clustered], n[:, :heads])
             for c, i, n in zip(cache.centroids, cache.labels, cache.counts, strict=True)]
    initial = SimpleNamespace(cluster_state=state, sink=cache.sink, prefix=cache.prefix,
                              clustered=cache.clustered, max_clusters=cache.max_clusters)
    index = calibrate_multipole_index(initial, observations)
    return index


def repeat_index(index, batch):
    index['centroids'] = {layer: c.repeat(1, batch, 1, 1) for layer, c in index['centroids'].items()}
    index['labels'] = {layer: labels.repeat(1, batch, 1) for layer, labels in index['labels'].items()}


def decode_author_batch(model, data, args):
    from multipoleattention import kmeans_ops_sequential

    batch = data['batch_size']
    # The author's reduce-overhead distance wrapper keeps private graph pools
    # for growing refinement shapes. Keep compilation and release that scratch
    # after each update. The whole-model decode graph is independent of this.
    distance = kmeans_ops_sequential._sq_euclid
    kmeans_ops_sequential._sq_euclid = torch.compile(
        distance._torchdynamo_orig_callable, options={'triton.cudagraphs': False})
    configure_model(model, batch, len(data['ids']), data['max_new_length'])
    original = {k: getattr(model, k) for k in MODEL_HOOKS}
    inputs = torch.tensor([data['ids']] * batch, device='cuda')
    first = torch.full((batch, 1), data['first'], device='cuda', dtype=torch.long)
    starts = np.zeros(batch, dtype=int)

    def run_arms():
        return decode_author_batch_once(model, data, args, original, inputs, first, starts)

    if getattr(args, 'matched_full_model_timing', False):
        return repeat_decode_arms(data['methods'], run_arms)
    return run_arms()


def decode_author_batch_once(model, data, args, original, inputs, first, starts):
    from benchmarks.retro_multipole_batch_adapter import attach_multipole_batch
    from benchmarks.retro_squeezed_adapter import attach_squeezed

    results = {}
    index = None
    methods = data['methods']
    if 'multipole-512' in methods:
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        attach_multipole_batch(model, budget=512, fixed_centroids=args.multipole_centroids)
        model.attention_type = 'RetroMultipole'
        model.init_kv_cache(starts, {})
        cache = model.kv_cache
        cache.install_repeated_prefix(data['layers'])
        torch.cuda.synchronize()
        setup = time.perf_counter() - start
        initial = cache.fixed_centroids or cache.clustered // cache.tokens_per_centroid
        initial_indexed = cache.clustered
        if MULTIPOLE_P90 in methods:
            index = reused_index(cache, data['routing_observations'])
            repeat_index(index, batch=model.batch_size)
        arm = run_decode(model, 'multipole-512', first, inputs, data['task'], args)
        arm.update(cache_setup_s=setup, clustering_distance_cuda_graph=False,
                   initial_centroids_per_kv_head=initial, clustered_tokens_on_install=initial_indexed,
                   centroids_per_kv_head=cache.fixed_centroids or cache.clustered // cache.tokens_per_centroid,
                   index_updates=cache.update_count,
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        results['multipole-512'] = arm
        del cache
        release_cache(model, original)

    for method in (MULTIPOLE_P90, GPU_SQUEEZED_P80):
        if method not in methods:
            continue
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        if method == GPU_SQUEEZED_P80:
            from benchmarks.h100_squeezed_index import build_batched_gpu

            index = build_batched_gpu(data['layers'], OBS, requested_clusters=args.squeezed_centroids)
            repeat_index(index, batch=model.batch_size)
        attach_squeezed(model, index, int(method[-2:]) / 100, packed_batch=True)
        model.attention_type = 'RetroSqueezed'
        model.init_kv_cache(starts, {})
        model.kv_cache.install_prefix(data['layers'])
        torch.cuda.synchronize()
        setup = time.perf_counter() - start
        sq = run_decode(model, method, first, inputs, data['task'], args)
        sq.update(cache_setup_s=setup, reused_initial_clusters=method == MULTIPOLE_P90,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), squeezed_index={
            k: index[k] for k in ('clusters', 'threshold', 'seconds', 'calibration_tokens', 'observation_window',
                                  'exact_sink_tokens', 'routing_position', 'fit_backend', 'indexed_tokens') if k in index
        })
        results[method] = sq
        index = None
        release_cache(model, original)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--length', type=int, default=32768)
    p.add_argument('--timing-prefix-length', type=int, default=30720)
    p.add_argument('--warmup-steps', type=int, default=512)
    p.add_argument('--windows', type=int, default=3)
    p.add_argument('--steps-per-window', type=int, default=512)
    p.add_argument('--multipole-centroids', type=int, default=2900)
    p.add_argument('--independent-squeezed', action='store_true', help='Use the native-key p80 index')
    p.add_argument('--squeezed-centroids', type=int, help='Override the 5% independent index width')
    p.add_argument('--paper-multipole', action='store_true', help='Use the online 1:16 index')
    p.add_argument('--matched-full-model-timing', action='store_true',
                   help='Time only Multipole and Squeezed on the whole prompt with fresh decode caches')
    a = p.parse_args()
    a.total_steps = a.warmup_steps + a.windows * a.steps_per_window
    if not (a.batch_size >= 2 and a.warmup_steps > 0 and a.windows > 0 and a.steps_per_window > 0
            and a.total_steps < a.timing_prefix_length <= a.length):
        p.error('invalid batch size or timing window')
    if a.matched_full_model_timing:
        if a.timing_prefix_length != a.length or (a.warmup_steps, a.windows, a.steps_per_window) != (256, 1, 1024):
            p.error('--matched-full-model-timing requires the full prompt, 256 warmup steps and one 1,024-step window')
    elif a.timing_prefix_length + a.total_steps > a.length:
        p.error('legacy timing windows must end within the context length')
    required = {'LOD_H100_THRESHOLD_IMPLEMENTATION': 'materialized', 'LOD_H100_THRESHOLD_RULE': 'region',
                'LOD_H100_PREFILL_RULE': 'region', 'LOD_H100_PREFILL_IMPLEMENTATION': 'fast'}
    if not a.matched_full_model_timing and any(os.environ.get(key) != v for key, v in required.items()):
        p.error('the LoD arm requires materialized threshold decode and fast threshold prefill')
    a.cohort = None
    a.methods = METHODS[2:] if a.matched_full_model_timing else METHODS
    if not a.matched_full_model_timing and (a.independent_squeezed or a.paper_multipole):
        p.error('paper settings require full-model timing')
    if a.squeezed_centroids is not None and (not a.independent_squeezed or a.squeezed_centroids < 1):
        p.error('--squeezed-centroids requires --independent-squeezed and a positive count')
    if a.independent_squeezed:
        a.methods = ['multipole-512', GPU_SQUEEZED_P80]
    if a.paper_multipole:
        a.multipole_centroids = None
    a.profile_quality = False
    a.profile_subset_kernels = False
    a.prewarm_multipole_worker = False
    a.stop_tokens = ()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(LLAMA)
    case, = read_cases(a, tokenizer)
    ids, task = case['ids'], case['task']
    digest = hashlib.sha256(struct.pack(f'<{len(ids)}I', *ids)).hexdigest()
    model = load_shared_model(LLAMA, max(65536, len(ids) + a.total_steps + 1), tokenizer, None)
    configure_prefill_mlp_chunk(model)
    original = {k: getattr(model, k) for k in MODEL_HOOKS}
    inputs = torch.tensor([ids], dtype=torch.long, device='cuda')
    configure_model(model, 1, len(ids), a.total_steps + 1)
    prefill_start = time.perf_counter()
    first, layers, observations = capture_prefill(
        model, inputs, True, False, True, None, capture_routing_observations=not a.independent_squeezed,
        capture_native_observations=a.independent_squeezed)
    shared_prefix_s = time.perf_counter() - prefill_start
    release_cache(model, original)

    record = dict(input_tokens=len(ids), input_token_sha256=digest, decoders={})
    result = dict(stage='running', model_name=LLAMA, model_revision=model.config._commit_hash,
                  batch_size=a.batch_size, dtype='bfloat16', context_limit=a.length,
                  timing_prefix_length=len(ids), warmup_steps=a.warmup_steps, methods=a.methods,
                  prefill='dense causal' if a.matched_full_model_timing else 'method-specific',
                  prefill_by_method={method: ('LoD threshold' if method == 'lod' else 'dense causal')
                                     for method in a.methods},
                  records=[record])
    if a.multipole_centroids is not None:
        result['multipole_fixed_centroids_per_kv_head'] = a.multipole_centroids
    if a.squeezed_centroids is not None:
        result['squeezed_fixed_centroids_per_kv_head'] = a.squeezed_centroids
    decoders = record['decoders']
    configure_model(model, a.batch_size, len(ids), a.total_steps + 1)
    if a.matched_full_model_timing:
        forced = json.dumps(ids[-a.total_steps:], separators=(',', ':')).encode()
        record['teacher_forced_token_sha256'] = hashlib.sha256(forced).hexdigest()
        record['shared_prefix_preparation_s'] = shared_prefix_s
        result.update(matched_full_model_timing=True, timed_decode_steps=a.steps_per_window,
                      measurement_warmups=1, measurement_repeats=2)
    else:
        inputs = inputs.repeat(a.batch_size, 1)
        first = first.repeat(a.batch_size, 1)
        attach_lod_batch(model, 'two-tier')
        model.attention_type = 'LoD'
        model.init_kv_cache(np.zeros(a.batch_size, dtype=int), {})
        repeated_first = model.sampling(model.prefill_forward(inputs_ids=inputs))
        for pool in model.kv_cache.pools:
            pool.catch_up_many([(request, len(ids)) for request in range(a.batch_size)])
        decoders['lod'] = run_decode(model, 'lod', repeated_first, inputs, task, a)
        save(a.output, result)
        print('LoD', decoders['lod']['median_decode_ms'], flush=True)
        release_cache(model, original)
        install_dense(model, layers)
        decoders['dense'] = run_decode(model, 'dense', first, inputs, task, a)
        save(a.output, result)
        print('Dense', decoders['dense']['median_decode_ms'], flush=True)
        release_cache(model, original)

    # The author runtime needs almost the full GPU at B=8. Transfer prefill
    # state through tmpfs and let it load weights after releasing ours.
    worker_config = SimpleNamespace(**{
        k: getattr(model, k) for k in ('batch_size', 'model_name', 'max_length', 'max_new_length')
    })
    del model, original
    gc.collect()
    torch.cuda.empty_cache()
    decoders.update(run_multipole_methods(worker_config, layers, ids, first, task, a, None, None, observations))
    result['stage'] = 'complete'
    save(a.output, result)
    for method, props in decoders.items():
        print(method, props['median_decode_ms'], flush=True)


if __name__ == '__main__':
    main()
