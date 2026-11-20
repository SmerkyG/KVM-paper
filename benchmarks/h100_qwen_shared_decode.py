"""Run connected decoders on a dense prompt pass, with separate LoD prefill.

Multipole can share the dense KV tensors with an author-compatible worker.
"""
import argparse
import atexit
from collections import Counter
from contextlib import contextmanager, nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import types

import numpy as np
import torch
from transformers import AutoTokenizer

from benchmarks.h100_comparison import configure_prefill_mlp_chunk
from benchmarks.h100_threshold_next_report import cleanup
from benchmarks.retro_lod_adapter import attach_lod
from benchmarks.retro_native_graph import NativeDecodeGraph
from benchmarks.retro_qwen3_adapter import RetroQwen3Model
from benchmarks.retro_ruler_run import LIMITS, save, score
from model_hub.llama import LlamaModel


MODEL = 'Qwen/Qwen3-8B'
LLAMA = 'meta-llama/Llama-3.1-8B-Instruct'
YARN = {'rope_type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings': 32768}
OBS = 100
MULTIPOLE_P90 = 'multipole-p90'
GPU_SQUEEZED_P80 = 'gpu-squeezed-p80'
GPU_SQUEEZED_5PCT_P80 = 'gpu-squeezed-5pct-p80'
GPU_SQUEEZED_METHODS = (GPU_SQUEEZED_P80, GPU_SQUEEZED_5PCT_P80)
SQUEEZED_WORKER_METHODS = (MULTIPOLE_P90, *GPU_SQUEEZED_METHODS)
MULTIPOLE_ROUTING_POSITION = 2048
MULTIPOLE_METHOD_PREFIXES = ('subset-', 'multipole-')
WORKER_METHOD_PREFIXES = (*MULTIPOLE_METHOD_PREFIXES, 'gpu-squeezed-')
MULTIPOLE_BUDGET_ARMS = ('multipole-512', 'multipole-12000')
METHODS = ('lod', 'dense', 'squeezed-p90', 'squeezed-p80', 'subset-512', 'subset-2048',
           'subset-squeezed-512', *MULTIPOLE_BUDGET_ARMS, *SQUEEZED_WORKER_METHODS,
           'squeezed-1531-p90', 'squeezed-2900-p90')


def load_shared_model(model_name, max_length, tokenizer, rope):
    if model_name == LLAMA:
        return LlamaModel(model_name, max_length, torch.bfloat16, 'cuda:0', tokenizer)
    return RetroQwen3Model(model_name, max_length, torch.bfloat16, 'cuda:0', tokenizer, rope_scaling=rope)


def read_cases(args, tokenizer):
    if args.cohort is None:
        manifest = json.loads(args.manifest.read_text())
        ids = manifest['contexts'][str(args.length)]['prompts'][0]['prompt_token_ids']
        if len(ids) != args.length:
            raise ValueError('timing prompt length differs from manifest')
        yield {'ids': ids[:args.timing_prefix_length], 'task': 'prolong-timing'}
        return
    cohort = json.loads(args.cohort.read_text())
    counts = Counter()
    for row in cohort['samples']:
        task = row['task']
        if task not in args.tasks or counts[task] >= args.limit_per_task:
            continue
        if hashlib.sha256(row['prompt'].encode()).hexdigest() != row['prompt_sha256']:
            raise ValueError('cohort prompt hash differs')
        shard = counts[task] % args.cohort_shards
        counts[task] += 1
        if shard != args.cohort_shard:
            continue
        yield {**{k: row[k] for k in ('task', 'source_row', 'index', 'answers', 'prompt_sha256')},
               'ids': tokenizer(row['prompt'])['input_ids']}
    if any(counts[task] != args.limit_per_task for task in args.tasks):
        raise ValueError(f'incomplete quality cohort: {counts}')


def index_for(layers, args, clusters=None):
    home = Path.home()
    digest = hashlib.sha256(Path(__file__).with_name('h100_squeezed_index.py').read_bytes())
    if clusters is not None:
        digest.update(f'clusters={clusters}'.encode())
    for k, _, q, _ in layers:
        for x in (k, q):
            digest.update(memoryview(x.contiguous().view(torch.uint8).numpy()))
    cache_path = args.index_dir / (digest.hexdigest() + '.pt') if args.index_dir else None
    if cache_path is not None and cache_path.exists():
        return torch.load(cache_path, map_location='cpu', weights_only=True), True
    with tempfile.TemporaryDirectory(prefix='qwen-squeezed-', dir='/dev/shm') as d:
        source, output = Path(d) / 'qk.pt', Path(d) / 'index.pt'
        torch.save({'layers': [(k, q) for k, _, q, _ in layers]}, source)
        command = [str(home / '.local/bin/uv'), 'run', '--offline', '--no-project', '--python',
                   str(home / 'comparison/.squeezed-venv/bin/python'), 'python', '-m',
                   'benchmarks.h100_squeezed_index', '--input', str(source), '--output', str(output)]
        if clusters is not None:
            command += ['--clusters', str(clusters)]
        print('Building fixed prefix index:', command, flush=True)
        subprocess.run(command, check=True)
        index = torch.load(output, map_location='cpu', weights_only=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp = cache_path.with_suffix('.tmp')
        torch.save(index, temp)
        temp.replace(cache_path)
    return index, False


@contextmanager
def shareable_cuda_allocations():
    # PyTorch 2.5 cannot export tensors in expandable CUDA allocator segments.
    settings = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
    torch.cuda.memory._set_allocator_settings('expandable_segments:False')
    try:
        yield
    finally:
        torch.cuda.memory._set_allocator_settings(settings)


def capture_prefill(model, inputs, capture_host, need_index, need_raw, profile, gpu_ipc=False,
                    capture_routing_observations=False, *, capture_native_observations=False, lod=False):
    start = time.perf_counter()
    if lod:
        if capture_host or need_index or need_raw or gpu_ipc or capture_routing_observations or capture_native_observations:
            raise ValueError('LoD prefill is reserved for LoD decoding')
        attach_lod(model, 'two-tier')
        model.attention_type = 'LoD'
    else:
        model.attention_type = 'Full_Flash_Attn'
    model.init_kv_cache(np.zeros(1, dtype=int), {})
    layers = []
    routing_observations = []
    prefill = model.prefill_attention
    position = model.position_embedd
    current_raw = None
    capture_events = []
    attention_events = []

    def timed_call(events, fn, *args):
        if profile is None:
            return fn(*args)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        result = fn(*args)
        end.record()
        events.append((begin, end))
        return result

    def to_host(x):
        return x.transpose(1, 2).contiguous().cpu()

    def share(x):
        return x.transpose(1, 2).clone(memory_format=torch.contiguous_format)

    def capture_position(this, q, k):
        nonlocal current_raw
        if capture_routing_observations:
            with shareable_cuda_allocations() if gpu_ipc else nullcontext():
                positions = torch.full((1, OBS), MULTIPOLE_ROUTING_POSITION, device=q.device, dtype=torch.long)
                queries, _ = this.apply_rotary_pos_emb(q[:, -OBS:].clone(), k[:, -OBS:].clone(), positions)
                queries = queries.view(1, OBS, model.num_heads, model.head_dim).transpose(1, 2).contiguous()
            routing_observations.append(queries if gpu_ipc else queries.cpu())
        keys = k.reshape(1, inputs.shape[1], model.num_key_value_heads, model.head_dim)
        if gpu_ipc:
            with shareable_cuda_allocations():
                current_raw = share(keys)
        else:
            current_raw = timed_call(capture_events, to_host, keys)
        return position(q, k)

    def capture_layer(q, k, v):
        observations = to_host(q[:, -OBS:]) if need_index or capture_native_observations else None
        layers.append((to_host(k), to_host(v), observations, current_raw))

    def capture(this, q, k, v, layer):
        if capture_host:
            if layer != len(layers):
                raise RuntimeError('unexpected layer capture order')
            timed_call(capture_events, capture_layer, q, k, v)
        elif gpu_ipc:
            with shareable_cuda_allocations():
                observations = share(q[:, -OBS:]) if capture_native_observations else None
                layers.append((share(k), share(v), observations, current_raw))
        return timed_call(attention_events, prefill, q, k, v, layer)

    if need_raw:
        model.position_embedd = types.MethodType(capture_position, model)
    if capture_host or gpu_ipc or profile is not None:
        model.prefill_attention = types.MethodType(capture, model)
    if profile is not None:
        torch.cuda.synchronize()
        prefill_start = time.perf_counter()
        profile['lod_cache_allocation_s'] = prefill_start - start
    try:
        token = model.sampling(model.prefill_forward(inputs_ids=inputs)).clone()
    finally:
        model.prefill_attention = prefill
        model.position_embedd = position
    if profile is not None:
        torch.cuda.synchronize()
        completion_start = time.perf_counter()
        profile['prefill_and_capture_s'] = completion_start - prefill_start
        profile['prefill_capture_s'] = sum(begin.elapsed_time(end) for begin, end in capture_events) / 1000
        profile['prefill_attention_and_state_s'] = sum(begin.elapsed_time(end) for begin, end in attention_events) / 1000
    if lod:
        for pool in model.kv_cache.pools:
            pool.catch_up(0, model.kv_cache.context)
    torch.cuda.synchronize()
    if profile is not None:
        profile['prefill_cache_completion_s'] = time.perf_counter() - completion_start
        if lod:
            profile['lod_cache_completion_s'] = profile['prefill_cache_completion_s']
    return token, layers, routing_observations


def install_decode(model, method, layers, index, *, multipole_centroids=None, initial_clusters=None):
    if method == 'lod':
        return
    if method == 'dense':
        model.attention_type = 'Full_Flash_Attn'
    elif method.startswith('squeezed-') or method in SQUEEZED_WORKER_METHODS:
        from benchmarks.retro_squeezed_adapter import attach_squeezed
        attach_squeezed(model, index, int(method[-2:]) / 100)
        model.attention_type = 'RetroSqueezed'
    else:
        from benchmarks.retro_multipole_adapter import attach_multipole
        replacement = method.startswith('multipole-')
        attach_multipole(model, budget=int(method.rsplit('-', 1)[-1]),
                         replacement=replacement, tokens_per_centroid=16 if replacement else 8,
                         fixed_centroids=multipole_centroids,
                         squeezed_kernels=method.startswith('subset-squeezed-'))
        model.attention_type = 'RetroMultipole'
    model.init_kv_cache(np.zeros(1, dtype=int), {})
    cache = model.kv_cache
    for layer, (k, v, _, raw) in enumerate(layers):
        cache_args = {}
        if method.startswith(MULTIPOLE_METHOD_PREFIXES) and method != MULTIPOLE_P90:
            cache.current_raw_k = raw.transpose(1, 2).contiguous().cuda()
            if initial_clusters is not None:
                cache_args['cluster_state'] = initial_clusters[layer]
        cache.prefill_update_kv_cache(None, k.transpose(1, 2).cuda(), v.transpose(1, 2).cuda(), layer, 0, **cache_args)
        cache.sync(layer, 0)
    model.move()
    if cache.context != model.input_length:
        raise RuntimeError('decode cache did not receive the complete shared prefix')


def run_decode(model, method, first, inputs, task, args):
    cache = model.kv_cache
    start = time.perf_counter()
    graph = NativeDecodeGraph(model, first, model.attention_type) if method == 'dense' else None
    if graph is None:
        cache.capture_decode(model, first)
    step = graph.step if graph is not None else cache.graph_step
    checked = graph.checked_step if graph is not None else lambda token: cache.checked_graph_step(model, token)
    start_token = first if args.cohort else inputs[:, -args.total_steps:-args.total_steps + 1]
    token, check = checked(start_token)
    if check['graph_argmax'] != check['eager_argmax'] or check['max_abs'] != 0:
        raise RuntimeError(f'{method} graph/eager difference: {check}')
    result = {'graph_check': check}
    generation_start = time.perf_counter()
    if args.cohort:
        generated = [first.clone(), token.clone()]
        stop_tokens = getattr(args, 'stop_tokens', ())
        audit = getattr(args, 'audit_squeezed_selections', False) and method in GPU_SQUEEZED_METHODS
        if audit:
            if cache.selection_counts is None:
                raise RuntimeError('selection auditing requires SQUEEZED_REPORT_SELECTION=1')
            selections = [(1, cache.selection_counts.clone())]
        for _ in range(LIMITS[task] - 2):
            token = step(token)
            generated.append(token.clone())
            if audit and (len(generated) - 1) % 8 == 0:
                selections.append((len(generated) - 1, cache.selection_counts.clone()))
            if stop_tokens and len(generated) % 8 == 0:
                recent = torch.cat(generated[-8:], dim=-1)[0].tolist()
                if any(t in stop_tokens for t in recent):
                    break
        result['output_token_ids'] = torch.cat(generated, dim=-1)[0].tolist()
        if audit:
            result['squeezed_selection_samples'] = [
                {'output_index': i, 'mean_prefix_tokens': counts.float().mean().item()}
                for i, counts in selections
            ]
        if args.profile_quality:
            result['quality_profile'] = dict(graph_and_check_s=generation_start - start,
                                             generation_s=time.perf_counter() - generation_start)
    else:
        start = inputs.shape[1] - args.total_steps
        capture_tokens = getattr(args, 'matched_full_model_timing', False)
        token_ids = [first, token.clone()] if capture_tokens else None
        for i in range(1, args.warmup_steps):
            token = step(inputs[:, start + i:start + i + 1])
            if token_ids is not None:
                token_ids.append(token.clone())
        samples = []
        def updates():
            if method == 'lod' and model.batch_size > 1:
                return sum(pool.catch_up_rows for pool in cache.pools)
            return getattr(cache, 'update_count', getattr(cache, 'catch_up_events', 0))
        def centroid_slots():
            if method == 'lod':
                lengths = [int(row['state_len']) for row in cache.pools[-1].metadata[:model.batch_size]]
                if min(lengths) != max(lengths):
                    raise RuntimeError('repeated prompt produced different centroid counts')
                return lengths[0]
            if method.startswith('squeezed-') or method in SQUEEZED_WORKER_METHODS:
                return cache.clusters
            if method.startswith(MULTIPOLE_METHOD_PREFIXES):
                return cache.max_clusters if cache.fixed_centroids is not None else cache.clustered // cache.tokens_per_centroid
            return None

        is_squeezed = method.startswith('squeezed-') or method in SQUEEZED_WORKER_METHODS
        for i in range(args.windows):
            begin = start + args.warmup_steps + i * args.steps_per_window
            torch.cuda.synchronize()
            before = updates()
            centroid_slots_start = centroid_slots()
            update_times = getattr(cache, 'update_times', ())
            time_index = len(update_times)
            t0 = time.perf_counter()
            for j in range(begin, begin + args.steps_per_window):
                token = step(inputs[:, j:j + 1])
                if token_ids is not None:
                    token_ids.append(token.clone())
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            samples.append(dict(seconds=elapsed, steps=args.steps_per_window,
                                cache_update_events=updates() - before))
            if centroid_slots_start is not None:
                samples[-1]['last_layer_centroid_slots_per_kv_head'] = [centroid_slots_start, centroid_slots()]
            if len(update_times) > time_index:
                samples[-1]['cache_update_seconds'] = update_times[time_index:]
            if is_squeezed and cache.selection_counts is not None:
                counts = cache.selection_counts.cpu()
                samples[-1]['selected_prefix_tokens_per_query_head'] = dict(
                    mean=float(counts.float().mean()), min=int(counts.min()), max=int(counts.max()))
        result.update(windows=samples, median_decode_ms=statistics.median(1000 * s['seconds'] / s['steps'] for s in samples))
        if token_ids is not None:
            by_request = torch.cat(token_ids, dim=-1).tolist()
            result['output_token_ids'] = by_request[0] if model.batch_size == 1 else by_request
    if method.startswith(('squeezed-', *MULTIPOLE_METHOD_PREFIXES)):
        result['last_layer_selected_prefix_counts'] = cache.head_count.flatten().tolist()
    expected = model.input_length + (len(generated) - 1 if args.cohort else args.total_steps)
    if cache.context != expected:
        raise RuntimeError(f'{method} decode context {cache.context} != {expected}')
    if graph is not None:
        graph.close()
    return result


def profile_squeezed_lookup(cache, layers, widths):
    """Time only centroid scoring using a shared observation query per layer."""
    from benchmarks.multipole_subset_profile import graph_us
    from benchmarks.retro_squeezed_adapter import centroid_scores

    queries = [q[:, :, -1:, :].contiguous().cuda() for _, _, q, _ in layers]
    results = {}
    for n in widths:
        centers = [c[:, :, :n].contiguous() for c in cache.centroids]
        counts = [c[:, :, :n].contiguous() for c in cache.cluster_counts]
        scores = torch.empty((1, cache.heads, n, 1), device=cache.device, dtype=torch.float32)

        def lookup():
            for q, c, count, labels in zip(queries, centers, counts, cache.labels, strict=True):
                centroid_scores(q, c, count, labels, cache.scale, scores, scores, cache.threshold,
                                block_m=cache.block_m, num_warps=cache.num_warps)

        results[str(n)] = graph_us(lookup, repeats=16) / 1000
    return results


def multipole_worker(payload, output):
    """Run the authors' compiled clustering and graph in their Triton runtime."""
    import triton
    if triton.__version__ != '3.1.0':
        raise RuntimeError(f'Multipole worker requires the base Triton 3.1.0; found {triton.__version__}')
    start = time.perf_counter()
    if payload.suffix == '.ipc':
        import pickle
        with payload.open('rb') as stream:
            data = pickle.load(stream)
    else:
        data = torch.load(payload, map_location='cpu', weights_only=True, mmap=True)
    print(f'Multipole worker PID {os.getpid()}', flush=True)
    out = decode_multipole_payload(data, start)
    # Release imported CUDA IPC tensors while the producer still owns them.
    # Some of these are reachable through cycles in the graph and model.
    del data
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    save(output, dict(stage='complete', decoders=out))


def calibrate_multipole_index(cache, observations):
    """Fit the released p90 cutoff on the initial Multipole routing index."""
    from benchmarks.h100_squeezed_index import calibration_scores

    start = time.perf_counter()
    centroids, labels, samples = {}, {}, []
    for layer, ((c, assignments, counts), q) in enumerate(zip(cache.cluster_state, observations, strict=True)):
        scores = calibration_scores(q[0].cuda(), c[0].float(), assignments[0], counts[0])
        samples.append(scores.cpu().numpy().reshape(-1))
        centroids[layer] = c
        labels[layer] = torch.nn.functional.pad(assignments, (cache.sink, 0), value=c.size(2))
    threshold = float(np.quantile(np.concatenate(samples), 0.9))
    if not 0 < threshold < 1:
        raise RuntimeError(f'invalid p90 threshold {threshold}')
    return dict(centroids=centroids, labels=labels, threshold={0.9: threshold},
                observation_window=cache.prefix - cache.sink - cache.clustered,
                calibration_tokens=OBS, exact_sink_tokens=cache.sink,
                routing_position=MULTIPOLE_ROUTING_POSITION, clusters=cache.max_clusters,
                seconds=time.perf_counter() - start)


def decode_multipole_payload(data, start):
    args = types.SimpleNamespace(**data['options'])
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    model_name = data.get('model_name', MODEL)
    if state := data.get('model_state'):
        model_type = LlamaModel if model_name == LLAMA else RetroQwen3Model
        model = model_type.__new__(model_type)
        model.__dict__.update(state)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = load_shared_model(model_name, data['max_length'], tokenizer, data['rope_scaling'])
    model.input_length = len(data['ids'])
    model.max_new_length = data['max_new_length']
    if data.get('batch_size', 1) > 1:
        from benchmarks.h100_llama_shared_batch import decode_author_batch
        return decode_author_batch(model, data, args)
    model.batch_size = model.prefill_bsz = 1
    model.prefill_method = 'full'
    original = {k: getattr(model, k) for k in ('init_kv_cache', 'decode_attention', 'move', 'position_embedd')}
    inputs = torch.tensor([data['ids']], device='cuda', dtype=torch.long)
    first = torch.tensor([[data['first']]], device='cuda', dtype=torch.long)
    torch.cuda.synchronize()
    worker_setup_s = time.perf_counter() - start
    def run_arms():
        return decode_multipole_arms(model, data, args, original, inputs, first, worker_setup_s)

    if getattr(args, 'matched_full_model_timing', False):
        return repeat_decode_arms(data['methods'], run_arms)
    return run_arms()


def repeat_decode_arms(methods, run_arms):
    repetitions = {method: [] for method in methods}
    for iteration in range(3):
        torch.manual_seed(0)
        for method, arm in run_arms().items():
            arm.update(iteration=iteration, warmup=iteration == 0)
            repetitions[method].append(arm)
            print(json.dumps(dict(method=method, iteration=iteration, warmup=arm['warmup'],
                                  decode_ms=arm['median_decode_ms'])), flush=True)
    return {method: dict(repetitions=rows,
                         median_decode_ms=statistics.median(r['median_decode_ms'] for r in rows[1:]))
            for method, rows in repetitions.items()}


def decode_multipole_arms(model, data, args, original, inputs, first, worker_setup_s):
    out = {}
    initial_indexes = {}
    methods = data['methods']
    paired_subset = 'subset-squeezed-512' in methods and 'subset-512' in methods
    paired_budgets = all(method in methods for method in MULTIPOLE_BUDGET_ARMS)
    fit_only = MULTIPOLE_P90 in methods and 'multipole-512' not in methods
    # Calibrate on the same initial index even when only Squeezed decodes.
    run_order = ['multipole-512', *methods] if fit_only else methods
    p90_index = None
    for method in run_order:
        for k, v in original.items():
            setattr(model, k, v)
        start = time.perf_counter()
        index_pair = None
        if paired_subset and method in ('subset-512', 'subset-squeezed-512'):
            index_pair = 'subset'
        elif paired_budgets and method in MULTIPOLE_BUDGET_ARMS:
            index_pair = 'budget'
        initial_clusters = initial_indexes.get(index_pair)
        reused = initial_clusters is not None
        if method == MULTIPOLE_P90 and p90_index is None:
            raise RuntimeError('Multipole must supply the initial index before p90 decode')
        index = p90_index if method == MULTIPOLE_P90 else None
        if method in GPU_SQUEEZED_METHODS:
            from benchmarks.h100_squeezed_index import build_batched_gpu

            fit_seed = getattr(args, 'squeezed_fit_seed', None)
            if fit_seed is not None:
                torch.manual_seed(fit_seed)
            count = args.squeezed_centroids if method == GPU_SQUEEZED_P80 else None
            index = build_batched_gpu(data['layers'], OBS, requested_clusters=count)
        install_decode(model, method, data['layers'], index, multipole_centroids=args.multipole_centroids,
                       initial_clusters=initial_clusters if reused else None)
        if index_pair is not None and not reused:
            initial_indexes[index_pair] = tuple(model.kv_cache.cluster_state)
        torch.cuda.synchronize()
        cache_setup_s = time.perf_counter() - start
        if method == 'multipole-512' and MULTIPOLE_P90 in methods:
            p90_index = calibrate_multipole_index(model.kv_cache, data['routing_observations'])
            if fit_only:
                model.kv_cache = None
                gc.collect()
                torch.cuda.empty_cache()
                continue
        if index_pair is not None:
            torch.manual_seed(0)
        if method in MULTIPOLE_BUDGET_ARMS:
            initial_centroids = model.kv_cache.cluster_state[-1][0].size(2)
            initial_indexed_tokens = model.kv_cache.clustered
        model.kv_cache.profile_decode = args.profile_subset_kernels and method.startswith('subset-')
        decode_start = time.perf_counter()
        out[method] = run_decode(model, method, first, inputs, data['task'], args)
        if MULTIPOLE_P90 in data['methods']:
            out[method]['decode_and_graph_s'] = time.perf_counter() - decode_start
        if index_pair is not None:
            out[method]['reused_initial_clusters'] = reused
        if model.kv_cache.profile_decode:
            from benchmarks.multipole_subset_profile import profile
            out[method]['kernel_profile'] = profile(model.kv_cache)
        out[method]['cache_setup_s'] = cache_setup_s
        if method in SQUEEZED_WORKER_METHODS:
            out[method]['reused_initial_clusters'] = method == MULTIPOLE_P90
            out[method]['squeezed_index'] = {k: index[k] for k in (
                'clusters', 'threshold', 'seconds', 'calibration_tokens', 'observation_window',
                'exact_sink_tokens', 'routing_position', 'indexed_tokens', 'fit_backend') if k in index}
        elif method in MULTIPOLE_BUDGET_ARMS:
            out[method]['initial_centroids_per_kv_head'] = initial_centroids
            out[method]['clustered_tokens_on_install'] = initial_indexed_tokens
            out[method]['centroids_per_kv_head'] = model.kv_cache.cluster_state[-1][0].size(2)
            out[method]['index_updates'] = model.kv_cache.update_count
        out[method]['worker_pid'] = os.getpid()
        if args.profile_quality:
            out[method]['quality_profile']['worker_setup_s'] = worker_setup_s
        model.kv_cache = None
        gc.collect()
        torch.cuda.empty_cache()
    return out


def run_multipole_methods(model, layers, ids, first, task, args, rope, worker_model_state, routing_observations):
    # Torch 2.5 Inductor needs the author's Triton 3.1, whereas LoD kernels
    # need Triton 3.7. Preserve compilation and account for index updates
    # by handing the exact same prefix to a second process in this GPU job.
    home = Path.home()
    options = {k: getattr(args, k) for k in ('windows', 'steps_per_window', 'warmup_steps', 'total_steps',
                                           'profile_quality', 'stop_tokens', 'multipole_centroids', 'profile_subset_kernels')}
    options['cohort'] = args.cohort is not None
    options['matched_full_model_timing'] = getattr(args, 'matched_full_model_timing', False)
    options['squeezed_centroids'] = getattr(args, 'squeezed_centroids', None)
    options['squeezed_fit_seed'] = getattr(args, 'squeezed_fit_seed', None)
    options['audit_squeezed_selections'] = getattr(args, 'audit_squeezed_selections', False)
    with tempfile.TemporaryDirectory(prefix='qwen-multipole-', dir='/dev/shm') as d:
        ipc = layers[0][0].is_cuda
        payload, output = Path(d) / ('prefix.ipc' if ipc else 'prefix.pt'), Path(d) / 'decode.json'
        data = dict(layers=layers, ids=ids, first=int(first[0].item()), task=task,
            methods=[m for m in args.methods if m.startswith(WORKER_METHOD_PREFIXES)], options=options,
            max_length=model.max_length, max_new_length=model.max_new_length, rope_scaling=rope,
            batch_size=model.batch_size, model_name=model.model_name, model_state=worker_model_state,
            routing_observations=routing_observations)
        if ipc:
            from torch.multiprocessing.reductions import ForkingPickler
            with payload.open('wb') as stream:
                ForkingPickler(stream, 4).dump(data)
        else:
            torch.save(data, payload)
        command = [str(home / '.local/bin/uv'), 'run', '--offline', '--no-project', '--python',
            str(home / 'comparison/.venv/bin/python'), '--with', 'transformers==4.51.3',
            'python', '-m', 'benchmarks.h100_qwen_shared_decode',
            '--subset-payload', str(payload), '--output', str(output)]
        env = {k: v for k, v in os.environ.items() if k not in ('VIRTUAL_ENV', 'UV_PYTHON')}
        if args.prewarm_multipole_worker:
            worker_command = command[:-4] + ['--multipole-worker-stdin']
            worker = getattr(args, '_multipole_worker', None)
            if worker is None:
                worker = subprocess.Popen(worker_command, stdin=subprocess.PIPE, text=True, env=env)
            worker.stdin.write(json.dumps([str(payload), str(output)]) + '\n')
            worker.stdin.close()
            # Import the runtime for the next prompt while this worker decodes.
            # The next worker waits for its payload before touching the GPU.
            args._multipole_worker = subprocess.Popen(worker_command, stdin=subprocess.PIPE, text=True, env=env)
            code = worker.wait()
            if code:
                raise subprocess.CalledProcessError(code, worker.args)
        else:
            print('Decoding shared prefix in author-compatible runtime:', command, flush=True)
            subprocess.run(command, check=True, env=env)
        result = json.loads(output.read_text())
    assert result['stage'] == 'complete'
    return result['decoders']


def close_multipole_worker(worker):
    worker.stdin.close()
    try:
        worker.wait(timeout=15)
    except subprocess.TimeoutExpired:
        worker.kill()
        worker.wait()


def resume_quality(metadata_path, output, result):
    meta = json.loads(metadata_path.read_text())
    command = meta['command']
    prior_path = Path(command[command.index('--output') + 1])
    if prior_path.resolve() == output.resolve():
        raise ValueError('resumed run must use a new output file')
    raw = prior_path.read_bytes()
    prior = json.loads(raw)
    for key, value in result.items():
        if key == 'stop_at_eos' and value and not prior.get(key):
            # Scoring already cuts both generation policies at the first EOS.
            continue
        if key not in ('stage', 'records', 'multipole_worker') and prior.get(key) != value:
            raise ValueError(f'cannot resume with a different {key}')
    records = {(r['task'], r['source_row']): r for r in prior['records']}
    if len(records) != len(prior['records']) or not records:
        raise ValueError('checkpoint must contain unique completed quality records')
    result['resumed_from'] = dict(run_id=meta['id'], hardware=meta['hardware'],
                                  file=prior_path.name, sha256=hashlib.sha256(raw).hexdigest(),
                                  records=len(records))
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--length', type=int)
    p.add_argument('--model-name', choices=(MODEL, LLAMA), default=MODEL)
    p.add_argument('--subset-payload', type=Path)
    p.add_argument('--multipole-worker-stdin', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--output', type=Path)
    p.add_argument('--methods', nargs='+', choices=METHODS, default=['lod', 'dense', 'squeezed-p90'])
    p.add_argument('--cohort', type=Path)
    p.add_argument('--cohort-shards', type=int, default=1)
    p.add_argument('--cohort-shard', type=int, default=0,
                   help='Select this zero-based shard from each task in the frozen cohort')
    p.add_argument('--resume-from-run', type=Path,
                   help='Read a prior cluster-run run.json and reuse its completed matching quality records')
    p.add_argument('--tasks', nargs='+', choices=tuple(LIMITS), default=['niah_multikey_2', 'niah_multikey_3', 'cwe', 'qa_1'])
    p.add_argument('--limit-per-task', type=int, default=1)
    p.add_argument('--profile-quality', action='store_true')
    p.add_argument('--stop-at-eos', action='store_true',
                   help='Stop quality generation after the first EOS/EOT, checked every eight tokens')
    p.add_argument('--dense-kv-source', choices=('device', 'host', 'verify'),
                   help='Keep the native dense cache on the GPU, rebuild it from host KV, or check the two routes')
    p.add_argument('--manifest', type=Path, default=Path('/home/ubuntu/comparison/results/qwen-opt-v1-manifest-0-prepare.json'))
    p.add_argument('--timing-prefix-length', type=int, help='Use this many tokens from the timing manifest prefix')
    p.add_argument('--index-dir', type=Path)
    p.add_argument('--squeezed-centroids', type=int, help='Override the 5% index width for GPU Squeezed')
    p.add_argument('--squeezed-fit-seed', type=int, help='Reset the random seed before each GPU Squeezed fit')
    p.add_argument('--audit-squeezed-selections', action='store_true',
                   help='Record attended prefix counts during quality generation')
    p.add_argument('--profile-subset-kernels', action='store_true',
                   help='Profile lookup, selection, and sparse attention on the final layer after decode timing')
    p.add_argument('--prewarm-multipole-worker', action='store_true',
                   help='Preload the next one-prompt Multipole worker while the current prompt runs')
    p.add_argument('--multipole-centroids', type=int,
                   help='Keep this many centroids per KV head in the Multipole and subset arms')
    p.add_argument('--windows', type=int, default=3)
    p.add_argument('--steps-per-window', type=int, default=512)
    p.add_argument('--warmup-steps', type=int, default=8)
    p.add_argument('--matched-full-model-timing', action='store_true',
                   help='Repeat Llama Multipole and Squeezed timing from fresh caches and retain token IDs')
    a = p.parse_args()
    if a.multipole_worker_stdin:
        for line in sys.stdin:
            payload, output = map(Path, json.loads(line))
            multipole_worker(payload, output)
        return
    if a.output is None:
        p.error('--output is required')
    if a.subset_payload:
        multipole_worker(a.subset_payload, a.output)
        return
    if a.length is None:
        p.error('--length is required')
    if a.profile_subset_kernels and (a.cohort is not None or not any(m.startswith('subset-') for m in a.methods)):
        p.error('--profile-subset-kernels requires a timing run with a subset method')
    if a.profile_quality and a.cohort is None:
        p.error('--profile-quality requires a quality cohort')
    if (not 0 <= a.cohort_shard < a.cohort_shards
            or a.cohort_shards > a.limit_per_task
            or (a.cohort_shards > 1 and a.cohort is None)):
        p.error('invalid quality cohort shard')
    if a.stop_at_eos and a.cohort is None:
        p.error('--stop-at-eos requires a quality cohort')
    if a.resume_from_run and (a.cohort is None or a.profile_quality):
        p.error('--resume-from-run requires a quality cohort without runtime profiling')
    a.total_steps = a.warmup_steps + a.windows * a.steps_per_window
    if a.timing_prefix_length is not None and a.cohort is not None:
        p.error('--timing-prefix-length applies to the timing prompt')
    if a.timing_prefix_length is None:
        a.timing_prefix_length = a.length
    if a.matched_full_model_timing and (
            a.cohort is not None or a.model_name != LLAMA
            or not set(a.methods) <= {'multipole-512', *SQUEEZED_WORKER_METHODS}
            or a.timing_prefix_length != a.length
            or (a.warmup_steps, a.windows, a.steps_per_window) != (256, 1, 1024)):
        p.error('--matched-full-model-timing requires supported Llama baselines on the full prompt '
                'with --warmup-steps 256 --windows 1 --steps-per-window 1024')
    if (min(a.warmup_steps, a.windows, a.steps_per_window, a.limit_per_task) < 1
            or not a.total_steps < a.timing_prefix_length <= a.length):
        p.error('invalid decode window or sample count')
    if len(set(a.methods)) != len(a.methods) or ('lod' in a.methods and a.methods[0] != 'lod'):
        p.error('methods must be unique and LoD, if selected, must run first')
    index_widths = {method: int(fields[1]) if len(fields) == 3 else None
                    for method in a.methods if (fields := method.split('-'))[0] == 'squeezed'}
    lookup_widths = sorted({n for n in index_widths.values() if n is not None})
    if 'lod' in a.methods and (not os.environ.get('LOD_H100_PREFILL_RULE')
                               or os.environ.get('LOD_H100_PREFILL_IMPLEMENTATION') != 'fast'):
        p.error('LoD requires the optimized threshold prefill configuration')
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(a.model_name)
    rope = (dict(YARN, factor=2.0 if a.length <= 65536 else 4.0)
            if a.model_name == MODEL and a.length > 32768 else None)
    need_index = any(k.startswith('squeezed-') for k in a.methods)
    need_raw = any(k.startswith(MULTIPOLE_METHOD_PREFIXES) for k in a.methods)
    native_squeezed = any(m in GPU_SQUEEZED_METHODS for m in a.methods)
    if a.squeezed_centroids is not None and (a.squeezed_centroids < 1 or GPU_SQUEEZED_P80 not in a.methods):
        p.error('--squeezed-centroids requires a positive count and gpu-squeezed-p80')
    if a.squeezed_fit_seed is not None and (a.squeezed_fit_seed < 0 or not native_squeezed):
        p.error('--squeezed-fit-seed requires a nonnegative seed and GPU Squeezed')
    if a.audit_squeezed_selections and (a.cohort is None or not native_squeezed):
        p.error('--audit-squeezed-selections requires GPU Squeezed and a quality cohort')
    need_worker = need_raw or native_squeezed
    shared_p90 = MULTIPOLE_P90 in a.methods
    if (shared_p90 and 'multipole-512' in a.methods
            and a.methods.index('multipole-512') > a.methods.index(MULTIPOLE_P90)):
        p.error('when benchmarking both methods, multipole-512 must precede multipole-p90')
    if a.prewarm_multipole_worker:
        if not need_worker:
            p.error('--prewarm-multipole-worker requires an author-compatible decode method')
        a._multipole_worker = None
        atexit.register(lambda: close_multipole_worker(a._multipole_worker) if a._multipole_worker else None)
    if a.multipole_centroids is not None and (a.multipole_centroids < 1 or not need_raw):
        p.error('--multipole-centroids requires a positive count and a Multipole or subset method')
    gpu_ipc = need_worker and os.environ.get('MULTIPOLE_GPU_IPC') == '1'
    share_weights = gpu_ipc and os.environ.get('MULTIPOLE_SHARE_WEIGHTS') == '1'
    if a.dense_kv_source is None:
        a.dense_kv_source = 'host' if need_index or (need_raw and not gpu_ipc) else 'device'
    capture_host = need_index or (need_worker and not gpu_ipc) or ('dense' in a.methods and a.dense_kv_source != 'device')
    if gpu_ipc and capture_host:
        p.error('Multipole GPU IPC requires GPU-resident dense KV and no Squeezed host index')
    with shareable_cuda_allocations() if share_weights else nullcontext():
        model = load_shared_model(a.model_name, max(131072, a.length + a.total_steps + 1), tokenizer, rope)
    # Export the read-only model tensors before attaching runtime-specific
    # methods and caches. The worker builds its own graph in Triton 3.1.
    worker_model_state = {k: v for k, v in vars(model).items() if k != 'tokenizer'} if share_weights else None
    configure_prefill_mlp_chunk(model)
    original = {k: getattr(model, k) for k in ('init_kv_cache', 'prefill_attention', 'decode_attention', 'move', 'position_embedd')}
    prefill_by_method = {method: ('LoD threshold' if method == 'lod' else 'dense causal')
                         for method in a.methods}
    result = dict(stage='running', model_name=a.model_name, model_revision=model.config._commit_hash, dtype='bfloat16',
                  rope_max_position_embeddings=model.config.max_position_embeddings,
                  batch_size=1, rope_scaling=rope, context_limit=a.length, methods=a.methods,
                  prefill=next(iter(prefill_by_method.values())) if len(set(prefill_by_method.values())) == 1
                  else 'method-specific',
                  prefill_by_method=prefill_by_method,
                  timing='decode-only; contiguous teacher-forced windows; cache writes and required updates included',
                  timing_prefix_length=a.timing_prefix_length if not a.cohort else None,
                  warmup_steps=a.warmup_steps if not a.cohort else None,
                  dense_kv_source=a.dense_kv_source if 'dense' in a.methods else None,
                  captures_host_kv=capture_host,
                  cohort_sha256=hashlib.sha256(a.cohort.read_bytes()).hexdigest() if a.cohort else None,
                  records=[])
    if need_worker:
        result['multipole_handoff'] = 'cuda-ipc' if gpu_ipc else 'host'
        result['multipole_weights'] = 'cuda-ipc' if share_weights else 'reload'
        if a.prewarm_multipole_worker:
            result['multipole_worker'] = 'prewarmed'
    if a.multipole_centroids is not None:
        result['multipole_fixed_centroids_per_kv_head'] = a.multipole_centroids
    if a.squeezed_centroids is not None:
        result['squeezed_fixed_centroids_per_kv_head'] = a.squeezed_centroids
    if a.squeezed_fit_seed is not None:
        result['squeezed_fit_seed'] = a.squeezed_fit_seed
    if a.matched_full_model_timing:
        result.update(matched_full_model_timing=True, timed_decode_steps=a.steps_per_window,
                      measurement_warmups=1, measurement_repeats=2)
    if a.stop_at_eos:
        result['stop_at_eos'] = True
    if a.cohort_shards > 1:
        result.update(cohort_shards=a.cohort_shards, cohort_shard=a.cohort_shard)
    prior_records = resume_quality(a.resume_from_run, a.output, result) if a.resume_from_run else {}
    stop_text = ('<|eot_id|>', '<|end_of_text|>') if a.model_name == LLAMA else ('<|im_end|>', '<|endoftext|>')
    stops = set(tokenizer.convert_tokens_to_ids(list(stop_text)))
    if None in stops or len(stops) != len(stop_text):
        raise RuntimeError('Tokenizer does not define the expected EOS/EOT tokens')
    a.stop_tokens = tuple(sorted(stops)) if a.stop_at_eos else ()
    for case in read_cases(a, tokenizer):
        case_start = time.perf_counter()
        ids = case.pop('ids')
        token_hash = hashlib.sha256(struct.pack(f'<{len(ids)}I', *ids)).hexdigest()
        if prior := prior_records.pop((case['task'], case.get('source_row')), None):
            if (any(prior[k] != v for k, v in case.items()) or prior['input_token_sha256'] != token_hash
                    or set(prior['decoders']) != set(a.methods)):
                raise ValueError('checkpoint record does not match the requested prompt and methods')
            result['records'].append(prior)
            continue
        inputs = torch.tensor([ids], dtype=torch.long, device='cuda')
        model.input_length = len(ids)
        model.max_new_length = LIMITS[case['task']] if a.cohort else a.total_steps + 1
        model.batch_size = model.prefill_bsz = 1
        model.prefill_method = 'full'
        if model.input_length + model.max_new_length > model.max_length:
            raise ValueError('requested sequence exceeds the model cache')
        for k, v in original.items():
            setattr(model, k, v)
        if a.profile_quality:
            torch.cuda.reset_peak_memory_stats()
        profile = {} if a.profile_quality else None
        prefill_start = time.perf_counter()
        lod_first = a.methods[0] == 'lod'
        first, layers, routing_observations = capture_prefill(
            model, inputs, capture_host and not lod_first, need_index and not lod_first,
            need_raw and not lod_first, profile, gpu_ipc and not lod_first,
            capture_routing_observations=shared_p90 and not lod_first,
            capture_native_observations=native_squeezed and not lod_first, lod=lod_first)
        row = dict(**case, input_tokens=len(ids), prefill_token_id=int(first.item()),
                   input_token_sha256=token_hash, decoders={})
        if shared_p90 or native_squeezed:
            row['lod_prefill_s' if lod_first else 'prefill_s'] = time.perf_counter() - prefill_start
        if a.matched_full_model_timing:
            row['teacher_forced_token_sha256'] = hashlib.sha256(
                json.dumps(ids[-a.total_steps:], separators=(',', ':')).encode()).hexdigest()
        indexes = {}
        multipole_results = None
        if profile is not None:
            row['quality_profile'] = profile
        native_dense_cache = model.kv_cache if not lod_first and 'dense' in a.methods else None
        dense_prefilled = not lod_first
        for method in a.methods:
            index = None
            setup_start = time.perf_counter()
            uses_multipole = method.startswith(WORKER_METHOD_PREFIXES)
            if method != 'lod':
                model.kv_cache = None
                gc.collect()
                torch.cuda.empty_cache()
                for k, v in original.items():
                    setattr(model, k, v)
                if not dense_prefilled:
                    prefill_start = time.perf_counter()
                    first, layers, routing_observations = capture_prefill(
                        model, inputs, capture_host, need_index, need_raw, None, gpu_ipc,
                        capture_routing_observations=shared_p90, capture_native_observations=native_squeezed)
                    row['prefill_s'] = time.perf_counter() - prefill_start
                    row['dense_prefill_token_id'] = int(first.item())
                    if 'dense' in a.methods:
                        native_dense_cache = model.kv_cache
                    model.kv_cache = None
                    dense_prefilled = True
                if uses_multipole and multipole_results is None:
                    multipole_results = run_multipole_methods(
                        model, layers, ids, first, case['task'], a, rope, worker_model_state, routing_observations)
                if method in index_widths:
                    width = index_widths[method]
                    if width not in indexes:
                        indexes[width] = index_for(layers, a, width)
                    index, reused = indexes[width]
                if method == 'dense' and a.dense_kv_source != 'host':
                    if a.dense_kv_source == 'verify':
                        for (k, v, _, _), keys, values in zip(
                                layers, native_dense_cache.key_cache, native_dense_cache.value_cache, strict=True):
                            for host, device in ((k, keys), (v, values)):
                                if not torch.equal(host.transpose(1, 2), device[:, :len(ids)].cpu()):
                                    raise RuntimeError('native dense cache differs from captured prefill KV')
                    model.attention_type = 'Full_Flash_Attn'
                    model.kv_cache = native_dense_cache
                elif not uses_multipole:
                    install_decode(model, method, layers, index)
                if method == 'dense':
                    native_dense_cache = None
            setup_s = time.perf_counter() - setup_start
            measured = multipole_results[method] if uses_multipole else run_decode(model, method, first, inputs, case['task'], a)
            if index is not None:
                index_meta = {k: index[k] for k in ('seconds', 'clusters', 'cluster_percent', 'threshold', 'observation_window')}
                index_meta['reused'] = reused
                if width is None:
                    row['squeezed_index'] = index_meta
                else:
                    measured['squeezed_index'] = index_meta
                    if not a.cohort and len(lookup_widths) > 1 and width == lookup_widths[-1]:
                        measured['lookup_only_ms_per_token'] = profile_squeezed_lookup(model.kv_cache, layers, lookup_widths)
            if a.cohort:
                out = measured['output_token_ids']
                stop = next((i for i, t in enumerate(out) if t in stops), len(out))
                if profile is not None:
                    measured['quality_profile'].update(
                        arm_setup_s=setup_s, stop_at_token=stop + 1 if stop < len(out) else None)
                measured['prediction'] = tokenizer.decode(out[:stop], skip_special_tokens=True)
                if a.model_name == LLAMA:
                    measured['prediction'] = cleanup(measured['prediction'])
                measured['score'] = score(case['task'], measured['prediction'], case['answers'])
            row['decoders'][method] = measured
            print(json.dumps(dict(task=case['task'], source_row=case.get('source_row'), method=method,
                **{k: measured[k] for k in ('score', 'median_decode_ms') if k in measured})), flush=True)
        row['elapsed_s'] = time.perf_counter() - case_start
        if profile is not None:
            profile['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        result['records'].append(row)
        save(a.output, result)
        del layers, routing_observations, indexes, index
        model.kv_cache = None
        gc.collect()
        if gpu_ipc:
            torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
    if prior_records:
        raise ValueError('checkpoint contains prompts outside the requested quality panel')
    result['stage'] = 'complete'
    save(a.output, result)


if __name__ == '__main__':
    main()
