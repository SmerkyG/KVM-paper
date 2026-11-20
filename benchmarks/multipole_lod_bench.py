"""Benchmark a production LoD layer at the Multipole paper's synthetic boundary.

The prefix is random BF16 KV with Qwen3-8B head geometry. LoD builds its
actual region/page cache before timing. A graph captures the connected decode
path, while catch-up and cache updates run outside the graph as in serving.
This is not full-model inference and is not quality matched to Multipole.
"""
import argparse
import json
import math
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmarks.retro_lod_adapter import install_lod_modules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--length', type=int, default=131072)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--mode', choices=('two-tier', 'three-tier-bf16', 'three-tier-int4'), default='three-tier-bf16')
    parser.add_argument('--warmup-steps', type=int, default=512)
    parser.add_argument('--steps', type=int, default=512)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--attention-only', action='store_true',
                        help='Graph attention with existing local KV; exclude cache writes and updates')
    parser.add_argument('--remote-only', action='store_true',
                        help='Exclude local, sink, and new KV from attention-only timing')
    parser.add_argument('--exclude-new-kv', action='store_true',
                        help='Include existing local and sink KV but exclude the new KV item')
    parser.add_argument('--paper-component-bench', action='store_true',
                        help='Also time remote attention with the author do_bench settings')
    parser.add_argument('--paper-update-only', action='store_true',
                        help='Time the isolated region-state update with the author do_bench settings')
    parser.add_argument('--diagnose-attention', action='store_true',
                        help='Compare eager and graph event timings with and without L2 clearing')
    parser.add_argument('--profile-window-updates', action='store_true',
                        help='Record update and sampled graph events inside connected decode windows')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.paper_component_bench and not (args.remote_only and args.attention_only):
        parser.error('--paper-component-bench requires --attention-only --remote-only')
    if args.diagnose_attention and not (args.remote_only and args.attention_only):
        parser.error('--diagnose-attention requires --attention-only --remote-only')
    if args.paper_update_only and (args.attention_only or args.remote_only):
        parser.error('--paper-update-only is a separate benchmark')
    if args.remote_only and not args.attention_only:
        parser.error('--remote-only requires --attention-only')
    if args.exclude_new_kv and (not args.attention_only or args.remote_only):
        parser.error('--exclude-new-kv requires --attention-only without --remote-only')
    if min(args.batch, args.length, args.steps, args.repeats) < 1 or args.warmup_steps < 0:
        parser.error('batch, length, steps, and repeats must be positive; warmup may be zero')

    install_lod_modules()
    from vllm_lod_plugin.config import VLLMLODSettings
    from vllm_lod_plugin.pool import VLLMLayerLODPool

    torch.manual_seed(105)
    torch.set_grad_enabled(False)
    b, h, kv, dim = args.batch, 32, 8, 128
    device = torch.device('cuda:0')
    capacity = args.length + max(args.warmup_steps + args.steps * args.repeats,
                                 2048 if args.remote_only else 0) + 16
    layer = SimpleNamespace(num_heads=h, num_kv_heads=kv, head_size=dim, head_size_v=dim,
                            impl=SimpleNamespace(scale=1.0 / math.sqrt(dim)))
    settings = VLLMLODSettings.production(mode=args.mode, pool_size=b, request_capacity=capacity)
    pool = VLLMLayerLODPool(layer, settings=settings, max_requests=b, request_capacity=capacity,
                            active_indices=torch.arange(b, device=device, dtype=torch.int32),
                            dtype=torch.bfloat16, device=device,
                            has_query_norm=True, has_key_norm=True)
    # Build from each independent request's KV, not a repeated model prompt.
    for slot in range(b):
        k = torch.randn((1, kv, args.length, dim), device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        storage = pool._initial_prefill_storage((slot,))
        if storage is None:
            raise RuntimeError('pool-backed prefix construction is unavailable')
        pool.engine._lod_prefill_storage = storage
        try:
            cache = pool.engine.build_cache_from_bf16(k, v, finalize_cache_for_decode=True)
        finally:
            del pool.engine._lod_prefill_storage
        pool.install_rows((slot,), cache)
        pool.engine.reset_runtime_cache()
        del k, v, cache
        print(f'prepared request {slot + 1}/{b}', flush=True)

    q = torch.randn((b, h, dim), device=device, dtype=torch.bfloat16)
    new_k = torch.randn((b, kv, dim), device=device, dtype=torch.bfloat16)
    new_v = torch.randn_like(new_k)
    out = torch.empty_like(q)
    meta = SimpleNamespace(num_actual_tokens=b, max_seq_len=args.length + 1)
    pool.catch_up_many([(slot, args.length) for slot in range(b)])
    if args.paper_update_only:
        import triton.testing
        # Supply a real production-shaped pending update, with cache assembly
        # outside timing. The isolated function updates summaries and assigns
        # ownership but deliberately does not append semantic pages.
        pool.decode(q, new_k, new_v, meta, out)
        pool.local_lens.sub_(1)
        torch.cuda.synchronize()
        append_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(append_graph):
            pool.decode(q, new_k, new_v, meta, out)
        context = args.length
        interval = int(pool.engine.decode_state_update_len)
        for _ in range(interval + 2):
            _, target = pool._catch_up_target(0, context)
            if target > int(pool.metadata[0]['coverage']):
                break
            pool.catch_up_many([(slot, context) for slot in range(b)])
            append_graph.replay()
            context += 1
        else:
            raise RuntimeError('failed to prepare pending state update')
        # Page-overflow metadata may differ by request but is irrelevant to
        # this state-only operation. Batch only rows with equal state geometry.
        metadata = pool.metadata[0]
        scalar_keys = ('state_len', 'scheduled_state_len', 'coverage', 'recent_len', 'total_len')
        if any(any(int(pool.metadata[slot][name]) != int(metadata[name]) for name in scalar_keys)
               for slot in range(1, b)):
            raise RuntimeError('state-only fixture has unequal state geometry')
        state = {name: tensor[:b] for name, tensor in pool.state.items()
                 if isinstance(tensor, torch.Tensor) and tensor.ndim}
        state.update({name: int(metadata[name]) for name in scalar_keys})
        state['state_capacity'] = pool.state_capacity
        coverage = int(state['coverage'])
        overflow_len = target - coverage
        if overflow_len != interval:
            raise RuntimeError(f'unexpected state update size: {overflow_len}, interval {interval}')
        orig_k, orig_v, orig_counts = state['state_k'], state['state_v'], state['counts']
        norm = state.get('key_norm_sums')
        overflow_k = state['recent_k'][..., :overflow_len, :]
        overflow_v = state['recent_v'][..., :overflow_len, :]
        kwargs = {'state_len': int(state['state_len']),
                  'ctx_len': int(pool.engine.local_len - pool.engine.chunk_len + target),
                  'available_context': target,
                  'state_capacity': int(state['state_capacity']),
                  'clustering_query_scale': None,
                  'scheduled_state_len': int(state.get('scheduled_state_len', state['state_len']))}
        def update():
            # Match the author's functional benchmark: each call receives the
            # same original state and allocates its output rather than
            # repeatedly adding the same tokens into a mutable cache.
            return pool.engine._update_state(orig_k.clone(), orig_v.clone(), orig_counts.clone(),
                                             None if norm is None else norm.clone(),
                                             overflow_k, overflow_v, **kwargs)
        # Prefill can leave reusable scratch allocated for a different phase.
        # Initialize the isolated benchmark's scratch once before verification.
        for name in ('_lod_state_update_buffers', '_lod_state_maxsim_buffers'):
            if hasattr(pool.engine, name):
                delattr(pool.engine, name)
        update()
        torch.cuda.synchronize()
        first = update()
        torch.cuda.synchronize()
        frozen_first_v = first[1].clone()
        second = update()
        torch.cuda.synchronize()
        if not torch.equal(first[1], frozen_first_v):
            raise RuntimeError('later update modified the prior update output')
        if first[3] != second[3] or not torch.equal(first[4], second[4]):
            raise RuntimeError('isolated state update is not repeatable')
        active = first[3]
        # Zero-count slots are inactive and their storage need not be initialized.
        if not torch.equal(first[2][..., :active, :], second[2][..., :active, :]):
            raise RuntimeError('isolated state counts are not repeatable')
        mask = first[2][..., :active, :] > 0
        first_active = [first[i][..., :active, :].masked_select(mask) for i in range(2)]
        second_active = [second[i][..., :active, :].masked_select(mask) for i in range(2)]
        summary_diffs = [float((x.float() - y.float()).abs().max().item())
                         for x, y in zip(first_active, second_active)]
        if not all(torch.allclose(x.float(), y.float(), atol=0.02, rtol=0.001)
                   for x, y in zip(first_active, second_active)):
            vdelta = (first[1][..., :active, :].float() - second[1][..., :active, :].float()).abs()
            index = torch.nonzero(vdelta == vdelta.max(), as_tuple=False)[0].tolist()
            assignments = torch.nonzero(first[4][index[0], index[1]] == index[2], as_tuple=False).flatten()
            third = update()
            torch.cuda.synchronize()
            summaries = []
            for sample in (first, second, third):
                ref = orig_v.float() * (orig_counts > 0)
                ref.scatter_add_(2, sample[4].unsqueeze(-1).expand_as(overflow_v), overflow_v.float())
                delta = (sample[1].float() - ref).abs()
                summaries.append((float(delta.max().item()), int((delta > 0.05).sum().item())))
            raise RuntimeError(f'isolated state summaries are not repeatable: {summary_diffs}; '
                               f'index={index}; first={first[1][tuple(index)].item()}; '
                               f'second={second[1][tuple(index)].item()}; '
                               f'third={third[1][tuple(index)].item()}; '
                               f'input={orig_v[tuple(index)].item()}; '
                               f'count={first[2][index[0],index[1],index[2],0].item()}; '
                               f'old_count={orig_counts[index[0],index[1],index[2],0].item()}; '
                               f'inputs={overflow_v[index[0],index[1],assignments,index[3]].tolist()}; '
                               f'reference_diffs={summaries}')
        if not torch.all(first[2].sum(dim=2) == orig_counts.sum(dim=2) + overflow_len):
            raise RuntimeError('isolated state update does not conserve counts')
        if not all(torch.isfinite(x).all() for x in first_active):
            raise RuntimeError('isolated state update produced non-finite summaries')
        expected_v = orig_v.float() * (orig_counts > 0)
        expected_v.scatter_add_(2, first[4].unsqueeze(-1).expand_as(overflow_v), overflow_v.float())
        reference_v = expected_v[..., :active, :].masked_select(mask)
        reference_diff = float((first_active[1].float() - reference_v).abs().max().item())
        if not torch.allclose(first_active[1].float(), reference_v, atol=0.05, rtol=0.01):
            raise RuntimeError(f'isolated state value summary does not match reference: {reference_diff}')
        del first, second
        quantiles = triton.testing.do_bench(update, warmup=500, rep=500, quantiles=[0.2, 0.5, 0.8])
        result = {'measurement_scope': 'isolated LoD region-state update, not complete cache maintenance',
                  'batch': b, 'prefix_tokens': args.length, 'update_context': context,
                  'mode': args.mode, 'model_geometry': 'Qwen3-8B', 'weights_loaded': False,
                  'update_tokens': overflow_len, 'interval_tokens': interval,
                  'state_len': kwargs['state_len'], 'kv_dtype': str(orig_k.dtype),
                  'quantiles_ms_20_50_80': [float(x) for x in quantiles],
                  'median_ms': float(quantiles[1]), 'amortized_ms': float(quantiles[1]) / interval,
                  'includes': ['state input clones', 'key and value summary update', 'owner assignment'],
                  'excludes': ['page archive update', 'local cache movement', 'per-token KV write', 'attention'],
                  'same_inputs_repeatable': True, 'repeat_max_abs_diffs': summary_diffs,
                  'count_conservation_checked': True, 'value_reference_max_abs': reference_diff}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        return
    if args.attention_only:
        local_lengths = pool.local_lens.clone()
        if args.remote_only or args.exclude_new_kv:
            # Override only the benchmark call boundary, preserving the serving
            # implementation and its defaults. The kernel supports empty fields.
            import vllm_lod_plugin.pool as pool_module
            original_attention = pool_module.fused_decode_paged_lod_attention
            zero_local_lengths = torch.zeros_like(pool.local_lens)
            empty_sink_k = pool.state['sink_k'][:, :, :0, :]
            empty_sink_v = pool.state['sink_v'][:, :, :0, :]
            def remote_attention(*pos, **kw):
                kw.update(new_k=None, new_v=None, store_new_kv=False, advance_local_lens=False)
                if args.remote_only:
                    kw.update(sink_k=empty_sink_k, sink_v=empty_sink_v,
                              local_lens=zero_local_lengths)
                return original_attention(*pos, **kw)
            pool_module.fused_decode_paged_lod_attention = remote_attention
        def attention():
            return pool.decode(q, new_k, new_v, meta, out,
                               store_new_kv=False, advance_local_lens=False)
        attention()
        torch.cuda.synchronize()
        if args.remote_only:
            # Perturb the excluded KV and verify it cannot affect the output.
            baseline = out.clone()
            excluded = [pool.state[name] for name in ('recent_k', 'recent_v', 'sink_k', 'sink_v')
                        if isinstance(pool.state.get(name), torch.Tensor)]
            saved = [t.clone() for t in excluded]
            for tensor in excluded:
                tensor.normal_()
            attention()
            exclusion_diff = float((out.float() - baseline.float()).abs().max().item())
            for tensor, original in zip(excluded, saved):
                tensor.copy_(original)
            if exclusion_diff != 0:
                raise RuntimeError(f'excluded local/sink KV affected output: {exclusion_diff}')
        else:
            exclusion_diff = None
        inclusion_diffs = None
        if args.exclude_new_kv:
            baseline = out.clone()
            inclusion_diffs = {}
            for field in ('recent_v', 'sink_v'):
                tensor = pool.state[field]
                original = tensor.clone()
                tensor.fill_(100)
                attention()
                inclusion_diffs[field] = float((out.float() - baseline.float()).abs().max().item())
                tensor.copy_(original)
                if inclusion_diffs[field] == 0:
                    raise RuntimeError(f'{field} has no effect on local/sink attention')
            attention()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            attention()
        graph.replay()
        graph_out = out.clone()
        attention()
        diff = float((out.float() - graph_out.float()).abs().max().item())
        if not torch.isfinite(out).all() or diff > 0.02 or not torch.equal(pool.local_lens, local_lengths):
            raise RuntimeError(f'LoD attention-only graph/eager check failed: {diff}')
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.repeats):
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start = time.perf_counter()
            start_event.record()
            for _ in range(args.steps):
                graph.replay()
            end_event.record()
            torch.cuda.synchronize()
            samples.append({'wall_ms_per_batch_step': (time.perf_counter() - start) * 1000 / args.steps,
                            'gpu_timeline_ms_per_batch_step': start_event.elapsed_time(end_event) / args.steps})
        result = {'measurement_scope': 'synthetic single-layer attention-only graph, not full-model inference',
                  'batch': b, 'prefix_tokens': args.length, 'mode': args.mode,
                  'model_geometry': 'Qwen3-8B', 'weights_loaded': False, 'quality_matched': False,
                  'kv_writes_included': False, 'cache_updates_included': False,
                  'local_attention_included': not args.remote_only,
                  'sink_attention_included': not args.remote_only,
                  'sink_tokens': int(pool.state['sink_k'].shape[2]) if not args.remote_only else 0,
                  'new_kv_attention_included': not (args.remote_only or args.exclude_new_kv),
                  'excluded_kv_perturbation_max_abs': exclusion_diff,
                  'included_kv_perturbation_max_abs': inclusion_diffs,
                  'local_lengths': local_lengths.tolist(),
                  'same_cache_graph_eager_max_abs': diff, 'records': samples,
                  'median_wall_ms_per_batch_step': statistics.median(x['wall_ms_per_batch_step'] for x in samples)}
        if args.diagnose_attention:
            # CUDA events bracket only attention, with optional L2 clearing
            # before the start event. This preserves the paper timing boundary.
            cache_flush = torch.empty((int(256e6 // 4),), dtype=torch.int32, device=device)
            def event_panel(fn, flush):
                starts, ends = [], []
                for _ in range(80):
                    if flush:
                        cache_flush.zero_()
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    fn()
                    end_event.record()
                    starts.append(start_event)
                    ends.append(end_event)
                torch.cuda.synchronize()
                values = [x.elapsed_time(y) for x, y in zip(starts, ends)]
                return {'median_ms': statistics.median(values),
                        'p20_ms': sorted(values)[int(.2 * len(values))],
                        'p80_ms': sorted(values)[int(.8 * len(values))]}
            panels = {}
            for name, fn in (('graph', graph.replay), ('eager', attention)):
                for flush in (False, True):
                    panels[name + ('_cold' if flush else '_warm')] = event_panel(fn, flush)
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                attention()
                torch.cuda.synchronize()
            with tempfile.TemporaryDirectory() as tmp:
                trace_path = Path(tmp) / 'attention.json'
                prof.export_chrome_trace(str(trace_path))
                trace = json.loads(trace_path.read_text())
            kernels = [event for event in trace['traceEvents']
                       if event.get('cat') == 'kernel' and event.get('dur') is not None]
            if not kernels:
                raise RuntimeError('attention profiler did not record CUDA kernels')
            result['attention_diagnostic'] = {'event_panels': panels,
                'profile_kernel_count': len(kernels),
                'profile_kernel_sum_ms': sum(e['dur'] for e in kernels) / 1000,
                'profile_kernel_span_ms': (max(e['ts'] + e['dur'] for e in kernels) -
                                           min(e['ts'] for e in kernels)) / 1000,
                'profile_kernels': [{'name': e.get('name'), 'duration_us': e['dur']} for e in kernels]}
        if args.paper_component_bench:
            import triton.testing
            quantiles = triton.testing.do_bench(attention, warmup=500, rep=500, quantiles=[0.2, 0.5, 0.8])
            result['paper_style_attention_quantiles_ms_20_50_80'] = [float(x) for x in quantiles]
        if args.diagnose_attention:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result), flush=True)
            return
        if args.remote_only:
            # Separately time real cache catch-up at its production interval.
            # Decode during preparation supplies the entries being incorporated;
            # it is outside these update samples and outside the graph above.
            pool_module.fused_decode_paged_lod_attention = original_attention
            pool.decode(q, new_k, new_v, meta, out)
            pool.local_lens.sub_(1)
            torch.cuda.synchronize()
            append_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(append_graph):
                pool.decode(q, new_k, new_v, meta, out)
            context = args.length
            interval = int(pool.engine.decode_state_update_len)
            update_samples = []
            for step in range(6 * interval):
                requests = [(slot, context) for slot in range(b)]
                _, target = pool._catch_up_target(0, context)
                pending = target > int(pool.metadata[0]['coverage'])
                if pending:
                    torch.cuda.synchronize()
                    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start = time.perf_counter()
                    start_event.record()
                    pool.catch_up_many(requests)
                    end_event.record()
                    torch.cuda.synchronize()
                    if step >= 2 * interval:
                        update_samples.append({'wall_ms': (time.perf_counter() - start) * 1000,
                                               'gpu_timeline_ms': start_event.elapsed_time(end_event),
                                               'context': context})
                else:
                    pool.catch_up_many(requests)
                append_graph.replay()
                context += 1
            if len(update_samples) < 3:
                raise RuntimeError(f'Expected repeated LoD updates, saw {update_samples}')
            update_median = statistics.median(x['wall_ms'] for x in update_samples)
            result['separate_update'] = {'interval_tokens': interval, 'warmup_tokens': 2 * interval,
                                         'samples': update_samples, 'median_wall_ms': update_median,
                                         'amortized_wall_ms_per_token': update_median / interval,
                                         'scope': 'production catch-up including host bookkeeping and cache maintenance'}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        return
    # Initialize graph scratch and undo the warmup append.
    pool.decode(q, new_k, new_v, meta, out)
    pool.local_lens.sub_(1)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pool.decode(q, new_k, new_v, meta, out)

    # The same initialized cache and position must give the same output for
    # eager and graph execution; both writes append the same K/V.
    graph.replay()
    graph_out = out.clone()
    pool.local_lens.sub_(1)
    pool.decode(q, new_k, new_v, meta, out)
    diff = float((out.float() - graph_out.float()).abs().max().item())
    if not torch.isfinite(out).all() or diff > 0.02:
        raise RuntimeError(f'LoD same-cache graph/eager mismatch: max abs {diff}')
    context = args.length + 1

    def run_steps(n):
        nonlocal context
        before = pool.catch_up_batches
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        start_event.record()
        profiled_updates = []
        graph_samples = []
        interval = int(pool.engine.decode_state_update_len)
        for _ in range(n):
            requests = [(slot, context) for slot in range(b)]
            # For this aligned synthetic prefix, update boundaries occur at
            # interval multiples. Avoid scanning host metadata on every step.
            pending = args.profile_window_updates and context % interval == 0
            if pending:
                if not any(pool._catch_up_target(slot, context)[1] > int(pool.metadata[slot]['coverage'])
                           for slot in range(b)):
                    raise RuntimeError('expected aligned catch-up boundary was absent')
                before_update = pool.catch_up_batches
                update_start_event = torch.cuda.Event(enable_timing=True)
                update_end_event = torch.cuda.Event(enable_timing=True)
                update_start = time.perf_counter()
                update_start_event.record()
                pool.catch_up_many(requests)
                update_end_event.record()
                profiled_updates.append((context, (time.perf_counter() - update_start) * 1000,
                                         pool.catch_up_batches - before_update,
                                         update_start_event, update_end_event))
            else:
                pool.catch_up_many(requests)
            if args.profile_window_updates and context % 64 == 1:
                graph_start, graph_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                graph_start.record()
                graph.replay()
                graph_end.record()
                graph_samples.append((context, graph_start, graph_end))
            else:
                graph.replay()
            context += 1
        end_event.record()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return {'steps': n, 'wall_ms_per_batch_step': elapsed * 1000 / n,
                'gpu_timeline_ms_per_batch_step': start_event.elapsed_time(end_event) / n,
                'catch_up_batches': pool.catch_up_batches - before,
                'profiled_updates': [{'context': pos, 'host_submit_ms': host,
                                      'catch_up_batches': groups, 'gpu_timeline_ms': first.elapsed_time(last)}
                                     for pos, host, groups, first, last in profiled_updates],
                'sampled_graphs': [{'context': pos, 'gpu_timeline_ms': first.elapsed_time(last)}
                                   for pos, first, last in graph_samples],
                'context_start': context - n, 'context_end': context}

    warmup = run_steps(args.warmup_steps) if args.warmup_steps else None
    if warmup and warmup['catch_up_batches'] == 0:
        raise RuntimeError('warmup must include a cache update before timing')
    records = [run_steps(args.steps) for _ in range(args.repeats)]
    if not all(row['catch_up_batches'] > 0 for row in records):
        raise RuntimeError('timed window did not include a LoD cache update')
    pool.catch_up_many([(slot, context) for slot in range(b)])
    graph.replay()
    graph_out = out.clone()
    pool.local_lens.sub_(1)
    pool.decode(q, new_k, new_v, meta, out)
    post_update_diff = float((out.float() - graph_out.float()).abs().max().item())
    if not torch.isfinite(out).all() or post_update_diff > 0.02:
        raise RuntimeError(f'LoD post-update graph/eager mismatch: max abs {post_update_diff}')
    result = {'measurement_scope': 'synthetic single attention layer, not full-model inference',
              'model_geometry': 'Qwen3-8B', 'batch': b, 'prefix_tokens': args.length,
              'mode': args.mode, 'query_heads': h, 'kv_heads': kv, 'head_dim': dim,
              'weights_loaded': False, 'seed': 105, 'quality_matched': False,
              'same_cache_graph_eager_max_abs': diff, 'warmup': warmup, 'records': records,
              'post_update_graph_eager_max_abs': post_update_diff,
              'median_wall_ms_per_batch_step': statistics.median(x['wall_ms_per_batch_step'] for x in records),
              'median_gpu_timeline_ms_per_batch_step': statistics.median(x['gpu_timeline_ms_per_batch_step'] for x in records),
              'update_interval': int(pool.engine.decode_state_update_len),
              'route_geometry': {'group_size': int(pool.engine.decode_route_group_size),
                                 'warps': int(pool.engine.decode_route_num_warps),
                                 'page_block': int(pool.engine.decode_block_n)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
