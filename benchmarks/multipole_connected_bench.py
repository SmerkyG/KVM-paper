"""Time connected graph-captured Multipole decode with actual cache updates.

Uses the authors' generation kernels and clustering through the existing
batch adapter. Synthetic one-layer Qwen3-8B geometry; no model is loaded.
"""
import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmarks.retro_multipole_batch_adapter import RetroMultipoleBatchCache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--length', type=int, default=131072)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--budget', type=int, default=512)
    p.add_argument('--warmup-steps', type=int, default=512)
    p.add_argument('--steps', type=int, default=512)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--diagnostic-graph-replays', type=int, default=0,
                   help='Also measure a fixed-position graph-only diagnostic after the timed windows')
    p.add_argument('--update-only', action='store_true',
                   help='Separately time complete cache updates after two warmup updates')
    p.add_argument('--author-initial-clusters', action='store_true',
                   help='Initialize update fixtures with the author clustering routine')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.author_initial_clusters and not a.update_only:
        p.error('--author-initial-clusters requires --update-only')
    if min(a.length, a.batch, a.budget, a.steps, a.repeats) <= 0 or a.warmup_steps < 0:
        p.error('length, batch, budget, steps and repeats must be positive')
    torch.manual_seed(105)
    torch.set_grad_enabled(False)
    b, h, kh, d = a.batch, 32, 8, 128
    model = SimpleNamespace(batch_size=b, num_gpus=1, num_layers=1, num_heads=h,
                            num_key_value_heads=kh, head_dim=d, device_map='cuda:0',
                            input_length=a.length,
                            max_new_length=a.warmup_steps + a.steps * a.repeats + 16,
                            dtype=torch.bfloat16)
    cache = RetroMultipoleBatchCache(model, budget=a.budget)
    n, clusters = cache.clustered, cache.clustered // 16
    initialized = cache.capacity if a.update_only else a.length
    cache.k[0][:, :, :initialized].normal_()
    cache.v[0][:, :, :initialized].normal_()
    cache.raw_k[0][:, :, :initialized].copy_(cache.k[0][:, :, :initialized])
    if a.author_initial_clusters:
        from multipoleattention.clustering import run_clustering_online
        for slot in range(b):
            start, end = slot * kh, (slot + 1) * kh
            raw = cache.raw_k[0][:, start:end, cache.sink:cache.sink + n].contiguous()
            centers, labels, counts, _ = run_clustering_online(raw, [clusters], 1, rank=0)
            cache._install_clusters(0, centers[0], labels[0], counts[0], start)
            del raw, centers, labels, counts
            print(f'initialized author clusters for request {slot + 1}/{b}', flush=True)
    else:
        labels = torch.randint(clusters, (1, b * kh, n), device='cuda')
        raw = cache.raw_k[0][:, :, cache.sink:cache.sink + n]
        counts = torch.zeros((1, b * kh, clusters), dtype=torch.int64, device='cuda')
        counts.scatter_add_(2, labels, torch.ones_like(labels))
        sums = torch.zeros((1, b * kh, clusters, d), dtype=torch.float32, device='cuda')
        sums.scatter_add_(2, labels.unsqueeze(-1).expand_as(raw), raw.float())
        centers = (sums / counts.clamp_min(1).unsqueeze(-1)).bfloat16()
        cache._install_clusters(0, centers, labels, counts)
        del labels, raw, counts, sums, centers
    torch.cuda.empty_cache()
    local_n, tail = a.length - n, cache.sink + n
    for source, dest in ((cache.k[0], cache.local_k[0]), (cache.v[0], cache.local_v[0])):
        local = torch.cat((source[:, :, :cache.sink], source[:, :, tail:a.length]), 2)
        dest[:, :local_n].copy_(local.reshape(b, kh, local_n, d).transpose(1, 2))
    q = torch.randn((b, 1, h, d), device='cuda', dtype=torch.bfloat16)
    new_k = torch.randn((b, 1, kh, d), device='cuda', dtype=torch.bfloat16)
    new_v = torch.randn_like(new_k)
    cache.current_raw_k = new_k.clone()
    cache.current_window_q = q.clone()
    cache.context = a.length
    if a.update_only:
        # Same boundary as the LoD catch-up: all state needed by the next
        # remote attention step, plus host metadata and local-cache movement.
        # Fill the next local window before each timed call, as if decode had
        # written those synthetic K/V tokens. That preparation is not timed.
        updates = []
        for iteration in range(6):
            cache.context = cache.sink + cache.clustered + 2 * cache.interval - 1
            if cache.context >= cache.capacity:
                raise RuntimeError('insufficient synthetic KV capacity for update panel')
            old_len = cache.context - cache.sink - cache.clustered
            for source, dest in ((cache.k[0], cache.local_k[0]), (cache.v[0], cache.local_v[0])):
                local = torch.cat((source[:, :, :cache.sink],
                                   source[:, :, cache.sink + cache.clustered:cache.context]), 2)
                dest[:, :cache.sink + old_len].copy_(local.reshape(b, kh, cache.sink + old_len, d).transpose(1, 2))
            torch.cuda.synchronize()
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start = time.perf_counter()
            start_event.record()
            cache._update_clusters()
            end_event.record()
            torch.cuda.synchronize()
            if iteration >= 2:
                updates.append({'wall_ms': (time.perf_counter() - start) * 1000,
                                'gpu_timeline_ms': start_event.elapsed_time(end_event),
                                'context': cache.context})
        nclusters = cache.clustered // 16
        counts = cache.counts[0][0, :, :nclusters]
        if not torch.all(counts.sum(dim=1) == cache.clustered):
            raise RuntimeError('updated Multipole cluster counts do not cover the remote cache')
        labels = cache.labels[0][0, 0, :cache.clustered]
        if labels.min() < 0 or labels.max() >= nclusters:
            raise RuntimeError('updated Multipole labels are out of range')
        values = cache.v[0][0, 0, cache.sink:cache.sink + cache.clustered].float()
        expected = torch.zeros((nclusters, d), dtype=torch.float32, device='cuda')
        expected.scatter_add_(0, labels[:, None].expand_as(values), values)
        observed = cache.value_centroids[0][0, 0, :nclusters] * counts[0, :, None]
        max_diff = float((expected - observed).abs().max().item())
        if not torch.allclose(expected, observed, atol=0.001, rtol=0.001):
            raise RuntimeError(f'updated Multipole value summaries disagree: max abs {max_diff}')
        median = statistics.median(x['wall_ms'] for x in updates)
        result = {'measurement_scope': 'synthetic single-layer complete cache update',
                  'implementation': 'RetroInfer Multipole adapter using author online cluster update',
                  'batch': b, 'prefix_tokens': a.length, 'model_geometry': 'Qwen3-8B',
                  'weights_loaded': False, 'update_interval_tokens': cache.interval,
                  'initial_clusters': 'author clustering' if a.author_initial_clusters else 'random labels',
                  'warmup_updates': 2, 'samples': updates, 'median_wall_ms': median,
                  'amortized_wall_ms_per_token': median / cache.interval,
                  'count_coverage_checked_all_kv_heads': True,
                  'value_summary_max_abs_error_one_kv_head': max_diff,
                  'includes': ['cluster update', 'label and count installation',
                               'value summary recomputation', 'local-cache movement', 'host metadata'],
                  'excludes': ['per-token KV append', 'attention', 'model execution']}
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        return
    cache._set_position(cache.context)
    # Capture the complete attention step, not individual paper components.
    eager = cache.decode_attention(q, new_k, new_v, 0)
    if not torch.isfinite(eager).all():
        raise RuntimeError('Multipole eager warmup output is non-finite')
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = cache.decode_attention(q, new_k, new_v, 0)

    def compare():
        cache._set_position(cache.context)
        graph.replay()
        observed = graph_out.clone()
        eager_out = cache.decode_attention(q, new_k, new_v, 0)
        torch.cuda.synchronize()
        if not torch.isfinite(observed).all() or not torch.isfinite(eager_out).all():
            raise RuntimeError('Multipole graph/eager output is non-finite')
        difference = (observed.float() - eager_out.float()).abs()
        return {'max_abs': float(difference.max()), 'mean_abs': float(difference.mean()),
                'selected_tokens_min': int(cache.head_count.min()),
                'selected_tokens_max': int(cache.head_count.max())}

    check_before = compare()
    if check_before['max_abs'] > 0.05 or check_before['selected_tokens_max'] > a.budget:
        raise RuntimeError(f'Multipole graph/eager check failed: {check_before}')

    def run_steps(count):
        before = cache.update_count
        torch.cuda.synchronize()
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        start_event.record()
        start_context = cache.context
        for _ in range(count):
            if cache.context + 1 - cache.sink - cache.clustered >= 2 * cache.interval:
                cache._update_clusters()
            cache._set_position(cache.context)
            graph.replay()
            cache.context += 1
        end_event.record()
        torch.cuda.synchronize()
        result = {'steps': count, 'wall_ms_per_batch_step': (time.perf_counter() - start) * 1000 / count,
                'gpu_timeline_ms_per_batch_step': start_event.elapsed_time(end_event) / count,
                'cluster_updates': cache.update_count - before,
                'context_start': start_context, 'context_end': cache.context}
        if cache.update_times:
            result['profiled_cluster_update_ms'] = [1000 * t for t in cache.update_times[before:]]
        if cache.update_stage_times:
            result['profiled_update_stages_ms'] = [
                {name: 1000 * elapsed for name, elapsed in stages.items()}
                for stages in cache.update_stage_times[before:]
            ]
        return result

    warmup = run_steps(a.warmup_steps) if a.warmup_steps else None
    if not warmup or warmup['cluster_updates'] == 0:
        raise RuntimeError('warmup must include a cluster update')
    records = [run_steps(a.steps) for _ in range(a.repeats)]
    if any(r['cluster_updates'] != a.steps // cache.interval for r in records):
        raise RuntimeError(f'unexpected cluster update count: {records}')
    check_after = compare()
    if check_after['max_abs'] > 0.05 or check_after['selected_tokens_max'] > a.budget:
        raise RuntimeError(f'Multipole post-update graph/eager check failed: {check_after}')
    graph_only = None
    if a.diagnostic_graph_replays:
        cache._set_position(cache.context)
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        start_event.record()
        for _ in range(a.diagnostic_graph_replays):
            graph.replay()
        end_event.record()
        torch.cuda.synchronize()
        graph_only = {'replays': a.diagnostic_graph_replays,
                      'wall_ms_per_replay': (time.perf_counter() - start) * 1000 / a.diagnostic_graph_replays,
                      'gpu_timeline_ms_per_replay': start_event.elapsed_time(end_event) / a.diagnostic_graph_replays,
                      'fixed_position': cache.context}
    result = {'measurement_scope': 'synthetic single attention layer, not full-model inference',
              'implementation': 'author generation kernels via RetroInfer batch adapter',
              'model_geometry': 'Qwen3-8B', 'weights_loaded': False, 'quality_matched': False,
              'batch': b, 'prefix_tokens': a.length, 'budget': a.budget,
              'synthetic_cluster_assignments': True, 'update_interval': cache.interval,
              'same_cache_graph_eager_before': check_before, 'same_cache_graph_eager_after': check_after,
              'warmup': warmup, 'records': records, 'graph_only_diagnostic': graph_only,
              'median_wall_ms_per_batch_step': statistics.median(r['wall_ms_per_batch_step'] for r in records),
              'median_gpu_timeline_ms_per_batch_step': statistics.median(r['gpu_timeline_ms_per_batch_step'] for r in records)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
