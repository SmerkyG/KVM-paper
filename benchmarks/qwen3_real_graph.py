"""Graph paper components and LoD using an in-memory Qwen3 layer fixture."""
import argparse
import importlib.util
import json
import math
import os
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch


def graph_time(fn, *, replays=512):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    graph.replay()
    observed = out.clone()
    expected = fn().clone()
    torch.cuda.synchronize()
    diff = float((observed.float() - expected.float()).abs().max())
    if not torch.isfinite(observed).all() or diff > 0.05:
        raise RuntimeError(f'graph/eager mismatch {diff}')
    graph.replay()
    repeat = float((out.float() - observed.float()).abs().max())
    if repeat > 0.05:
        raise RuntimeError(f'graph replay mismatch {repeat}')
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(2):
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        start_event.record()
        for _ in range(replays):
            graph.replay()
        end_event.record()
        torch.cuda.synchronize()
        samples.append({'wall_ms': (time.perf_counter() - start) * 1000 / replays,
                        'gpu_ms': start_event.elapsed_time(end_event) / replays})
    return {'median_ms': statistics.median(x['wall_ms'] for x in samples),
            'samples': samples, 'graph_eager_max_abs': diff, 'replay_max_abs': repeat,
            'device_used_bytes_at_graph_setup': total_bytes - free_bytes}


def lod_panel(data):
    from benchmarks.retro_lod_adapter import install_lod_modules
    install_lod_modules()
    from vllm_lod_plugin.config import VLLMLODSettings
    import vllm_lod_plugin.pool as pool_module
    from vllm_lod_plugin.pool import VLLMLayerLODPool
    n = data['metadata']['length']
    q = data['q_rot'][:, :, 0, :].to('cuda').contiguous()
    k = data['k_rot'].to('cuda').contiguous()
    v = data['v'].to('cuda').contiguous()
    z = torch.zeros((1, 8, 128), device='cuda', dtype=torch.bfloat16)
    layer = SimpleNamespace(num_heads=32, num_kv_heads=8, head_size=128, head_size_v=128,
                            impl=SimpleNamespace(scale=1.0 / math.sqrt(128)))
    rows = []
    for mode in ('two-tier', 'three-tier-bf16', 'three-tier-int4'):
        settings = VLLMLODSettings.production(mode=mode, pool_size=1, request_capacity=n + 2048)
        pool = VLLMLayerLODPool(layer, settings=settings, max_requests=1, request_capacity=n + 2048,
                                active_indices=torch.zeros(1, device='cuda', dtype=torch.int32),
                                dtype=torch.bfloat16, device=torch.device('cuda'),
                                has_query_norm=True, has_key_norm=True)
        storage = pool._initial_prefill_storage((0,))
        pool.engine._lod_prefill_storage = storage
        try:
            cache = pool.engine.build_cache_from_bf16(k, v, finalize_cache_for_decode=True)
        finally:
            del pool.engine._lod_prefill_storage
        pool.install_rows((0,), cache)
        pool.engine.reset_runtime_cache()
        pool.catch_up_many([(0, n)])
        out = torch.empty_like(q)
        meta = SimpleNamespace(num_actual_tokens=1, max_seq_len=n + 1)
        original = pool_module.fused_decode_paged_lod_attention
        for scope in ('remote', 'local'):
            # The zero local-length tensor must be allocated before capture.
            zeros = torch.zeros_like(pool.local_lens)
            def fixed_override(*pos, **kw):
                kw.update(new_k=None, new_v=None, store_new_kv=False, advance_local_lens=False)
                if scope == 'remote':
                    kw.update(sink_k=pool.state['sink_k'][:, :, :0, :],
                              sink_v=pool.state['sink_v'][:, :, :0, :], local_lens=zeros)
                return original(*pos, **kw)
            pool_module.fused_decode_paged_lod_attention = fixed_override
            def fn():
                pool.decode(q, z, z, meta, out, store_new_kv=False, advance_local_lens=False)
                return out
            before = fn().clone()
            changes = {}
            for name in ('recent_v', 'sink_v'):
                tensor = pool.state[name]
                saved = tensor.clone()
                tensor.add_(10)
                changes[name] = float((fn().float() - before.float()).abs().max())
                tensor.copy_(saved)
            if scope == 'remote' and any(changes.values()):
                raise RuntimeError(f'remote output included excluded KV: {changes}')
            if scope == 'local' and any(x <= 0 for x in changes.values()):
                raise RuntimeError(f'local/sink KV had no effect: {changes}')
            timing = graph_time(fn)
            rows.append({'mode': mode, 'scope': scope, 'local_sink_perturbation': changes, **timing})
            pool_module.fused_decode_paged_lod_attention = original
            print(json.dumps(rows[-1]), flush=True)
        del pool, cache, storage, out
        torch.cuda.empty_cache()
    return rows


def author_panel(data):
    from multipoleattention.kmeans_ops_sequential import block_kmeans
    source = Path('/home/ubuntu/comparison/MultipoleAttention/benchmark/benchmark_kernels.py').read_text().split('BLOCK_KV_GLOBAL = 512\n')[0]
    source = source.replace('import matplotlib.pyplot as plt\n', '').replace('import pytest\n', '')
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'author.py'
        path.write_text(source)
        spec = importlib.util.spec_from_file_location('author_real_benchmark', path)
        author = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(author)
        author.BLOCK_KV_GLOBAL = 2048
        q = data['q_rot'].to('cuda').reshape(1, 8, 4, 128).contiguous()
        k = data['k_rot'].to('cuda').contiguous()
        v = data['v'].to('cuda').contiguous()
        scale = 1 / math.sqrt(128)
        n = k.shape[2]
        if n < 1024:
            raise ValueError('author panel needs at least 1024 prefix tokens')
        dense = graph_time(lambda: author.flash_attention_optimized(q, k, v, scale))
        dense_out = author.flash_attention_optimized(q, k, v, scale)
        logits = q[0, 0].float() @ k[0, 0].float().T * scale
        ref = logits.softmax(-1) @ v[0, 0].float()
        dense['reference_max_abs'] = float((dense_out[0, 0].float() - ref).abs().max())
        if dense['reference_max_abs'] > 0.03:
            raise RuntimeError(f'dense reference mismatch {dense}')
        rows = [{'method': 'dense', 'scope': 'all', **dense}]
        print(json.dumps(rows[-1]), flush=True)
        for scope in ('remote', 'local'):
            if scope == 'remote':
                rk, rv = k, v
            else:
                rk, rv = k[:, :, 1:-256, :].contiguous(), v[:, :, 1:-256, :].contiguous()
                lk = torch.cat((k[:, :, :1], k[:, :, -256:]), dim=2).contiguous()
                lv = torch.cat((v[:, :, :1], v[:, :, -256:]), dim=2).contiguous()
            m = rk.shape[2]
            for ratio in (16, 8):
                torch.manual_seed(105)
                centers_list, labels_list = [], []
                offset = 0
                for start in range(0, m, 8192):
                    block = rk[0, :, start:min(start + 8192, m)].contiguous()
                    centers, _, labels = block_kmeans(block, block.shape[1] // ratio)
                    centers_list.append(centers)
                    labels_list.append(labels.long() + offset)
                    offset += centers.shape[1]
                labels = torch.cat(labels_list, dim=1).unsqueeze(0).contiguous()
                centers = torch.cat(centers_list, dim=1).unsqueeze(0).to(torch.bfloat16).contiguous()
                count = torch.zeros((1, 8, offset), device='cuda', dtype=torch.int64)
                count.scatter_add_(2, labels, torch.ones_like(labels))
                key_sums = torch.zeros((1, 8, offset, 128), device='cuda', dtype=torch.float32)
                key_sums.scatter_add_(2, labels.unsqueeze(-1).expand_as(rk), rk.float())
                centers = (key_sums / count.clamp_min(1).unsqueeze(-1)).to(torch.bfloat16).contiguous()
                values = torch.zeros((1, 8, offset, 128), device='cuda', dtype=torch.float32)
                values.scatter_add_(2, labels.unsqueeze(-1).expand_as(rv), rv.float())
                values = (values / count.clamp_min(1).unsqueeze(-1)).to(torch.bfloat16).contiguous()
                if not torch.all(count.sum(2) == m):
                    raise RuntimeError('cluster labels do not cover the remote prefix')
                for budget in (512, int(.05*m), int(.1*m)):
                    author.SPARSITY_RATIO = (budget + 0.01) / m
                    def lookup():
                        return author.centroid_lookup_kernel(q, centers, labels, count, scale, None, 4)
                    indices, mask, qk = lookup()
                    indices = indices.clone(); mask = mask.clone(); qk = qk.clone()
                    if indices.shape[-1] != budget or not torch.any(indices >= 0):
                        raise RuntimeError(f'invalid selected indices: {indices.shape}, budget={budget}')
                    for replacement in ((True, False) if ratio == 16 else (False,)):
                        out = torch.zeros_like(q, dtype=torch.float32)
                        maximum = torch.full(q.shape[:-1], -float('inf'), device='cuda', dtype=torch.float32)
                        denom = torch.zeros_like(maximum)
                        if scope == 'local':
                            local_idx = torch.arange(lk.shape[2], device='cuda', dtype=torch.int64).view(1, 1, -1).expand(1, 8, -1).contiguous()
                        def fn():
                            lookup()
                            if replacement:
                                author.centroid_replacement_kernel(q, centers, values, labels, mask, count, scale, None, qk)
                            out.zero_(); maximum.fill_(-float('inf')); denom.zero_()
                            if scope == 'local':
                                state = author.sparse_flash_attn(q, lk, lv, local_idx, out, maximum, denom, scale)
                            else:
                                state = (out, maximum, denom)
                            return author.sparse_flash_attn(q, rk, rv, indices, *state, scale)[0]
                        # Independently verify the selected sparse output for
                        # one KV head and all four of its query heads.
                        selected = indices[0, 0]
                        selected = selected[selected >= 0].long()
                        ref_k, ref_v = rk[0, 0, selected], rv[0, 0, selected]
                        if scope == 'local':
                            ref_k = torch.cat((lk[0, 0], ref_k), dim=0)
                            ref_v = torch.cat((lv[0, 0], ref_v), dim=0)
                        ref_logits = q[0, 0].float() @ ref_k.float().T * scale
                        reference = ref_logits.softmax(-1) @ ref_v.float()
                        ref_error = float((fn()[0, 0].float() - reference).abs().max())
                        if ref_error > 0.03:
                            raise RuntimeError(f'selected-attention reference mismatch: {ref_error}')
                        row = {'method': 'multipole' if replacement else 'squeezed-variant',
                               'scope': scope, 'tokens_per_centroid': ratio, 'budget': budget,
                               'valid_selected_min': int((indices >= 0).sum(-1).min()),
                               'valid_selected_max': int((indices >= 0).sum(-1).max()),
                               'sparse_reference_max_abs': ref_error,
                               **graph_time(fn)}
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                del centers_list, labels_list, centers, labels, count, key_sums, values
                torch.cuda.empty_cache()
        return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=('author', 'lod', 'threshold'), required=True)
    p.add_argument('--payload-fd', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_grad_enabled(False)
    with os.fdopen(os.dup(a.payload_fd), 'rb') as stream:
        data = torch.load(stream, map_location='cpu', weights_only=True)
    if a.kind == 'threshold':
        from benchmarks.qwen3_threshold_real_graph import panel
        rows = panel(data)
    else:
        rows = author_panel(data) if a.kind == 'author' else lod_panel(data)
    result = {'metadata': data['metadata'], 'kind': a.kind,
              'measurement_scope': 'one layer, fixed model-produced Q/K/V; graph-only',
              'records': rows}
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
