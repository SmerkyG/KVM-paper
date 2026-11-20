"""Graph exactly the three component functions from the author paper benchmark.

The callbacks use the author's independent synthetic inputs, including dummy
merge state for sparse attention. They are not a connected attention pipeline.
"""
import argparse
import importlib.util
import json
import math
import pathlib
import sys
import tempfile
import time

import torch
import triton


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--author-benchmark', required=True, type=pathlib.Path)
    p.add_argument('--output', required=True, type=pathlib.Path)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--length', type=int, default=131072)
    p.add_argument('--fraction', type=float, default=0.05)
    p.add_argument('--tokens-per-centroid', type=int, default=16)
    p.add_argument('--split', type=int, default=2048)
    p.add_argument('--replays', type=int, default=512)
    p.add_argument('--local-tokens', type=int, default=256,
                   help='Add a separately initialized dense local field to a second graph')
    p.add_argument('--sink-tokens', type=int, default=0,
                   help='Include this many sink entries in the local field')
    p.add_argument('--dense-only', action='store_true',
                   help='Capture the author paper benchmark dense FlashDecoding fixture')
    p.add_argument('--omit-replacement', action='store_true',
                   help='Measure the Multipole-derived Squeezed variant without centroid replacement')
    a = p.parse_args()
    if a.tokens_per_centroid < 1 or a.length < a.tokens_per_centroid:
        p.error('tokens-per-centroid must be positive and no greater than the remote length')
    if min(a.local_tokens, a.sink_tokens) < 0:
        p.error('local and sink token counts cannot be negative')
    torch.set_grad_enabled(False)
    torch.manual_seed(105)
    text = a.author_benchmark.read_text().split('BLOCK_KV_GLOBAL = 512\n')[0]
    text = text.replace('import matplotlib.pyplot as plt\n', '').replace('import pytest\n', '')
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / 'author_benchmark.py'
        path.write_text(text)
        spec = importlib.util.spec_from_file_location('author_benchmark', path)
        author = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(author)
        author.N_CTX_Q, author.N_CTX_KV = 1, a.length
        author.H_Q, author.H_KV, author.D_HEAD = 32, 8, 128
        author.N_CENTROIDS = a.length // a.tokens_per_centroid
        author.BATCH, author.BLOCK_KV_GLOBAL, author.SPARSITY_RATIO = a.batch, a.split, a.fraction

        if a.dense_only:
            callbacks = []
            original_do_bench = triton.testing.do_bench
            def collect_dense(fn, *args, **kwargs):
                callbacks.append(fn)
                return [0.0, 0.0, 0.0]
            triton.testing.do_bench = collect_dense
            try:
                author.test_flash_decoding()
            finally:
                triton.testing.do_bench = original_do_bench
            if len(callbacks) != 1:
                raise RuntimeError('expected one author dense callback')
            dense = callbacks[0]
            dense()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = dense()
            graph.replay()
            observed = captured.clone()
            expected = dense()
            diff = float((observed.float() - expected.float()).abs().max().item())
            if not torch.isfinite(observed).all() or diff > 0.02:
                raise RuntimeError(f'author dense graph/eager mismatch: {diff}')
            cells = dict(zip(dense.__code__.co_freevars, (c.cell_contents for c in dense.__closure__)))
            q_ref, k_ref, v_ref, scale_ref = (cells[name] for name in ('q', 'k', 'v', 'sm_scale'))
            reference_diff = 0.0
            # Independently check one KV head per request; avoid expanding GQA.
            for slot in range(a.batch):
                head = slot % 8
                logits = torch.matmul(q_ref[slot, head].float(), k_ref[slot, head].float().T) * scale_ref
                reference = torch.matmul(logits.softmax(-1), v_ref[slot, head].float())
                reference_diff = max(reference_diff, float((observed[slot, head].float() - reference).abs().max().item()))
            if reference_diff > 0.02:
                raise RuntimeError(f'author dense independent reference mismatch: {reference_diff}')
            for _ in range(20):
                graph.replay()
            torch.cuda.synchronize()
            samples = []
            for _ in range(2):
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start = time.perf_counter()
                start_event.record()
                for _ in range(a.replays):
                    graph.replay()
                end_event.record()
                torch.cuda.synchronize()
                samples.append({'wall_ms_per_graph': (time.perf_counter() - start) * 1000 / a.replays,
                                'gpu_timeline_ms_per_graph': start_event.elapsed_time(end_event) / a.replays})
            result = {'measurement_scope': 'synthetic single-layer author dense FlashDecoding graph',
                      'batch': a.batch, 'prefix_tokens': a.length, 'model_geometry': 'Qwen3-8B',
                      'weights_loaded': False, 'graph_eager_max_abs': diff,
                      'independent_reference_max_abs': reference_diff,
                      'replays': a.replays, 'samples': samples}
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result), flush=True)
            return

        # The authors construct each benchmark's inputs and then call do_bench
        # on the component closure. Intercept those closures without changing
        # the operations they contain.
        callbacks = []
        original_do_bench = triton.testing.do_bench
        def collect(fn, *args, **kwargs):
            if sys._getframe(1).f_code.co_name in (
                'test_centroid_lookup', 'test_centroid_replacement', 'test_sparse_attn'
            ):
                callbacks.append(fn)
                return [0.0, 0.0, 0.0]
            return original_do_bench(fn, *args, **kwargs)
        triton.testing.do_bench = collect
        try:
            author.test_centroid_lookup()
            author.test_centroid_replacement()
            author.test_sparse_attn()
        finally:
            triton.testing.do_bench = original_do_bench
        if len(callbacks) != 3:
            raise RuntimeError(f'expected 3 author component callbacks, got {len(callbacks)}')
        lookup, replacement, sparse = callbacks
        sparse_cells = dict(zip(sparse.__code__.co_freevars, (c.cell_contents for c in sparse.__closure__)))
        # Author's sparse test uses unspecified dummy merge buffers. Initialize
        # them to a valid empty state, outside the measured graph.
        for name in ('o', 'M', 'L'):
            if name not in sparse_cells:
                raise RuntimeError(f'author sparse fixture missing {name}')
        def reset_merge():
            sparse_cells['o'].zero_()
            sparse_cells['M'].fill_(-float('inf'))
            sparse_cells['L'].zero_()
        reset_merge()
        measured_callbacks = (lookup, sparse) if a.omit_replacement else callbacks
        for fn in measured_callbacks:
            fn()
        torch.cuda.synchronize()
        reset_merge()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_lookup = lookup()
            graph_replacement = None if a.omit_replacement else replacement()
            # The author sparse function overwrites its merge inputs.
            # Reinitialize them inside every replay to keep inputs constant.
            reset_merge()
            graph_sparse = sparse()

        reset_merge()
        graph.replay()
        graph_results = (tuple(x.clone() for x in graph_lookup),
                         tuple(x.clone() for x in graph_sparse)) if a.omit_replacement else (
                         tuple(x.clone() for x in graph_lookup), graph_replacement.clone(),
                         tuple(x.clone() for x in graph_sparse))
        reset_merge()
        eager_results = (lookup(), sparse()) if a.omit_replacement else (lookup(), replacement(), sparse())
        torch.cuda.synchronize()
        diffs = []
        for graph_group, eager_group in zip(graph_results, eager_results):
            if not isinstance(graph_group, tuple):
                graph_group, eager_group = (graph_group,), (eager_group,)
            for observed, expected in zip(graph_group, eager_group):
                if observed.dtype == torch.bool or not observed.is_floating_point():
                    if not torch.equal(observed, expected):
                        raise RuntimeError('author graph and eager selections disagree')
                else:
                    if not torch.isfinite(observed).all() or not torch.isfinite(expected).all():
                        raise RuntimeError('author graph or eager produced non-finite results')
                    diffs.append(float((observed.float() - expected.float()).abs().max()))
        if max(diffs, default=0.0) > 0.05:
            raise RuntimeError(f'author graph/eager mismatch {diffs}')
        graph.replay()
        first_sparse = tuple(x.clone() for x in graph_sparse)
        graph.replay()
        sparse_replay_diff = max(float((x.float() - y.float()).abs().max())
                                 for x, y in zip(first_sparse, graph_sparse))
        if sparse_replay_diff > 0.05:
            raise RuntimeError(f'author sparse graph replay changed outputs: {sparse_replay_diff}')

        def measure(captured):
            reset_merge()
            for _ in range(20):
                captured.replay()
            torch.cuda.synchronize()
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start = time.perf_counter()
            start_event.record()
            for _ in range(a.replays):
                captured.replay()
            end_event.record()
            torch.cuda.synchronize()
            return {'wall_ms_per_graph': (time.perf_counter() - start) * 1000 / a.replays,
                    'gpu_timeline_ms_per_graph': start_event.elapsed_time(end_event) / a.replays}
        paper_only = measure(graph)

        if a.local_tokens == 0 and a.sink_tokens == 0:
            result = {'measurement_scope': 'synthetic single-layer, independent author component fixtures',
                      'batch': a.batch, 'prefix_tokens': a.length, 'fraction': a.fraction, 'split': a.split,
                      'tokens_per_centroid': a.tokens_per_centroid, 'centroids': author.N_CENTROIDS,
                      'model_geometry': 'Qwen3-8B', 'weights_loaded': False,
                      'connected_pipeline': False, 'cache_update_in_graph': False,
                      'centroid_replacement_included': not a.omit_replacement,
                      'local_attention_included': False, 'sink_attention_included': False,
                      'sparse_dummy_merge_initialized_empty': True,
                      'sparse_merge_buffers_reset_inside_graph': True,
                      'sparse_replay_max_abs': sparse_replay_diff,
                      'graph_eager_max_abs': max(diffs, default=0.0),
                      'replays': a.replays, 'paper_only_graph': paper_only}
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result), flush=True)
            return

        # Use the exact same query and the author's sparse-attention operation
        # for the local/sink field; pass its output to the original sparse
        # operation so the two fields are merged in the same softmax.
        local_count = a.local_tokens + a.sink_tokens
        q_local = sparse_cells['q']
        k_local = torch.randn((a.batch, 8, local_count, 128), dtype=torch.bfloat16, device='cuda')
        v_local = torch.randn_like(k_local)
        indices = torch.arange(local_count, dtype=torch.int64, device='cuda').view(1, 1, -1).expand(a.batch, 8, -1).contiguous()
        o_local = torch.zeros(q_local.shape, dtype=torch.float32, device='cuda')
        m_local = torch.full(q_local.shape[:-1], -float('inf'), dtype=torch.float32, device='cuda')
        l_local = torch.zeros_like(m_local)
        def reset_local():
            o_local.zero_(); m_local.fill_(-float('inf')); l_local.zero_()
        def local():
            return author.sparse_flash_attn(q_local, k_local, v_local, indices,
                                             o_local, m_local, l_local, 1 / math.sqrt(128))
        def remote_with_local(local_result):
            return author.sparse_flash_attn(sparse_cells['q'], sparse_cells['k'], sparse_cells['v'],
                                             sparse_cells['k_idx'], *local_result, sparse_cells['sm_scale'])
        reset_local()
        local_eager = local()
        local_snapshot = local_eager[0].clone()
        logits = torch.matmul(q_local.float(), k_local.float().transpose(-1, -2)) / math.sqrt(128)
        local_reference = torch.matmul(logits.softmax(-1), v_local.float())
        local_reference_diff = float((local_snapshot.float() - local_reference).abs().max().item())
        if local_reference_diff > 0.02:
            raise RuntimeError(f'local/sink independent reference mismatch: {local_reference_diff}')
        torch.cuda.synchronize()
        reset_merge(); reset_local()
        graph_local = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_local):
            lookup()
            if not a.omit_replacement:
                replacement()
            # Both attention calls update the supplied output/max/denominator
            # tensors in place. Reinitialize them for every replay, as the
            # released generation path does for each new token.
            reset_local()
            local_graph_output = local()
            combined_graph_output = remote_with_local(local_graph_output)
        reset_merge(); reset_local()
        graph_local.replay()
        graph_combined_output = tuple(x.clone() for x in combined_graph_output)
        reset_local()
        local_eager = local()
        combined_eager = remote_with_local(local_eager)
        torch.cuda.synchronize()
        combined_diffs = [float((x.float() - y.float()).abs().max()) for x, y in zip(graph_combined_output, combined_eager)]
        if any(not torch.isfinite(x).all() for x in graph_combined_output) or max(combined_diffs) > 0.05:
            raise RuntimeError(f'combined local/remote graph/eager mismatch: {combined_diffs}')
        replay_diffs = []
        for _ in range(3):
            graph_local.replay()
            replay_diffs.append(float((combined_graph_output[0].float() - graph_combined_output[0].float()).abs().max().item()))
        if max(replay_diffs) > 0.05:
            raise RuntimeError(f'local/sink graph replay changed outputs: {replay_diffs}')
        saved_local_v = v_local.clone()
        inclusion_diffs = {}
        for name, field in (('local', slice(0, a.local_tokens)),
                            ('sink', slice(a.local_tokens, local_count))):
            if field.start == field.stop:
                continue
            v_local[:, :, field].fill_(100)
            graph_local.replay()
            inclusion_diffs[name] = float((combined_graph_output[0].float() - graph_combined_output[0].float()).abs().max().item())
            v_local.copy_(saved_local_v)
            if inclusion_diffs[name] == 0:
                raise RuntimeError(f'{name} values had no effect on merged output')
        reset_local()
        with_local = measure(graph_local)
        result = {'measurement_scope': 'synthetic single-layer, independent author component fixtures',
                  'batch': a.batch, 'prefix_tokens': a.length, 'fraction': a.fraction, 'split': a.split,
                  'tokens_per_centroid': a.tokens_per_centroid, 'centroids': author.N_CENTROIDS,
                  'model_geometry': 'Qwen3-8B', 'weights_loaded': False,
                  'connected_pipeline': False, 'cache_update_in_graph': False,
                  'centroid_replacement_included': not a.omit_replacement,
                  'sparse_dummy_merge_initialized_empty': True,
                  'sparse_merge_buffers_reset_inside_graph': True,
                  'sparse_replay_max_abs': sparse_replay_diff,
                  'graph_eager_max_abs': max(diffs, default=0.0),
                  'local_independent_reference_max_abs': local_reference_diff,
                  'combined_graph_eager_max_abs': max(combined_diffs),
                  'combined_replay_max_abs': max(replay_diffs),
                  'local_sink_perturbation_max_abs': inclusion_diffs,
                  'local_tokens': a.local_tokens, 'sink_tokens': a.sink_tokens,
                  'total_tokens': a.length + local_count,
                  'local_remote_softmax_merged': True,
                  'local_merge_buffers_reset_inside_graph': True,
                  'replays': a.replays, 'paper_only_graph': paper_only,
                  'paper_plus_local_graph': with_local}
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
