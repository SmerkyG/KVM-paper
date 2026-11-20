"""Compare Multipole's published dense attention with RetroInfer's FlashAttention.

Both receive the same BF16 Q/K/V data in their native KV layouts. Measure
isolated warmed calls using the paper's Triton timing method and separately
measure replay of each entire attention call inside a CUDA graph.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import tempfile

import torch
import triton
from flash_attn import flash_attn_with_kvcache

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--author-benchmark', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
p.add_argument('--batches', default='1,4,8,16')
p.add_argument('--lengths', default='32768,131072')
p.add_argument('--splits', default='512,2048')
a = p.parse_args()
source = a.author_benchmark.read_text().split('BLOCK_KV_GLOBAL = 512\n')[0]
source = source.replace('import matplotlib.pyplot as plt\n', '').replace('import pytest\n', '')
torch.set_grad_enabled(False)
torch.manual_seed(2026)

def measured(fn):
    return float(triton.testing.do_bench(fn, warmup=500, rep=500, quantiles=[0.2, 0.5, 0.8])[1])

def graph_time(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    for _ in range(10):
        graph.replay()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    for start, end in zip(starts, ends):
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    result = statistics.median(start.elapsed_time(end) for start, end in zip(starts, ends))
    return result

with tempfile.TemporaryDirectory() as temp:
    path = Path(temp) / 'author_benchmark.py'
    path.write_text(source)
    spec = importlib.util.spec_from_file_location('author_benchmark', path)
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    rows = []
    for length in map(int, a.lengths.split(',')):
        for batch in map(int, a.batches.split(',')):
            q = torch.randn((batch, 1, 32, 128), device='cuda', dtype=torch.bfloat16)
            k = torch.randn((batch, length, 8, 128), device='cuda', dtype=torch.bfloat16)
            v = torch.randn_like(k)
            ka = k.transpose(1, 2).contiguous()
            va = v.transpose(1, 2).contiguous()
            qa = q.reshape(batch, 8, 4, 128)
            lens = torch.full((batch,), length, device='cuda', dtype=torch.int32)
            ref = lambda: flash_attn_with_kvcache(q, k, v, cache_seqlens=lens)
            output = ref().reshape(batch, 8, 4, 128).float()
            times = {'flashattention': {'do_bench_ms': measured(ref), 'graph_ms': graph_time(ref)}}
            for split in map(int, a.splits.split(',')):
                author.BLOCK_KV_GLOBAL = split
                fn = lambda: author.flash_attention_optimized(qa, ka, va, 1/math.sqrt(128))
                actual = fn().float()
                difference = (actual - output).abs()
                torch.testing.assert_close(actual, output, atol=0.005, rtol=0.05)
                times[f'author_{split}'] = {'do_bench_ms': measured(fn), 'graph_ms': graph_time(fn),
                    'max_abs_error': float(difference.max()), 'mean_abs_error': float(difference.mean())}
            row = {'length': length, 'batch': batch, 'timings': times}
            rows.append(row)
            a.output.write_text(json.dumps({'measurement_scope': 'synthetic single attention layer, not full-model inference',
                                            'results': rows}, indent=2) + '\n')
            print(json.dumps(row), flush=True)
            del q, k, v, ka, va, qa, lens, output, actual, difference, ref, fn
            torch.cuda.empty_cache()
