"""Measure the authors' published attention components on an H100.

Loads their benchmark functions without its main loop and measures each
component at fixed splits.
"""
import argparse
import importlib.util
import json
import math
import pathlib
import tempfile

import torch
import triton

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--author-benchmark', required=True, type=pathlib.Path)
p.add_argument('--output', required=True, type=pathlib.Path)
p.add_argument('--batches', default='1,4,8,16')
p.add_argument('--splits', default='512,2048')
p.add_argument('--length', type=int, default=131072)
p.add_argument('--fractions', default='0.1,0.05', help='Fraction of KV tokens retained')
a = p.parse_args()
source = a.author_benchmark.read_text().split('BLOCK_KV_GLOBAL = 512\n')[0]
source = source.replace('import matplotlib.pyplot as plt\n', '').replace('import pytest\n', '')
with tempfile.TemporaryDirectory() as temp:
    path = pathlib.Path(temp) / 'author_benchmark.py'
    path.write_text(source)
    spec = importlib.util.spec_from_file_location('author_benchmark', path)
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    author.N_CTX_Q, author.N_CTX_KV = 1, a.length
    author.H_Q, author.H_KV, author.D_HEAD = 32, 8, 128
    author.N_CENTROIDS = a.length // 16
    torch.set_grad_enabled(False)
    rows = []
    for batch in map(int, a.batches.split(',')):
        author.BATCH = batch
        for split in map(int, a.splits.split(',')):
            author.BLOCK_KV_GLOBAL = split
            stages = {}
            for ratio in map(float, a.fractions.split(',')):
                author.SPARSITY_RATIO = ratio
                stages[str(ratio)] = {name: float(fn()[1]) for name, fn in (
                    ('dense_ms', author.test_flash_decoding),
                    ('lookup_ms', author.test_centroid_lookup),
                    ('replacement_ms', author.test_centroid_replacement),
                    ('sparse_ms', author.test_sparse_attn))}
                print(json.dumps({'batch': batch, 'split': split, 'fraction': ratio, **stages[str(ratio)]}), flush=True)
                torch.cuda.empty_cache()
            rows.append({'batch': batch, 'split': split, 'stages': stages})
            a.output.write_text(json.dumps({'measurement_scope': 'synthetic single-layer component measurements, not full-model inference',
                                            'length': a.length, 'results': rows}, indent=2) + '\n')
