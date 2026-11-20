"""Time the authors' 8K-cluster online update and amortize over 128 steps."""
import argparse
import json
import pathlib
import torch
import triton
from multipoleattention.kmeans_ops_sequential import sequential_kmeans_append, block_kmeans

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--batches', default='1,4,8,16')
p.add_argument('--tokens-per-centroid', type=int, default=16)
p.add_argument('--output', required=True, type=pathlib.Path)
a = p.parse_args()
if a.tokens_per_centroid < 1 or 8192 % a.tokens_per_centroid or 128 % a.tokens_per_centroid:
    p.error('tokens-per-centroid must divide 8192 and 128')
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision('high')
torch.manual_seed(0)
rows = []
for batch in map(int, a.batches.split(',')):
    data = torch.randn(batch * 8, 8192, 128, device='cuda', dtype=torch.float32)
    new = torch.randn(batch * 8, 128, 128, device='cuda', dtype=torch.float32)
    all_data = torch.cat((data, new), dim=1)
    centroids, counts, labels = block_kmeans(data, 8192 // a.tokens_per_centroid)
    fn = lambda: sequential_kmeans_append(new, centroids, counts, all_data, 128 // a.tokens_per_centroid, lloyd_iters=3)
    update_ms = float(triton.testing.do_bench(fn, warmup=500, rep=500, quantiles=[0.2, 0.5, 0.8])[1])
    row = {'batch': batch, 'tokens_per_centroid': a.tokens_per_centroid,
           'centroids': 8192 // a.tokens_per_centroid, 'new_centroids': 128 // a.tokens_per_centroid,
           'update_ms': update_ms, 'amortized_ms': update_ms / 128}
    rows.append(row)
    a.output.write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(row), flush=True)
    del data, new, all_data, centroids, counts, labels
    torch.cuda.empty_cache()
