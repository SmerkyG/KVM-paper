"""Inspect sparse index ordering and numerical repeatability on a fixed input."""
from types import SimpleNamespace
import torch
from benchmarks.retro_multipole_adapter import RetroMultipoleCache

torch.manual_seed(277)
torch.set_grad_enabled(False)
h, kh, d, prefix = 32, 8, 128, 65536
model = SimpleNamespace(batch_size=1, num_gpus=1, num_layers=1, num_heads=h,
    num_key_value_heads=kh, head_dim=d, device_map='cuda:0', input_length=prefix,
    max_new_length=8, dtype=torch.bfloat16)
cache = RetroMultipoleCache(model, budget=512)
n = cache.clustered
nc = n // 16
labels = torch.randint(nc, (1, kh, n), device='cuda')
raw = torch.randn((1, kh, n, d), device='cuda', dtype=torch.bfloat16)
counts = torch.zeros((1, kh, nc), device='cuda', dtype=torch.long)
counts.scatter_add_(2, labels, torch.ones_like(labels))
sums = torch.zeros((1, kh, nc, d), device='cuda')
sums.scatter_add_(2, labels.unsqueeze(-1).expand_as(raw), raw.float())
centers = (sums / counts.clamp_min(1).unsqueeze(-1)).bfloat16()
cache.k[0][:, :, :prefix].normal_(); cache.v[0][:, :, :prefix].normal_()
tail = cache.sink+n
cache.local_k[0][:, :prefix-n].copy_(torch.cat((cache.k[0][:, :, :cache.sink],cache.k[0][:, :, tail:prefix]),2).transpose(1,2))
cache.local_v[0][:, :prefix-n].copy_(torch.cat((cache.v[0][:, :, :cache.sink],cache.v[0][:, :, tail:prefix]),2).transpose(1,2))
cache._install_clusters(0, centers, labels, counts)
q = torch.randn((1,1,h,d), device='cuda', dtype=torch.bfloat16)
k = torch.randn((1,1,kh,d), device='cuda', dtype=torch.bfloat16)
v = torch.randn_like(k)
cache.current_raw_k = k.clone(); cache.current_window_q = torch.randn_like(q)
cache._set_position(prefix)
cache.decode_attention(q,k,v,0)
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    out = cache.decode_attention(q,k,v,0)
outputs = []; index_rows = []; count_rows = []
for i in range(20):
    graph.replay()
    outputs.append(out.clone())
    index_rows.append(cache.indices.clone())
    count_rows.append(cache.head_count.clone())
for row in count_rows[1:]: torch.testing.assert_close(row, count_rows[0], atol=0, rtol=0)
for row in index_rows[1:]:
    for head in range(kh):
        count = int(count_rows[0][0,head])
        torch.testing.assert_close(row[0,head,:count].sort().values, index_rows[0][0,head,:count].sort().values, atol=0, rtol=0)
max_diff = max(float((row.float()-outputs[0].float()).abs().max()) for row in outputs[1:])
reordered = sum(not torch.equal(row,index_rows[0]) for row in index_rows[1:])
print(f'fixed-input repeatability: max attention difference={max_diff}; changed raw index order={reordered}/19; selected sets identical', flush=True)
