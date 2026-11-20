"""Graph-time complete dense and adapted Multipole attention on matched KV data.

One synthetic Llama-3.1-shaped attention layer, including new-KV writes and
local attention in Multipole. Cluster labels are synthetic but counts, key
centroids, and value centroids are derived from the same input KV. Measures
steady-state replay; periodic updates are measured separately by the model run.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
import torch
from flash_attn import flash_attn_with_kvcache
from benchmarks.retro_multipole_batch_adapter import RetroMultipoleBatchCache

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--length', type=int, default=32768)
p.add_argument('--batch', type=int, default=8)
p.add_argument('--budget', type=int, default=512)
p.add_argument('--time-updates', action='store_true')
p.add_argument('--output', required=True, type=Path)
a = p.parse_args()
torch.manual_seed(105)
torch.set_grad_enabled(False)
b, h, kh, dim, prefix = a.batch, 32, 8, 128, a.length
model = SimpleNamespace(batch_size=b, num_gpus=1, num_layers=1, num_heads=h,
    num_key_value_heads=kh, head_dim=dim, device_map='cuda:0', input_length=prefix,
    max_new_length=768, dtype=torch.bfloat16)
cache = RetroMultipoleBatchCache(model, budget=a.budget)
n = cache.clustered
clusters = n // 16
cache.k[0][:,:,:prefix].normal_()
cache.v[0][:,:,:prefix].normal_()
cache.raw_k[0][:,:,:prefix].copy_(cache.k[0][:,:,:prefix])
labels = torch.randint(clusters, (1,b*kh,n), device='cuda')
raw = cache.raw_k[0][:,:,cache.sink:cache.sink+n]
counts = torch.zeros((1,b*kh,clusters), dtype=torch.int64, device='cuda')
counts.scatter_add_(2, labels, torch.ones_like(labels))
sums = torch.zeros((1,b*kh,clusters,dim), dtype=torch.float32, device='cuda')
sums.scatter_add_(2, labels.unsqueeze(-1).expand_as(raw), raw.float())
centroids = (sums/counts.clamp_min(1).unsqueeze(-1)).bfloat16()
cache._install_clusters(0, centroids, labels, counts)
local_n = prefix - n
tail = cache.sink + n
cache.local_k[0][:,:local_n].copy_(torch.cat((cache.k[0][:,:,:cache.sink],cache.k[0][:,:,tail:prefix]),2).reshape(b,kh,local_n,dim).transpose(1,2))
cache.local_v[0][:,:local_n].copy_(torch.cat((cache.v[0][:,:,:cache.sink],cache.v[0][:,:,tail:prefix]),2).reshape(b,kh,local_n,dim).transpose(1,2))
q = torch.randn((b,1,h,dim), device='cuda', dtype=torch.bfloat16)
k_new = torch.randn((b,1,kh,dim), device='cuda', dtype=torch.bfloat16)
v_new = torch.randn_like(k_new)
cache.current_raw_k = k_new.clone()
cache.current_window_q = q.clone()
cache.context = prefix
cache._set_position(prefix)
dense_k = torch.cat((cache.k[0][:,:,:prefix].reshape(b,kh,prefix,dim).transpose(1,2),k_new),dim=1).contiguous()
dense_v = torch.cat((cache.v[0][:,:,:prefix].reshape(b,kh,prefix,dim).transpose(1,2),v_new),dim=1).contiguous()
lengths = torch.full((b,),prefix+1,dtype=torch.int32,device='cuda')

def dense():
    dense_k.index_copy_(1, cache.all_index, k_new)
    dense_v.index_copy_(1, cache.all_index, v_new)
    return flash_attn_with_kvcache(q,dense_k,dense_v,cache_seqlens=lengths)
def multipole():
    return cache.decode_attention(q,k_new,v_new,0)

def timed(fn):
    for _ in range(5):
        result = fn()
    assert torch.isfinite(result).all()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fn()
    for _ in range(20): graph.replay()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(300)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(300)]
    for start,end in zip(starts,ends):
        start.record(); graph.replay(); end.record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) for start,end in zip(starts,ends))

result = {'measurement_scope':'synthetic single attention layer, not full-model inference',
    'batch':b,'prefix':prefix,'budget':a.budget,'clusters_per_kv_head':clusters,
    'dense_ms':timed(dense),'multipole_complete_attention_ms':timed(multipole)}
result['selected_tokens_min'] = int(cache.head_count.min().item())
result['selected_tokens_max'] = int(cache.head_count.max().item())
if a.time_updates:
    # Restore identical shape and data before each update so that compilation
    # for growing cluster shapes is not mistaken for steady-state execution.
    for arr in (cache.raw_k[0], cache.k[0], cache.v[0]):
        arr[:, :, prefix:].normal_()
    state = [cache.centroids[0], cache.value_centroids[0], cache.counts[0],
             cache.labels[0], cache.local_k[0], cache.local_v[0]]
    saved = [x.clone() for x in state]
    times = []
    for _ in range(6):
        cache.clustered = n
        cache.context = prefix
        for tensor, original in zip(state, saved):
            tensor.copy_(original)
        torch.cuda.synchronize()
        start = time.perf_counter()
        cache._update_clusters()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    result['update_ms_including_three_warmups'] = times
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
