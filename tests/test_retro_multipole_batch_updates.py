"""Validate the author's online update maintains the active KV frontier."""
from types import SimpleNamespace
import torch
from benchmarks.retro_multipole_batch_adapter import RetroMultipoleBatchCache

torch.manual_seed(812)
torch.set_grad_enabled(False)
batch, h, kh, d, prefix = 4, 32, 8, 128, 512
model = SimpleNamespace(batch_size=batch, num_gpus=1, num_layers=1, num_heads=h,
    num_key_value_heads=kh, head_dim=d, device_map='cuda:0', input_length=prefix,
    max_new_length=280, dtype=torch.bfloat16)
cache = RetroMultipoleBatchCache(model, budget=64)
for request in range(batch):
    k = torch.randn((1,prefix,kh,d), dtype=torch.bfloat16, device='cuda')
    v = torch.randn_like(k)
    cache.current_raw_k = torch.randn((1,prefix,kh*d), dtype=torch.bfloat16, device='cuda')
    cache.prefill_update_kv_cache(None,k,v,0,request)
for update in range(2):
    while cache.context + 1 - cache.sink - cache.clustered < 2 * cache.interval:
        raw = torch.randn((batch,kh,d), dtype=torch.bfloat16, device='cuda')
        key = torch.randn_like(raw); value = torch.randn_like(raw)
        slot = cache.context - cache.clustered
        cache.raw_k[0][:,:,cache.context].copy_(raw.reshape(1,batch*kh,d))
        cache.k[0][:,:,cache.context].copy_(key.reshape(1,batch*kh,d))
        cache.v[0][:,:,cache.context].copy_(value.reshape(1,batch*kh,d))
        cache.local_k[0][:,slot].copy_(key)
        cache.local_v[0][:,slot].copy_(value)
        cache.context += 1
    cache._update_clusters()
    clusters = cache.clustered // 16
    center = cache.centroids[0][:,:,:clusters]
    labels = cache.labels[0][:,:,:cache.clustered]
    counts = cache.counts[0][:,:,:clusters]
    assert labels.shape[2] == cache.clustered
    torch.testing.assert_close(counts.sum(dim=2), torch.full_like(counts.sum(dim=2), cache.clustered))
    actual_counts = torch.zeros_like(counts)
    actual_counts.scatter_add_(2, labels, torch.ones_like(labels))
    torch.testing.assert_close(counts, actual_counts)
    values = cache.v[0][:,:,cache.sink:cache.sink+cache.clustered]
    sums = torch.zeros_like(center, dtype=torch.float32)
    sums.scatter_add_(2, labels.unsqueeze(-1).expand_as(values), values.float())
    expected = sums / actual_counts.clamp_min(1).unsqueeze(-1)
    torch.testing.assert_close(cache.value_centroids[0][:,:,:center.shape[2]], expected)
    local_len = cache.context - cache.clustered
    expected_k = torch.cat((cache.k[0][:,:,:cache.sink],cache.k[0][:,:,cache.sink+cache.clustered:cache.context]),dim=2).reshape(batch,kh,local_len,d).transpose(1,2)
    expected_v = torch.cat((cache.v[0][:,:,:cache.sink],cache.v[0][:,:,cache.sink+cache.clustered:cache.context]),dim=2).reshape(batch,kh,local_len,d).transpose(1,2)
    torch.testing.assert_close(cache.local_k[0][:,:local_len], expected_k, rtol=0, atol=0)
    torch.testing.assert_close(cache.local_v[0][:,:local_len], expected_v, rtol=0, atol=0)
    assert int(cache.active_length) == cache.clustered
    print(f'PASS batch={batch}, update={update+1}, context={cache.context}, clustered={cache.clustered}, local={local_len}', flush=True)
