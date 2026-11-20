"""Check Multipole's graphed GQA frontier against explicit attention."""
from types import SimpleNamespace
import math
import os
import torch

from benchmarks.retro_multipole_batch_adapter import RetroMultipoleBatchCache
from multipoleattention.kernel_wrappers import centroid_lookup


def main():
    torch.manual_seed(271)
    torch.set_grad_enabled(False)
    batch, h, kh, d, prefix, steps = int(os.environ.get("MULTIPOLE_TEST_BATCH", "4")), 32, 8, 128, 512, 8
    total_kh = batch * kh
    model = SimpleNamespace(batch_size=batch, num_gpus=1, num_layers=1, num_heads=h,
        num_key_value_heads=kh, head_dim=d, device_map='cuda:0', input_length=prefix,
        max_new_length=steps, dtype=torch.bfloat16)
    for replacement, budget in ((True, 64), (False, 64), (True, 1000000)):
        cache = RetroMultipoleBatchCache(model, budget=budget, replacement=replacement)
        n = cache.clustered
        clusters = n // 16
        labels = torch.randint(clusters, (1, total_kh, n), device='cuda')
        raw = torch.randn((1, total_kh, n, d), device='cuda', dtype=torch.bfloat16)
        sums = torch.zeros((1, total_kh, clusters, d), device='cuda', dtype=torch.float32)
        counts = torch.zeros((1, total_kh, clusters), device='cuda', dtype=torch.int64)
        counts.scatter_add_(2, labels, torch.ones_like(labels))
        sums.scatter_add_(2, labels.unsqueeze(-1).expand_as(raw), raw.float())
        centers = (sums / counts.clamp_min(1).unsqueeze(-1)).bfloat16()
        cache.k[0][:, :, :prefix].normal_()
        cache.v[0][:, :, :prefix].normal_()
        tail = cache.sink + n
        cache.local_k[0][:, :prefix-n].copy_(torch.cat((cache.k[0][:, :, :cache.sink], cache.k[0][:, :, tail:prefix]), 2).reshape(batch, kh, prefix-n, d).transpose(1, 2))
        cache.local_v[0][:, :prefix-n].copy_(torch.cat((cache.v[0][:, :, :cache.sink], cache.v[0][:, :, tail:prefix]), 2).reshape(batch, kh, prefix-n, d).transpose(1, 2))
        cache._install_clusters(0, centers, labels, counts)
        q = torch.randn((batch, 1, h + 2 * kh, d), device='cuda', dtype=torch.bfloat16)[:, :, :h]
        qw = torch.randn_like(q)
        k = torch.randn((batch, 1, kh, d), device='cuda', dtype=torch.bfloat16)
        v = torch.randn_like(k)
        cache.current_raw_k = k.clone()
        cache.current_window_q = qw
        cache._set_position(prefix)
        cache.decode_attention(q, k, v, 0)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = cache.decode_attention(q, k, v, 0)
        for step in range(steps):
            q.normal_(); qw.normal_(); k.normal_(); v.normal_()
            cache.current_raw_k.copy_(k)
            cache._set_position(prefix + step)
            graph.replay()
            actual_copy = actual.clone()
            q_window = qw.view(1, total_kh, h // kh, d)
            selected, qk = cache._lookup(q_window, 0)
            counts_all = cache.counts[0].float()
            if budget <= n:
                _, author_selected, _ = centroid_lookup(q_window, cache.centroids[0],
                    cache.labels[0], cache.counts[0], cache.scale, budget, h // kh)
                torch.testing.assert_close(selected & (counts_all > 0),
                    author_selected & (counts_all > 0), atol=0, rtol=0)
            assert ((selected * counts_all).sum(dim=2) <= budget).all()
            assert (cache.head_count <= budget).all()
            if budget > n:
                assert selected[0, :, :clusters][counts[0] > 0].all()
            reference = torch.empty_like(actual, dtype=torch.float32)
            mask = selected.gather(2, cache.labels[0])[:, :, :n]
            for flat_head in range(batch * h):
                request = flat_head // h
                head = flat_head % h
                kv = flat_head // (h // kh)
                group = flat_head % (h // kh)
                remote_k = cache.k[0][0, kv, cache.sink:cache.sink+n][mask[0, kv]]
                remote_v = cache.v[0][0, kv, cache.sink:cache.sink+n][mask[0, kv]]
                local_k = cache.local_k[0][request, :prefix-n+step+1, kv % kh]
                local_v = cache.local_v[0][request, :prefix-n+step+1, kv % kh]
                keys = torch.cat((remote_k, local_k)).float()
                values = torch.cat((remote_v, local_v)).float()
                logits = keys @ q[request, 0, head].float() * cache.scale
                if replacement:
                    use = (~selected[0, kv]) & (cache.counts[0][0, kv] > 0)
                    summary_logits = qk[0, kv, group, use].float() * math.log(2) + cache.counts[0][0, kv, use].float().log()
                    logits = torch.cat((logits, summary_logits))
                    values = torch.cat((values, cache.value_centroids[0][0, kv, use]))
                reference[request, 0, head] = logits.softmax(0) @ values
            torch.testing.assert_close(actual_copy.float(), reference, atol=0.04, rtol=0.04)
        print(f'PASS batch={batch}, replacement={replacement}, budget={budget}: {steps} graph steps match explicit grouped-KV attention', flush=True)


if __name__ == '__main__':
    main()
