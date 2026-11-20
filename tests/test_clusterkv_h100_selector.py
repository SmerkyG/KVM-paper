"""Validate ClusterKV's FP16 centroid lookup, prefix scan, and cluster means."""
from types import SimpleNamespace

import torch

from clusterkv import _clusterkv_knl as kernels

from benchmarks.retro_clusterkv_adapter import RetroClusterKVCache


@torch.inference_mode()
def main():
    device = 'cuda'
    generator = torch.Generator(device=device).manual_seed(7)
    for n in (400, 408, 424):
        kv_heads, heads, dim, budget = 8, 32, 128, 1023
        counts = torch.randint(0, 5, (kv_heads, n), generator=generator, device=device, dtype=torch.int32)
        counts[:, :3] += 1024
        prefix = counts.cumsum(1, dtype=torch.int32)
        length = int(prefix.max())
        indices = torch.arange(length, dtype=torch.int32, device=device).repeat(kv_heads, 1)
        q = torch.zeros((1, heads, dim), device=device, dtype=torch.float16)
        q[:, ::2, 0] = 1
        q[:, 1::2, 0] = -1
        centers = torch.zeros((kv_heads, n, dim), device=device, dtype=torch.float16)
        centers[:, :, 0] = torch.stack([
            torch.randperm(n, device=device, generator=generator) for _ in range(kv_heads)
        ]).half() - n // 2
        near = torch.empty((heads, n), dtype=torch.int32, device=device)
        sizes = torch.empty_like(near)
        selected = torch.full((heads, budget), -1, dtype=torch.int32, device=device)
        kernels.get_neigh_c(q, centers, counts, near, sizes)
        kernels.get_sel_indices(near, sizes, prefix, indices, selected)

        group = heads // kv_heads
        rank = (q[0, :, :1] * centers[:, :, 0].repeat_interleave(group, dim=0)).argsort(descending=True)
        torch.testing.assert_close(near.long(), rank)
        for head in range(heads):
            bank = head // group
            ends = prefix[bank].tolist()
            expected = []
            for cluster in rank[head].tolist():
                end = ends[cluster]
                start = ends[cluster - 1] if cluster else 0
                expected.extend(range(start, end))
                if len(expected) >= budget:
                    break
            torch.testing.assert_close(selected[head].cpu(), torch.tensor(expected[:budget], dtype=torch.int32))
        print(f'PASS {n} centroids with a {budget}-token exact budget', flush=True)

    keys = torch.full((8, 64, 128), 0.5, device=device, dtype=torch.float16)
    labels = torch.arange(64, device=device, dtype=torch.int32).repeat(8, 1) % 4
    centers = torch.zeros((8, 4, 128), device=device, dtype=torch.float16)
    kernels.update_centroids(keys, labels, centers, 64, torch.cuda.current_stream().cuda_stream)
    torch.testing.assert_close(centers, torch.full_like(centers, 0.5), rtol=0, atol=0)
    print('PASS FP16 centroid update uses one count per key', flush=True)

    model = SimpleNamespace(batch_size=1, num_gpus=1, dtype=torch.float16, num_layers=4, num_heads=32,
                            num_key_value_heads=8, head_dim=128, device_map=device, input_length=2048,
                            max_new_length=12)
    cache = RetroClusterKVCache(model, clusters=400, budget=1024, iterations=2, sink=16,
                                window=8, window_clusters=4)
    q = torch.randn((1, 1, 32, 128), device=device, dtype=torch.float16)
    k = torch.randn((1, 2048, 8, 128), device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    for layer in range(model.num_layers):
        cache.prefill_update_kv_cache(None, k, v, layer, 0)
        cache.sync(layer, 0)
    ctrl = cache.controller
    head_to_kv = torch.arange(model.num_heads, device=device) // 4
    for step in range(1, 11):
        for layer in range(model.num_layers):
            cache.decode_update_kv_cache(k[:, :1], v[:, :1], layer)
            out = cache.decode_attention(q, layer)
        selected = torch.cat((ctrl.sink_indices, ctrl.sel_token_indices, ctrl.cur_win_indices), dim=-1)
        assert (selected >= 0).all() and (selected < ctrl.kv_seqlen).all()
        ordered = torch.sort(selected, dim=-1).values
        assert (ordered[:, 1:] > ordered[:, :-1]).all()
        assert (ctrl.sel_token_indices < ctrl.cur_win_indices[:, :1]).all()
        keys, values = ctrl.get_kv(model.num_layers - 1)
        keys = keys[selected, 0, head_to_kv[:, None]].float()
        values = values[selected, 0, head_to_kv[:, None]].float()
        weights = (q[0, 0].float()[:, None] * keys).sum(dim=-1).mul(128 ** -0.5).softmax(dim=-1)
        expected = (weights[:, :, None] * values).sum(dim=1)
        torch.testing.assert_close(out[0, 0].float(), expected, rtol=0.02, atol=0.004)
        print(f'PASS direct attention at decode step {step}', flush=True)
    assert cache.update_count == 1
    print('PASS grouped sparse attention and disjoint selection across a window update', flush=True)

if __name__ == '__main__':
    main()
