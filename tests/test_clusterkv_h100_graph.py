"""Check ClusterKV device planning against the released planner and direct attention."""
from types import SimpleNamespace

import torch

from benchmarks.retro_clusterkv_adapter import RetroClusterKVCache
from benchmarks.retro_clusterkv_graph import ClusterKVDeviceDecode


@torch.inference_mode()
def main():
    torch.manual_seed(7)
    model = SimpleNamespace(batch_size=1, num_gpus=1, dtype=torch.float16, num_layers=4,
                            num_heads=32, num_key_value_heads=8, head_dim=128,
                            device_map='cuda', input_length=2048, max_new_length=24)
    cache = RetroClusterKVCache(model, clusters=400, budget=1024, iterations=2, sink=16,
                                window=8, window_clusters=4)
    ctrl = cache.controller
    def rand(*shape):
        return torch.randn(*shape, device='cuda', dtype=torch.float16)

    q, k, v = rand(1, 1, 32, 128), rand(1, 2048, 8, 128), rand(1, 2048, 8, 128)
    for layer in range(model.num_layers):
        cache.prefill_update_kv_cache(None, k, v, layer, 0)
        cache.sync(layer, 0)
    torch.cuda.current_stream().wait_stream(ctrl.build_cluster_stream)
    planned = ClusterKVDeviceDecode(ctrl, 1024)
    planned.selected = torch.empty_like(planned.selected)
    new_k, new_v = rand(1, 1, 8, 128), rand(1, 1, 8, 128)
    heads = torch.arange(32, device='cuda') // 4

    def step():
        planned.prepare()
        for layer in range(model.num_layers):
            planned.append(new_k, new_v, layer)
            out[layer] = planned.attention(q, layer)

    out = [None] * model.num_layers
    planned.position.fill_(ctrl.prompt_len)
    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()

    for offset in range(18):
        pos = ctrl.prompt_len + offset
        planned.position.fill_(pos)
        graph.replay()
        for layer in range(model.num_layers):
            cache.decode_update_kv_cache(new_k, new_v, layer)
            released = cache.decode_attention(q, layer)
            torch.testing.assert_close(out[layer], released, rtol=0.005, atol=0.001)
        selected = torch.cat((ctrl.sink_indices, ctrl.sel_token_indices, ctrl.cur_win_indices), dim=-1)
        actual = planned.indices[:, :selected.shape[1] - 1]
        torch.testing.assert_close(actual, selected[:, :-1])
        keys = ctrl.kv_cache[-1][selected, 0, 0, heads[:, None]].float()
        values = ctrl.kv_cache[-1][selected, 1, 0, heads[:, None]].float()
        weights = (q[0, 0].float()[:, None] * keys).sum(dim=-1).mul(128 ** -0.5).softmax(dim=-1)
        expected = (weights[:, :, None] * values).sum(dim=1)
        torch.testing.assert_close(out[-1][0, 0].float(), expected, rtol=0.01, atol=0.002)
        if (offset + 1) % ctrl.window == 0:
            # The reference already built the window. Copy those exact means
            # so FP16 atomic reduction order cannot affect the selector check.
            key_start = pos + 1 - ctrl.sink - ctrl.window
            planned.key_start = key_start
            planned.centroid_start = ctrl.nlist + (cache.update_count - 1) * ctrl.window_nlist
            for layer in range(2, ctrl.num_layers):
                n = ctrl.window_nlist
                planned.update_metadata(layer, ctrl.centroids[layer][:, -n:], ctrl.cluster_size[layer][:, -n:],
                                        ctrl.cluster_size_ps[layer][:, -n:] - key_start,
                                        ctrl.cluster_key_indices[layer][:, -ctrl.window:])
    assert cache.update_count == 2
    print('PASS graph vs released planner, exact attention and per-head indices across two window updates')


if __name__ == '__main__':
    main()
