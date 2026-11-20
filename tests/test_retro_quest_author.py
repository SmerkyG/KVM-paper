"""Compare graph-compatible page bookkeeping with the released QUEST controller."""
from types import SimpleNamespace
import argparse

import torch
import quest.utils as quest
from benchmarks.retro_quest_author_adapter import RetroAuthorQuestCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared-kv', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(917)
    torch.set_grad_enabled(False)
    model = SimpleNamespace(batch_size=1, num_gpus=1, num_layers=3,
        num_heads=32, num_key_value_heads=8, head_dim=128, device_map='cuda:0',
        input_length=240, max_new_length=48, dtype=torch.float16)
    budget_tokens = 32
    cache = RetroAuthorQuestCache(model, budget_tokens, shared_kv=args.shared_kv)
    expected_heads = model.num_key_value_heads if args.shared_kv else model.num_heads
    assert cache.controller.kv_cache.pool.buf.shape[-2] == expected_heads
    assert cache.controller.metadata_cache.pool.buf.shape[-2] == expected_heads
    reference = quest.InferenceController(num_layers=1, num_heads=32, head_dim=128,
        page_size=16, page_budget=2, max_seq_len=320,
        dtype=torch.float16, device='cuda')
    k = torch.randn((1, 240, 8, 128), dtype=torch.float16, device='cuda')
    v = torch.randn_like(k)
    reference.prepare_metadata(240)
    reference.begin_forward(240)
    quest.append_kv(k[0].repeat_interleave(4, dim=1).contiguous(),
                    v[0].repeat_interleave(4, dim=1).contiguous(), reference, 0)
    for layer in range(3):
        cache.prefill_update_kv_cache(None, k, v, layer, 0)
    cache._initialize_decode()
    cache.graph_mode = True
    q_static = torch.empty((1, 1, 32, 128), dtype=torch.float16, device='cuda')
    k_static = torch.empty((1, 1, 8, 128), dtype=torch.float16, device='cuda')
    v_static = torch.empty_like(k_static)
    q_static.normal_()
    k_static.normal_()
    v_static.normal_()
    cache._set_position(cache.context)
    cache.phase = 1
    cache.decode_attention(q_static, k_static, v_static, 2)
    torch.cuda.synchronize()
    graphs, outputs = [], []
    pool = None
    for phase in range(1, 17):
        cache.phase = phase
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            out = cache.decode_attention(q_static, k_static, v_static, 2)
        if pool is None:
            pool = graph.pool()
        graphs.append(graph)
        outputs.append(out)
    for step in range(34):
        if cache.context > cache.prompt_length and cache.context % 16 == 0:
            cache._archive_completed_page()
        cache._set_position(cache.context)
        cache.phase = cache.context % 16 + 1
        q = torch.randn((1, 1, 32, 128), dtype=torch.float16, device='cuda')
        nk = torch.randn((1, 1, 8, 128), dtype=torch.float16, device='cuda')
        nv = torch.randn_like(nk)
        q_static.copy_(q)
        k_static.copy_(nk)
        v_static.copy_(nv)
        graphs[cache.phase - 1].replay()
        got = outputs[cache.phase - 1].clone()
        cache.context += 1
        reference.prepare_metadata(1)
        reference.begin_forward(1)
        quest.append_kv(nk[0].repeat_interleave(4, dim=1).contiguous(),
                        nv[0].repeat_interleave(4, dim=1).contiguous(), reference, 0)
        scores = quest.decode_estimate(q.reshape(1, 32, 128), reference, 0)
        quest.decode_topk(scores, reference)
        expected = quest.decode_sparse_attn(q.reshape(1, 32, 128), reference, 0,
                                             reference.topk_dindices_buffer).reshape_as(q)
        torch.testing.assert_close(got.float(), expected.float(), atol=0.005, rtol=0.005)
        reference.end_forward()
    print(f'PASS shared_kv={args.shared_kv}: graphed author kernels match the expanded controller over 34 steps across KV and metadata page rollovers', flush=True)


if __name__ == '__main__':
    main()
