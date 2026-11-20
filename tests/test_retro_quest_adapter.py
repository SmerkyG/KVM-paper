"""GPU reference check for RetroInfer QUEST cache writes and page boundaries."""
import math
from types import SimpleNamespace

import torch

from benchmarks.retro_quest_adapter import RetroQuestCache


def check(budget):
    torch.manual_seed(731)
    model = SimpleNamespace(batch_size=1, num_gpus=1, num_layers=3,
        num_heads=8, num_key_value_heads=2, head_dim=128, device_map='cuda:0',
        input_length=33, max_new_length=32, dtype=torch.bfloat16)
    cache = RetroQuestCache(model, budget)
    keys = torch.randn((1, 33, 2, 128), device='cuda', dtype=torch.bfloat16)
    vals = torch.randn_like(keys)
    for layer in range(3):
        cache.prefill_update_kv_cache(None, keys, vals, layer, 0)
    for step in range(18):
        q = torch.randn((1, 1, 8, 128), device='cuda', dtype=torch.bfloat16)
        k = torch.randn((1, 1, 2, 128), device='cuda', dtype=torch.bfloat16)
        v = torch.randn_like(k)
        keys = torch.cat((keys, k), dim=1)
        vals = torch.cat((vals, v), dim=1)
        for layer in range(3):
            cache.decode_update_kv_cache(k, v, layer)
            got = cache.decode_attention(q, k, v, layer)
            ref = torch.empty_like(got)
            for h in range(8):
                kh = h // 4
                qh = q[0, 0, h].float()
                kk = keys[0, :, kh].float()
                vv = vals[0, :, kh].float()
                if layer >= 2:
                    count = (kk.shape[0] + 15) // 16
                    page_scores = []
                    for page in range(count):
                        block = kk[page * 16:(page + 1) * 16]
                        page_scores.append(torch.maximum(qh * block.amin(0),
                                                          qh * block.amax(0)).sum())
                    scores = torch.stack(page_scores)
                    scores[-1] = float('inf')
                    pages = scores.topk(min(count, (budget + 15) // 16)).indices
                    ix = torch.cat([torch.arange(int(p) * 16,
                        min((int(p) + 1) * 16, kk.shape[0]), device='cuda') for p in pages])
                    kk, vv = kk[ix], vv[ix]
                ref[0, 0, h] = torch.softmax(kk @ qh / math.sqrt(128), dim=0) @ vv
            torch.testing.assert_close(got.float(), ref.float(), atol=0.008, rtol=0.03)
    print(f'PASS budget={budget}, 18 decode steps across a page boundary', flush=True)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    check(32)
    check(128)
