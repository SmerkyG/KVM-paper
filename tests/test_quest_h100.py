"""GPU verification for the benchmark-only QUEST implementation.

Run as a cluster-run experiment with the H100 vLLM environment.
"""
import math
import os
import torch
from vllm_lod_plugin.quest_kernels import quest_attention, update_bounds


def check(length, budget, partial_prefix):
    torch.manual_seed(23 + length)
    device = os.environ.get('QUEST_TEST_DEVICE', 'cuda')
    page_size, kv_heads, heads, dim = 16, 2, 8, 128
    pages = (length + page_size - 1) // page_size
    physical = torch.randperm(pages + 3, device=device)[:pages].int()
    table = torch.zeros((1, pages + 3), dtype=torch.int32, device=device)
    table[0, :pages] = physical
    # vLLM's NHD physical storage presents a non-contiguous logical BHND view.
    cache = torch.zeros((pages + 3, page_size, kv_heads, dim*2), device=device,
                        dtype=torch.bfloat16).transpose(1, 2)
    k = torch.randn((length, kv_heads, dim), device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    slots = (physical[torch.arange(length, device=device)//page_size] * page_size
             + torch.arange(length, device=device)%page_size).long()
    for start, end in ((0, partial_prefix), (partial_prefix, length)):
        if end == start:
            continue
        cache[slots[start:end]//page_size, :, slots[start:end]%page_size, :dim] = k[start:end]
        cache[slots[start:end]//page_size, :, slots[start:end]%page_size, dim:] = v[start:end]
        if start == 0:
            bounds = torch.empty((pages+3, kv_heads, 2, dim), device=device, dtype=torch.bfloat16)
        update_bounds(cache, slots[start:end], bounds)
    q = torch.randn((1, heads, dim), device=device, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    quest_attention(q, cache, table, torch.tensor([length], device=device, dtype=torch.int32),
                    bounds, out, budget, 1/math.sqrt(dim))
    expected = torch.empty_like(out)
    for h in range(heads):
        kh = h//(heads//kv_heads)
        bounds_lo = torch.stack([k[i:i+page_size,kh].float().amin(0) for i in range(0,length,page_size)])
        bounds_hi = torch.stack([k[i:i+page_size,kh].float().amax(0) for i in range(0,length,page_size)])
        qh = q[0,h].float()
        scores = torch.maximum(qh*bounds_lo, qh*bounds_hi).sum(-1)
        scores[-1] = torch.inf
        chosen = scores.topk(min(pages, (budget+15)//16)).indices
        indices = torch.cat([torch.arange(int(p)*page_size, min((int(p)+1)*page_size,length), device=device) for p in chosen])
        probs = torch.softmax((k[indices,kh].float() @ qh) / math.sqrt(dim), dim=0)
        expected[0,h] = probs @ v[indices,kh].float()
    torch.testing.assert_close(out.float(), expected.float(), atol=0.006, rtol=0.025)
    print(f'PASS length={length} budget={budget} split_update={partial_prefix}', flush=True)


def check_graph():
    torch.manual_seed(91)
    cache = torch.randn((5, 16, 2, 256), device='cuda', dtype=torch.bfloat16).transpose(1, 2)
    slots = torch.arange(48, device='cuda', dtype=torch.int64)
    bounds = torch.empty((5, 2, 2, 128), device='cuda', dtype=torch.bfloat16)
    table = torch.tensor([[0, 1, 2, 3, 4]], device='cuda', dtype=torch.int32)
    length = torch.tensor([48], device='cuda', dtype=torch.int32)
    query = torch.randn((1, 8, 128), device='cuda', dtype=torch.bfloat16)
    output = torch.empty_like(query)
    update_bounds(cache, slots, bounds)
    quest_attention(query, cache, table, length, bounds, output, 32, 1/math.sqrt(128))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        update_bounds(cache, slots, bounds)
        quest_attention(query, cache, table, length, bounds, output, 32, 1/math.sqrt(128))
    query.copy_(torch.randn_like(query))
    cache[2, :, 3, :].copy_(torch.randn_like(cache[2, :, 3, :]))
    graph.replay()
    graph_out = output.clone()
    update_bounds(cache, slots, bounds)
    quest_attention(query, cache, table, length, bounds, output, 32, 1/math.sqrt(128))
    torch.testing.assert_close(graph_out.float(), output.float(), atol=0.001, rtol=0.001)
    print('PASS graph replay after query and cache changes', flush=True)


if __name__ == '__main__':
    for length, budget, split in ((33, 128, 13), (123, 32, 61), (319, 144, 256), (513, 1024, 512)):
        check(length, budget, split)
    if os.environ.get('QUEST_TEST_DEVICE', 'cuda') == 'cuda':
        check_graph()
