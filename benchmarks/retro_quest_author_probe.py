"""Probe author QUEST CUDA kernels for full graph capture on a tiny cache."""
import os
import torch
import quest.utils as quest

torch.set_grad_enabled(False)
c = quest.InferenceController(num_layers=1, num_heads=32, head_dim=128,
    page_size=16, page_budget=2, max_seq_len=96, dtype=torch.float16, device='cuda')
c.prepare_metadata(48)
c.begin_forward(48)
k = torch.randn((48, 32, 128), device='cuda', dtype=torch.float16)
v = torch.randn_like(k)
quest.append_kv(k, v, c, 0)
c.end_forward()
c.prepare_metadata(1)
c.begin_forward(1)
q = torch.randn((1, 32, 128), device='cuda', dtype=torch.float16)
nk = torch.randn_like(q)
nv = torch.randn_like(q)

def step():
    quest.append_kv(nk, nv, c, 0)
    scores = quest.decode_estimate(q, c, 0)
    quest.decode_topk(scores, c)
    return quest.decode_sparse_attn(q, c, 0, c.topk_dindices_buffer)

eager = step()
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
stream = torch.cuda.default_stream() if os.environ.get('QUEST_PROBE_DEFAULT_STREAM') else None
with torch.cuda.graph(graph, stream=stream):
    captured = step()
q.copy_(torch.randn_like(q))
graph.replay()
torch.cuda.synchronize()
replayed = captured.clone()
eager = step()
torch.cuda.synchronize()
diff = (eager - replayed).abs().max().item()
print('graph max difference after changing query', diff, flush=True)
if diff > 0.002:
    raise AssertionError('author kernels were not correctly captured')
c.end_forward()
