"""Compare batched shared-KV QUEST with independent original controllers."""
from types import SimpleNamespace
import torch
import quest.utils as quest
from benchmarks.retro_quest_author_batch_adapter import RetroAuthorQuestBatchCache

torch.manual_seed(991)
torch.set_grad_enabled(False)
batch, prefix, heads, kv, dim = 2, 240, 32, 8, 128
model = SimpleNamespace(batch_size=batch, num_gpus=1, num_layers=3, num_heads=heads,
    num_key_value_heads=kv, head_dim=dim, device_map='cuda:0',
    input_length=prefix, max_new_length=48, dtype=torch.float16)
cache = RetroAuthorQuestBatchCache(model, 32, True)
references = [quest.InferenceController(num_layers=1, num_heads=heads, head_dim=dim,
    page_size=16, page_budget=2, max_seq_len=320, dtype=torch.float16, device='cuda')
    for _ in range(batch)]
for i in range(batch):
    k = torch.randn((1,prefix,kv,dim), dtype=torch.float16, device='cuda')
    v = torch.randn_like(k)
    for layer in range(3):
        cache.prefill_update_kv_cache(None,k,v,layer,i)
    ref=references[i]
    ref.prepare_metadata(prefix); ref.begin_forward(prefix)
    quest.append_kv(k[0].repeat_interleave(4,dim=1).contiguous(),
                    v[0].repeat_interleave(4,dim=1).contiguous(),ref,0)
for c in cache.requests: c._initialize_decode()
q=torch.randn((batch,1,heads,dim),dtype=torch.float16,device='cuda')
k=torch.randn((batch,1,kv,dim),dtype=torch.float16,device='cuda')
v=torch.randn_like(k)
cache._set_position(cache.context)
cache.decode_attention(q,k,v,2)
torch.cuda.synchronize()
graphs,outputs=[],[]
pool=None
for phase in range(1,17):
    for c in cache.requests: c.phase=phase
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,pool=pool):
        out=cache.decode_attention(q,k,v,2)
    if pool is None: pool=graph.pool()
    graphs.append(graph); outputs.append(out)
for step in range(34):
    previous=cache.context
    if previous > prefix and previous % 16 == 0:
        for c in cache.requests:
            c.context=previous
            c._archive_completed_page()
    cache._set_position(previous)
    phase=previous%16+1
    q.normal_(); k.normal_(); v.normal_()
    graphs[phase-1].replay()
    got=outputs[phase-1].clone()
    cache.context+=1
    for i,ref in enumerate(references):
        ref.prepare_metadata(1); ref.begin_forward(1)
        quest.append_kv(k[i].repeat_interleave(4,dim=1).contiguous(),
                        v[i].repeat_interleave(4,dim=1).contiguous(),ref,0)
        scores=quest.decode_estimate(q[i].reshape(1,heads,dim),ref,0)
        quest.decode_topk(scores,ref)
        expected=quest.decode_sparse_attn(q[i].reshape(1,heads,dim),ref,0,
            ref.topk_dindices_buffer).reshape_as(q[i:i+1])
        torch.testing.assert_close(got[i:i+1].float(),expected.float(),atol=0.005,rtol=0.005)
        ref.end_forward()
print('PASS independent requests match original QUEST across 34 graphed steps and page rollovers',flush=True)
