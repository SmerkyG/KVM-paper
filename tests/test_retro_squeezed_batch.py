"""Check batched Squeezed outputs against explicit selected-KV attention."""
from types import SimpleNamespace
import math
import torch
from benchmarks.retro_squeezed_batch_adapter import RetroSqueezedBatchCache

torch.manual_seed(204)
torch.set_grad_enabled(False)
batch,heads,kv,dim,prefix,obs,clusters=2,32,8,128,256,8,16
model=SimpleNamespace(batch_size=batch,num_gpus=1,num_layers=1,num_heads=heads,
    num_key_value_heads=kv,head_dim=dim,device_map='cuda:0',input_length=prefix,
    max_new_length=24,dtype=torch.bfloat16)
index={'centroids':{0:torch.randn(1,heads,clusters,dim)},
       'labels':{0:torch.randint(clusters,(1,heads,prefix-obs))},
       'threshold':{0.9:0.0035},'observation_window':obs}
cache=RetroSqueezedBatchCache(model,index,0.9)
for i in range(batch):
    k=torch.randn(1,prefix,kv,dim,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)
    cache.prefill_update_kv_cache(None,k,v,0,i)
q=torch.randn(batch,1,heads,dim,device='cuda',dtype=torch.bfloat16)
k=torch.randn(batch,1,kv,dim,device='cuda',dtype=torch.bfloat16)
v=torch.randn_like(k)
cache._set_position(prefix)
cache.decode_attention(q,k,v,0)
torch.cuda.synchronize()
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph): out=cache.decode_attention(q,k,v,0)
for step in range(20):
    q.normal_(); k.normal_(); v.normal_()
    cache._set_position(prefix+step)
    graph.replay(); got=out.clone()
    for i,request in enumerate(cache.requests):
        mask=request.select(q[i:i+1].transpose(1,2).contiguous(),0)
        expected=[]
        for head in range(heads):
            kh=head//4
            fk=request.fixed_k[0][0,kh][mask[0,head]]
            fv=request.fixed_v[0][0,kh][mask[0,head]]
            dk=request.dynamic_k[0][0,:step+1,kh]
            dv=request.dynamic_v[0][0,:step+1,kh]
            keys=torch.cat((fk,dk)).float(); values=torch.cat((fv,dv)).float()
            expected.append(torch.softmax(keys @ q[i,0,head].float()/math.sqrt(dim),0) @ values)
        expected=torch.stack(expected).reshape(1,1,heads,dim)
        torch.testing.assert_close(got[i:i+1].float(),expected,atol=0.035,rtol=0.035)
print('PASS distinct requests match explicit selected-KV attention across 20 graph steps',flush=True)
