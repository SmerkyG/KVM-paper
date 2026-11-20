"""Exercise fast threshold prefill, bounds, caps, and concentrated coarse mass."""
import argparse
import json
import os
import types
from pathlib import Path
import torch
from benchmarks.retro_lod_adapter import RetroLODCache, install_lod_modules


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rule', choices=('entry', 'region'), default='entry')
    p.add_argument('--regions',type=int,default=128,help='Use non-tile-aligned counts to check parallel packing')
    p.add_argument('--stress-layout',action='store_true',help='Include empty and ineligible oversized regions')
    p.add_argument('--multiplier',type=float,default=2.,help='Per-region cutoff; use a large value to test no exact entries')
    p.add_argument('--queries',type=int,default=8)
    p.add_argument('--tile',type=int,default=4,help='Internal tile; use larger values to check expert prefixes across query groups')
    a=p.parse_args();install_lod_modules()
    from benchmarks.h100_threshold_prefill_fast import threshold_prefill_fast
    torch.set_grad_enabled(False);torch.manual_seed(117)
    model=types.SimpleNamespace(batch_size=1,num_gpus=1,input_length=1024,max_new_length=32,
        num_layers=1,num_heads=32,num_key_value_heads=8,head_dim=128,device_map='cuda:0',dtype=torch.bfloat16)
    engine=RetroLODCache(model,mode='two-tier').pools[0].engine
    if a.regions<1:p.error('--regions must be positive')
    sizes=torch.tensor([3,13,30,2,9,5,19,7,4,8,6,11],device='cuda').repeat((a.regions+11)//12)[:a.regions]
    if a.stress_layout:
        sizes[::7]=0
        if a.regions>1:sizes[1]=1025
    tokens,slots=int(sizes.sum().item()),len(sizes)
    owner=torch.repeat_interleave(torch.arange(slots,device='cuda'),sizes)
    owners=owner[None,None,:].expand(1,8,-1).contiguous()
    k=torch.randn((1,8,tokens,128),device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    page=engine._new_page_cache(k,v,owners,state_capacity=slots,sequence_capacity=tokens+256,virtual_k=k,virtual_v=v)
    sk=torch.zeros((1,8,slots,128),device='cuda',dtype=torch.bfloat16);sv=torch.zeros_like(sk)
    for i in range(slots):
        sk[:,:,i]=k[:,:,owner==i].float().sum(2).to(sk.dtype)
        sv[:,:,i]=v[:,:,owner==i].float().sum(2).to(sv.dtype)
    counts=sizes.float()[None,None,:,None].expand(1,8,-1,-1).contiguous()
    if a.queries<1 or a.tile<1:p.error('queries and internal tile must be positive')
    baseq=torch.randn((1,32,a.queries,128),device='cuda',dtype=torch.bfloat16)
    lk=torch.randn((1,8,5,128),device='cuda',dtype=torch.bfloat16);lv=torch.randn_like(lk)
    sinkk=torch.randn((1,8,1,128),device='cuda',dtype=torch.bfloat16);sinkv=torch.randn_like(sinkk)
    os.environ['LOD_H100_FAST_VERIFY']='1';os.environ['LOD_H100_FAST_TILE']=str(a.tile)
    results=[]
    cases = [(0,16,1),(8,16,1),(16,16,1),(0,1,1),(16,1,1),(8,16,32),(16,16,32),(0,16,32)]
    if os.environ.get('LOD_H100_FAST_FULL_SELECT') == '1':
        if a.rule != 'region':
            raise ValueError('full-selection fixture requires per-region routing')
        cases = [(0,16,1), (0,1,1), (0,16,32)]
    for cap,work,scale in cases:
        os.environ['LOD_H100_PREFILL_MAX_REGIONS']=str(cap)
        q=(baseq.float()*scale).to(baseq.dtype)
        scores=torch.matmul(q.float(),lk.repeat_interleave(4,1).float().transpose(-1,-2))*engine.scaling
        ll=torch.logsumexp(scores,-1);lo=torch.matmul(scores.softmax(-1),lv.repeat_interleave(4,1).float()).to(q.dtype)
        engine._lod_fast_prefill_audit=[]
        out=threshold_prefill_fast(engine,q,lk,lv,sk,sv,counts,None,k,v,state_len=slots,state_capacity=slots,
            page_cache=page,local_branch=(lo,ll),sink_k=sinkk,sink_v=sinkv,rule=a.rule,multiplier=a.multiplier if a.rule=='region' else 16.,
            entry_bound=16,work_bound=work,tile_queries=4)
        torch.cuda.synchronize()
        if not bool(torch.isfinite(out).all().item()):raise RuntimeError('nonfinite synthetic output')
        results.append({'cap':cap,'work_bound':work,'query_scale':scale,'checks':engine._lod_fast_prefill_audit})
    if a.rule=='entry' and not any(x['fallback_queries'] for r in results if r['query_scale']==32 for x in r['checks']):
        raise RuntimeError('concentrated case did not exercise numerical fallback')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'stage':'complete','cases':results},indent=2)+'\n')
    print(json.dumps(results),flush=True)

if __name__=='__main__':main()
