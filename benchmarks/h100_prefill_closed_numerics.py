"""Compare coarse arithmetic on identical live Q/K/V; diagnostic, not timing.

No states are saved to disk. FP32 PyTorch projects the same closed frontier,
separately from the original BF16 online full-coarse/subtraction arithmetic.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import triton


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from benchmarks.retro_lod_adapter import install_lod_modules
    install_lod_modules()
    from benchmarks import h100_threshold_prefill_fast as fast
    from benchmarks import h100_threshold_prefill_direct as direct
    producer=fast._produce;project=direct.project;last={};records=[];calls=0

    class Producer:
        def __getitem__(self,grid):
            def call(*args,**kwargs):
                last.update(args=args,kwargs=kwargs)
                return producer[grid](*args,**kwargs)
            return call

    def compare(scores,lengths,bits,mean_v,output,lse,**kwargs):
        nonlocal calls
        project(scores,lengths,bits,mean_v,output,lse,**kwargs)
        index=calls;calls+=1
        if index not in (0,16,48,80):return
        h,_,s=scores.shape;t=min(32,kwargs['t']);g=kwargs['g'];kv=h//g;cap=kwargs['cap'];d=mean_v.size(-1)
        raw=scores[:,:t].contiguous();mask=bits[:,:t].contiguous()
        query=last['args'][0][:,:,:t].contiguous()
        full=torch.empty((h,t,d),device=scores.device,dtype=torch.float32)
        full_lse=torch.empty((h,t),device=scores.device,dtype=torch.float32)
        route=torch.empty_like(full_lse);same_raw=torch.empty_like(raw)
        args=last['args']
        producer[(h,triton.cdiv(t,64))](query,args[1],args[2],lengths,same_raw,full,full_lse,route,
            t,s,cap,g,d,args[13],64,64,COARSE=True,num_warps=4,num_stages=2)
        size=lengths.view(kv,cap)[:,:s].float().repeat_interleave(g,0)[:,None,:]
        r=torch.arange(s,device=scores.device)
        opened=((mask[:,:,r//8].to(torch.int32)>>(r%8))&1).bool()
        native=(raw+size.clamp_min(1).log()).masked_fill(size<=0,-float('inf'))
        closed=native.masked_fill(opened,-float('inf'))
        closed_lse=closed.logsumexp(-1)
        prob=torch.exp(closed-closed_lse[:,:,None]).nan_to_num()
        values=mean_v.view(kv,s,d).float()
        ref=torch.bmm(prob.view(kv,g*t,s),values).view(h,t,d)
        removed=torch.exp(native-full_lse[:,:,None]).masked_fill(~opened,0)
        removed_value=torch.bmm(removed.view(kv,g*t,s),values).view(h,t,d)
        remaining=(1-removed.sum(-1)).clamp_min(0)
        original=(full-removed_value)/remaining.clamp_min(1.e-30)[:,:,None]
        original=torch.where((remaining<1.e-4)[:,:,None],ref,original)
        item=dict(call=index,regions=s,queries=t,score_max_abs=float((same_raw-raw).abs().max()),
            original_subtraction_max_abs=float((original-ref).abs().max()),
            original_subtraction_rmse=float((original-ref).square().mean().sqrt()),
            minimum_closed_mass=float(remaining.min()),fallback_queries=int((remaining<1.e-4).sum()))
        for arithmetic,precision,centered in (('direct','bf16',False),('direct','split-bf16',False),('online64','split-bf16',False),
                                             ('direct','scaled-bf16',False),('direct','scaled-fp16',False),('direct','scaled-fp16',True),
                                             ('direct','offset-bf16',False),('direct','offset-fp16',False)):
            os.environ['LOD_H100_CLOSED_PRECISION']=precision
            os.environ['LOD_H100_CLOSED_ARITHMETIC']=arithmetic
            os.environ['LOD_H100_CLOSED_VALUE_CENTER']=str(int(centered))
            result=torch.empty_like(full);result_lse=torch.empty_like(full_lse)
            project(raw,lengths,mask,mean_v,result,result_lse,t=t,s=s,cap=cap,g=g,buffers={})
            item[arithmetic+'-'+precision+('-centered' if centered else '')]=dict(max_abs=float((result-ref).abs().max()),
                rmse=float((result-ref).square().mean().sqrt()),
                max_lse_error=float((result_lse-closed_lse).abs().max()),
                vs_original_max_abs=float((result-original).abs().max()))
        weight=torch.exp(closed-closed.max(-1,keepdim=True).values).nan_to_num()
        lost=weight.masked_fill(weight.to(torch.float16)!=0,0.).sum(-1)/weight.sum(-1).clamp_min(1.e-30)
        item['fp16_max_underflow_mass_fraction']=float(lost.max())
        item['mean_values_max_abs']=float(values.abs().max())
        os.environ['LOD_H100_CLOSED_PRECISION']='bf16'
        os.environ['LOD_H100_CLOSED_ARITHMETIC']='direct'
        os.environ['LOD_H100_CLOSED_VALUE_CENTER']='0'
        records.append(item);print(item,flush=True)

    fast._produce=Producer();direct.project=compare
    from benchmarks import h100_comparison
    sys.argv=['h100_comparison','--method','retro-lod-bf16','--lod-graph','--checkpoint',a.checkpoint,
        '--manifest',str(a.manifest),'--length','65536','--tokens','2','--decode-warmup-steps','0',
        '--warmups','0','--repeats','1','--finish-lod-prefill-cache','--replay-prompt-tokens','--capture-token-ids',
        '--output',str(a.output.with_suffix('.model.json'))]
    h100_comparison.main()
    assert len(records)==4,(calls,len(records))
    a.output.write_text(json.dumps(dict(stage='complete',scope=__doc__,records=records,
        manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()
