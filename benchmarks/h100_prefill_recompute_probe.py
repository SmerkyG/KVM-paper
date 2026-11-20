"""Tune fused closed selection on live 64K inputs, without saved Q/K/V snapshots."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import torch
import triton


class ProbeComplete(Exception):pass


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoint',required=True);a=p.parse_args()
    from benchmarks.retro_lod_adapter import install_lod_modules
    install_lod_modules()
    from benchmarks import h100_threshold_prefill_recompute as mod
    original=mod.select_closed;records=[];fixture={}
    class Intercept:
        def __getitem__(self,grid):
            def call(*args,**kwargs):
                t,s=args[14:16]
                if t!=4096 or s<2500:return original[grid](*args,**kwargs)
                q=args[0];h=q.size(1);slots,counts,entries,hc,offsets,coarse,lse=args[7:14]
                hc.zero_();original[grid](*args,**kwargs);torch.cuda.synchronize()
                reference=[x.clone()for x in (slots,counts,entries,hc,coarse,lse)]
                valid=torch.arange(slots.size(-1),device=q.device)[None,None,None,:]<counts.view(1,h,t,1)
                fixture.update(query_shape=list(q.shape),regions=s,mean_opened_regions=float(counts.float().mean()))
                for m,n,ag,w,st in ((32,64,False,4,2),(16,64,False,4,2),(16,128,False,4,2),
                    (32,128,False,4,2),(64,64,False,4,2),(32,64,True,4,2),(32,128,True,4,2),
                    (16,128,False,8,2),(32,128,False,8,2),(16,256,False,4,1),(32,64,False,8,2)):
                    params=(*args[:22],m,n,ag);opts=kwargs|dict(num_warps=w,num_stages=st)
                    def fn():
                        hc.zero_();return original[(h,triton.cdiv(t,m))](*params,**opts)
                    item=dict(m=m,n=n,aggregate=ag,warps=w,stages=st)
                    try:
                        fn();torch.cuda.synchronize()
                        for x,y in zip((counts,entries,hc),reference[1:4]):torch.testing.assert_close(x,y,atol=0,rtol=0)
                        assert torch.equal(slots[valid],reference[0][valid])
                        err=float((coarse-reference[4]).abs().max());le=float((lse-reference[5]).abs().max())
                        assert err<=.04 and le<=2.e-5,(err,le)
                        heads=torch.arange(h,device=q.device)[:,None,None]
                        keys=((heads*s+slots[0])*t+offsets[0])[valid[0]].sort().values
                        expert=keys//t;rank=keys%t
                        assert not keys.numel()or(int(rank[0])==0 and bool(torch.where(expert[1:]==expert[:-1],rank[1:]==rank[:-1]+1,rank[1:]==0).all()))
                        for _ in range(3):fn()
                        values=[]
                        for _ in range(3):
                            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
                            for _ in range(5):fn()
                            end.record();end.synchronize();values.append(start.elapsed_time(end)/5)
                        item.update(valid=True,median_ms=statistics.median(values),samples_ms=values,max_abs_error=err,max_lse_error=le)
                    except(triton.OutOfResources,AssertionError)as e:item.update(valid=False,error=str(e))
                    records.append(item);print(item,flush=True)
                raise ProbeComplete()
            return call
    mod.select_closed=Intercept()
    from benchmarks import h100_comparison
    sys.argv=['h100_comparison','--method','retro-lod-bf16','--lod-graph','--checkpoint',a.checkpoint,
        '--manifest',str(a.manifest),'--length','65536','--tokens','2','--decode-warmup-steps','0',
        '--warmups','0','--repeats','1','--finish-lod-prefill-cache','--replay-prompt-tokens','--capture-token-ids',
        '--output',str(a.output.with_suffix('.unused.json'))]
    try:h100_comparison.main()
    except ProbeComplete:
        a.output.write_text(json.dumps(dict(stage='complete',scope=__doc__,fixture=fixture,records=records),indent=2)+'\n');return
    raise RuntimeError('fixture not reached')


if __name__=='__main__':main()
