"""Tune the threshold producer on real in-memory 64K prefill inputs.

Diagnostic only: stops model execution after checking and timing equivalent
producer shapes. No Q/K/V snapshot is saved and no model speed is inferred.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import torch
import triton


class ProbeComplete(Exception):pass


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoint',required=True)
    a=p.parse_args()
    from benchmarks.retro_lod_adapter import install_lod_modules
    install_lod_modules()
    from benchmarks import h100_threshold_prefill_fast as fast
    original=fast._produce;records=[];fixture={}
    class Intercept:
        def __getitem__(self,grid):
            def call(*args,**kwargs):
                if args[9]<2500 or args[8]!=4096:return original[grid](*args,**kwargs)
                original[grid](*args,**kwargs);torch.cuda.synchronize()
                coarse=kwargs.get('COARSE',True)
                outputs=[args[i] for i in ((4,5,6,7) if coarse else (4,7))]+[kwargs['InvRMS']]
                reference=[v.clone() for v in outputs]
                fixture.update(query_shape=list(args[0].shape),scores_shape=list(args[4].shape))
                for m,n,w,stages in ((64,64,4,2),(32,128,4,2),(64,128,4,2),(64,128,8,2),(128,64,4,2),
                              (128,64,8,2),(128,128,8,2),(32,256,4,2),(64,256,4,2),(64,256,8,2),(64,64,8,2),
                              (32,256,4,1),(64,256,4,1),(64,128,4,1),(128,128,8,1)):
                    opts=kwargs|dict(num_warps=w,num_stages=stages)
                    params=(*args[:14],m,n)
                    def fn():return original[(args[0].size(1),triton.cdiv(args[8],m))](*params,**opts)
                    item=dict(m=m,n=n,warps=w,stages=stages)
                    try:
                        fn();torch.cuda.synchronize()
                        errors=[float((v-r).abs().max()) for v,r in zip(outputs,reference)]
                        assert (errors[0]<0.0001 and errors[1]<=0.04 and errors[2]<0.002 and errors[3]<0.0001 and errors[4]==0) if coarse else (errors[0]<0.0001 and errors[1]<0.0001 and errors[2]==0),errors
                        for _ in range(3):fn()
                        values=[]
                        for _ in range(3):
                            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(10):fn()
                            end.record();end.synchronize();values.append(start.elapsed_time(end)/10)
                        item.update(valid=True,max_abs_differences=errors,median_ms=statistics.median(values),samples_ms=values)
                    except (triton.OutOfResources,AssertionError) as e:
                        item.update(valid=False,error=str(e))
                    records.append(item);print(item,flush=True)
                raise ProbeComplete()
            return call
    fast._produce=Intercept()
    from benchmarks import h100_comparison
    sys.argv=['h100_comparison','--method','retro-lod-bf16','--lod-graph','--checkpoint',a.checkpoint,
        '--manifest',str(a.manifest),'--length','65536','--tokens','2','--decode-warmup-steps','0',
        '--warmups','0','--repeats','1','--finish-lod-prefill-cache','--replay-prompt-tokens','--capture-token-ids',
        '--output',str(a.output.with_suffix('.unused.json'))]
    try:h100_comparison.main()
    except ProbeComplete:
        a.output.write_text(json.dumps(dict(stage='complete',scope=__doc__,fixture=fixture,records=records,
            manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest()),indent=2)+'\n');return
    raise RuntimeError('model did not reach producer fixture')


if __name__=='__main__':main()
