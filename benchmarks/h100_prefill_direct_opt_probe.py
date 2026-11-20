"""Tune fused selection on live 128K inputs, then stop without a model timing.

Accepts h100_comparison arguments. No Q/K/V fixture is written to disk.
"""
import json
from pathlib import Path
import statistics
import sys

import torch
import triton


class ProbeComplete(Exception):pass


def main():
    from benchmarks.retro_lod_adapter import install_lod_modules
    install_lod_modules()
    from benchmarks import h100_threshold_prefill_direct as direct
    from benchmarks.h100_prefill_direct_fused import select_normalize
    original=direct.project;records=[];fixture={}
    output=Path(sys.argv[sys.argv.index('--output')+1])

    def project(scores,lengths,bits,mean_v,out,lse,*,t,s,cap,g,buffers,**kwargs):
        result=original(scores,lengths,bits,mean_v,out,lse,t=t,s=s,cap=cap,g=g,buffers=buffers,**kwargs)
        if s<4096 or t!=4096:return result
        inv,route_z,nregions,slots,counts,entries,hc,offsets,c=kwargs['routing']
        h=scores.size(0);kv=h//g
        probability=buffers['direct_probabilities'][:h*t*s].view(h,t,s)
        denominator=buffers['direct_denominator'][:h*t].view(h,t)
        log=buffers['direct_log_sizes'][:kv*s].view(kv,s)
        reference=[v.clone() for v in (probability,denominator,lse,bits,counts,slots)]
        fixture.update(scores=list(scores.shape),width=slots.size(-1),region_slots=s)
        for warps in (4,8,16):
            def fn():
                hc.zero_()
                return select_normalize[(h*t,)](scores,inv,route_z,lengths,log,nregions,
                    slots,counts,entries,hc,offsets,bits,probability,denominator,lse,
                    t,s,cap,g,slots.size(-1),float(c),triton.next_power_of_2(max(s,8)),num_warps=warps)
            item=dict(warps=warps)
            try:
                kernel=fn();torch.cuda.synchronize()
                assert torch.equal(bits,reference[3]) and torch.equal(counts,reference[4])
                valid=torch.arange(slots.size(-1),device=slots.device)[None,None,None,:]<counts.view(1,h,t,1)
                assert torch.equal(slots[valid],reference[5][valid])
                errors=[float((v-r).abs().max())for v,r in zip((probability,denominator,lse),reference[:3])]
                assert errors[0]<=.001 and errors[1]<=.001 and errors[2]<=1.e-5,errors
                samples=[]
                for _ in range(3):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):fn()
                    end.record();end.synchronize();samples.append(start.elapsed_time(end)/10)
                item.update(valid=True,samples_ms=samples,median_ms=statistics.median(samples),
                            n_regs=kernel.n_regs,n_spills=kernel.n_spills,errors=errors)
            except (triton.OutOfResources,AssertionError) as exc:
                item.update(valid=False,error=str(exc))
            records.append(item);print(json.dumps(item),flush=True)
        raise ProbeComplete()

    direct.project=project
    from benchmarks import h100_comparison
    try:h100_comparison.main()
    except ProbeComplete:
        output.write_text(json.dumps(dict(stage='complete',scope=__doc__,fixture=fixture,records=records),indent=2)+'\n')
        return
    raise RuntimeError('model did not reach the live probe fixture')


if __name__=='__main__':main()
