"""Compare old and new routing/projection on identical live model inputs.

Stops at the first eligible call, or traces until a difference when requested.
This diagnostic is not a model timing.
Accepts h100_comparison arguments and saves differences, not Q/K/V states.
"""
import json
import os
from pathlib import Path
import sys

import torch
import triton


class ProbeComplete(Exception):pass


def main():
    from benchmarks.retro_lod_adapter import install_lod_modules
    install_lod_modules()
    from benchmarks import h100_threshold_prefill_fast as fast
    from benchmarks import h100_threshold_prefill_direct as direct
    from benchmarks.h100_threshold_prefill_prefix import select
    from benchmarks import h100_threshold_prefill_closed as closed
    produce=fast._produce;project=direct.project;exact=closed.exact_attention;fixture={};report={};history=[]
    find_difference=os.environ.get('LOD_H100_NUMERICS_FIND_DIFFERENCE')=='1'
    output=Path(sys.argv[sys.argv.index('--output')+1]);minimum=int(os.environ.get('LOD_H100_PROBE_MIN_REGIONS','2048'))
    class Intercept:
        def __getitem__(self,grid):
            def call(*args,**kwargs):
                result=produce[grid](*args,**kwargs)
                if args[9]<minimum or args[8]!=4096:
                    fixture.clear();return result
                old=list(args);old[4]=torch.empty_like(args[4]);old[7]=torch.empty_like(args[7])
                inv=torch.empty_like(kwargs['InvRMS']);bits=torch.empty(args[4].size(0),args[8],triton.cdiv(args[9],8),device=args[4].device,dtype=torch.uint8)
                produce[grid](*old,**(kwargs|dict(InvRMS=inv,CandidateBits=bits)))
                fixture.update(scores=old[4],z=old[7],inv=inv,bits=bits)
                return result
            return call
    fast._produce=Intercept()

    def diff(a,b):
        af=a.float();bf=b.float()
        return dict(max_abs=float((af-bf).abs().nan_to_num().max()),different=int((a!=b).sum()))

    def intercept(scores,lengths,bits,mean_v,out,lse,*,t,s,cap,g,buffers,**kwargs):
        result=project(scores,lengths,bits,mean_v,out,lse,t=t,s=s,cap=cap,g=g,buffers=buffers,**kwargs)
        if not fixture:return result
        inv,z,nregions,slots,counts,entries,hc,offsets,c=kwargs['routing'];h=scores.size(0)
        old_slots=torch.empty_like(slots);old_counts=torch.empty_like(counts);old_entries=torch.empty_like(entries)
        old_hc=torch.zeros_like(hc);old_offsets=torch.empty_like(offsets);old_bits=fixture['bits']
        select(fixture['scores'],fixture['inv'],fixture['z'],lengths,nregions,old_slots,old_counts,old_entries,old_hc,old_offsets,old_bits,
               t=t,s=s,cap=cap,g=g,multiplier=c,buffers={})
        report.update(shape=list(scores.shape),scores=diff(scores,fixture['scores']),routing_z=diff(z,fixture['z']),
            inverse=diff(inv,fixture['inv']),selection_bits=diff(bits,old_bits),counts=diff(counts,old_counts))
        p=buffers['direct_probabilities'][:h*t*s];den=buffers['direct_denominator'][:h*t]
        for label,ss,bb in (('same_inputs',scores,bits),('legacy',fixture['scores'],old_bits)):
            other={};oo=torch.empty_like(out);ll=torch.empty_like(lse)
            project(ss,lengths,bb,mean_v,oo,ll,t=t,s=s,cap=cap,g=g,buffers=other,prepared=None,routing=None)
            report[label]=dict(probabilities=diff(p,other['direct_probabilities'][:h*t*s]),
                denominator=diff(den,other['direct_denominator'][:h*t]),lse=diff(lse,ll),output=diff(out,oo))
        fixture['legacy_routed']=(old_slots,old_counts.view(1,h,t),old_entries,None,None,old_hc,old_offsets)
        if not find_difference:raise ProbeComplete()
        return result
    direct.project=intercept

    def exact_intercept(engine,q,page,state_len,routed,buffers,**kwargs):
        out=exact(engine,q,page,state_len,routed,buffers,**kwargs)
        if not fixture:return out
        other=exact(engine,q,page,state_len,fixture['legacy_routed'],{},**kwargs)
        report['exact_output']=diff(out[0],other[0]);report['exact_lse']=diff(out[1],other[1])
        history.append(dict(report));report.clear()
        def changed(value):
            return (value.get('different',0)!=0 or any(changed(v)for v in value.values()if isinstance(v,dict)))
        if changed(history[-1]):raise ProbeComplete()
        return out
    if find_difference:closed.exact_attention=exact_intercept
    from benchmarks import h100_comparison
    try:h100_comparison.main()
    except ProbeComplete:
        records=history if find_difference else report
        output.write_text(json.dumps(dict(stage='complete',scope=__doc__,records=records),indent=2)+'\n')
        print(json.dumps(dict(calls=len(history),first_difference=history[-1]if history else report)),flush=True);return
    if find_difference:
        output.write_text(json.dumps(dict(stage='complete',scope=__doc__,records=history),indent=2)+'\n')
        print(json.dumps(dict(calls=len(history),first_difference=None)),flush=True);return
    raise RuntimeError('model did not reach the diagnostic fixture')


if __name__=='__main__':main()
