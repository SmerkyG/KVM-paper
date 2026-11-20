"""Benchmark unchanged threshold packing on live model-produced prefill scores.

Runs through cluster-run. Stops the model intentionally after collecting the
diagnostic; it is not a model latency or quality result. No Q/K/V snapshot is
saved. Every candidate sees the same query, scores, lengths and normalizer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import torch


class ProbeComplete(Exception):pass


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--length',type=int,default=128000);p.add_argument('--min-regions',type=int,default=5000)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--backends',nargs='+',choices=('scan','hybrid','bias','sparse','bitmap','row'),default=['scan','hybrid','bias'])
    a=p.parse_args()
    from benchmarks.retro_lod_adapter import install_lod_modules
    install_lod_modules()
    from benchmarks import h100_threshold_prefill_scan as scan
    from benchmarks import h100_threshold_prefill_fast as fast
    original=scan.select;results=[];fixture={}

    def intercepted(scores,inv,logz,lengths,nregions,slots,counts,entries,hc,offsets,bits,**kw):
        if kw['s']<a.min_regions or kw['t']!=4096:
            return original(scores,inv,logz,lengths,nregions,slots,counts,entries,hc,offsets,bits,**kw)
        h,t,s=scores.shape;g=kw['g'];q=kw['q'];width=slots.size(-1)
        def baseline():
            hc.zero_()
            fast._select_region_full[(h*t,)](q,scores,logz,lengths,nregions,slots,counts,entries,hc,offsets,bits,
                t,s,kw['cap'],g,q.size(-1),width,float(kw['multiplier']),__import__('triton').next_power_of_2(max(s,8)),
                1,False,num_warps=8)
        baseline();torch.cuda.synchronize()
        reference=[x.clone() for x in (slots,counts,entries,hc,bits)]
        valid=torch.arange(width,device=scores.device)[None,None,None,:]<counts.view(1,h,t,1)
        def validate():
            for x,y in zip((counts,entries,hc,bits),reference[1:]):torch.testing.assert_close(x,y,atol=0,rtol=0)
            assert torch.equal(slots[valid],reference[0][valid]),'selected region ordering changed'
            selected=slots[0];off=offsets[0]
            heads=torch.arange(h,device=scores.device)[:,None,None]
            keys=((heads*s+selected)*t+off)[valid[0]].sort().values
            if keys.numel():
                expert=keys//t;rank=keys%t
                assert int(rank[0])==0
                assert bool(torch.where(expert[1:]==expert[:-1],rank[1:]==rank[:-1]+1,rank[1:]==0).all()),'expert offsets collide or leave gaps'
        def bench(fn):
            for _ in range(3):fn()
            values=[]
            for _ in range(3):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(10):fn()
                end.record();end.synchronize();values.append(start.elapsed_time(end)/10)
            return dict(median_ms=statistics.median(values),samples_ms=values)
        fixture.update(query_shape=list(q.shape),scores_shape=list(scores.shape),multiplier=kw['multiplier'],
            mean_regions=float(counts.float().mean()),maximum_regions=int(counts.max()),
            mean_entries=float(entries.float().mean()),reference_bits_sha256=hashlib.sha256(bits.cpu().numpy().tobytes()).hexdigest())
        results.append(dict(backend='full-row',**bench(baseline)))
        if 'row' in a.backends:
            for m,w in ((1,2),(1,4),(2,2),(2,4),(4,4),(4,8),(8,8)):
                def fn():
                    hc.zero_()
                    grid=(h,__import__('triton').cdiv(t,m)) if m>1 else (h*t,)
                    fast._select_region_full[grid](q,scores,logz,lengths,nregions,slots,counts,entries,hc,offsets,bits,
                        t,s,kw['cap'],g,q.size(-1),width,float(kw['multiplier']),__import__('triton').next_power_of_2(max(s,8)),
                        m,m>1,num_warps=w)
                fn();validate();results.append(dict(backend='row',m=m,n=__import__('triton').next_power_of_2(max(s,8)),warps=w,**bench(fn)))
                print(results[-1],flush=True)
        for m,n in ((8,128),(16,128),(32,128),(64,128),(16,256),(32,256)):
            if 'scan' not in a.backends:break
            opts=kw|dict(m=m,n=n,warps=4)
            def fn():return original(scores,inv,logz,lengths,nregions,slots,counts,entries,hc,offsets,bits,**opts)
            fn();validate();results.append(dict(backend='scan',m=m,n=n,warps=4,**bench(fn)))
            print(results[-1],flush=True)
        for m,n,warps in ((16,128,4),(16,64,4),(16,128,8),(32,128,8),(32,256,8),(64,128,8),(32,256,16)):
            if 'hybrid' not in a.backends:break
            opts=kw|dict(m=m,n=n,warps=warps,atomic_experts=True)
            def fn():return original(scores,inv,logz,lengths,nregions,slots,counts,entries,hc,offsets,bits,**opts)
            fn();validate();results.append(dict(backend='hybrid',m=m,n=n,warps=warps,**bench(fn)))
            print(results[-1],flush=True)
        from benchmarks.h100_threshold_prefill_bias import select as biased
        for warps in (4,8,16):
            if 'bias' not in a.backends:break
            def fn():return biased(scores,inv,logz,lengths,nregions,slots,counts,entries,hc,offsets,bits,**(kw|dict(warps=warps)))
            fn();validate();results.append(dict(backend='bias',warps=warps,**bench(fn)))
            print(results[-1],flush=True)
        if 'sparse' in a.backends:
            from benchmarks.h100_threshold_prefill_sparse import select as sparse
            for m,n,pm in ((8,128,4),(16,128,4),(32,128,4),(16,256,4),(32,256,4),(32,256,8),(16,128,8),(16,128,2)):
                opts=kw|dict(m=m,n=n,warps=4,pack_m=pm)
                def fn():return sparse(scores,inv,logz,lengths,nregions,slots,counts,entries,hc,offsets,bits,**opts)
                fn();validate();results.append(dict(backend='sparse',m=m,n=n,pack_m=pm,warps=4,**bench(fn)))
                print(results[-1],flush=True)
        if 'bitmap' in a.backends:
            from benchmarks.h100_threshold_prefill_bitmap import select as bitmap
            for m,n in ((8,128),(16,128),(32,128),(16,256),(32,256)):
                opts=kw|dict(m=m,n=n,warps=4)
                def fn():return bitmap(scores,inv,logz,lengths,nregions,slots,counts,entries,hc,offsets,bits,**opts)
                fn();validate();results.append(dict(backend='bitmap',m=m,n=n,warps=4,**bench(fn)))
                print(results[-1],flush=True)
        raise ProbeComplete()
    scan.select=intercepted
    os.environ.update(LOD_H100_FAST_SCAN_SELECT='1',LOD_H100_FAST_FULL_SELECT='1')
    from benchmarks import h100_comparison
    sys.argv=['h100_comparison','--method','retro-lod-bf16','--lod-graph','--checkpoint',a.checkpoint,
        '--manifest',str(a.manifest),'--length',str(a.length),'--tokens','2','--decode-warmup-steps','0',
        '--warmups','0','--repeats','1','--finish-lod-prefill-cache','--replay-prompt-tokens','--capture-token-ids',
        '--output',str(a.output.with_suffix('.unused.json'))]
    try:h100_comparison.main()
    except ProbeComplete:
        a.output.write_text(json.dumps(dict(stage='complete',scope=__doc__,fixture=fixture,records=results,
            manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest()),indent=2)+'\n')
        return
    raise RuntimeError('model did not reach the requested live selection fixture')


if __name__=='__main__':main()
