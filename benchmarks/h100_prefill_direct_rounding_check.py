"""Regression check for strict routing decisions at FP32 rounding boundaries."""
import argparse
import json
import math
from pathlib import Path
import random

import torch
import triton
import triton.language as tl

from benchmarks.h100_prefill_direct_fused import _legacy_proxy


@triton.jit
def _probe(Raw,Inv,Log,Output,Fused,N:tl.constexpr,B:tl.constexpr):
    r=tl.program_id(0)*B+tl.arange(0,B)
    raw=tl.load(Raw+r,mask=r<N,other=0.);inv=tl.load(Inv+r,mask=r<N,other=1.)
    log=tl.load(Log+r,mask=r<N,other=0.)
    tl.store(Output+r,_legacy_proxy(raw,inv,log),mask=r<N)
    tl.store(Fused+r,tl.fma(raw,inv,log),mask=r<N)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rng=random.Random(337);n=1024
    raw=torch.tensor([rng.uniform(-8.,8.)for _ in range(n)],device='cuda',dtype=torch.float32)
    inv=torch.tensor([rng.uniform(.25,4.)for _ in range(n)],device='cuda',dtype=torch.float32)
    log=torch.tensor([math.log(1+i%79)for i in range(n)],device='cuda',dtype=torch.float32)
    # Independent eager PyTorch operations materialize the rounded product.
    product=raw*inv;expected=product+log
    output=torch.empty_like(raw);fused=torch.empty_like(raw)
    _probe[(triton.cdiv(n,128),)](raw,inv,log,output,fused,n,128,num_warps=4)
    assert torch.equal(output,expected)
    flips=int((fused>expected).sum())
    assert flips>0,'fixture failed to exercise an FMA-induced strict-threshold flip'
    assert not bool((output>expected).any())
    data=dict(stage='complete',kind='strict-proxy-rounding',samples=n,
        legacy_max_abs=float((output-expected).abs().max()),fma_differences=int((fused!=expected).sum()),
        fma_false_positive_decisions=flips)
    a.output.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data),flush=True)


if __name__=='__main__':main()
