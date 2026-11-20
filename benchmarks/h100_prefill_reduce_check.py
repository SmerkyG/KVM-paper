"""Check separate route-count reducers against a full PyTorch softmax."""
import argparse
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from benchmarks.retro_lod_adapter import install_lod_modules
    install_lod_modules()
    from benchmarks.h100_prefill_reduce_buckets import reduce
    torch.set_grad_enabled(False);torch.manual_seed(271)
    rows,width,d=6173,256,128
    counts=torch.tensor([0,1,31,32,33,63,64,65,127,128,129,255],dtype=torch.int32,device='cuda').repeat((rows+11)//12)[:rows]
    values=torch.randn((rows,width,d),dtype=torch.bfloat16,device='cuda')
    lse=torch.randn((rows,width),device='cuda')*3
    valid=torch.arange(width,device='cuda')[None,:]<counts[:,None]
    masked=lse.masked_fill(~valid,-torch.inf);expected_lse=torch.logsumexp(masked,-1)
    safe=torch.where(counts>0,expected_lse,0.)
    weights=torch.exp(masked-safe[:,None]);expected=(weights[:,:,None]*values.float()).sum(1)
    out=torch.empty((rows,d),device='cuda',dtype=torch.bfloat16);ol=torch.empty((rows,),device='cuda')
    reduce(values,lse,counts,out,ol,width=width,d=d,buffers={})
    error=float((out.float()-expected).abs().max())
    rounded=expected.to(out.dtype)
    torch.testing.assert_close(out,rounded,atol=.0001,rtol=.008)
    assert torch.isfinite(out).all()
    rounding=float((rounded.float()-expected).abs().max())
    torch.testing.assert_close(ol,expected_lse,atol=2.e-5,rtol=2.e-5)
    a.output.write_text(json.dumps(dict(stage='complete',cases=[dict(rows=rows,width=width,max_abs_error=error,bf16_rounding_error=rounding,max_abs_rounded_reference_error=float((out.float()-rounded.float()).abs().max()))]),indent=2)+'\n')


if __name__=='__main__':main()
