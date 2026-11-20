"""Compare the Qwen coherence rescale with independent PyTorch arithmetic."""
import argparse
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from benchmarks.retro_lod_adapter import install_lod_modules
    install_lod_modules()
    from lod_attention.kernels.state_rescale import rescale
    torch.manual_seed(315);cases=[]
    for n in (1,193,4096):
        for strided in (False,True):
            x=torch.randn((1,8,n*2 if strided else n,128),device='cuda',dtype=torch.bfloat16)
            r=torch.rand((*x.shape[:-1],1),device='cuda',dtype=torch.float32)*5
            r[...,0,0]=0.
            if n>1:r[...,1,0]=-1.
            if strided:x=x[:,:,::2];r=r[:,:,::2]
            expected=(x.float()/r.clamp_min(1.e-12)).to(x.dtype);actual=rescale(x,r)
            different=int((actual.view(torch.int16)!=expected.view(torch.int16)).sum())
            assert different==0,(n,strided,different)
            cases.append(dict(tokens=n,strided=strided,elements=x.numel(),different_bits=different))
    a.output.write_text(json.dumps(dict(stage='complete',cases=cases),indent=2)+'\n')


if __name__=='__main__':main()
