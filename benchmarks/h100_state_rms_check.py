"""Compare fused normalization bit-for-bit with the independent PyTorch steps."""
import argparse
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from benchmarks.retro_lod_adapter import install_lod_modules
    install_lod_modules()
    from lod_attention.kernels.state_rms import normalize
    torch.manual_seed(924);records=[]
    for n in (193,4096):
        x=torch.randn((1,8,n,128),dtype=torch.bfloat16,device='cuda')
        x[:,:,0]=0.;x[:,:,1,0]*=1000.;x[:,:,2,:64]*=1.e-5
        x[:,:,3].fill_(1.);x[:,:,4,:64].zero_()
        f=x.float();expected=(f*f.square().mean(-1,keepdim=True).clamp_min(1.e-12).sqrt().reciprocal()).to(x.dtype)
        for style in (0,3):
            debug={};out=normalize(x,style,diagnostics=debug)
            mean=f.square().mean(-1);root=mean.clamp_min(1.e-12).sqrt();inv=root.reciprocal()
            records.append(dict(tokens=n,style=style,elements=x.numel(),
                different_bits=int((out.view(torch.int16)!=expected.view(torch.int16)).sum()),
                mean_differences=int((debug['mean']!=mean).sum()),root_differences=int((debug['root']!=root).sum()),
                inverse_differences=int((debug['inverse']!=inv).sum()),
                root_same_mean_difference=int((debug['root']!=debug['mean'].clamp_min(1.e-12).sqrt()).sum()),
                inverse_same_root_difference=int((debug['inverse']!=debug['root'].reciprocal()).sum()),
                max_abs=float((out.float()-expected.float()).abs().max())))
    assert all(r[k]==0 for r in records if r['style']==3 for k in
        ('different_bits','mean_differences','root_differences','inverse_differences'))
    assert any(r['different_bits']>0 for r in records if r['style']==0)
    a.output.write_text(json.dumps(dict(stage='complete',records=records),indent=2)+'\n')
    print(json.dumps(records),flush=True)


if __name__=='__main__':main()
