"""Check packed BF16 maxima against PyTorch, including tie and NaN semantics."""
import argparse
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from benchmarks.retro_lod_adapter import install_lod_modules
    install_lod_modules()
    from lod_attention.kernels.state_maxsim import maximum
    torch.manual_seed(731);records=[]
    for s in (1,33,513,2048,4096):
        x=torch.randn((2,8,19,s),device='cuda',dtype=torch.bfloat16)
        x[0,0,0].fill_(-float('inf'));x[0,0,1].zero_();x[0,0,1,0]=-0.
        x[0,0,2].fill_(1.);x[0,0,3].fill_(float('inf'))
        x[0,0,4,0]=float('nan');x[0,0,4,-1]=float('nan')
        x[0,0,5,-1]=float('nan')
        actual,index=maximum(x);expected=x.max(-1)
        assert torch.equal(index,expected.indices),(s,'indices')
        assert torch.equal(actual.isnan(),expected.values.isnan()),(s,'NaN positions')
        # NaN payload bits are not stable across PyTorch reduction shapes.
        # Every finite result and every index must still match bit-for-bit.
        diff=(actual.view(torch.int16)!=expected.values.view(torch.int16))&~actual.isnan()
        if bool(diff.any()):
            print(dict(columns=s,coordinates=diff.nonzero().tolist(),actual_bits=actual.view(torch.int16)[diff].tolist(),
                expected_bits=expected.values.view(torch.int16)[diff].tolist()),flush=True)
        assert not bool(diff.any()),(s,'value bits')
        records.append(dict(columns=s,rows=index.numel(),different_indices=0,different_finite_value_bits=0,matching_nan_positions=True))
    a.output.write_text(json.dumps(dict(stage='complete',cases=records),indent=2)+'\n')


if __name__=='__main__':main()
