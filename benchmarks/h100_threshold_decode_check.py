"""Validate optimized uncapped decode against the independent PyTorch frontier."""
import argparse
import json
import os
from pathlib import Path
import torch
from benchmarks.h100_threshold_synthetic_check import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation',choices=('packed','indexed','combined','shared','materialized'),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_grad_enabled(False)
    os.environ['LOD_H100_THRESHOLD_IMPLEMENTATION']=a.implementation
    if a.implementation == 'materialized':
        from benchmarks.retro_lod_adapter import install_lod_modules
        install_lod_modules()
    cases=[(2,0.1,67,False,False),(3,4,67,False,False),(4,16,127,False,False),
           (5,32,127,False,False),(6,0.1,67,True,False),(7,0.1,129,False,True),
           (8,16,127,False,False,'entry'),(9,16,256,True,False,'entry',True),
           (10,16,127,False,False,'entry',False,24),
           (11,16,127,False,False,'entry',False,32),
           (11,16,127,False,False,'entry',False,32,1)]
    if a.implementation == 'materialized':
        cases = cases[:6]
    records=[]
    for case in cases:
        row=run(*case);records.append(row);print(json.dumps(row),flush=True)
    a.output.write_text(json.dumps(dict(stage='complete',implementation=a.implementation,records=records),indent=2)+'\n')


if __name__=='__main__':main()
