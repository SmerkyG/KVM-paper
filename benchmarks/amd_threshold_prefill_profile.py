"""Diagnose prefill stage costs at Qwen geometry; synthetic inputs only."""
import argparse
import json
import os
from pathlib import Path
import torch
from triton.testing import do_bench_cudagraph
from lod_attention.kernels.threshold_prefill import threshold_prefill_route_coarse_attention


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--warps',default='4,8,16')
    p.add_argument('--backends',default='blas')
    p.add_argument('--tune-groups',action='store_true')
    a=p.parse_args();torch.set_grad_enabled(False);torch.manual_seed(1651)
    q=torch.randn((1,24,16384,256),device='cuda',dtype=torch.bfloat16)*4
    c=torch.randint(1,257,(1,4,3547,1),device='cuda').float()
    k=torch.randn((1,4,3547,256),device='cuda',dtype=torch.bfloat16)*c
    v=torch.randn_like(k)*c
    rows=[]
    configs=([(backend,int(warps),64,64,4) for backend in a.backends.split(',') for warps in a.warps.split(',')]
             if not a.tune_groups else [('blas',8,m,n,w) for m,n,w in
                 [(64,64,4),(64,128,4),(64,128,8),(128,64,4),(128,64,8),(128,128,8),(256,64,8),(256,128,8)]])
    for backend,warps,gm,gn,gw in configs:
        os.environ['LOD_AMD_PREFILL_THRESHOLD']='16';os.environ['LOD_AMD_PREFILL_MATMUL']=backend
        os.environ['LOD_AMD_PREFILL_SELECT_WARPS']=str(warps)
        os.environ['LOD_AMD_PREFILL_GROUP_M']=str(gm);os.environ['LOD_AMD_PREFILL_GROUP_N']=str(gn)
        os.environ['LOD_AMD_PREFILL_GROUP_WARPS']=str(gw)
        buffers={}
        def run():
            return threshold_prefill_route_coarse_attention(q,k,v,c,state_len=3547,
                kv_group_size=6,scale=256**-.5,normalize_route_query=True,buffers=buffers)
        run();torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            run();torch.cuda.synchronize()
        table=prof.key_averages().table(sort_by='self_device_time_total',row_limit=18)
        row=dict(backend=backend,select_warps=warps,group_m=gm,group_n=gn,group_warps=gw,milliseconds=do_bench_cudagraph(run,rep=200),profile=table)
        print(json.dumps(row),flush=True);rows.append(row)
        del buffers;torch.cuda.empty_cache()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(stage='complete',scope=__doc__,records=rows),indent=2)+'\n')


if __name__=='__main__':main()
