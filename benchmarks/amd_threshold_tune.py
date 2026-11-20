"""Tune threshold decode on a fixed synthetic 64K AMD cache, not model speed."""
import argparse
import json
import math
import types
from pathlib import Path

import torch
from benchmarks.amd_threshold_decode import threshold_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', choices=('qwen','k2'), required=True)
    parser.add_argument('--union', action='store_true', help='Try the GQA-shared union worker')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.union:
        from benchmarks.amd_threshold_union import threshold_attention as attention
    else:
        attention = threshold_attention
    torch.manual_seed(18)
    torch.set_grad_enabled(False)
    device='cuda'; kv=8; group,dim=(6,256) if args.geometry=='qwen' else (8,128)
    heads=kv*group; slots=4096; size=16; capacity=slots*size; local=256; localcap=512
    centroids=torch.randn((1,kv,slots,dim),device=device,dtype=torch.bfloat16)*2
    leaf_k=centroids.repeat_interleave(size,dim=2).contiguous()
    leaf_v=torch.randn_like(leaf_k)
    state_k=(centroids.float()*size).to(torch.bfloat16)
    state_v=leaf_v.view(1,kv,slots,size,dim).float().sum(dim=3).to(torch.bfloat16)
    counts=torch.full((1,kv,slots,1),size,device=device,dtype=torch.float32)
    page_indices=torch.arange(capacity,device=device,dtype=torch.int32).view(1,1,slots,size).expand(1,kv,-1,-1).contiguous()
    roots=torch.arange(slots,device=device,dtype=torch.int32).view(1,1,slots,1).expand(1,kv,-1,-1).contiguous()
    dirs=torch.full((1,kv,slots,64),-1,device=device,dtype=torch.int32)
    dirs[:,:,:,0]=torch.arange(slots,device=device,dtype=torch.int32)
    lengths=torch.full((1,kv,slots),size,device=device,dtype=torch.int32)
    state={'state_k':state_k,'state_v':state_v,'counts':counts,
           'recent_k':torch.randn((1,kv,localcap,dim),device=device,dtype=torch.bfloat16),
           'recent_v':torch.randn((1,kv,localcap,dim),device=device,dtype=torch.bfloat16),
           'sink_k':torch.randn((1,kv,1,dim),device=device,dtype=torch.bfloat16),
           'sink_v':torch.randn((1,kv,1,dim),device=device,dtype=torch.bfloat16),
           'page_cache':{'paged_page_directory':True,'slot_pages':roots,'overflow_page_values':dirs,
                         'page_indices':page_indices,'leaf_k':leaf_k,'leaf_v':leaf_v,'slot_lengths':lengths}}
    pool=types.SimpleNamespace(kv_heads=kv,settings=types.SimpleNamespace(levels=2),state_capacity=slots,
          dtype=torch.bfloat16,state=state,device=device,decode_local_limit=local,
          engine=types.SimpleNamespace(scaling=1/math.sqrt(dim)),
          state_lens=torch.tensor([slots],device=device,dtype=torch.int32),
          local_lens=torch.tensor([local],device=device,dtype=torch.int32))
    q=torch.randn((1,1,heads,dim),device=device,dtype=torch.bfloat16)
    k=torch.randn((1,1,kv,dim),device=device,dtype=torch.bfloat16); v=torch.randn_like(k); out=torch.empty_like(q)
    records=[]
    for multiplier in (4,16,32):
        expected=None
        for tile in (32,64,128):
            for splits in (16,32,64):
                def invoke():
                    attention(pool,q,k,v,out,multiplier,tile=tile,splits=splits)
                    pool.local_lens.sub_(1)
                invoke(); torch.cuda.synchronize()
                if expected is None: expected=out.clone()
                error=float((out.float()-expected.float()).abs().max())
                scratch=pool._threshold_union_scratch if args.union else pool._threshold_scratch
                selected=scratch['count'].float()
                avg=float(selected.mean()); maximum=int(selected.max())
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(8):invoke()
                for _ in range(4):graph.replay()
                torch.cuda.synchronize(); begin=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(50):graph.replay()
                end.record(); torch.cuda.synchronize()
                row={'multiplier':multiplier,'tile':tile,'splits':splits,'union':args.union,
                     'microseconds':begin.elapsed_time(end)*1000/400,
                     'mean_open_regions':avg,'max_open_regions':maximum,'max_abs_vs_first':error}
                records.append(row);print(json.dumps(row),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'stage':'complete','geometry':args.geometry,'synthetic':True,
                                      'context':capacity,'count_scope':'per KV-head union' if args.union else 'per query head',
                                      'records':records},indent=2)+'\n')


if __name__=='__main__':
    main()
