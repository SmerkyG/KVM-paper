"""Reproduce BF16 residual cancellation with a valid, nearly fully opened region.

257 identical BF16 keys of value 3 have exact sum 771, stored as BF16 772.
Sixteen full 16-entry pages sum to 768, leaving a single original key of 3.
Region-minus-opened subtraction invents key 4; summing the remaining page gives
key 3. This can turn the rare token's correct 1/257 attention into almost 1.
The GPU check runs both frontiers and an independent dense PyTorch attention.
"""
import argparse
import functools
import json
import math
from pathlib import Path
import struct
import types


def bf16(x):
    bits=struct.unpack('<I',struct.pack('<f',x))[0]
    bits+=0x7fff+((bits>>16)&1)
    return struct.unpack('<f',struct.pack('<I',bits&0xffff0000))[0]


def example():
    n=257;k=3.;opened=256;stored=bf16(n*k);removed=16*bf16(16*k)
    region_key=(stored-removed)/(n-opened);page_key=bf16(k)
    relative=math.exp((region_key-k)*math.sqrt(128))
    return dict(n=n,opened_entries=opened,true_key=k,stored_region_sum=stored,
        opened_page_sum=removed,region_subtraction_key=region_key,remaining_page_key=page_key,
        dense_expected=1/n,region_subtraction_expected=relative/(opened+relative))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);p.add_argument('--cpu-demo',action='store_true');a=p.parse_args()
    demo=example()
    if a.cpu_demo:
        print(json.dumps(demo,indent=2));return
    if a.output is None:p.error('--output is required for the GPU check')
    import torch
    from benchmarks.retro_lod_adapter import install_lod_modules
    install_lod_modules()
    from benchmarks.amd_threshold_three_tier import threshold_attention
    from benchmarks.h100_threshold_pages import page_threshold_attention
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    device='cuda';H,G,D,S,L,P,W=2,4,128,67,257,17,8
    def zeros(*shape,dtype=torch.bfloat16):return torch.zeros(shape,device=device,dtype=dtype)
    lk=torch.full((1,H,L,D),3.,device=device,dtype=torch.bfloat16);lv=zeros(1,H,L,D);lv[:,:,-1]=1
    indices=torch.arange(P*16,device=device,dtype=torch.int32).reshape(1,1,P,16).expand(1,H,-1,-1).clone();indices.masked_fill_(indices>=L,-1)
    pc=torch.full((1,H,P),16,device=device,dtype=torch.int32);pc[:,:,-1]=1
    pk=(pc.float()[...,None]*3).expand(-1,-1,-1,D).to(torch.bfloat16).contiguous();pv=zeros(1,H,P,D);pv[:,:,-1]=1
    table=torch.full((1,H,S,32),-1,device=device,dtype=torch.int32);table[:,:,0,:P]=torch.arange(P,device=device,dtype=torch.int32)
    lengths=zeros(1,H,S,dtype=torch.int32);lengths[:,:,0]=L
    sk=zeros(1,H,S,D);sv=zeros(1,H,S,D);sk[:,:,0]=bf16(3*L);sv[:,:,0]=1
    page=dict(leaf_k=lk,leaf_v=lv,page_indices=indices,page_counts=pc,page_sum_k=pk,page_sum_v=pv,slot_pages=table,slot_lengths=lengths,
        overflow_page_keys=torch.full((1,H,256),-1,device=device,dtype=torch.int32),
        overflow_page_values=torch.full((1,H,256),-1,device=device,dtype=torch.int32),overflow_used=zeros(dtype=torch.int32),leaf_quant_bits=0)
    st=dict(state_k=sk,state_v=sv,counts=lengths.float()[...,None],page_cache=page,
        recent_k=zeros(1,H,128,D),recent_v=zeros(1,H,128,D),sink_k=zeros(1,H,1,D),sink_v=zeros(1,H,1,D))
    pool=types.SimpleNamespace(settings=types.SimpleNamespace(levels=3,quant_group_size=32),dtype=torch.bfloat16,kv_heads=H,state_capacity=S,state=st,
        decode_local_limit=64,engine=types.SimpleNamespace(scaling=D**-.5,leaf_hash_probes=8,leaf_block_n=32),
        state_lens=torch.ones(1,device=device,dtype=torch.int32),local_lens=zeros(1,dtype=torch.int32))
    q=torch.ones((1,1,H*G,D),device=device,dtype=torch.bfloat16);k=zeros(1,1,H,D);v=torch.zeros_like(k);stats=torch.empty((H*G,W,4),device=device,dtype=torch.int32)
    keys=torch.cat((lk[0,0].float(),zeros(2,D).float()));values=torch.cat((lv[0,0].float(),zeros(2,D).float()))
    dense=(q[0,0,0].float()@keys.T*D**-.5).softmax(-1)@values
    rows=[]
    for residual in ('region','pages'):
        out=torch.empty_like(q);pool.local_lens.zero_()
        consumer=functools.partial(page_threshold_attention,multiplier=.5,stats_buffer=stats,residual=residual)
        threshold_attention(pool,q,k,v,out,.01,workers=W,precise=True,round_counts_bf16=True,page_attention=consumer)
        torch.cuda.synchronize();value=out[0,0,0].float();work=stats.sum(1)
        assert bool((work[:,3]==256).all()),work
        if residual=='region':assert float(value.min())>.99
        else:torch.testing.assert_close(value,dense,rtol=.005,atol=1.e-5)
        rows.append(dict(residual=residual,output=float(value.mean()),max_abs_vs_dense=float((value-dense).abs().max()),work=work.cpu().tolist()))
    a.output.write_text(json.dumps(dict(stage='complete',scope='controlled BF16 numerical regression, not a model quality score',analytic=demo,dense=float(dense.mean()),records=rows),indent=2)+'\n')
    print(json.dumps(dict(stage='complete',analytic=demo,records=rows),indent=2))


if __name__=='__main__':main()
