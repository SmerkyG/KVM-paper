"""Threshold packing with sparse tile staging and deterministic expert prefixes.

Each selected pair is staged once with its tile-local expert offset. The final
pass visits output ranks rather than unpacking the entire query/region mask.
Tile capacity follows the existing mathematical route-width bound, not a cap.
"""
import math
import torch
import triton
import triton.language as tl

from benchmarks.h100_threshold_prefill_scan import _query_prefix, _expert_prefix


@triton.jit
def _stage(Scores, Inv, LogZ, Lengths, NRegions, Bits, QCounts, QEntries, Expert, Pairs,
           T:tl.constexpr,S:tl.constexpr,CAP:tl.constexpr,G:tl.constexpr,
           RG:tl.constexpr,QG:tl.constexpr,K:tl.constexpr,C:tl.constexpr,M:tl.constexpr,N:tl.constexpr):
    head=tl.program_id(0);qg=tl.program_id(1);rg=tl.program_id(2)
    q=qg*M+tl.arange(0,M);r=rg*N+tl.arange(0,N)
    row=head.to(tl.int64)*T+q
    inv=tl.load(Inv+row,mask=q<T,other=0.)
    z=tl.load(LogZ+row,mask=q<T,other=0.)
    nonempty=tl.load(NRegions+head//G).to(tl.float32)
    cutoff=z+tl.log(C/tl.maximum(nonempty,1.))
    size=tl.load(Lengths+head//G*CAP+r,mask=r<S,other=0).to(tl.int32)
    dot=tl.load(Scores+row[:,None]*S+r[None,:],mask=(q[:,None]<T)&(r[None,:]<S),other=0.)
    opened=(q[:,None]<T)&(r[None,:]<S)&(size[None,:]>0)&(size[None,:]<=1024)&(
        dot*inv[:,None]+tl.log(tl.maximum(size[None,:].to(tl.float32),1.))>cutoff[:,None])
    hit=opened.to(tl.int32)
    rank=tl.cumsum(hit,1)-1
    expert_rank=tl.cumsum(hit,0)-1
    tl.store(Pairs+(row[:,None]*RG+rg)*K+rank,r[None,:]|(expert_rank<<16),mask=opened)
    tl.store(QCounts+row*RG+rg,tl.sum(hit,1),mask=q<T)
    tl.store(QEntries+row*RG+rg,tl.sum(tl.where(opened,size[None,:],0),1),mask=q<T)
    tl.store(Expert+(head*QG+qg)*S+r,tl.sum(hit,0),mask=r<S)
    bit=tl.arange(0,8)
    packed=tl.sum(tl.reshape(hit,(M,N//8,8))*(1<<bit[None,None,:]),2).to(tl.uint8)
    byte=rg*(N//8)+tl.arange(0,N//8)
    tl.store(Bits+row[:,None]*tl.cdiv(S,8)+byte[None,:],packed,
             mask=(q[:,None]<T)&(byte[None,:]<tl.cdiv(S,8)))


@triton.jit
def _pack(Pairs, Prefix, Expert, Counts, Slots, Offsets,
          T:tl.constexpr,S:tl.constexpr,WIDTH:tl.constexpr,RG:tl.constexpr,QG:tl.constexpr,
          K:tl.constexpr,QM:tl.constexpr,M:tl.constexpr,R:tl.constexpr,STEPS:tl.constexpr):
    head=tl.program_id(0);q=tl.program_id(1)*M+tl.arange(0,M)
    row=head.to(tl.int64)*T+q
    r=tl.arange(0,R)
    count=tl.load(Counts+row,mask=q<T,other=0)
    valid=(q[:,None]<T)&(r[None,:]<count[:,None])
    lo=tl.zeros((M,R),tl.int32);hi=tl.full((M,R),RG,tl.int32)
    for _ in tl.static_range(STEPS):
        mid=(lo+hi)//2
        before=tl.load(Prefix+row[:,None]*RG+mid,mask=valid&(mid<RG),other=2147483647)
        left=before<=r[None,:]
        lo=tl.where(left,mid+1,lo);hi=tl.where(left,hi,mid)
    tile=lo-1
    before=tl.load(Prefix+row[:,None]*RG+tile,mask=valid,other=0)
    packed=tl.load(Pairs+(row[:,None]*RG+tile)*K+r[None,:]-before,mask=valid,other=0)
    region=packed&65535;local_offset=packed>>16
    base=tl.load(Expert+(head*QG+q[:,None]//QM)*S+region,mask=valid,other=0)
    tl.store(Slots+row[:,None]*WIDTH+r[None,:],region,mask=valid)
    tl.store(Offsets+row[:,None]*WIDTH+r[None,:],base+local_offset,mask=valid)


def select(scores,inv,logz,lengths,nregions,slots,counts,entries,head_counts,offsets,bits,
           *,t,s,cap,g,multiplier,buffers,m=32,n=256,warps=4,q=None,pack_m=4,**kwargs):
    if s>=65536:raise ValueError('sparse staging requires fewer than 65536 regions')
    if m not in (8,16,32,64) or n not in (64,128,256):raise ValueError('unsupported sparse staging tile')
    h=scores.size(0);rg=triton.cdiv(s,n);qg=triton.cdiv(t,m);width=slots.size(-1);k=min(n,width)
    def ws(name,shape):
        key='sparse_'+name;numel=math.prod(shape);old=buffers.get(key)
        if old is None or old.numel()<numel:
            old=torch.empty(numel,dtype=torch.int32,device=scores.device);buffers[key]=old
        return old[:numel].view(shape)
    qc=ws('query_counts',(h*t,rg));qe=ws('query_entries',(h*t,rg));expert=ws('expert_counts',(h,qg,s))
    pairs=ws('pairs',(h*t,rg,k))
    _stage[(h,qg,rg)](scores,inv,logz,lengths,nregions,bits,qc,qe,expert,pairs,t,s,cap,g,rg,qg,k,
        float(multiplier),m,n,num_warps=warps)
    _query_prefix[(triton.cdiv(h*t,16),)](qc,qe,counts,entries,h*t,rg,16,triton.next_power_of_2(rg),num_warps=4)
    _expert_prefix[(h,triton.cdiv(s,32))](expert,head_counts,s,qg,triton.next_power_of_2(qg),32,num_warps=4)
    _pack[(h,triton.cdiv(t,pack_m))](pairs,qc,expert,counts,slots,offsets,t,s,width,rg,qg,k,m,pack_m,
        triton.next_power_of_2(width),rg.bit_length(),num_warps=4)
