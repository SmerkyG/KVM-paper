"""Reuse region-only threshold arithmetic before full-row query packing."""
import torch
import triton
import triton.language as tl


@triton.jit
def _prepare(Lengths,NRegions,LogSize,Shift,S:tl.constexpr,CAP:tl.constexpr,C:tl.constexpr,B:tl.constexpr):
    kh=tl.program_id(0);region=tl.program_id(1)*B+tl.arange(0,B)
    size=tl.load(Lengths+kh*CAP+region,mask=region<S,other=0).to(tl.int32)
    log=tl.where((size>0)&(size<=1024),tl.log(tl.maximum(size.to(tl.float32),1.)),-float('inf'))
    tl.store(LogSize+kh*S+region,log,mask=region<S)
    if tl.program_id(1)==0:
        nonempty=tl.load(NRegions+kh).to(tl.float32)
        tl.store(Shift+kh,tl.log(C/tl.maximum(nonempty,1.)))


@triton.jit
def _select(Scores,Inv,LogZ,Lengths,LogSize,Shift,Slots,Counts,Entries,HeadCounts,Offsets,Bits,
            T:tl.constexpr,S:tl.constexpr,CAP:tl.constexpr,G:tl.constexpr,WIDTH:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);head=row//T;kh=head//G
    region=tl.arange(0,B)
    score=tl.load(Scores+row*S+region,mask=region<S,other=0.)
    bias=tl.load(LogSize+kh*S+region,mask=region<S,other=-float('inf'))
    inv=tl.load(Inv+row);cutoff=tl.load(LogZ+row)+tl.load(Shift+kh)
    opened=(region<S)&(score*inv+bias>cutoff)
    rank=tl.cumsum(opened.to(tl.int32),0)-1
    tl.store(Slots+row*WIDTH+tl.maximum(rank,0),region,mask=opened)
    offset=tl.atomic_add(HeadCounts+head*S+region,1,mask=opened,sem='relaxed')
    tl.store(Offsets+row*WIDTH+tl.maximum(rank,0),offset,mask=opened)
    size=tl.load(Lengths+kh*CAP+region,mask=opened,other=0).to(tl.int32)
    tl.store(Counts+row,tl.sum(opened.to(tl.int32),0));tl.store(Entries+row,tl.sum(size,0))
    bit=tl.arange(0,8)
    packed=tl.sum(tl.reshape(opened.to(tl.int32),(B//8,8))*(1<<bit[None,:]),1).to(tl.uint8)
    byte=tl.arange(0,B//8)
    tl.store(Bits+row*tl.cdiv(S,8)+byte,packed,mask=byte<tl.cdiv(S,8))


def select(scores,inv,logz,lengths,nregions,slots,counts,entries,head_counts,offsets,bits,
           *,t,s,cap,g,multiplier,buffers,warps=8,**kwargs):
    h=scores.size(0);kv=h//g
    def ws(name,shape):
        key='bias_'+name;numel=1
        for size in shape:numel*=size
        old=buffers.get(key)
        if old is None or old.numel()<numel:
            old=torch.empty(numel,device=scores.device,dtype=torch.float32);buffers[key]=old
        return old[:numel].view(shape)
    bias=ws('log_size',(kv,s));shift=ws('shift',(kv,))
    _prepare[(kv,triton.cdiv(s,256))](lengths,nregions,bias,shift,s,cap,float(multiplier),256,num_warps=4)
    head_counts.zero_()
    _select[(h*t,)](scores,inv,logz,lengths,bias,shift,slots,counts,entries,head_counts,offsets,bits,
        t,s,cap,g,slots.size(-1),triton.next_power_of_2(max(s,8)),num_warps=warps)
