"""Compact query IDs into route-count buckets; reduce with separate kernels.

No routing decisions change. Only four tiny device counters use atomics, one
reservation per input block and bucket. No count is copied to the CPU.
"""
import torch
import triton
import triton.language as tl
from lod_attention.kernels.paged_prefill import _reduce_small_frontier


@triton.jit
def _group(Counts,IDs,Buckets,ROWS:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0)*B+tl.arange(0,B)
    count=tl.load(Counts+row,mask=row<ROWS,other=0)
    kind=tl.where(count<=32,0,tl.where(count<=64,1,tl.where(count<=128,2,3)))
    for k in tl.static_range(4):
        hit=(row<ROWS)&(kind==k);rank=tl.cumsum(hit.to(tl.int32),0)-1
        base=tl.atomic_add(Buckets+k,tl.sum(hit.to(tl.int32),0),sem='relaxed')
        tl.store(IDs+k*ROWS+base+rank,row,mask=hit)


@triton.jit
def _reduce(Values,LSE,Counts,IDs,Buckets,Output,OutputLSE,
            ROWS:tl.constexpr,WIDTH:tl.constexpr,D:tl.constexpr,K:tl.constexpr,R:tl.constexpr,W:tl.constexpr,WORKERS:tl.constexpr):
    number=tl.load(Buckets+K)
    for index in range(tl.program_id(0),number,WORKERS):
        row=tl.load(IDs+K*ROWS+index).to(tl.int64);count=tl.load(Counts+row)
        if R:
            _reduce_small_frontier(Values,LSE,Output,OutputLSE,row,count,WIDTH,R,D)
        else:
            ranks=tl.arange(0,W);dim=tl.arange(0,D);lane=tl.arange(0,16)
            score=tl.load(LSE+row*WIDTH+ranks,mask=ranks<count,other=-float('inf'))
            maximum=tl.max(score,0);safe=tl.where(maximum!=-float('inf'),maximum,0.)
            den=tl.sum(tl.where(ranks<count,tl.exp(score-safe),0.),0)
            num=tl.zeros((D,),tl.float32)
            for start in range(tl.cdiv(count,16)):
                rank=start*16+lane
                tile_score=tl.load(LSE+row*WIDTH+rank,mask=rank<count,other=-float('inf'))
                weight=tl.where(rank<count,tl.exp(tile_score-safe),0.)
                value=tl.load(Values+(row*WIDTH+rank[:,None])*D+dim[None,:],mask=rank[:,None]<count,other=0).to(tl.float32)
                num+=tl.sum(weight[:,None]*value,0)
            tl.store(Output+row*D+dim,num/tl.maximum(den,1.e-30))
            tl.store(OutputLSE+row,tl.where(den>0,safe+tl.log(den),-float('inf')))


def reduce(values,lse,counts,output,output_lse,*,width,d,buffers):
    rows=counts.numel()
    def ws(name,n):
        key='reduce_bucket_'+name;old=buffers.get(key)
        if old is None or old.numel()<n:
            old=torch.empty(n,dtype=torch.int32,device=counts.device);buffers[key]=old
        return old[:n]
    ids=ws('ids',rows*4);sizes=ws('sizes',4);sizes.zero_()
    _group[(triton.cdiv(rows,256),)](counts,ids,sizes,rows,256,num_warps=4)
    workers=min(rows,1056)
    for k,r in enumerate((32,64,128,0)):
        _reduce[(workers,)](values,lse,counts,ids,sizes,output,output_lse,
            rows,width,d,k,r,triton.next_power_of_2(width),workers,num_warps=4)
