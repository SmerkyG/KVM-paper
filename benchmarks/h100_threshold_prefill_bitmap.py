"""Use bitmap population counts for query and expert ranks in threshold packing.

Only word counts participate in the local query prefix scan. Expert ranks come
from a 32-bit query-tile bitmap, avoiding a second cross-query prefix scan over
the dense boolean tile. Selection, ordering and eligibility are unchanged.
"""
import math
import torch
import triton
import triton.language as tl
from benchmarks.h100_threshold_prefill_scan import _query_prefix, _expert_prefix


@triton.jit
def popc(x):
    return tl.inline_asm_elementwise('popc.b32 $0, $1;',constraints='=r,r',args=[x],dtype=tl.int32,is_pure=True,pack=1)


@triton.jit
def _tiles(Scores,Inv,LogZ,Lengths,NRegions,Bits,QCounts,QEntries,Expert,Masks,
           T:tl.constexpr,S:tl.constexpr,CAP:tl.constexpr,G:tl.constexpr,
           RG:tl.constexpr,QG:tl.constexpr,C:tl.constexpr,M:tl.constexpr,N:tl.constexpr):
    head=tl.program_id(0);qg=tl.program_id(1);rg=tl.program_id(2)
    qi=tl.arange(0,M);q=qg*M+qi;r=rg*N+tl.arange(0,N)
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
    tl.store(QCounts+row*RG+rg,tl.sum(hit,1),mask=q<T)
    tl.store(QEntries+row*RG+rg,tl.sum(tl.where(opened,size[None,:],0),1),mask=q<T)
    word=tl.sum(hit.to(tl.uint32)*(tl.full((),1,tl.uint32)<<qi[:,None]),0).to(tl.uint32)
    tl.store(Masks+(head*QG+qg)*S+r,word,mask=r<S)
    tl.store(Expert+(head*QG+qg)*S+r,popc(word),mask=r<S)
    bit=tl.arange(0,8)
    packed=tl.sum(tl.reshape(hit,(M,N//8,8))*(1<<bit[None,None,:]),2).to(tl.uint8)
    byte=rg*(N//8)+tl.arange(0,N//8)
    tl.store(Bits+row[:,None]*tl.cdiv(S,8)+byte[None,:],packed,
             mask=(q[:,None]<T)&(byte[None,:]<tl.cdiv(S,8)))


@triton.jit
def _pack(Bits,QPrefix,Expert,Masks,Counts,Slots,Offsets,
          T:tl.constexpr,S:tl.constexpr,WIDTH:tl.constexpr,RG:tl.constexpr,QG:tl.constexpr,
          M:tl.constexpr,N:tl.constexpr):
    head=tl.program_id(0);qg=tl.program_id(1);rg=tl.program_id(2)
    qi=tl.arange(0,M);q=qg*M+qi;ri=tl.arange(0,N);r=rg*N+ri
    row=head.to(tl.int64)*T+q
    qp=tl.load(QPrefix+row*RG+rg,mask=q<T,other=0)
    nxt=tl.load(QPrefix+row*RG+rg+1,mask=(q<T)&(rg+1<RG),other=0)
    total=tl.load(Counts+row,mask=q<T,other=0)
    after=tl.where(rg+1<RG,nxt,total)
    if tl.sum(after-qp,0)>0:
        byte=rg*(N//8)+tl.arange(0,N//32)[:,None]*4+tl.arange(0,4)[None,:]
        data=tl.load(Bits+row[:,None,None]*tl.cdiv(S,8)+byte[None,:,:],
            mask=(q[:,None,None]<T)&(byte[None,:,:]<tl.cdiv(S,8)),other=0).to(tl.uint32)
        words=tl.sum(data<<(8*tl.arange(0,4)[None,None,:]),2).to(tl.uint32)
        counts=popc(words);prefix=tl.cumsum(counts,1)-counts
        word=tl.reshape(tl.broadcast_to(words[:,:,None],(M,N//32,32)),(M,N))
        before=tl.reshape(tl.broadcast_to(prefix[:,:,None],(M,N//32,32)),(M,N))
        lower=(tl.full((),1,tl.uint32)<<(ri%32))-1
        rank=qp[:,None]+before+popc(word&lower[None,:])
        opened=(q[:,None]<T)&(r[None,:]<S)&(((word>>(ri[None,:]%32))&1)!=0)
        expert=tl.load(Masks+(head*QG+qg)*S+r,mask=r<S,other=0).to(tl.uint32)
        lower_query=(tl.full((),1,tl.uint32)<<qi)-1
        local=popc(expert[None,:]&lower_query[:,None])
        base=tl.load(Expert+(head*QG+qg)*S+r,mask=r<S,other=0)
        tl.store(Slots+row[:,None]*WIDTH+rank,r[None,:],mask=opened)
        tl.store(Offsets+row[:,None]*WIDTH+rank,base[None,:]+local,mask=opened)


def select(scores,inv,logz,lengths,nregions,slots,counts,entries,head_counts,offsets,bits,
           *,t,s,cap,g,multiplier,buffers,m=16,n=128,warps=4,q=None,**kwargs):
    if m not in (8,16,32) or n not in (64,128,256):raise ValueError('unsupported bitmap-rank tile')
    h=scores.size(0);rg=triton.cdiv(s,n);qg=triton.cdiv(t,m)
    def ws(name,shape):
        key='bitmap_'+name;numel=math.prod(shape);old=buffers.get(key)
        if old is None or old.numel()<numel:
            old=torch.empty(numel,dtype=torch.int32,device=scores.device);buffers[key]=old
        return old[:numel].view(shape)
    qc=ws('query_counts',(h*t,rg));qe=ws('query_entries',(h*t,rg))
    expert=ws('expert_counts',(h,qg,s));masks=ws('expert_masks',(h,qg,s))
    grid=(h,qg,rg)
    _tiles[grid](scores,inv,logz,lengths,nregions,bits,qc,qe,expert,masks,t,s,cap,g,rg,qg,float(multiplier),m,n,num_warps=warps)
    _query_prefix[(triton.cdiv(h*t,16),)](qc,qe,counts,entries,h*t,rg,16,triton.next_power_of_2(rg),num_warps=4)
    _expert_prefix[(h,triton.cdiv(s,32))](expert,head_counts,s,qg,triton.next_power_of_2(qg),32,num_warps=4)
    _pack[grid](bits,qc,expert,masks,counts,slots,offsets,t,s,slots.size(-1),rg,qg,m,n,num_warps=warps)
