"""Select regions and compute closed-summary attention in one saved-score scan.

A query tile streams small region tiles. Packing reserves one expert range per
region, and per-query ranks remain in region order. Coarse attention is computed
directly over unopened summaries, without full-field subtraction or fallback.
"""
import triton
import triton.language as tl


@triton.jit
def select_closed(Scores,InvRMS,RouteLSE,Lengths,NRegions,MeanV,Slots,Counts,Entries,HeadCounts,Offsets,
                  Coarse,CoarseLSE,T:tl.constexpr,S:tl.constexpr,CAP:tl.constexpr,G:tl.constexpr,D:tl.constexpr,
                  WIDTH:tl.constexpr,C:tl.constexpr,M:tl.constexpr,N:tl.constexpr):
    head=tl.program_id(0);kh=head//G
    query=tl.program_id(1)*M+tl.arange(0,M);row=head.to(tl.int64)*T+query
    col=tl.arange(0,N);dim=tl.arange(0,D)
    inv=tl.load(InvRMS+row,mask=query<T,other=0)
    logz=tl.load(RouteLSE+row,mask=query<T,other=0)
    nonempty=tl.load(NRegions+kh).to(tl.float32)
    cutoff=logz+tl.log(C/tl.maximum(nonempty,1.))
    chosen=tl.zeros((M,),tl.int32);entries=tl.zeros((M,),tl.int32)
    maximum=tl.full((M,),-float('inf'),tl.float32);den=tl.zeros((M,),tl.float32)
    acc=tl.zeros((M,D),tl.float32)
    for block in range(tl.cdiv(S,N)):
        region=block*N+col
        count=tl.load(Lengths+kh*CAP+region,mask=region<S,other=0).to(tl.int32)
        valid=(query[:,None]<T)&(region[None,:]<S)&(count[None,:]>0)
        raw=tl.load(Scores+row[:,None]*S+region[None,:],mask=(query[:,None]<T)&(region[None,:]<S),other=0)
        logc=tl.log(tl.maximum(count.to(tl.float32),1.))
        opened=valid&(count[None,:]<=1024)&(raw*inv[:,None]+logc[None,:]>cutoff[:,None])
        rank=chosen[:,None]+tl.cumsum(opened.to(tl.int32),1)-1
        tl.store(Slots+row[:,None]*WIDTH+rank,region[None,:],mask=opened)
        total=tl.sum(opened.to(tl.int32),0)
        base=tl.atomic_add(HeadCounts+head*S+region,total,mask=(region<S)&(total>0),sem='relaxed')
        offset=base[None,:]+tl.cumsum(opened.to(tl.int32),0)-1
        tl.store(Offsets+row[:,None]*WIDTH+rank,offset,mask=opened)
        chosen+=tl.sum(opened.to(tl.int32),1)
        entries+=tl.sum(tl.where(opened,count[None,:],0),1)
        closed=valid&~opened
        score=tl.where(closed,raw+logc[None,:],-float('inf'))
        nxt=tl.maximum(maximum,tl.max(score,1));safe=tl.where(nxt!=-float('inf'),nxt,0.)
        adjust=tl.where(maximum!=-float('inf'),tl.exp(maximum-safe),0.)
        probability=tl.where(closed,tl.exp(score-safe[:,None]),0.)
        values=tl.load(MeanV+(kh*S+region[:,None])*D+dim[None,:],mask=(region[:,None]<S)&(count[:,None]>0),other=0)
        acc=acc*adjust[:,None]+tl.dot(probability.to(values.dtype),values)
        den=den*adjust+tl.sum(probability,1);maximum=nxt
    tl.store(Counts+row,chosen,mask=query<T);tl.store(Entries+row,entries,mask=query<T)
    tl.store(Coarse+row[:,None]*D+dim[None,:],acc/tl.maximum(den[:,None],1.e-30),mask=query[:,None]<T)
    tl.store(CoarseLSE+row,tl.where(den>0,maximum+tl.log(den),-float('inf')),mask=query<T)
