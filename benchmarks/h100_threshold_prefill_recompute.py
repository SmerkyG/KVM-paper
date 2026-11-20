"""Second QK pass: fuse strict selection, expert packing and closed summaries.

The first pass produces only routing LSE. Recomputing QK from the small centroid
cache avoids materializing and rereading the large query-by-region score matrix.
Neither selected regions nor exact entries are capped by this implementation.
"""
import triton
import triton.language as tl


@triton.jit
def select_closed(Q,K,V,Lengths,NRegions,Inv,RouteLSE,Slots,Counts,Entries,HeadCounts,Offsets,Coarse,LSE,
                  T:tl.constexpr,S:tl.constexpr,CAP:tl.constexpr,G:tl.constexpr,D:tl.constexpr,
                  WIDTH:tl.constexpr,SCALE:tl.constexpr,C:tl.constexpr,M:tl.constexpr,N:tl.constexpr,AGGREGATE:tl.constexpr=True):
    head=tl.program_id(0);kh=head//G
    query=tl.program_id(1)*M+tl.arange(0,M);row=head.to(tl.int64)*T+query
    col=tl.arange(0,N);dim=tl.arange(0,D)
    q=tl.load(Q+row[:,None]*D+dim[None,:],mask=query[:,None]<T,other=0)
    inv=tl.load(Inv+row,mask=query<T,other=0.)
    cutoff=tl.load(RouteLSE+row,mask=query<T,other=0.)+tl.log(C/tl.maximum(tl.load(NRegions+kh).to(tl.float32),1.))
    selected=tl.zeros((M,),tl.int32);entries=tl.zeros((M,),tl.int32)
    maximum=tl.full((M,),-float('inf'),tl.float32);den=tl.zeros((M,),tl.float32)
    num=tl.zeros((M,D),tl.float32)
    for start in range(tl.cdiv(S,N)):
        r=start*N+col
        size=tl.load(Lengths+kh*CAP+r,mask=r<S,other=0).to(tl.int32)
        valid=(query[:,None]<T)&(r[None,:]<S)&(size[None,:]>0)
        k=tl.load(K+(kh*S+r[None,:])*D+dim[:,None],mask=(r[None,:]<S)&(size[None,:]>0),other=0)
        raw=tl.dot(q,k).to(tl.float32)*SCALE
        log=tl.log(tl.maximum(size.to(tl.float32),1.))
        opened=valid&(size[None,:]<=1024)&(raw*inv[:,None]+log[None,:]>cutoff[:,None])
        hit=opened.to(tl.int32)
        rank=selected[:,None]+tl.cumsum(hit,1)-1
        tl.store(Slots+row[:,None]*WIDTH+rank,r[None,:],mask=opened)
        if AGGREGATE:
            total=tl.sum(hit,0)
            base=tl.atomic_add(HeadCounts+head*S+r,total,mask=(r<S)&(total>0),sem='relaxed')
            offset=base[None,:]+tl.cumsum(hit,0)-1
        else:
            pointer=tl.broadcast_to(HeadCounts+head*S+r[None,:],(M,N))
            offset=tl.atomic_add(pointer,1,mask=opened,sem='relaxed')
        tl.store(Offsets+row[:,None]*WIDTH+rank,offset,mask=opened)
        selected+=tl.sum(hit,1);entries+=tl.sum(tl.where(opened,size[None,:],0),1)
        closed=valid&~opened
        score=tl.where(closed,raw+log[None,:],-float('inf'))
        nxt=tl.maximum(maximum,tl.max(score,1));safe=tl.where(nxt!=-float('inf'),nxt,0.)
        adjust=tl.where(maximum!=-float('inf'),tl.exp(maximum-safe),0.)
        p=tl.where(closed,tl.exp(score-safe[:,None]),0.)
        value=tl.load(V+(kh*S+r[:,None])*D+dim[None,:],mask=(r[:,None]<S)&(size[:,None]>0),other=0)
        num=num*adjust[:,None]+tl.dot(p.to(value.dtype),value)
        den=den*adjust+tl.sum(p,1);maximum=nxt
    tl.store(Counts+row,selected,mask=query<T);tl.store(Entries+row,entries,mask=query<T)
    tl.store(Coarse+row[:,None]*D+dim[None,:],num/tl.maximum(den[:,None],1.e-30),mask=query[:,None]<T)
    tl.store(LSE+row,tl.where(den>0,maximum+tl.log(den),-float('inf')),mask=query<T)
