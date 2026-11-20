"""Tiled, sort-free region-threshold packing using device counter reservations.

Each query/region tile owns its score slice. It reserves one compact query range
per nonempty row and one expert range per nonempty region, then places all
selected pairs. No selected region is capped or copied to the host.
"""
import triton
import triton.language as tl


@triton.jit
def select(Scores, InvRMS, RouteLSE, Lengths, NRegions, Slots, Counts, Entries,
           HeadCounts, Offsets, Bits,
           T:tl.constexpr,S:tl.constexpr,CAP:tl.constexpr,G:tl.constexpr,WIDTH:tl.constexpr,
           C:tl.constexpr,M:tl.constexpr,N:tl.constexpr):
    head=tl.program_id(0);kh=head//G
    query=tl.program_id(1)*M+tl.arange(0,M);region=tl.program_id(2)*N+tl.arange(0,N)
    row=head.to(tl.int64)*T+query
    inv=tl.load(InvRMS+row,mask=query<T,other=0)
    logz=tl.load(RouteLSE+row,mask=query<T,other=0)
    nonempty=tl.load(NRegions+kh).to(tl.float32)
    cutoff=logz+tl.log(C/tl.maximum(nonempty,1.))
    count=tl.load(Lengths+kh*CAP+region,mask=region<S,other=0).to(tl.int32)
    score=tl.load(Scores+row[:,None]*S+region[None,:],mask=(query[:,None]<T)&(region[None,:]<S),other=0)
    opened=(query[:,None]<T)&(region[None,:]<S)&(count[None,:]>0)&(count[None,:]<=1024)&(
        score*inv[:,None]+tl.log(tl.maximum(count[None,:].to(tl.float32),1.))>cutoff[:,None])
    per_query=tl.sum(opened.to(tl.int32),1)
    first=tl.atomic_add(Counts+row,per_query,mask=(query<T)&(per_query>0),sem='relaxed')
    rank=first[:,None]+tl.cumsum(opened.to(tl.int32),1)-1
    tl.store(Slots+row[:,None]*WIDTH+rank,region[None,:],mask=opened)
    per_region=tl.sum(opened.to(tl.int32),0)
    base=tl.atomic_add(HeadCounts+head*S+region,per_region,mask=(region<S)&(per_region>0),sem='relaxed')
    expert=base[None,:]+tl.cumsum(opened.to(tl.int32),0)-1
    tl.store(Offsets+row[:,None]*WIDTH+rank,expert,mask=opened)
    work=tl.sum(tl.where(opened,count[None,:],0),1)
    tl.atomic_add(Entries+row,work,mask=(query<T)&(work>0),sem='relaxed')
    bit=tl.arange(0,8)
    packed=tl.sum(tl.reshape(opened.to(tl.int32),(M,N//8,8))*(1<<bit[None,None,:]),2).to(tl.uint8)
    byte=tl.program_id(2)*(N//8)+tl.arange(0,N//8)
    tl.store(Bits+row[:,None]*tl.cdiv(S,8)+byte[None,:],packed,mask=(query[:,None]<T)&(byte[None,:]<tl.cdiv(S,8)))
