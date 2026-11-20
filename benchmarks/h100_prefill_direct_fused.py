"""Direct-summary kernels that share final routing and normalization work."""
import triton
import triton.language as tl


@triton.jit
def _legacy_proxy(raw,inv,log):
    # The legacy sparse selector rounds raw*inv before adding log(n_r).
    # Pin that rounding: a contracted FMA changes strict boundary decisions.
    scaled=tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;',constraints='=f,f,f',
        args=[raw,inv],dtype=tl.float32,is_pure=True,pack=1)
    return scaled+log


@triton.jit
def select_normalize(Scores,Inv,RouteZ,Lengths,LogSize,NRegions,
                     Slots,Counts,Entries,HeadCounts,Offsets,Bits,Probability,Denominator,LSE,
                     T:tl.constexpr,S:tl.constexpr,CAP:tl.constexpr,G:tl.constexpr,
                     WIDTH:tl.constexpr,C:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);head=row//T;kh=head//G;r=tl.arange(0,B)
    size=tl.load(Lengths+kh*CAP+r,mask=r<S,other=0).to(tl.int32)
    log=tl.load(LogSize+kh*S+r,mask=r<S,other=-float('inf'))
    raw=tl.load(Scores+row*S+r,mask=r<S,other=0.)
    inv=tl.load(Inv+row)
    cutoff=tl.load(RouteZ+row)+tl.log(C/tl.maximum(tl.load(NRegions+kh).to(tl.float32),1.))
    opened=(r<S)&(size>0)&(size<=1024)&(_legacy_proxy(raw,inv,log)>cutoff)
    # Finish the softmax before the scan/scatter so raw/log/weight vectors
    # need not remain live across the atomic expert reservations.
    closed=(r<S)&(size>0)&~opened
    logit=tl.where(closed,raw+log,-float('inf'))
    maximum=tl.max(logit,0);safe=tl.where(maximum!=-float('inf'),maximum,0.)
    weight=tl.where(closed,tl.exp(logit-safe),0.);den=tl.sum(weight,0)
    tl.store(Probability+row*S+r,weight,mask=r<S)
    tl.store(Denominator+row,den)
    tl.store(LSE+row,tl.where(den>0,safe+tl.log(den),-float('inf')))
    rank=tl.cumsum(opened.to(tl.int32),0)-1
    tl.store(Slots+row*WIDTH+rank,r,mask=opened)
    offset=tl.atomic_add(HeadCounts+head*S+r,1,mask=opened,sem='relaxed')
    tl.store(Offsets+row*WIDTH+rank,offset,mask=opened)
    tl.store(Counts+row,tl.sum(opened.to(tl.int32),0))
    tl.store(Entries+row,tl.sum(tl.where(opened,size,0),0))
    byte=tl.arange(0,B//8)
    packed=tl.sum(tl.reshape(opened.to(tl.int32),(B//8,8))*(1<<tl.arange(0,8)[None,:]),1).to(tl.uint8)
    tl.store(Bits+row*tl.cdiv(S,8)+byte,packed,mask=byte<tl.cdiv(S,8))


@triton.jit
def normalize_bitmap(Scores,Inv,RouteZ,Lengths,LogSize,NRegions,Bits,WordIDs,WordCounts,
                     Probability,Denominator,LSE,T:tl.constexpr,S:tl.constexpr,CAP:tl.constexpr,
                     G:tl.constexpr,C:tl.constexpr,B:tl.constexpr,COMPACT_WORDS:tl.constexpr=True):
    row=tl.program_id(0).to(tl.int64);head=row//T;kh=head//G;r=tl.arange(0,B)
    size=tl.load(Lengths+kh*CAP+r,mask=r<S,other=0).to(tl.int32)
    log=tl.load(LogSize+kh*S+r,mask=r<S,other=-float('inf'))
    raw=tl.load(Scores+row*S+r,mask=r<S,other=0.)
    cutoff=tl.load(RouteZ+row)+tl.log(C/tl.maximum(tl.load(NRegions+kh).to(tl.float32),1.))
    opened=(r<S)&(size>0)&(size<=1024)&(_legacy_proxy(raw,tl.load(Inv+row),log)>cutoff)
    closed=(r<S)&(size>0)&~opened
    logit=tl.where(closed,raw+log,-float('inf'))
    maximum=tl.max(logit,0);safe=tl.where(maximum!=-float('inf'),maximum,0.)
    weight=tl.where(closed,tl.exp(logit-safe),0.);den=tl.sum(weight,0)
    tl.store(Probability+row*S+r,weight,mask=r<S)
    tl.store(Denominator+row,den)
    tl.store(LSE+row,tl.where(den>0,safe+tl.log(den),-float('inf')))
    byte=tl.arange(0,B//8)
    packed=tl.sum(tl.reshape(opened.to(tl.int32),(B//8,8))*(1<<tl.arange(0,8)[None,:]),1).to(tl.uint8)
    tl.store(Bits+row*tl.cdiv(S,8)+byte,packed,mask=byte<tl.cdiv(S,8))
    # Compact words rather than all region IDs. Sparse packing below performs
    # atomic reservations only for set bits and never reloads routing scores.
    if COMPACT_WORDS:
        word=tl.arange(0,B//32)
        live=tl.sum(tl.reshape(opened.to(tl.int32),(B//32,32)),1)>0
        rank=tl.cumsum(live.to(tl.int32),0)-1
        tl.store(WordIDs+row*tl.cdiv(S,32)+rank,word,mask=live)
        tl.store(WordCounts+row,tl.sum(live.to(tl.int32),0))


@triton.jit
def pack_bitmap(Lengths,Bits,IDs,WordCounts,Slots,Counts,Entries,HeadCounts,Offsets,
                T:tl.constexpr,S:tl.constexpr,CAP:tl.constexpr,G:tl.constexpr,WIDTH:tl.constexpr,W:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);head=row//T;kh=head//G
    count=tl.load(WordCounts+row);group=tl.arange(0,4);lane=tl.arange(0,32);byte=tl.arange(0,4)
    selected=tl.full((),0,tl.int32);entries=tl.full((),0,tl.int32)
    for start in range(tl.cdiv(count,4)):
        pos=start*4+group
        word=tl.load(IDs+row*W+pos,mask=pos<count,other=0)
        b=word[:,None]*4+byte[None,:]
        data=tl.load(Bits+row*tl.cdiv(S,8)+b,mask=(pos[:,None]<count)&(b<tl.cdiv(S,8)),other=0).to(tl.uint32)
        mask=tl.sum(data<<(8*byte[None,:]),1).to(tl.uint32)
        region=word[:,None]*32+lane[None,:]
        hit=(pos[:,None]<count)&(region<S)&(((mask[:,None]>>lane[None,:])&1)!=0)
        active=tl.reshape(hit,(128,));r=tl.reshape(region,(128,))
        rank=selected+tl.cumsum(active.to(tl.int32),0)-1
        tl.store(Slots+row*WIDTH+rank,r,mask=active)
        offset=tl.atomic_add(HeadCounts+head*S+r,1,mask=active,sem='relaxed')
        tl.store(Offsets+row*WIDTH+rank,offset,mask=active)
        size=tl.load(Lengths+kh*CAP+r,mask=active,other=0).to(tl.int32)
        selected+=tl.sum(active.to(tl.int32),0);entries+=tl.sum(size,0)
    tl.store(Counts+row,selected);tl.store(Entries+row,entries)


@triton.jit
def _route_frontier(Values,LSE,row,count,WIDTH:tl.constexpr,D:tl.constexpr,R:tl.constexpr):
    rank=tl.arange(0,R);dim=tl.arange(0,D)
    score=tl.load(LSE+row*WIDTH+rank,mask=rank<count,other=-float('inf'))
    maximum=tl.max(score,0);safe=tl.where(maximum!=-float('inf'),maximum,0.)
    weight=tl.where(rank<count,tl.exp(score-safe),0.)
    value=tl.load(Values+(row*WIDTH+rank[:,None])*D+dim[None,:],mask=rank[:,None]<count,other=0.).to(tl.float32)
    den=tl.sum(weight,0)
    out=tl.sum(value*weight[:,None],0)/tl.maximum(den,1.e-30)
    return out,tl.where(den>0,safe+tl.log(den),-float('inf'))


@triton.jit
def _route_frontier_large(Values,LSE,row,count,WIDTH:tl.constexpr,D:tl.constexpr,R:tl.constexpr):
    rank=tl.arange(0,R);dim=tl.arange(0,D);lane=tl.arange(0,16)
    score=tl.load(LSE+row*WIDTH+rank,mask=rank<count,other=-float('inf'))
    maximum=tl.max(score,0);safe=tl.where(maximum!=-float('inf'),maximum,0.)
    den=tl.sum(tl.where(rank<count,tl.exp(score-safe),0.),0)
    num=tl.zeros((D,),tl.float32)
    for start in range(tl.cdiv(count,16)):
        r=start*16+lane
        tile=tl.load(LSE+row*WIDTH+r,mask=r<count,other=-float('inf'))
        weight=tl.where(r<count,tl.exp(tile-safe),0.)
        value=tl.load(Values+(row*WIDTH+r[:,None])*D+dim[None,:],mask=r[:,None]<count,other=0.).to(tl.float32)
        num+=tl.sum(value*weight[:,None],0)
    return num/tl.maximum(den,1.e-30),tl.where(den>0,safe+tl.log(den),-float('inf'))


@triton.jit
def reduce_merge(Q,SinkK,SinkV,Coarse,CoarseLSE,Values,LSE,Counts,Local,LocalLSE,Output,
                 T:tl.constexpr,G:tl.constexpr,D:tl.constexpr,WIDTH:tl.constexpr,R:tl.constexpr,
                 SCALE:tl.constexpr,SINK:tl.constexpr,SKH:tl.constexpr,SKT:tl.constexpr,
                 SVH:tl.constexpr,SVT:tl.constexpr,LH:tl.constexpr,LT:tl.constexpr,
                 LLH:tl.constexpr,LLT:tl.constexpr,OH:tl.constexpr,OT:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);head=row//T;query=row%T;kh=head//G;dim=tl.arange(0,D)
    count=tl.load(Counts+row)
    if count<=32:
        exact,exact_lse=_route_frontier(Values,LSE,row,count,WIDTH,D,32)
    elif count<=64:
        exact,exact_lse=_route_frontier(Values,LSE,row,count,WIDTH,D,64)
    elif count<=128:
        exact,exact_lse=_route_frontier(Values,LSE,row,count,WIDTH,D,128)
    else:
        exact,exact_lse=_route_frontier_large(Values,LSE,row,count,WIDTH,D,R)
    # Preserve the BF16 intermediate rounding of the separate native reducer.
    exact=exact.to(Q.dtype.element_ty).to(tl.float32)
    q=tl.load(Q+row*D+dim).to(tl.float32)
    if SINK==1:
        key=tl.load(SinkK+kh*SKH+dim).to(tl.float32)
        sink_value=tl.load(SinkV+kh*SVH+dim).to(tl.float32)
        sink_lse=tl.sum(q*key,0)*SCALE
    else:
        mx=tl.full((),-float('inf'),tl.float32);den=tl.full((),0.,tl.float32);num=tl.zeros((D,),tl.float32)
        for i in tl.static_range(SINK):
            key=tl.load(SinkK+kh*SKH+i*SKT+dim).to(tl.float32)
            value=tl.load(SinkV+kh*SVH+i*SVT+dim).to(tl.float32)
            score=tl.sum(q*key,0)*SCALE;nxt=tl.maximum(mx,score)
            old=tl.exp(mx-nxt);new=tl.exp(score-nxt)
            den=den*old+new;num=num*old+value*new;mx=nxt
        sink_lse=mx+tl.log(den);sink_value=num/den
    coarse_lse=tl.load(CoarseLSE+row)
    local_lse=tl.load(LocalLSE+head*LLH+query*LLT)
    maximum=tl.maximum(tl.maximum(tl.maximum(coarse_lse,sink_lse),exact_lse),local_lse)
    cw=tl.exp(coarse_lse-maximum);sw=tl.exp(sink_lse-maximum)
    ew=tl.exp(exact_lse-maximum);lw=tl.exp(local_lse-maximum)
    coarse=tl.load(Coarse+row*D+dim).to(tl.float32)
    local=tl.load(Local+head*LH+query*LT+dim).to(tl.float32)
    denominator=cw+sw;numerator=cw*coarse
    numerator+=sw*sink_value
    denominator+=ew;numerator+=ew*exact
    denominator+=lw;numerator+=lw*local
    tl.store(Output+head*OH+query*OT+dim,numerator/denominator)
