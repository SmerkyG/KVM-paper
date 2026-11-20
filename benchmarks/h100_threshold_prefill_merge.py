"""Merge variable exact routes and coarse-summary corrections in one pass."""
import triton
import triton.language as tl

from benchmarks.h100_threshold_prefill_fast import _merge


@triton.jit
def merge_routes(Q,Scores,V,Lengths,Slots,Counts,Full,FullLSE,Exact,ExactLSE,
                 Local,LocalLSE,SinkK,SinkV,Out,Flags,
                 T:tl.constexpr,S:tl.constexpr,CAP:tl.constexpr,G:tl.constexpr,D:tl.constexpr,
                 WIDTH:tl.constexpr,SCALE:tl.constexpr,SINK:tl.constexpr,
                 LK0:tl.constexpr,LK1:tl.constexpr,LL0:tl.constexpr,LL1:tl.constexpr,
                 O0:tl.constexpr,O1:tl.constexpr,M:tl.constexpr,R:tl.constexpr,W:tl.constexpr,
                 STABILITY:tl.constexpr,STATIC:tl.constexpr=False):
    head=tl.program_id(0);kh=head//G
    query=tl.program_id(1)*M+tl.arange(0,M);row=head.to(tl.int64)*T+query
    dim=tl.arange(0,D);rank=tl.arange(0,W)
    count=tl.load(Counts+row,mask=query<T,other=0)
    fl=tl.load(FullLSE+row,mask=query<T,other=-float('inf'))
    ll=tl.load(LocalLSE+head*LL0+query*LL1,mask=query<T,other=-float('inf'))
    el=tl.load(ExactLSE+row[:,None]*WIDTH+rank[None,:],
        mask=(query[:,None]<T)&(rank[None,:]<count[:,None]),other=-float('inf'))
    maximum=tl.maximum(tl.maximum(fl,ll),tl.max(el,1))
    q=tl.load(Q+row[:,None]*D+dim[None,:],mask=query[:,None]<T,other=0).to(tl.float32)
    for i in tl.static_range(SINK):
        sk=tl.load(SinkK+(kh*SINK+i)*D+dim).to(tl.float32)
        maximum=tl.maximum(maximum,tl.sum(q*sk[None,:],1)*SCALE)
    safe=tl.where(maximum!=-float('inf'),maximum,0.)
    cw=tl.exp(fl-safe);lw=tl.exp(ll-safe)
    fv=tl.load(Full+row[:,None]*D+dim[None,:],mask=query[:,None]<T,other=0).to(tl.float32)
    lv=tl.load(Local+head*LK0+query[:,None]*LK1+dim[None,:],mask=query[:,None]<T,other=0).to(tl.float32)
    numerator=cw[:,None]*fv+lw[:,None]*lv
    denominator=cw+lw;removed=tl.zeros((M,),tl.float32)
    r=tl.arange(0,R)
    for start in range(1 if STATIC else tl.cdiv(tl.max(count,0),R)):
        rr=start*R+r
        valid=(query[:,None]<T)&(rr[None,:]<count[:,None])
        position=row[:,None]*WIDTH+rr[None,:]
        score=tl.load(ExactLSE+position,mask=valid,other=-float('inf'))
        ev=tl.load(Exact+position[:,:,None]*D+dim[None,None,:],mask=valid[:,:,None],other=0).to(tl.float32)
        ew=tl.where(valid,tl.exp(score-safe[:,None]),0.)
        slot=tl.load(Slots+position,mask=valid,other=0).to(tl.int32)
        size=tl.load(Lengths+kh*CAP+slot,mask=valid,other=1).to(tl.float32)
        raw=tl.load(Scores+row[:,None]*S+slot,mask=valid,other=0)
        weight=tl.where(valid,tl.exp(raw+tl.log(size)-safe[:,None]),0.)
        value=tl.load(V+(kh*S+slot[:,:,None])*D+dim[None,None,:],mask=valid[:,:,None],other=0).to(tl.float32)
        numerator+=tl.sum(ew[:,:,None]*ev-weight[:,:,None]*value,1)
        delta=tl.sum(weight,1);removed+=delta
        denominator+=tl.sum(ew,1)-delta
    for i in tl.static_range(SINK):
        sk=tl.load(SinkK+(kh*SINK+i)*D+dim).to(tl.float32)
        sv=tl.load(SinkV+(kh*SINK+i)*D+dim).to(tl.float32)
        weight=tl.exp(tl.sum(q*sk[None,:],1)*SCALE-safe)
        numerator+=weight[:,None]*sv[None,:];denominator+=weight
    fallback=(cw>0)&(cw-removed<STABILITY*cw)&(count>0)
    tl.store(Flags+row,fallback.to(tl.uint8),mask=query<T)
    tl.store(Out+head*O0+query[:,None]*O1+dim[None,:],
        numerator/tl.maximum(denominator[:,None],1.e-30),mask=query[:,None]<T)


@triton.jit
def merge_bucket(Q,Scores,V,Lengths,Slots,Counts,Full,FullLSE,Exact,ExactLSE,
                 Local,LocalLSE,SinkK,SinkV,Out,Flags,
                 T:tl.constexpr,S:tl.constexpr,CAP:tl.constexpr,G:tl.constexpr,D:tl.constexpr,
                 WIDTH:tl.constexpr,SCALE:tl.constexpr,SINK:tl.constexpr,
                 LK0:tl.constexpr,LK1:tl.constexpr,LL0:tl.constexpr,LL1:tl.constexpr,
                 O0:tl.constexpr,O1:tl.constexpr,STABILITY:tl.constexpr,W:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)*T+tl.program_id(1)
    count=tl.load(Counts+row)
    if count<=32:
        merge_routes(Q,Scores,V,Lengths,Slots,Counts,Full,FullLSE,Exact,ExactLSE,
            Local,LocalLSE,SinkK,SinkV,Out,Flags,T,S,CAP,G,D,WIDTH,SCALE,SINK,
            LK0,LK1,LL0,LL1,O0,O1,1,32,32,STABILITY,True)
    elif count<=64:
        merge_routes(Q,Scores,V,Lengths,Slots,Counts,Full,FullLSE,Exact,ExactLSE,
            Local,LocalLSE,SinkK,SinkV,Out,Flags,T,S,CAP,G,D,WIDTH,SCALE,SINK,
            LK0,LK1,LL0,LL1,O0,O1,1,64,64,STABILITY,True)
    elif count<=128:
        merge_routes(Q,Scores,V,Lengths,Slots,Counts,Full,FullLSE,Exact,ExactLSE,
            Local,LocalLSE,SinkK,SinkV,Out,Flags,T,S,CAP,G,D,WIDTH,SCALE,SINK,
            LK0,LK1,LL0,LL1,O0,O1,1,128,128,STABILITY,True)
    else:
        merge_routes(Q,Scores,V,Lengths,Slots,Counts,Full,FullLSE,Exact,ExactLSE,
            Local,LocalLSE,SinkK,SinkV,Out,Flags,T,S,CAP,G,D,WIDTH,SCALE,SINK,
            LK0,LK1,LL0,LL1,O0,O1,1,16,W,STABILITY,False)


@triton.jit
def repair_routes(Q,Scores,K,V,Lengths,Slots,Counts,Full,FullLSE,Exact,ExactLSE,
                  Local,LocalLSE,SinkK,SinkV,Out,Flags,Bits,
                  T:tl.constexpr,S:tl.constexpr,CAP:tl.constexpr,G:tl.constexpr,D:tl.constexpr,
                  WIDTH:tl.constexpr,SCALE:tl.constexpr,SINK:tl.constexpr,
                  LK0:tl.constexpr,LK1:tl.constexpr,LL0:tl.constexpr,LL1:tl.constexpr,
                  O0:tl.constexpr,O1:tl.constexpr,STABILITY:tl.constexpr,ROWS:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)*128+tl.arange(0,128)
    active=tl.load(Flags+row,mask=row<ROWS,other=0)!=0
    while tl.sum(active.to(tl.int32),0)>0:
        chosen=tl.min(tl.where(active,row,9223372036854775807),0)
        _merge(Q,Scores,K,V,Lengths,Slots,Counts,Full,FullLSE,Exact,ExactLSE,
            Local,LocalLSE,SinkK,SinkV,Out,Flags,Bits,
            T,S,CAP,G,D,WIDTH,SCALE,SINK,True,LK0,LK1,LL0,LL1,O0,O1,32,STABILITY,True,ROW_OVERRIDE=chosen)
        active=active&(row!=chosen)
