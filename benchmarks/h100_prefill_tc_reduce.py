"""Use tensor cores for batches of small route frontiers.

A block diagonal weight matrix gathers independent query frontiers into one
product. A three-part BF16 expansion closely preserves FP32 softmax weights;
large frontiers continue through the uncapped original reducer.
"""
import triton
import triton.language as tl


@triton.jit
def reduce(Values,LSE,Counts,Output,OutputLSE,ROWS:tl.constexpr,WIDTH:tl.constexpr,D:tl.constexpr,M:tl.constexpr,
           PARTS:tl.constexpr=3,RANKS:tl.constexpr=32):
    q=tl.program_id(0).to(tl.int64)*M+tl.arange(0,M);r=tl.arange(0,RANKS)
    count=tl.load(Counts+q,mask=q<ROWS,other=0)
    active=(q<ROWS)&(count<=RANKS)
    valid=active[:,None]&(r[None,:]<count[:,None])
    score=tl.load(LSE+q[:,None]*WIDTH+r[None,:],mask=valid,other=-float('inf'))
    mx=tl.max(score,1);safe=tl.where(mx!=-float('inf'),mx,0.)
    w=tl.where(valid,tl.exp(score-safe[:,None]),0.);den=tl.sum(w,1)
    flat=tl.reshape(w,(M*RANKS,));col=tl.arange(0,M*RANKS);m=tl.arange(0,16);d=tl.arange(0,D)
    diagonal=tl.where(m[:,None]==col[None,:]//RANKS,flat[None,:],0.)
    high=diagonal.to(tl.bfloat16);low=(diagonal-high.to(tl.float32)).to(tl.bfloat16)
    src=tl.program_id(0).to(tl.int64)*M+col//RANKS
    n=tl.load(Counts+src,mask=src<ROWS,other=0)
    live=(src<ROWS)&(n<=RANKS)&(col%RANKS<n)
    value=tl.load(Values+(src[:,None]*WIDTH+col[:,None]%RANKS)*D+d[None,:],mask=live[:,None],other=0)
    acc=tl.dot(high,value)+tl.dot(low,value)
    if PARTS==3:
        tail=((diagonal-high.to(tl.float32))-low.to(tl.float32)).to(tl.bfloat16)
        acc+=tl.dot(tail,value)
    # The padded MMA rows carry no query. All partition-mass reductions remain
    # FP32, and the final BF16 rounding happens only when writing Output.
    qout=tl.program_id(0).to(tl.int64)*M+m
    outden=tl.sum(tl.where(m[:,None]==tl.arange(0,M)[None,:],den[None,:],0.),1)
    result=acc/tl.maximum(outden[:,None],1.e-30)
    outcount=tl.load(Counts+qout,mask=(m<M)&(qout<ROWS),other=RANKS+1)
    tl.store(Output+qout[:,None]*D+d[None,:],result,mask=((m<M)&(qout<ROWS)&(outcount<=RANKS))[:,None])
    tl.store(OutputLSE+q,tl.where(den>0,safe+tl.log(den),-float('inf')),mask=active)
