"""Four-stage AMD decode with per-tile routing and persistent leaf indices.

Exact work remains per query head. Shared GQA score/coarse producers emit short
tile-local region lists, which exact workers consume directly without any
global prefix packing, page expansion, or query-time page-directory traversal.
"""
import os
import torch
import triton
import triton.language as tl
from benchmarks.amd_threshold_frontier import prepare_cache,scores_lse
from benchmarks.amd_threshold_merge import merge_split_d


@triton.jit
def coarse_select(Native,Route,PartLSE,PartCount,Counts,MeanV,Allowed,Selected,Number,Out,LSE,
                  S:tl.constexpr,G:tl.constexpr,D:tl.constexpr,C:tl.constexpr,TILE:tl.constexpr,NG:tl.constexpr,NGB:tl.constexpr):
    kh=tl.program_id(0);group=tl.program_id(1);lane=tl.arange(0,16);head=kh*G+lane;hm=lane<G
    i=group*TILE+tl.arange(0,TILE);dim=tl.arange(0,D);g=tl.arange(0,NGB)
    part=tl.load(PartLSE+head[:,None]*NG+g[None,:],mask=hm[:,None]&(g[None,:]<NG),other=-float('inf'))
    pm=tl.max(part,1);safe=tl.where(pm!=-float('inf'),pm,0.)
    z=tl.sum(tl.exp(part-safe[:,None]),1)
    nonempty=tl.sum(tl.load(PartCount+kh*NG+g,mask=g<NG,other=0),0)
    cut=tl.where(nonempty>0,safe+tl.log(z)+tl.log(C/nonempty),float('inf'))
    route=tl.load(Route+head[:,None]*S+i[None,:],mask=hm[:,None]&(i[None,:]<S),other=-float('inf'))
    count=tl.load(Counts+kh*S+i,mask=i<S,other=0).to(tl.float32)
    valid=(i<S)&(count>0);opened=hm[:,None]&valid[None,:]&(count[None,:]<=1024)&(route>cut[:,None])
    tl.store(Allowed+head[:,None]*S+i[None,:],opened.to(tl.int32),mask=hm[:,None]&(i[None,:]<S))
    rank=tl.cumsum(opened.to(tl.int32),1)-1;number=tl.sum(opened.to(tl.int32),1)
    tl.store(Selected+(head[:,None]*NG+group)*TILE+rank,i[None,:],mask=opened)
    tl.store(Number+head*NG+group,number,mask=hm)
    native=tl.load(Native+head[:,None]*S+i[None,:],mask=hm[:,None]&(i[None,:]<S),other=-float('inf'))
    closed=hm[:,None]&valid[None,:]&~opened;score=tl.where(closed,native,-float('inf'))
    mx=tl.max(score,1);safe=tl.where(mx!=-float('inf'),mx,0.)
    w=tl.where(closed,tl.exp(score-safe[:,None]),0.);den=tl.sum(w,1)
    value=tl.load(MeanV+(kh*S+i[:,None])*D+dim[None,:],mask=valid[:,None],other=0).to(tl.bfloat16)
    acc=tl.dot(w.to(tl.bfloat16),value,out_dtype=tl.float32)
    row=head*NG+group
    tl.store(Out+row[:,None]*D+dim[None,:],acc/tl.maximum(den[:,None],1.e-30),mask=hm[:,None])
    tl.store(LSE+row,tl.where(den>0,safe+tl.log(den),-float('inf')),mask=hm)


@triton.jit
def exact(Q,Selected,Number,Lengths,Fixed,Offsets,ArenaK,ArenaV,LocalK,LocalV,SinkK,SinkV,NewK,NewV,Saved,Out,LSE,
           S:tl.constexpr,G:tl.constexpr,D:tl.constexpr,TILE:tl.constexpr,NG:tl.constexpr,N:tl.constexpr,SPLITS:tl.constexpr,
           SCALE:tl.constexpr,FIXED_STRIDE:tl.constexpr,FIXED_BEGIN:tl.constexpr,ARENA_CAP:tl.constexpr,
           LOCAL_CAP:tl.constexpr,LOCAL_LIMIT:tl.constexpr,SINK:tl.constexpr):
    head=tl.program_id(0);group=tl.program_id(1);split=tl.program_id(2);kh=head//G
    dim=tl.arange(0,D);token=tl.arange(0,N);q=tl.load(Q+head*D+dim).to(tl.float32)
    count=tl.load(Number+head*NG+group)
    mx=tl.full((),-float('inf'),tl.float32);den=tl.full((),0.,tl.float32);acc=tl.zeros((D,),tl.float32)
    for rank in range(count):
        slot=tl.load(Selected+(head*NG+group)*TILE+rank);length=tl.load(Lengths+kh*S+slot)
        offset=tl.load(Offsets+kh*(S+1)+slot)
        for start in range(split*N,length,SPLITS*N):
            ordinal=start+token;where=FIXED_BEGIN+offset+ordinal
            physical=tl.load(Fixed+kh*FIXED_STRIDE+where,mask=(ordinal<length)&(where<FIXED_STRIDE),other=-1)
            ok=(ordinal<length)&(physical>=0)&(physical<ARENA_CAP)
            key=tl.load(ArenaK+tl.maximum(physical,0)[:,None]*D+dim[None,:],mask=ok[:,None],other=0).to(tl.float32)
            value=tl.load(ArenaV+tl.maximum(physical,0)[:,None]*D+dim[None,:],mask=ok[:,None],other=0).to(tl.float32)
            score=tl.where(ok,tl.sum(key*q[None,:],1)*SCALE,-float('inf'))
            nm=tl.maximum(mx,tl.max(score,0));safe=tl.where(nm!=-float('inf'),nm,0.)
            adjust=tl.exp(mx-safe);w=tl.where(ok,tl.exp(score-safe),0.)
            acc=acc*adjust+tl.sum(w[:,None]*value,0);den=den*adjust+tl.sum(w,0);mx=nm
    if group==0:
        local=tl.load(Saved)
        for start in range(split*N,LOCAL_LIMIT+1+SINK,SPLITS*N):
            loc=start+token;old=(loc<local)&(loc<LOCAL_LIMIT);current=loc==local
            sink=(loc>=LOCAL_LIMIT+1)&(loc<LOCAL_LIMIT+1+SINK)
            key=tl.load(LocalK+(kh*LOCAL_CAP+loc[:,None])*D+dim[None,:],mask=old[:,None],other=0).to(tl.float32)
            value=tl.load(LocalV+(kh*LOCAL_CAP+loc[:,None])*D+dim[None,:],mask=old[:,None],other=0).to(tl.float32)
            nk=tl.load(NewK+kh*D+dim).to(tl.float32);nv=tl.load(NewV+kh*D+dim).to(tl.float32)
            key=tl.where(current[:,None],nk[None,:],key);value=tl.where(current[:,None],nv[None,:],value)
            if SINK:
                sk=tl.load(SinkK+(kh*SINK+(loc-LOCAL_LIMIT-1)[:,None])*D+dim[None,:],mask=sink[:,None],other=0).to(tl.float32)
                sv=tl.load(SinkV+(kh*SINK+(loc-LOCAL_LIMIT-1)[:,None])*D+dim[None,:],mask=sink[:,None],other=0).to(tl.float32)
                key=tl.where(sink[:,None],sk,key);value=tl.where(sink[:,None],sv,value)
            active=old|current|sink;score=tl.where(active,tl.sum(key*q[None,:],1)*SCALE,-float('inf'))
            nm=tl.maximum(mx,tl.max(score,0));safe=tl.where(nm!=-float('inf'),nm,0.)
            adjust=tl.exp(mx-safe);w=tl.where(active,tl.exp(score-safe),0.)
            acc=acc*adjust+tl.sum(w[:,None]*value,0);den=den*adjust+tl.sum(w,0);mx=nm
    row=head*(NG*SPLITS)+group*SPLITS+split
    tl.store(Out+row*D+dim,acc/tl.maximum(den,1.e-30));tl.store(LSE+row,tl.where(den>0,mx+tl.log(den),-float('inf')))


def threshold_attention_tiled(pool,q,k,v,out,multiplier,*,tile=64,splits=1,rule='region',**_):
    if rule!='region':raise ValueError('tiled attention currently supports equal-region thresholds')
    cache=prepare_cache(pool);S=pool.state_capacity;KV=pool.kv_heads;D=q.size(-1);H=q.size(2);G=H//KV;NG=triton.cdiv(S,tile)
    st=pool.state;page=st['page_cache'];spec=(S,tile,splits);b=getattr(pool,'_threshold_tiled_scratch',None)
    if b is None or b['spec']!=spec:
        def f(*shape):return torch.empty(shape,device=q.device,dtype=torch.float32)
        def i(*shape):return torch.empty(shape,device=q.device,dtype=torch.int32)
        b=dict(spec=spec,mask_only=True,native=f(H,S),route=f(H,S),part_lse=f(H,NG),part_count=i(KV,NG),allowed=i(H,S),saved=i(1),
               slots=i(H,NG,tile),number=i(H,NG),co=f(H,NG,D),cl=f(H,NG),eo=f(H,NG*splits,D),el=f(H,NG*splits))
        pool._threshold_tiled_scratch=b
    pool._threshold_scratch=b;scale=float(pool.engine.scaling)
    scores_lse[(KV,NG)](q,cache['mean_k'],st['counts'],pool.state_lens,pool.local_lens,b['route'],b['part_lse'],b['part_count'],b['saved'],b['native'],
                        S,G,D,scale,tile,NG,True,num_warps=4)
    coarse_select[(KV,NG)](b['native'],b['route'],b['part_lse'],b['part_count'],st['counts'],cache['mean_v'],b['allowed'],b['slots'],b['number'],
                            b['co'],b['cl'],S,G,D,float(multiplier),tile,NG,triton.next_power_of_2(NG),num_warps=4)
    exact[(H,NG,splits)](q,b['slots'],b['number'],page['slot_lengths'],cache['fixed'],cache['offsets'],cache['arena_k'],cache['arena_v'],
        st['recent_k'],st['recent_v'],st['sink_k'],st['sink_v'],k,v,b['saved'],b['eo'],b['el'],
        S,G,D,tile,NG,int(os.environ.get('LOD_AMD_THRESHOLD_ENTRY_BLOCK','32')),splits,scale,cache['fixed'].size(-1),cache['fixed_begin'],
        cache['arena_k'].numel()//D,st['recent_k'].size(2),int(pool.decode_local_limit),st['sink_k'].size(2),
        num_warps=int(os.environ.get('LOD_AMD_THRESHOLD_ENTRY_WARPS','2')))
    bd=int(os.environ.get('LOD_AMD_THRESHOLD_MERGE_BLOCK','32')) or 32
    merge_split_d[(H,triton.cdiv(D,bd))](b['co'],b['cl'],b['eo'],b['el'],b['co'],b['cl'],k,v,
        st['recent_k'],st['recent_v'],pool.local_lens,b['saved'],out,D,G,KV,st['recent_k'].size(2),NG,NG*splits,0,
        triton.next_power_of_2(NG*(1+splits)),bd,num_warps=4)
    return out
