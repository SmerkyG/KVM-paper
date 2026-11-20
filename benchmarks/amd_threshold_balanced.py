"""Four-stage AMD threshold decode with balanced exact-entry traversal.

Coarse attention packs entries within each routing tile. Exact workers build
the short tile prefix in registers and share work by actual entry count. This
removes global packing without assigning a long region to one slow worker.
"""
import os
import torch
import triton
import triton.language as tl
from benchmarks.amd_threshold_frontier import prepare_cache,scores_lse
from benchmarks.amd_threshold_merge import merge_split_d


@triton.jit
def coarse_pack(Native,Route,PartLSE,PartCount,Counts,Lengths,MeanV,FixedOffsets,Allowed,PackedOffsets,PackedEnds,Number,TileTotal,Out,LSE,
                  S:tl.constexpr,G:tl.constexpr,D:tl.constexpr,C:tl.constexpr,TILE:tl.constexpr,NG:tl.constexpr,NGB:tl.constexpr):
    kh=tl.program_id(0);group=tl.program_id(1);lane=tl.arange(0,16);head=kh*G+lane;hm=lane<G
    i=group*TILE+tl.arange(0,TILE);dim=tl.arange(0,D);g=tl.arange(0,NGB)
    part=tl.load(PartLSE+head[:,None]*NG+g[None,:],mask=hm[:,None]&(g[None,:]<NG),other=-float('inf'))
    pm=tl.max(part,1);safe=tl.where(pm!=-float('inf'),pm,0.)
    z=tl.sum(tl.exp(part-safe[:,None]),1);nonempty=tl.sum(tl.load(PartCount+kh*NG+g,mask=g<NG,other=0),0)
    cut=tl.where(nonempty>0,safe+tl.log(z)+tl.log(C/nonempty),float('inf'))
    route=tl.load(Route+head[:,None]*S+i[None,:],mask=hm[:,None]&(i[None,:]<S),other=-float('inf'))
    count=tl.load(Counts+kh*S+i,mask=i<S,other=0).to(tl.float32);valid=(i<S)&(count>0)
    opened=hm[:,None]&valid[None,:]&(count[None,:]<=1024)&(route>cut[:,None])
    tl.store(Allowed+head[:,None]*S+i[None,:],opened.to(tl.int32),mask=hm[:,None]&(i[None,:]<S))
    rank=tl.cumsum(opened.to(tl.int32),1)-1;number=tl.sum(opened.to(tl.int32),1)
    size=tl.load(Lengths+kh*S+i,mask=i<S,other=0);offset=tl.load(FixedOffsets+kh*(S+1)+i,mask=i<S,other=0)
    ends=tl.cumsum(tl.where(opened,size[None,:],0),1)
    dst=(head[:,None]*NG+group)*TILE+rank
    tl.store(PackedOffsets+dst,tl.broadcast_to(offset[None,:],(16,TILE)),mask=opened)
    tl.store(PackedEnds+dst,ends,mask=opened)
    tl.store(Number+head*NG+group,number,mask=hm)
    tl.store(TileTotal+head*NG+group,tl.sum(tl.where(opened,size[None,:],0),1),mask=hm)
    native=tl.load(Native+head[:,None]*S+i[None,:],mask=hm[:,None]&(i[None,:]<S),other=-float('inf'))
    closed=hm[:,None]&valid[None,:]&~opened;score=tl.where(closed,native,-float('inf'))
    mx=tl.max(score,1);safe=tl.where(mx!=-float('inf'),mx,0.);w=tl.where(closed,tl.exp(score-safe[:,None]),0.)
    den=tl.sum(w,1);value=tl.load(MeanV+(kh*S+i[:,None])*D+dim[None,:],mask=valid[:,None],other=0).to(tl.bfloat16)
    acc=tl.dot(w.to(tl.bfloat16),value,out_dtype=tl.float32);row=head*NG+group
    tl.store(Out+row[:,None]*D+dim[None,:],acc/tl.maximum(den[:,None],1.e-30),mask=hm[:,None])
    tl.store(LSE+row,tl.where(den>0,safe+tl.log(den),-float('inf')),mask=hm)


@triton.jit
def leaves(Q,PackedOffsets,PackedEnds,Number,TileTotal,Fixed,ArenaK,ArenaV,Saved,Out,LSE,
            G:tl.constexpr,D:tl.constexpr,TILE:tl.constexpr,NG:tl.constexpr,NGB:tl.constexpr,N:tl.constexpr,WORKERS:tl.constexpr,SCALE:tl.constexpr,
            FIXED_STRIDE:tl.constexpr,FIXED_BEGIN:tl.constexpr,ARENA_CAP:tl.constexpr,LOCAL_CAP:tl.constexpr,LOCAL_LIMIT:tl.constexpr,SINK:tl.constexpr,
            LOCAL_OFFSET:tl.constexpr,SINK_OFFSET:tl.constexpr,SWAP:tl.constexpr,CG:tl.constexpr):
    head=tl.program_id(1) if SWAP else tl.program_id(0);worker=tl.program_id(0) if SWAP else tl.program_id(1);kh=head//G
    dim=tl.arange(0,D);token=tl.arange(0,N);g=tl.arange(0,NGB);r=tl.arange(0,TILE)
    q=tl.load(Q+head*D+dim).to(tl.float32)
    totals=tl.load(TileTotal+head*NG+g,mask=g<NG,other=0);ends=tl.cumsum(totals,0);remote=tl.sum(totals,0)
    local=tl.minimum(tl.load(Saved),LOCAL_LIMIT);total=remote+local+1+SINK
    mx=tl.full((),-float('inf'),tl.float32);den=tl.full((),0.,tl.float32);acc=tl.zeros((D,),tl.float32)
    for start in range(worker*N,total,WORKERS*N):
        entry=start+token;is_remote=entry<remote;physical=tl.full((N,),-1,tl.int32)
        if start<remote:
            low=tl.zeros((N,),tl.int32);high=tl.full((N,),NG,tl.int32)
            for _ in range(NGB.bit_length()):
                mid=(low+high)//2;end=tl.gather(ends,tl.minimum(mid,NGB-1),0)
                advance=is_remote&(mid<NG)&(end<=entry)
                low=tl.where(advance,mid+1,low);high=tl.where(advance,high,mid)
            group=tl.minimum(low,NG-1)
            before=tl.where(group>0,tl.gather(ends,tl.maximum(group-1,0),0),0);ordinal=entry-before
            count=tl.load(Number+head*NG+group,mask=is_remote,other=0)
            rank=tl.zeros((N,),tl.int32);region_before=tl.zeros((N,),tl.int32)
            if tl.max(count,0)>1:
                region_ends=tl.load(PackedEnds+(head*NG+group[:,None])*TILE+r[None,:],mask=is_remote[:,None]&(r[None,:]<count[:,None]),other=2147483647)
                precedes=(r[None,:]<count[:,None])&(region_ends<=ordinal[:,None])
                rank=tl.sum(precedes.to(tl.int32),1);region_before=tl.max(tl.where(precedes,region_ends,0),1)
            offset=tl.load(PackedOffsets+(head*NG+group)*TILE+rank,mask=is_remote,other=0)
            where=FIXED_BEGIN+offset+ordinal-region_before
            physical=tl.load(Fixed+kh*FIXED_STRIDE+where,mask=is_remote&(where<FIXED_STRIDE),other=-1)
        loc=entry-remote;old=(loc>=0)&(loc<local);current=loc==local;sink_idx=loc-local-1;sink=(sink_idx>=0)&(sink_idx<SINK)
        physical=tl.where(old,LOCAL_OFFSET+kh*LOCAL_CAP+loc,physical)
        physical=tl.where(current,LOCAL_OFFSET+kh*LOCAL_CAP+tl.load(Saved),physical)
        if SINK:physical=tl.where(sink,SINK_OFFSET+kh*SINK+sink_idx,physical)
        valid=(is_remote|old|current|sink)&(physical>=0)&(physical<ARENA_CAP)
        key=tl.load(ArenaK+tl.maximum(physical,0)[:,None]*D+dim[None,:],mask=valid[:,None],other=0,cache_modifier='.cg' if CG else '').to(tl.float32)
        score=tl.where(valid,tl.sum(key*q[None,:],1)*SCALE,-float('inf'))
        nm=tl.maximum(mx,tl.max(score,0));safe=tl.where(nm!=-float('inf'),nm,0.);adjust=tl.exp(mx-safe);w=tl.where(valid,tl.exp(score-safe),0.)
        value=tl.load(ArenaV+tl.maximum(physical,0)[:,None]*D+dim[None,:],mask=valid[:,None],other=0,cache_modifier='.cg' if CG else '').to(tl.float32)
        acc=acc*adjust+tl.sum(w[:,None]*value,0);den=den*adjust+tl.sum(w,0);mx=nm
    row=head*WORKERS+worker;tl.store(Out+row*D+dim,acc/tl.maximum(den,1.e-30))
    tl.store(LSE+row,tl.where(den>0,mx+tl.log(den),-float('inf')))


def threshold_attention_balanced(pool,q,k,v,out,multiplier,*,tile=64,splits=64,rule='region',**_):
    if rule!='region':raise ValueError('balanced workers currently support equal-region thresholds')
    cache=prepare_cache(pool);S=pool.state_capacity;KV=pool.kv_heads;D=q.size(-1);H=q.size(2);G=H//KV;NG=triton.cdiv(S,tile)
    st=pool.state;page=st['page_cache'];spec=(S,tile,splits);b=getattr(pool,'_threshold_balanced_scratch',None)
    if b is None or b['spec']!=spec:
        def f(*shape):return torch.empty(shape,device=q.device,dtype=torch.float32)
        def i(*shape):return torch.empty(shape,device=q.device,dtype=torch.int32)
        b=dict(spec=spec,mask_only=True,native=f(H,S),route=f(H,S),part_lse=f(H,NG),part_count=i(KV,NG),allowed=i(H,S),saved=i(1),
               offsets=i(H,NG,tile),ends=i(H,NG,tile),number=i(H,NG),total=i(H,NG),co=f(H,NG,D),cl=f(H,NG),eo=f(H,splits,D),el=f(H,splits))
        pool._threshold_balanced_scratch=b
    pool._threshold_scratch=b;scale=float(pool.engine.scaling)
    scores_lse[(KV,NG)](q,cache['mean_k'],st['counts'],pool.state_lens,pool.local_lens,b['route'],b['part_lse'],b['part_count'],b['saved'],b['native'],
        S,G,D,scale,tile,NG,True,NewK=k,NewV=v,LocalK=st['recent_k'],LocalV=st['recent_v'],LOCALCAP=st['recent_k'].size(2),APPEND=True,num_warps=4)
    coarse_pack[(KV,NG)](b['native'],b['route'],b['part_lse'],b['part_count'],st['counts'],page['slot_lengths'],cache['mean_v'],cache['offsets'],
        b['allowed'],b['offsets'],b['ends'],b['number'],b['total'],b['co'],b['cl'],S,G,D,float(multiplier),tile,NG,triton.next_power_of_2(NG),num_warps=4)
    swap=os.environ.get('LOD_AMD_THRESHOLD_GRID_SWAP','1')=='1'
    leaves[(splits,H) if swap else (H,splits)](q,b['offsets'],b['ends'],b['number'],b['total'],cache['fixed'],cache['arena_k'],cache['arena_v'],b['saved'],b['eo'],b['el'],
        G,D,tile,NG,triton.next_power_of_2(NG),int(os.environ.get('LOD_AMD_THRESHOLD_ENTRY_BLOCK','32')),splits,scale,
        cache['fixed'].size(-1),cache['fixed_begin'],cache['arena_k'].numel()//D,st['recent_k'].size(2),int(pool.decode_local_limit),st['sink_k'].size(2),
        cache['local_offset'],cache['sink_offset'],swap,os.environ.get('LOD_AMD_THRESHOLD_CACHE_CG','0')=='1',
        num_warps=int(os.environ.get('LOD_AMD_THRESHOLD_ENTRY_WARPS','2')))
    bd=int(os.environ.get('LOD_AMD_THRESHOLD_MERGE_BLOCK','32')) or 32
    merge_split_d[(H,triton.cdiv(D,bd))](b['co'],b['cl'],b['eo'],b['el'],b['co'],b['cl'],k,v,
        st['recent_k'],st['recent_v'],pool.local_lens,b['saved'],out,D,G,KV,st['recent_k'].size(2),NG,splits,0,
        triton.next_power_of_2(NG+splits),bd,num_warps=4)
    return out
