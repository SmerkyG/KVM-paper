"""Balanced exact workers with register-cached routing metadata and flat KV indices."""
import math
import os
import torch
import triton
import triton.language as tl
from benchmarks.amd_threshold_frontier import prepare_cache,scores_lse
from benchmarks.amd_threshold_ranked import coarse
from benchmarks.amd_threshold_merge import merge_split_d
from benchmarks.amd_threshold_decode import _local


@triton.jit
def pack(Route,PartLSE,PartCount,StateCounts,Lengths,Allowed,Selected,Ends,Count,S:tl.constexpr,G:tl.constexpr,C:tl.constexpr,
         NG:tl.constexpr,NGB:tl.constexpr,BLOCK:tl.constexpr):
    head=tl.program_id(0);kh=head//G;i=tl.arange(0,BLOCK);g=tl.arange(0,NGB)
    part=tl.load(PartLSE+head*NG+g,mask=g<NG,other=-float('inf'))
    mx=tl.max(part,0);safe=tl.where(mx!=-float('inf'),mx,0.)
    z=tl.sum(tl.exp(part-safe),0);nonempty=tl.sum(tl.load(PartCount+kh*NG+g,mask=g<NG,other=0),0)
    cut=tl.where(nonempty>0,safe+tl.log(z)+tl.log(C/nonempty),float('inf'))
    score=tl.load(Route+head*S+i,mask=i<S,other=-float('inf'))
    count=tl.load(StateCounts+kh*S+i,mask=i<S,other=0).to(tl.float32)
    opened=(i<S)&(count>0)&(count<=1024)&(score>cut)
    tl.store(Allowed+head*S+i,opened.to(tl.int32),mask=i<S)
    rank=tl.cumsum(opened.to(tl.int32),0)-1
    size=tl.load(Lengths+kh*S+i,mask=i<S,other=0);ends=tl.cumsum(tl.where(opened,size,0),0)
    tl.store(Selected+head*S+rank,i,mask=opened);tl.store(Ends+head*S+rank,ends,mask=opened)
    tl.store(Count+head,tl.sum(opened.to(tl.int32),0))


@triton.jit
def leaves(Q,Selected,Ends,Count,Fixed,Offsets,ArenaK,ArenaV,LocalK,LocalV,SinkK,SinkV,NewK,NewV,Saved,Out,LSE,
            S:tl.constexpr,G:tl.constexpr,D:tl.constexpr,N:tl.constexpr,WORKERS:tl.constexpr,BSEL:tl.constexpr,SCALE:tl.constexpr,
            FIXED_STRIDE:tl.constexpr,FIXED_BEGIN:tl.constexpr,ARENA_CAP:tl.constexpr,LOCAL_CAP:tl.constexpr,LOCAL_LIMIT:tl.constexpr,SINK:tl.constexpr,
            COMBINED:tl.constexpr,SWAP:tl.constexpr,LOCAL_OFFSET:tl.constexpr,SINK_OFFSET:tl.constexpr,CG:tl.constexpr):
    head=tl.program_id(1) if SWAP else tl.program_id(0)
    worker=tl.program_id(0) if SWAP else tl.program_id(1);kh=head//G
    dim=tl.arange(0,D);token=tl.arange(0,N);r=tl.arange(0,BSEL)
    q=tl.load(Q+head*D+dim).to(tl.float32);count=tl.load(Count+head)
    ends=tl.load(Ends+head*S+r,mask=r<count,other=2147483647)
    slots=tl.load(Selected+head*S+r,mask=r<count,other=0)
    offsets=tl.load(Offsets+kh*(S+1)+slots,mask=r<count,other=0)
    remote=tl.load(Ends+head*S+count-1,mask=count>0,other=0)
    total=remote
    if COMBINED:
        local=tl.minimum(tl.load(Saved),LOCAL_LIMIT);total+=local+1+SINK
    mx=tl.full((),-float('inf'),tl.float32);den=tl.full((),0.,tl.float32);acc=tl.zeros((D,),tl.float32)
    for start in range(worker*N,total,WORKERS*N):
        entry=start+token;is_remote=entry<remote
        physical=tl.full((N,),-1,tl.int32)
        if start<remote:
            low=tl.zeros((N,),tl.int32);high=tl.full((N,),count,tl.int32)
            for _ in range(BSEL.bit_length()):
                mid=(low+high)//2
                end=tl.gather(ends,tl.minimum(mid,BSEL-1),0)
                advance=is_remote&(mid<count)&(end<=entry)
                low=tl.where(advance,mid+1,low);high=tl.where(advance,high,mid)
            rank=tl.minimum(low,BSEL-1)
            offset=tl.gather(offsets,rank,0)
            before=tl.where(rank>0,tl.gather(ends,tl.maximum(rank-1,0),0),0)
            fixedpos=FIXED_BEGIN+offset+entry-before
            physical=tl.load(Fixed+kh*FIXED_STRIDE+fixedpos,mask=is_remote&(fixedpos<FIXED_STRIDE),other=-1)
        valid=is_remote&(physical>=0)&(physical<ARENA_CAP)
        if COMBINED and start+N>remote:
            loc=entry-remote;old=(loc>=0)&(loc<local);current=loc==local;sink_idx=loc-local-1;sink=(sink_idx>=0)&(sink_idx<SINK)
            physical=tl.where(old,LOCAL_OFFSET+kh*LOCAL_CAP+loc,physical)
            physical=tl.where(current,LOCAL_OFFSET+kh*LOCAL_CAP+tl.load(Saved),physical)
            if SINK:
                physical=tl.where(sink,SINK_OFFSET+kh*SINK+sink_idx,physical)
            valid|=old|current|sink
        key=tl.load(ArenaK+tl.maximum(physical,0)[:,None]*D+dim[None,:],mask=valid[:,None],other=0,
                    cache_modifier='.cg' if CG else '').to(tl.float32)
        score=tl.where(valid,tl.sum(key*q[None,:],1)*SCALE,-float('inf'))
        nm=tl.maximum(mx,tl.max(score,0));safe=tl.where(nm!=-float('inf'),nm,0.)
        adjust=tl.exp(mx-safe);w=tl.where(valid,tl.exp(score-safe),0.)
        value=tl.load(ArenaV+tl.maximum(physical,0)[:,None]*D+dim[None,:],mask=valid[:,None],other=0,
                      cache_modifier='.cg' if CG else '').to(tl.float32)
        acc=acc*adjust+tl.sum(w[:,None]*value,0);den=den*adjust+tl.sum(w,0);mx=nm
    row=head*WORKERS+worker
    tl.store(Out+row*D+dim,acc/tl.maximum(den,1.e-30));tl.store(LSE+row,tl.where(den>0,mx+tl.log(den),-float('inf')))


def threshold_attention_cached(pool,q,k,v,out,multiplier,*,tile=64,splits=64,rule='region',**_):
    if rule!='region':raise ValueError('cached workers currently support equal-region thresholds')
    cache=prepare_cache(pool);S=pool.state_capacity;KV=pool.kv_heads;D=q.size(-1);H=q.size(2);G=H//KV;NG=triton.cdiv(S,tile)
    st=pool.state;page=st['page_cache'];combined=os.environ.get('LOD_AMD_THRESHOLD_FUSE_LOCAL','0')=='1'
    nl=0 if combined else triton.cdiv(int(pool.decode_local_limit)+1+st['sink_k'].size(2),tile)
    spec=(S,tile,splits,nl);b=getattr(pool,'_threshold_cached_scratch',None)
    if b is None or b['spec']!=spec:
        def f(*shape):return torch.empty(shape,device=q.device,dtype=torch.float32)
        def i(*shape):return torch.empty(shape,device=q.device,dtype=torch.int32)
        b=dict(spec=spec,mask_only=True,native=f(H,S),route=f(H,S),part_lse=f(H,NG),part_count=i(KV,NG),allowed=i(H,S),saved=i(1),
               selected=i(H,S),ends=i(H,S),count=i(H),co=f(H,NG,D),cl=f(H,NG),eo=f(H,splits,D),el=f(H,splits),lo=f(H,max(1,nl),D),ll=f(H,max(1,nl)))
        pool._threshold_cached_scratch=b
    pool._threshold_scratch=b;scale=float(pool.engine.scaling)
    scores_lse[(KV,NG)](q,cache['mean_k'],st['counts'],pool.state_lens,pool.local_lens,b['route'],b['part_lse'],b['part_count'],b['saved'],b['native'],
                        S,G,D,scale,tile,NG,True,NewK=k,NewV=v,LocalK=st['recent_k'],LocalV=st['recent_v'],
                        LOCALCAP=st['recent_k'].size(2),APPEND=combined,num_warps=4)
    pack[(H,)](b['route'],b['part_lse'],b['part_count'],st['counts'],page['slot_lengths'],b['allowed'],b['selected'],b['ends'],b['count'],
                S,G,float(multiplier),NG,triton.next_power_of_2(NG),triton.next_power_of_2(S),
                num_warps=int(os.environ.get('LOD_AMD_THRESHOLD_PACK_WARPS','4')))
    coarse[(KV,NG)](b['native'],b['allowed'],st['counts'],cache['mean_v'],b['co'],b['cl'],S,G,D,tile,NG,num_warps=4)
    bsel=triton.next_power_of_2(min(S,math.ceil(S/multiplier)+2))
    swap=os.environ.get('LOD_AMD_THRESHOLD_GRID_SWAP','0')=='1'
    kernel=leaves[(splits,H) if swap else (H,splits)](q,b['selected'],b['ends'],b['count'],cache['fixed'],cache['offsets'],cache['arena_k'],cache['arena_v'],
        st['recent_k'],st['recent_v'],st['sink_k'],st['sink_v'],k,v,b['saved'],b['eo'],b['el'],
        S,G,D,int(os.environ.get('LOD_AMD_THRESHOLD_ENTRY_BLOCK','32')),splits,bsel,scale,cache['fixed'].size(-1),cache['fixed_begin'],
        cache['arena_k'].numel()//D,st['recent_k'].size(2),int(pool.decode_local_limit),st['sink_k'].size(2),combined,swap,
        cache['local_offset'],cache['sink_offset'],os.environ.get('LOD_AMD_THRESHOLD_CACHE_CG','0')=='1',
        num_warps=int(os.environ.get('LOD_AMD_THRESHOLD_ENTRY_WARPS','2')))
    pool._threshold_kernel_resources=dict(registers=kernel.n_regs,spills=kernel.n_spills,shared=kernel.metadata.shared)
    if not combined:
        _local[(H,nl)](q,st['recent_k'],st['recent_v'],st['sink_k'],st['sink_v'],k,v,b['saved'],b['lo'],b['ll'],
            KV,G,D,st['recent_k'].size(2),int(pool.decode_local_limit),st['sink_k'].size(2),scale,tile,nl,num_warps=4)
    bd=int(os.environ.get('LOD_AMD_THRESHOLD_MERGE_BLOCK','32')) or 32
    merge_split_d[(H,triton.cdiv(D,bd))](b['co'],b['cl'],b['eo'],b['el'],b['lo'],b['ll'],k,v,
        st['recent_k'],st['recent_v'],pool.local_lens,b['saved'],out,D,G,KV,st['recent_k'].size(2),NG,splits,nl,
        triton.next_power_of_2(NG+splits+nl),bd,num_warps=4)
    return out
