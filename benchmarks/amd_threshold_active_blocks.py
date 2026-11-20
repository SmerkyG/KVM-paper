"""Four-stage shared-GQA threshold decode with active persistent leaf blocks.

Each selected region marks the fixed-index blocks it overlaps. Exact workers
read only entries selected by at least one head and retain per-head masks.
There is no selected-region cap. The block-flag metadata has O(n / block_size)
capacity, so this variant makes no complete-execution subquadratic-work claim.
"""
import os
import torch
import triton
import triton.language as tl
from benchmarks.amd_threshold_frontier import prepare_cache,scores_lse


@triton.jit
def coarse_mark(Native,Route,PartLSE,PartCount,Counts,Lengths,MeanV,FixedOffsets,Allowed,RegionBits,Active,Out,LSE,
                  S:tl.constexpr,G:tl.constexpr,D:tl.constexpr,C:tl.constexpr,TILE:tl.constexpr,NG:tl.constexpr,NGB:tl.constexpr,
                  N:tl.constexpr,REMOTEBLOCKS:tl.constexpr,BLOCKS:tl.constexpr):
    kh=tl.program_id(0);group=tl.program_id(1);lane=tl.arange(0,16);head=kh*G+lane;hm=lane<G
    i=group*TILE+tl.arange(0,TILE);dim=tl.arange(0,D);g=tl.arange(0,NGB)
    part=tl.load(PartLSE+head[:,None]*NG+g[None,:],mask=hm[:,None]&(g[None,:]<NG),other=-float('inf'))
    pm=tl.max(part,1);safe=tl.where(pm!=-float('inf'),pm,0.);z=tl.sum(tl.exp(part-safe[:,None]),1)
    nonempty=tl.sum(tl.load(PartCount+kh*NG+g,mask=g<NG,other=0),0)
    cut=tl.where(nonempty>0,safe+tl.log(z)+tl.log(C/nonempty),float('inf'))
    route=tl.load(Route+head[:,None]*S+i[None,:],mask=hm[:,None]&(i[None,:]<S),other=-float('inf'))
    count=tl.load(Counts+kh*S+i,mask=i<S,other=0).to(tl.float32);valid=(i<S)&(count>0)
    opened=hm[:,None]&valid[None,:]&(count[None,:]<=1024)&(route>cut[:,None])
    tl.store(Allowed+head[:,None]*S+i[None,:],opened.to(tl.int32),mask=hm[:,None]&(i[None,:]<S))
    bits=tl.sum(tl.where(opened,1<<lane[:,None],0),0)
    tl.store(RegionBits+kh*S+i,bits,mask=i<S)
    begin=tl.load(FixedOffsets+kh*(S+1)+i,mask=i<S,other=0)
    length=tl.load(Lengths+kh*S+i,mask=i<S,other=0)
    first=begin//N;last=(begin+length-1)//N
    steps=tl.max(tl.where((bits!=0)&(length>0),last-first+1,0),0)
    for step in range(steps):
        block=first+step;mask=(bits!=0)&(length>0)&(block<=last)&(block<REMOTEBLOCKS)
        tl.atomic_or(Active+kh*BLOCKS+block,bits,mask=mask,sem='relaxed')
    native=tl.load(Native+head[:,None]*S+i[None,:],mask=hm[:,None]&(i[None,:]<S),other=-float('inf'))
    closed=hm[:,None]&valid[None,:]&~opened;score=tl.where(closed,native,-float('inf'))
    mx=tl.max(score,1);safe=tl.where(mx!=-float('inf'),mx,0.);w=tl.where(closed,tl.exp(score-safe[:,None]),0.);den=tl.sum(w,1)
    value=tl.load(MeanV+(kh*S+i[:,None])*D+dim[None,:],mask=valid[:,None],other=0).to(tl.bfloat16)
    acc=tl.dot(w.to(tl.bfloat16),value,out_dtype=tl.float32);row=head*NG+group
    tl.store(Out+row[:,None]*D+dim[None,:],acc/tl.maximum(den[:,None],1.e-30),mask=hm[:,None])
    tl.store(LSE+row,tl.where(den>0,safe+tl.log(den),-float('inf')),mask=hm)


@triton.jit
def exact_blocks(Q,Allowed,Active,Fixed,Owners,FixedOffsets,ArenaK,ArenaV,Saved,Out,LSE,
                  S:tl.constexpr,G:tl.constexpr,D:tl.constexpr,N:tl.constexpr,BLOCKS:tl.constexpr,REMOTEBLOCKS:tl.constexpr,
                  LEAFCAP:tl.constexpr,FIXED_STRIDE:tl.constexpr,FIXED_BEGIN:tl.constexpr,ARENA_CAP:tl.constexpr,
                  LOCAL_OFFSET:tl.constexpr,LOCAL_CAP:tl.constexpr,LOCAL_LIMIT:tl.constexpr,SINK_OFFSET:tl.constexpr,SINK:tl.constexpr,
                  SCALE:tl.constexpr,PRECISE:tl.constexpr):
    block=tl.program_id(0);kh=tl.program_id(1);bits=tl.load(Active+kh*BLOCKS+block)
    if bits==0:return
    lane=tl.arange(0,16);head=kh*G+lane;hm=lane<G;dim=tl.arange(0,D);token=tl.arange(0,N)
    q=tl.load(Q+head[:,None]*D+dim[None,:],mask=hm[:,None],other=0).to(tl.bfloat16)
    if block<REMOTEBLOCKS:
        ordinal=block*N+token;total=tl.load(FixedOffsets+kh*(S+1)+S);valid=ordinal<total
        owner=tl.load(Owners+kh*LEAFCAP+ordinal,mask=valid&(ordinal<LEAFCAP),other=0)
        entry_bits=tl.load(Allowed+kh*S+owner,mask=valid,other=0)
        needed=valid&(entry_bits!=0)
        member=hm[:,None]&needed[None,:]&((entry_bits[None,:]&(1<<lane[:,None]))!=0)
        where=FIXED_BEGIN+ordinal
        physical=tl.load(Fixed+kh*FIXED_STRIDE+where,mask=needed&(where<FIXED_STRIDE),other=-1)
        member&=(physical[None,:]>=0)&(physical[None,:]<ARENA_CAP)
    else:
        local=tl.minimum(tl.load(Saved),LOCAL_LIMIT);loc=(block-REMOTEBLOCKS)*N+token
        old=loc<local;current=loc==local;sink_idx=loc-local-1;sink=(sink_idx>=0)&(sink_idx<SINK)
        physical=tl.where(old,LOCAL_OFFSET+kh*LOCAL_CAP+loc,-1)
        physical=tl.where(current,LOCAL_OFFSET+kh*LOCAL_CAP+tl.load(Saved),physical)
        if SINK:physical=tl.where(sink,SINK_OFFSET+kh*SINK+sink_idx,physical)
        member=hm[:,None]&(old|current|sink)[None,:]
        needed=old|current|sink
    needed&=(physical>=0)&(physical<ARENA_CAP)
    key=tl.load(ArenaK+tl.maximum(physical,0)[None,:]*D+dim[:,None],mask=needed[None,:],other=0,cache_modifier='.cg').to(tl.bfloat16)
    score=tl.where(member,tl.dot(q,key,out_dtype=tl.float32)*(SCALE*1.4426950408889634),-float('inf'))
    mx=tl.max(score,1);safe=tl.where(mx!=-float('inf'),mx,0.);w=tl.where(member,tl.exp2(score-safe[:,None]),0.);den=tl.sum(w,1)
    value=tl.load(ArenaV+tl.maximum(physical,0)[:,None]*D+dim[None,:],mask=needed[:,None],other=0,cache_modifier='.cg').to(tl.bfloat16)
    acc=tl.dot(w.to(tl.bfloat16),value,out_dtype=tl.float32)
    if PRECISE:acc+=tl.dot((w-w.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16),value,out_dtype=tl.float32)
    row=head*BLOCKS+block;active=hm&((bits&(1<<lane))!=0)
    tl.store(Out+row[:,None]*D+dim[None,:],acc/tl.maximum(den[:,None],1.e-30),mask=active[:,None])
    tl.store(LSE+row,tl.where(den>0,(safe+tl.log2(den))*0.6931471805599453,-float('inf')),mask=active)


@triton.jit
def compact_blocks(Active,List,Number,BLOCKS:tl.constexpr,B:tl.constexpr):
    kh=tl.program_id(0);i=tl.arange(0,B);bits=tl.load(Active+kh*BLOCKS+i,mask=i<BLOCKS,other=0)
    valid=(i<BLOCKS)&(bits!=0);rank=tl.cumsum(valid.to(tl.int32),0)-1
    tl.store(List+kh*BLOCKS+rank,i,mask=valid);tl.store(Number+kh,tl.sum(valid.to(tl.int32),0))


@triton.jit
def exact_workers(Q,Allowed,Active,Fixed,Owners,FixedOffsets,ArenaK,ArenaV,Saved,Out,LSE,
                  S:tl.constexpr,G:tl.constexpr,D:tl.constexpr,N:tl.constexpr,BLOCKS:tl.constexpr,REMOTEBLOCKS:tl.constexpr,
                  LEAFCAP:tl.constexpr,FIXED_STRIDE:tl.constexpr,FIXED_BEGIN:tl.constexpr,ARENA_CAP:tl.constexpr,
                  LOCAL_OFFSET:tl.constexpr,LOCAL_CAP:tl.constexpr,LOCAL_LIMIT:tl.constexpr,SINK_OFFSET:tl.constexpr,SINK:tl.constexpr,
                  SCALE:tl.constexpr,PRECISE:tl.constexpr,WORKERS:tl.constexpr,CONTIG:tl.constexpr,List,Number,COMPACT:tl.constexpr):
    worker=tl.program_id(0);kh=tl.program_id(1)
    lane=tl.arange(0,16);head=kh*G+lane;hm=lane<G;dim=tl.arange(0,D);token=tl.arange(0,N)
    q=tl.load(Q+head[:,None]*D+dim[None,:],mask=hm[:,None],other=0).to(tl.bfloat16)
    mx=tl.full((16,),-float('inf'),tl.float32);den=tl.zeros((16,),tl.float32);acc=tl.zeros((16,D),tl.float32)
    begin=worker;end=BLOCKS;step=WORKERS
    if CONTIG:
        per=tl.cdiv(BLOCKS,WORKERS);begin=worker*per;end=tl.minimum(begin+per,BLOCKS);step=1
    if COMPACT:end=tl.load(Number+kh);begin=worker;step=WORKERS
    for ordinal_block in range(begin,end,step):
        block=ordinal_block
        if COMPACT:block=tl.load(List+kh*BLOCKS+ordinal_block)
        bits=tl.load(Active+kh*BLOCKS+block)
        if bits!=0:
            if block<REMOTEBLOCKS:
                ordinal=block*N+token;total=tl.load(FixedOffsets+kh*(S+1)+S);valid=ordinal<total
                owner=tl.load(Owners+kh*LEAFCAP+ordinal,mask=valid&(ordinal<LEAFCAP),other=0)
                entry_bits=tl.load(Allowed+kh*S+owner,mask=valid,other=0)
                needed=valid&(entry_bits!=0)
                member=hm[:,None]&needed[None,:]&((entry_bits[None,:]&(1<<lane[:,None]))!=0)
                where=FIXED_BEGIN+ordinal
                physical=tl.load(Fixed+kh*FIXED_STRIDE+where,mask=needed&(where<FIXED_STRIDE),other=-1)
                member&=(physical[None,:]>=0)&(physical[None,:]<ARENA_CAP)
            else:
                local=tl.minimum(tl.load(Saved),LOCAL_LIMIT);loc=(block-REMOTEBLOCKS)*N+token
                old=loc<local;current=loc==local;sink_idx=loc-local-1;sink=(sink_idx>=0)&(sink_idx<SINK)
                physical=tl.where(old,LOCAL_OFFSET+kh*LOCAL_CAP+loc,-1)
                physical=tl.where(current,LOCAL_OFFSET+kh*LOCAL_CAP+tl.load(Saved),physical)
                if SINK:physical=tl.where(sink,SINK_OFFSET+kh*SINK+sink_idx,physical)
                member=hm[:,None]&(old|current|sink)[None,:]
                needed=old|current|sink
            needed&=(physical>=0)&(physical<ARENA_CAP)
            key=tl.load(ArenaK+tl.maximum(physical,0)[None,:]*D+dim[:,None],mask=needed[None,:],other=0,cache_modifier='.cg').to(tl.bfloat16)
            score=tl.where(member,tl.dot(q,key,out_dtype=tl.float32)*(SCALE*1.4426950408889634),-float('inf'))
            nm=tl.maximum(mx,tl.max(score,1));safe=tl.where(nm!=-float('inf'),nm,0.)
            adjust=tl.exp2(mx-safe);w=tl.where(member,tl.exp2(score-safe[:,None]),0.)
            value=tl.load(ArenaV+tl.maximum(physical,0)[:,None]*D+dim[None,:],mask=needed[:,None],other=0,cache_modifier='.cg').to(tl.bfloat16)
            acc=acc*adjust[:,None]+tl.dot(w.to(tl.bfloat16),value,out_dtype=tl.float32)
            if PRECISE:acc+=tl.dot((w-w.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16),value,out_dtype=tl.float32)
            den=den*adjust+tl.sum(w,1);mx=nm
    row=head*WORKERS+worker
    tl.store(Out+row[:,None]*D+dim[None,:],acc/tl.maximum(den[:,None],1.e-30),mask=hm[:,None])
    tl.store(LSE+row,tl.where(den>0,(mx+tl.log2(den))*0.6931471805599453,-float('inf')),mask=hm)


@triton.jit
def merge(Co,Cl,Eo,El,Active,Saved,LocalLen,Out,G:tl.constexpr,D:tl.constexpr,NG:tl.constexpr,NE:tl.constexpr,BLOCK:tl.constexpr,BD:tl.constexpr):
    head=tl.program_id(0);db=tl.program_id(1);kh=head//G;lane=head%G;x=tl.arange(0,BLOCK);dim=db*BD+tl.arange(0,BD)
    ci=x<NG;ei=(x>=NG)&(x<NG+NE)
    flag=tl.load(Active+kh*NE+tl.maximum(x-NG,0),mask=ei,other=0);ei&=(flag&(1<<lane))!=0
    cpos=tl.minimum(x,NG-1);epos=tl.minimum(tl.maximum(x-NG,0),NE-1)
    pos=tl.where(ci,head*NG+cpos,tl.num_programs(0)*NG+head*NE+epos)
    l=tl.load(Cl+pos,mask=ci|ei,other=-float('inf'))
    mx=tl.max(l,0);w=tl.exp(l-mx);z=tl.sum(w,0)
    partial=tl.load(Co+pos[:,None]*D+dim[None,:],mask=(ci|ei)[:,None]&(dim[None,:]<D),other=0)
    value=tl.sum(w[:,None]*partial,0)/z;tl.store(Out+head*D+dim,value,mask=dim<D)
    if head==0 and db==0:tl.store(LocalLen,tl.load(Saved)+1)


def threshold_attention_active(pool,q,k,v,out,multiplier,*,tile=64,splits=64,rule='region',**_):
    if rule!='region':raise ValueError('active-block workers currently support equal-region thresholds')
    cache=prepare_cache(pool);S=pool.state_capacity;KV=pool.kv_heads;D=q.size(-1);H=q.size(2);G=H//KV;NG=triton.cdiv(S,tile)
    st=pool.state;page=st['page_cache'];n=int(os.environ.get('LOD_AMD_THRESHOLD_ENTRY_BLOCK','64'))
    localcap=st['recent_k'].size(2);limit=int(pool.decode_local_limit);sink=st['sink_k'].size(2)
    remote=triton.cdiv(cache['leaf_capacity'],n);blocks=remote+triton.cdiv(limit+1+sink,n);workers=int(os.environ.get('LOD_AMD_THRESHOLD_ACTIVE_WORKERS','0'));ne=workers or blocks;spec=(S,tile,n,workers)
    b=getattr(pool,'_threshold_active_scratch',None)
    if b is None or b['spec']!=spec:
        def f(*shape):return torch.empty(shape,device=q.device,dtype=torch.float32)
        def i(*shape):return torch.empty(shape,device=q.device,dtype=torch.int32)
        partial=f(H*(NG+ne),D);normalizers=f(H*(NG+ne))
        b=dict(spec=spec,mask_only=True,native=f(H,S),route=f(H,S),part_lse=f(H,NG),part_count=i(KV,NG),allowed=i(H,S),saved=i(1),
               active=i(KV,blocks),block_list=i(KV,blocks),block_count=i(KV),region_bits=i(KV,S),co=partial[:H*NG],cl=normalizers[:H*NG],eo=partial[H*NG:],el=normalizers[H*NG:])
        pool._threshold_active_scratch=b
    pool._threshold_scratch=b;scale=float(pool.engine.scaling)
    scores_lse[(KV,NG)](q,cache['mean_k'],st['counts'],pool.state_lens,pool.local_lens,b['route'],b['part_lse'],b['part_count'],b['saved'],b['native'],
        S,G,D,scale,tile,NG,True,NewK=k,NewV=v,LocalK=st['recent_k'],LocalV=st['recent_v'],LOCALCAP=localcap,APPEND=True,
        Active=b['active'],ACTIVECAP=blocks,REMOTEBLOCKS=remote,BLOCKN=n,LOCALLIMIT=limit,SINK=sink,CLEAR_ACTIVE=True,num_warps=int(os.environ.get('LOD_AMD_THRESHOLD_ROUTE_WARPS','4')))
    coarse_mark[(KV,NG)](b['native'],b['route'],b['part_lse'],b['part_count'],st['counts'],page['slot_lengths'],cache['mean_v'],cache['offsets'],
        b['allowed'],b['region_bits'],b['active'],b['co'],b['cl'],S,G,D,float(multiplier),tile,NG,triton.next_power_of_2(NG),n,remote,blocks,num_warps=int(os.environ.get('LOD_AMD_THRESHOLD_COARSE_WARPS','4')))
    compact=os.environ.get('LOD_AMD_THRESHOLD_COMPACT_BLOCKS','0')=='1'
    if compact:
        if not workers:raise ValueError('block compaction requires positive worker count')
        compact_blocks[(KV,)](b['active'],b['block_list'],b['block_count'],blocks,triton.next_power_of_2(blocks),num_warps=4)
    if workers:
        exact_workers[(workers,KV)](q,b['region_bits'],b['active'],cache['fixed'],cache['owners'],cache['offsets'],cache['arena_k'],cache['arena_v'],b['saved'],b['eo'],b['el'],
        S,G,D,n,blocks,remote,cache['leaf_capacity'],cache['fixed'].size(-1),cache['fixed_begin'],cache['arena_k'].numel()//D,
        cache['local_offset'],localcap,limit,cache['sink_offset'],sink,scale,os.environ.get('LOD_AMD_THRESHOLD_PRECISE','1')=='1',
        WORKERS=workers,CONTIG=os.environ.get('LOD_AMD_THRESHOLD_CONTIG','1')=='1',List=b['block_list'],Number=b['block_count'],COMPACT=compact,num_warps=int(os.environ.get('LOD_AMD_THRESHOLD_ENTRY_WARPS','2')),num_stages=1,waves_per_eu=int(os.environ.get('LOD_AMD_THRESHOLD_WAVES','0')),matrix_instr_nonkdim=int(os.environ.get('LOD_AMD_THRESHOLD_MATRIX','0')))
        from benchmarks.amd_threshold_merge import merge_split_d
        bd=int(os.environ.get('LOD_AMD_THRESHOLD_MERGE_BLOCK','16')) or 16
        merge_split_d[(H,triton.cdiv(D,bd))](b['co'],b['cl'],b['eo'],b['el'],b['co'],b['cl'],k,v,st['recent_k'],st['recent_v'],
            pool.local_lens,b['saved'],out,D,G,KV,localcap,NG,workers,0,triton.next_power_of_2(NG+workers),bd,num_warps=4)
    else:
        exact_blocks[(blocks,KV)](q,b['region_bits'],b['active'],cache['fixed'],cache['owners'],cache['offsets'],cache['arena_k'],cache['arena_v'],b['saved'],b['eo'],b['el'],
            S,G,D,n,blocks,remote,cache['leaf_capacity'],cache['fixed'].size(-1),cache['fixed_begin'],cache['arena_k'].numel()//D,
            cache['local_offset'],localcap,limit,cache['sink_offset'],sink,scale,os.environ.get('LOD_AMD_THRESHOLD_PRECISE','1')=='1',
            num_warps=int(os.environ.get('LOD_AMD_THRESHOLD_ENTRY_WARPS','2')),num_stages=1,waves_per_eu=int(os.environ.get('LOD_AMD_THRESHOLD_WAVES','0')),matrix_instr_nonkdim=int(os.environ.get('LOD_AMD_THRESHOLD_MATRIX','0')))
        bd=int(os.environ.get('LOD_AMD_THRESHOLD_MERGE_BLOCK','16'))
        merge[(H,triton.cdiv(D,bd))](b['co'],b['cl'],b['eo'],b['el'],b['active'],b['saved'],pool.local_lens,out,G,D,NG,blocks,triton.next_power_of_2(NG+blocks),bd,num_warps=4)
    return out
