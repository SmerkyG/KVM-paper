"""Fuse threshold/coarse attention and tile-local physical-index preparation.

Every routing tile owns its original persistent leaf-index range, so selected
entries can be compacted inside that range without a global packing barrier.
The exact consumer performs only the short tile-prefix lookup, sharing KV
reads across GQA heads and preserving per-head selection bits.
"""
import os
import torch
import triton
import triton.language as tl
from benchmarks.amd_threshold_frontier import prepare_cache,scores_lse
from benchmarks.amd_threshold_merge import merge_split_d


@triton.jit
def coarse_pack(Native,Route,PartLSE,PartCount,Counts,Lengths,MeanV,FixedOffsets,Allowed,PackedOffsets,PackedEnds,PackedBits,Number,TileTotal,Out,LSE,Fixed,Flat,FlatBits,
                  S:tl.constexpr,G:tl.constexpr,D:tl.constexpr,C:tl.constexpr,TILE:tl.constexpr,NG:tl.constexpr,NGB:tl.constexpr,CAP:tl.constexpr,INDEX_STRIDE:tl.constexpr,FIXED_BEGIN:tl.constexpr,CHUNK:tl.constexpr):
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
    bits=tl.sum(tl.where(opened,1<<lane[:,None],0),0);union=bits!=0
    rank=tl.cumsum(union.to(tl.int32),0)-1;number=tl.sum(union.to(tl.int32),0)
    size=tl.load(Lengths+kh*S+i,mask=i<S,other=0);offset=tl.load(FixedOffsets+kh*(S+1)+i,mask=i<S,other=0)
    ends=tl.cumsum(tl.where(union,size,0),0);dst=(kh*NG+group)*TILE+rank
    tl.store(PackedOffsets+dst,offset,mask=union);tl.store(PackedEnds+dst,ends,mask=union);tl.store(PackedBits+dst,bits,mask=union)
    tl.store(Number+kh*NG+group,number);tl.store(TileTotal+kh*NG+group,tl.sum(tl.where(union,size,0),0))
    native=tl.load(Native+head[:,None]*S+i[None,:],mask=hm[:,None]&(i[None,:]<S),other=-float('inf'))
    closed=hm[:,None]&valid[None,:]&~opened;score=tl.where(closed,native,-float('inf'))
    mx=tl.max(score,1);safe=tl.where(mx!=-float('inf'),mx,0.);w=tl.where(closed,tl.exp(score-safe[:,None]),0.)
    den=tl.sum(w,1);value=tl.load(MeanV+(kh*S+i[:,None])*D+dim[None,:],mask=valid[:,None],other=0).to(tl.bfloat16)
    acc=tl.dot(w.to(tl.bfloat16),value,out_dtype=tl.float32);row=head*NG+group
    tl.store(Out+row[:,None]*D+dim[None,:],acc/tl.maximum(den[:,None],1.e-30),mask=hm[:,None])
    tl.store(LSE+row,tl.where(den>0,safe+tl.log(den),-float('inf')),mask=hm)

    # Each tile owns its original persistent-index range. Its selected subset
    # fits there without a global packing barrier or a region-count cap.
    tilebase=tl.load(FixedOffsets+kh*(S+1)+group*TILE)
    r=tl.arange(0,CHUNK);token=tl.arange(0,1024);j=tl.arange(0,TILE)
    for start in range(0,number,CHUNK):
        wanted=start+r
        original=tl.max(tl.where(union[None,:]&(rank[None,:]==wanted[:,None]),j[None,:],-1),1)
        region_valid=original>=0;slot=group*TILE+tl.maximum(original,0)
        length=tl.load(Lengths+kh*S+slot,mask=region_valid,other=0)
        source=tl.load(FixedOffsets+kh*(S+1)+slot,mask=region_valid,other=0)
        end=tl.gather(ends,tl.maximum(original,0),0);before=end-length
        member=tl.gather(bits,tl.maximum(original,0),0)
        physical=tl.load(Fixed+kh*INDEX_STRIDE+FIXED_BEGIN+source[:,None]+token[None,:],mask=region_valid[:,None]&(token[None,:]<length[:,None]),other=-1)
        destination=kh*CAP+tilebase+before[:,None]+token[None,:]
        tl.store(Flat+destination,physical,mask=region_valid[:,None]&(token[None,:]<length[:,None]))
        tl.store(FlatBits+destination,tl.broadcast_to(member[:,None],(CHUNK,1024)),mask=region_valid[:,None]&(token[None,:]<length[:,None]))


@triton.jit
def leaves(Q,PackedOffsets,PackedEnds,PackedBits,Number,TileTotal,Fixed,ArenaK,ArenaV,Saved,Out,LSE,
            S:tl.constexpr,G:tl.constexpr,D:tl.constexpr,TILE:tl.constexpr,NG:tl.constexpr,NGB:tl.constexpr,N:tl.constexpr,WORKERS:tl.constexpr,SCALE:tl.constexpr,
            FIXED_STRIDE:tl.constexpr,FIXED_BEGIN:tl.constexpr,ARENA_CAP:tl.constexpr,LOCAL_CAP:tl.constexpr,LOCAL_LIMIT:tl.constexpr,SINK:tl.constexpr,
            LOCAL_OFFSET:tl.constexpr,SINK_OFFSET:tl.constexpr,PRECISE:tl.constexpr,CG:tl.constexpr):
    kh=tl.program_id(1);worker=tl.program_id(0);lane=tl.arange(0,16);head=kh*G+lane;hm=lane<G
    dim=tl.arange(0,D);token=tl.arange(0,N);g=tl.arange(0,NGB);r=tl.arange(0,TILE)
    q=tl.load(Q+head[:,None]*D+dim[None,:],mask=hm[:,None],other=0).to(tl.bfloat16)
    totals=tl.load(TileTotal+kh*NG+g,mask=g<NG,other=0);ends=tl.cumsum(totals,0);remote=tl.sum(totals,0)
    local=tl.minimum(tl.load(Saved),LOCAL_LIMIT);total=remote+local+1+SINK
    mx=tl.full((16,),-float('inf'),tl.float32);den=tl.zeros((16,),tl.float32);acc=tl.zeros((16,D),tl.float32)
    for start in range(worker*N,total,WORKERS*N):
        entry=start+token;is_remote=entry<remote;physical=tl.full((N,),-1,tl.int32);bits=tl.zeros((N,),tl.int32)
        if start<remote:
            low=tl.zeros((N,),tl.int32);high=tl.full((N,),NG,tl.int32)
            for _ in range(NGB.bit_length()):
                mid=(low+high)//2;end=tl.gather(ends,tl.minimum(mid,NGB-1),0)
                advance=is_remote&(mid<NG)&(end<=entry)
                low=tl.where(advance,mid+1,low);high=tl.where(advance,high,mid)
            group=tl.minimum(low,NG-1)
            before=tl.where(group>0,tl.gather(ends,tl.maximum(group-1,0),0),0);ordinal=entry-before
            base=tl.load(PackedEnds+kh*(S+1)+group*TILE,mask=is_remote,other=0)
            where=base+ordinal
            physical=tl.load(PackedOffsets+kh*FIXED_STRIDE+where,mask=is_remote&(where<FIXED_STRIDE),other=-1)
            bits=tl.load(PackedBits+kh*FIXED_STRIDE+where,mask=is_remote&(where<FIXED_STRIDE),other=0)
        loc=entry-remote;old=(loc>=0)&(loc<local);current=loc==local;sink_idx=loc-local-1;sink=(sink_idx>=0)&(sink_idx<SINK)
        physical=tl.where(old,LOCAL_OFFSET+kh*LOCAL_CAP+loc,physical)
        physical=tl.where(current,LOCAL_OFFSET+kh*LOCAL_CAP+tl.load(Saved),physical)
        if SINK:physical=tl.where(sink,SINK_OFFSET+kh*SINK+sink_idx,physical)
        bits=tl.where(old|current|sink,(1<<G)-1,bits)
        valid=(is_remote|old|current|sink)&(physical>=0)&(physical<ARENA_CAP)
        key=tl.load(ArenaK+tl.maximum(physical,0)[None,:]*D+dim[:,None],mask=valid[None,:],other=0,cache_modifier='.cg' if CG else '').to(tl.bfloat16)
        member=hm[:,None]&valid[None,:]&((bits[None,:]&(1<<lane[:,None]))!=0)
        score=tl.where(member,tl.dot(q,key,out_dtype=tl.float32)*(SCALE*1.4426950408889634),-float('inf'))
        nm=tl.maximum(mx,tl.max(score,1));safe=tl.where(nm!=-float('inf'),nm,0.);adjust=tl.exp2(mx-safe);w=tl.where(member,tl.exp2(score-safe[:,None]),0.)
        value=tl.load(ArenaV+tl.maximum(physical,0)[:,None]*D+dim[None,:],mask=valid[:,None],other=0,cache_modifier='.cg' if CG else '').to(tl.bfloat16)
        acc=acc*adjust[:,None]+tl.dot(w.to(tl.bfloat16),value,out_dtype=tl.float32)
        if PRECISE:acc+=tl.dot((w-w.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16),value,out_dtype=tl.float32)
        den=den*adjust+tl.sum(w,1);mx=nm
    row=head*WORKERS+worker;tl.store(Out+row[:,None]*D+dim[None,:],acc/tl.maximum(den[:,None],1.e-30),mask=hm[:,None])
    tl.store(LSE+row,tl.where(den>0,(mx+tl.log2(den))*0.6931471805599453,-float('inf')),mask=hm)


def threshold_attention_table(pool,q,k,v,out,multiplier,*,tile=64,splits=64,rule='region',**_):
    if rule!='region':raise ValueError('balanced workers currently support equal-region thresholds')
    cache=prepare_cache(pool);S=pool.state_capacity;KV=pool.kv_heads;D=q.size(-1);H=q.size(2);G=H//KV;NG=triton.cdiv(S,tile)
    st=pool.state;page=st['page_cache'];cap=cache['leaf_capacity'];spec=(S,tile,splits);b=getattr(pool,'_threshold_table_scratch',None)
    if b is None or b['spec']!=spec:
        def f(*shape):return torch.empty(shape,device=q.device,dtype=torch.float32)
        def i(*shape):return torch.empty(shape,device=q.device,dtype=torch.int32)
        b=dict(spec=spec,mask_only=True,native=f(H,S),route=f(H,S),part_lse=f(H,NG),part_count=i(KV,NG),allowed=i(H,S),saved=i(1),
               flat=i(KV,cap),flat_bits=i(KV,cap),offsets=i(KV,NG,tile),ends=i(KV,NG,tile),bits=i(KV,NG,tile),number=i(KV,NG),total=i(KV,NG),co=f(H,NG,D),cl=f(H,NG),eo=f(H,splits,D),el=f(H,splits))
        pool._threshold_table_scratch=b
    pool._threshold_scratch=b;scale=float(pool.engine.scaling)
    scores_lse[(KV,NG)](q,cache['mean_k'],st['counts'],pool.state_lens,pool.local_lens,b['route'],b['part_lse'],b['part_count'],b['saved'],b['native'],
        S,G,D,scale,tile,NG,True,NewK=k,NewV=v,LocalK=st['recent_k'],LocalV=st['recent_v'],LOCALCAP=st['recent_k'].size(2),APPEND=True,num_warps=4)
    coarse_pack[(KV,NG)](b['native'],b['route'],b['part_lse'],b['part_count'],st['counts'],page['slot_lengths'],cache['mean_v'],cache['offsets'],
        b['allowed'],b['offsets'],b['ends'],b['bits'],b['number'],b['total'],b['co'],b['cl'],cache['fixed'],b['flat'],b['flat_bits'],S,G,D,float(multiplier),tile,NG,triton.next_power_of_2(NG),cap,cache['fixed'].size(-1),cache['fixed_begin'],int(os.environ.get('LOD_AMD_THRESHOLD_TABLE_CHUNK','4')),num_warps=4)
    leaves[(splits,KV)](q,b['flat'],cache['offsets'],b['flat_bits'],b['number'],b['total'],cache['fixed'],cache['arena_k'],cache['arena_v'],b['saved'],b['eo'],b['el'],
        S,G,D,tile,NG,triton.next_power_of_2(NG),int(os.environ.get('LOD_AMD_THRESHOLD_ENTRY_BLOCK','64')),splits,scale,
        cap,0,cache['arena_k'].numel()//D,st['recent_k'].size(2),int(pool.decode_local_limit),st['sink_k'].size(2),
        cache['local_offset'],cache['sink_offset'],os.environ.get('LOD_AMD_THRESHOLD_PRECISE','1')=='1',os.environ.get('LOD_AMD_THRESHOLD_CACHE_CG','0')=='1',
        num_warps=int(os.environ.get('LOD_AMD_THRESHOLD_ENTRY_WARPS','4')),num_stages=1)
    bd=int(os.environ.get('LOD_AMD_THRESHOLD_MERGE_BLOCK','32')) or 32
    merge_split_d[(H,triton.cdiv(D,bd))](b['co'],b['cl'],b['eo'],b['el'],b['co'],b['cl'],k,v,
        st['recent_k'],st['recent_v'],pool.local_lens,b['saved'],out,D,G,KV,st['recent_k'].size(2),NG,splits,0,
        triton.next_power_of_2(NG+splits),bd,num_warps=4)
    return out
