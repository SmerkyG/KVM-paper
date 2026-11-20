"""Threshold decode with GQA-shared attention over freshly compacted physical indices.

Only small indices and per-head membership bits are materialized. KV remains
in its original persistent arena. Complete attention timing includes this
linear-work compaction, routing, coarse attention and final reduction.
"""
import os
import torch
import triton
import triton.language as tl
from benchmarks.amd_threshold_frontier import prepare_cache,route_cutoff,scores_lse
from benchmarks.amd_threshold_shared import coarse_pack
from benchmarks.amd_threshold_merge import merge_split_d


@triton.jit
def materialize(Offsets,Ends,Bits,Number,TileTotal,Fixed,Saved,Flat,FlatBits,Total,
                S:tl.constexpr,TILE:tl.constexpr,NG:tl.constexpr,NGB:tl.constexpr,INDEX_STRIDE:tl.constexpr,FIXED_BEGIN:tl.constexpr,
                CAP:tl.constexpr,LOCAL_CAP:tl.constexpr,LOCAL_LIMIT:tl.constexpr,SINK:tl.constexpr,G:tl.constexpr,
                LOCAL_OFFSET:tl.constexpr,SINK_OFFSET:tl.constexpr,SHARDS:tl.constexpr,
                CURRENT:tl.constexpr=True,INCLUDE_LOCAL:tl.constexpr=True,BATCH_KV_HEADS:tl.constexpr=0,
                PACK_WORKERS:tl.constexpr=64, VECTOR_PACK:tl.constexpr=False, CacheIndices=None):
    kh=tl.program_id(0);worker=tl.program_id(1);g=tl.arange(0,NGB);token=tl.arange(0,1024)
    row = kh // BATCH_KV_HEADS if BATCH_KV_HEADS else 0
    cache_row = tl.load(CacheIndices + row).to(tl.int32) if CacheIndices is not None else row
    cache_kh = cache_row * BATCH_KV_HEADS + kh % BATCH_KV_HEADS if BATCH_KV_HEADS else kh
    totals=tl.load(TileTotal+kh*NG+g,mask=g<NG,other=0);prefix=tl.cumsum(totals,0);remote=tl.sum(totals,0)
    # A fixed number of shards keeps total prefix work linear in NG.
    for group in range(worker,NG,PACK_WORKERS):
        before=tl.where(group>0,tl.sum(tl.gather(prefix,tl.maximum(group-1,0)[None],0),0),0)
        count=tl.load(Number+kh*NG+group)
        if VECTOR_PACK:
            # Copy four short regions in one tile. Loop over the longest
            # region so the same path also handles all 1,024 eligible entries.
            entry_in_tile = tl.arange(0, 256)
            for begin in range(tl.program_id(2) * 4, count, SHARDS * 4):
                rank = begin + tl.arange(0, 4)
                region = (kh * NG + group) * TILE + rank
                active = rank < count
                end = tl.load(Ends + region, mask=active, other=0)
                prev = tl.load(Ends + region - 1, mask=active & (rank > 0), other=0)
                length = end - prev
                offset = tl.load(Offsets + region, mask=active, other=0)
                bits = tl.load(Bits + region, mask=active, other=0)
                for block in range(0, tl.max(length, 0), 256):
                    entry = block + entry_in_tile
                    valid = active[:, None] & (entry[None, :] < length[:, None])
                    source = cache_kh * INDEX_STRIDE + FIXED_BEGIN + offset[:, None] + entry[None, :]
                    physical = tl.load(Fixed + source, mask=valid, other=-1)
                    dst = kh * CAP + before + prev[:, None] + entry[None, :]
                    tl.store(Flat + dst, physical, mask=valid)
                    tl.store(FlatBits + dst, bits[:, None], mask=valid)
        else:
            for rank in range(tl.program_id(2),count,SHARDS):
                i=(kh*NG+group)*TILE+rank
                end=tl.load(Ends+i);prev=tl.load(Ends+i-1,mask=rank>0,other=0);length=end-prev
                offset=tl.load(Offsets+i);bits=tl.load(Bits+i)
                physical=tl.load(Fixed+cache_kh*INDEX_STRIDE+FIXED_BEGIN+offset+token,mask=token<length,other=-1)
                tl.store(Flat+kh*CAP+before+prev+token,physical,mask=token<length)
                tl.store(FlatBits+kh*CAP+before+prev+token,bits,mask=token<length)
    if worker==0 and tl.program_id(2)==0:
        saved=tl.load(Saved+row)
        local=tl.minimum(saved,LOCAL_LIMIT) if INCLUDE_LOCAL else 0
        for start in range(0,local+CURRENT+SINK,1024):
            loc=start+token;old=loc<local;cur=(loc==local)&CURRENT;si=loc-local-CURRENT;sink=(si>=0)&(si<SINK)
            physical=tl.where(old,LOCAL_OFFSET+cache_kh*LOCAL_CAP+loc,-1)
            physical=tl.where(cur,LOCAL_OFFSET+cache_kh*LOCAL_CAP+saved,physical)
            if SINK:physical=tl.where(sink,SINK_OFFSET+cache_kh*SINK+si,physical)
            valid=old|cur|sink
            tl.store(Flat+kh*CAP+remote+loc,physical,mask=valid)
            tl.store(FlatBits+kh*CAP+remote+loc,(1<<G)-1,mask=valid)
        tl.store(Total+kh,remote+local+CURRENT+SINK)


@triton.jit
def consume(Q,PackedOffsets,PackedEnds,PackedBits,Number,TileTotal,Fixed,ArenaK,ArenaV,Saved,Out,LSE,
            G:tl.constexpr,D:tl.constexpr,TILE:tl.constexpr,NG:tl.constexpr,NGB:tl.constexpr,N:tl.constexpr,WORKERS:tl.constexpr,SCALE:tl.constexpr,
            FIXED_STRIDE:tl.constexpr,FIXED_BEGIN:tl.constexpr,ARENA_CAP:tl.constexpr,LOCAL_CAP:tl.constexpr,LOCAL_LIMIT:tl.constexpr,SINK:tl.constexpr,
            LOCAL_OFFSET:tl.constexpr,SINK_OFFSET:tl.constexpr,PRECISE:tl.constexpr,CG:tl.constexpr):
    kh=tl.program_id(1);worker=tl.program_id(0);lane=tl.arange(0,16);head=kh*G+lane;hm=lane<G
    dim=tl.arange(0,D);token=tl.arange(0,N);g=tl.arange(0,NGB);r=tl.arange(0,TILE)
    q=tl.load(Q+head[:,None]*D+dim[None,:],mask=hm[:,None],other=0).to(tl.bfloat16)
    total=tl.load(Number+kh);tiles=tl.cdiv(total,N);per=tl.cdiv(tiles,WORKERS)
    mx=tl.full((16,),-float('inf'),tl.float32);den=tl.zeros((16,),tl.float32);acc=tl.zeros((16,D),tl.float32)
    for tile in range(worker*per,tl.minimum((worker+1)*per,tiles)):
        entry=tile*N+token
        physical=tl.load(PackedOffsets+kh*FIXED_STRIDE+entry,mask=entry<total,other=-1)
        bits=tl.load(PackedBits+kh*FIXED_STRIDE+entry,mask=entry<total,other=0)
        valid=(entry<total)&(physical>=0)&(physical<ARENA_CAP)
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


def threshold_attention_materialized(pool,q,k,v,out,multiplier,*,tile=64,splits=128,rule='region',
                                     append=True,include_local=True,include_sink=True,optimized=False,
                                     entry_block=None,entry_warps=None,merge_block=None,precise=None,cache_indices=None,**_):
    if rule!='region':raise ValueError('materialized workers implement equal-region thresholds')
    batch=q.size(0)
    cache=prepare_cache(pool);S=pool.state_capacity;KV=batch*pool.kv_heads;D=q.size(-1);H=batch*q.size(2);G=H//KV;NG=triton.cdiv(S,tile)
    batch_heads=pool.kv_heads if batch>1 or cache_indices is not None else 0
    if (cache_indices is None and batch != pool.state_lens.numel()
            or cache_indices is not None and cache_indices.shape != (batch,)):
        raise ValueError('materialized decode requires one cache index per query row')
    precision = (os.environ.get('LOD_AMD_THRESHOLD_PRECISE','0')=='1') if precise is None else precise
    st=pool.state;page=st['page_cache'];localcap=st['recent_k'].size(2);limit=int(pool.decode_local_limit);sink=st['sink_k'].size(2)
    cap=cache['leaf_capacity']+limit+1+sink;spec=(S,tile,splits,cap,batch,optimized)
    buffers = vars(pool).setdefault('_threshold_materialized_scratch_by_shape', {})
    b = buffers.get(spec)
    if b is None:
        def f(*shape):return torch.empty(shape,device=q.device,dtype=torch.float32)
        def i(*shape):return torch.empty(shape,device=q.device,dtype=torch.int32)
        b=dict(spec=spec,mask_only=True,native=f(H,S),route=f(H,S),part_lse=f(H,NG),part_count=i(KV,NG),allowed=i(H,S),saved=i(batch),
               offsets=i(KV,NG,tile),ends=i(KV,NG,tile),bits=i(KV,NG,tile),number=i(KV,NG),total=i(KV,NG),
               flat=i(KV,cap),flat_bits=i(KV,cap),flat_total=i(KV),co=f(H,NG,D),cl=f(H,NG),eo=f(H,splits,D),el=f(H,splits))
        if optimized:
            b['cutoffs']=f(H)
        buffers[spec]=b
    pool._threshold_scratch=b;scale=float(pool.engine.scaling)
    scores_lse[(KV,NG)](q,cache['mean_k'],st['counts'],pool.state_lens,pool.local_lens,b['route'],b['part_lse'],b['part_count'],b['saved'],b['native'],
        S,G,D,scale,tile,NG,True,NewK=k,NewV=v,LocalK=st['recent_k'],LocalV=st['recent_v'],LOCALCAP=localcap,APPEND=append,
        BATCH_KV_HEADS=batch_heads,CacheIndices=cache_indices,num_warps=4)
    ngb=triton.next_power_of_2(NG)
    if optimized:
        route_cutoff[(H,)](b['part_lse'],b['part_count'],b['cutoffs'],NG,ngb,G,float(multiplier),num_warps=1)
    coarse_pack[(KV,NG)](b['native'],b['route'],b['part_lse'],b['part_count'],st['counts'],page['slot_lengths'],cache['mean_v'],cache['offsets'],
        b['allowed'],b['offsets'],b['ends'],b['bits'],b['number'],b['total'],b['co'],b['cl'],S,G,D,float(multiplier),tile,NG,ngb,Cutoffs=b.get('cutoffs'),BATCH_KV_HEADS=batch_heads,CacheIndices=cache_indices,Precise=precision and cache_indices is not None,num_warps=4)
    shards=1 if optimized else int(os.environ.get('LOD_AMD_THRESHOLD_MATERIALIZE_SHARDS','4'))
    pack_workers=32 if optimized else 64
    materialize[(KV,pack_workers,shards)](b['offsets'],b['ends'],b['bits'],b['number'],b['total'],cache['fixed'],b['saved'],b['flat'],b['flat_bits'],b['flat_total'],
        S,tile,NG,triton.next_power_of_2(NG),cache['fixed'].size(-1),cache['fixed_begin'],cap,localcap,limit,sink if include_sink else 0,G,cache['local_offset'],cache['sink_offset'],shards,
        CURRENT=append,INCLUDE_LOCAL=include_local,BATCH_KV_HEADS=batch_heads,
        PACK_WORKERS=pack_workers,VECTOR_PACK=optimized,CacheIndices=cache_indices,
        num_warps=int(os.environ.get('LOD_AMD_THRESHOLD_MATERIALIZE_WARPS','1')))
    consume[(splits,KV)](q,b['flat'],b['ends'],b['flat_bits'],b['flat_total'],b['total'],cache['fixed'],cache['arena_k'],cache['arena_v'],b['saved'],b['eo'],b['el'],
        G,D,tile,NG,triton.next_power_of_2(NG),int(os.environ.get('LOD_AMD_THRESHOLD_ENTRY_BLOCK','64')) if entry_block is None else entry_block,splits,scale,
        cap,0,cache['arena_k'].numel()//D,localcap,limit,sink,cache['local_offset'],cache['sink_offset'],
        precision,True,
        num_warps=int(os.environ.get('LOD_AMD_THRESHOLD_ENTRY_WARPS','1')) if entry_warps is None else entry_warps,num_stages=1)
    bd=(int(os.environ.get('LOD_AMD_THRESHOLD_MERGE_BLOCK','16')) or 16) if merge_block is None else merge_block
    merge_split_d[(H,triton.cdiv(D,bd))](b['co'],b['cl'],b['eo'],b['el'],b['co'],b['cl'],k,v,st['recent_k'],st['recent_v'],
        pool.local_lens,b['saved'],out,D,G,KV,localcap,NG,splits,0,triton.next_power_of_2(NG+splits),bd,APPEND=append,
        BATCH_KV_HEADS=batch_heads,CacheIndices=cache_indices,num_warps=4)
    return out
