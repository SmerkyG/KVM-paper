"""Threshold routing with an AITER-shaped, shared-KV complete-frontier consumer."""
import os
import torch
import triton
import triton.language as tl
from benchmarks.amd_threshold_frontier import prepare_cache,scores_lse
from benchmarks.amd_threshold_decode import _scan_groups
from lod_attention.kernels.aiter_page1_attention import reduce_page1_segments_advance_local


@triton.jit
def select_tiles(Route,PartLSE,PartCount,Counts,Lengths,Allowed,Slots,Masks,Offsets,TileCount,TilePages,
                 S:tl.constexpr,G:tl.constexpr,C:tl.constexpr,TILE:tl.constexpr,NG:tl.constexpr,NGB:tl.constexpr,SUB:tl.constexpr):
    seq=tl.program_id(0);kh=seq//(G//SUB);group=tl.program_id(1);lane=tl.arange(0,16);head=seq*SUB+lane;hm=lane<SUB
    j=tl.arange(0,TILE);i=group*TILE+j;g=tl.arange(0,NGB)
    part=tl.load(PartLSE+head[:,None]*NG+g[None,:],mask=hm[:,None]&(g[None,:]<NG),other=-float('inf'))
    pm=tl.max(part,1);safe=tl.where(pm!=-float('inf'),pm,0.)
    z=tl.sum(tl.exp(part-safe[:,None]),1)
    nonempty=tl.sum(tl.load(PartCount+kh*NG+g,mask=g<NG,other=0),0)
    cutoff=tl.where(nonempty>0,safe+tl.log(z)+tl.log(C/nonempty),float('inf'))
    r=tl.load(Route+head[:,None]*S+i[None,:],mask=hm[:,None]&(i[None,:]<S),other=-float('inf'))
    count=tl.load(Counts+kh*S+i,mask=i<S,other=0).to(tl.float32)
    opened=hm[:,None]&(i[None,:]<S)&(count[None,:]>0)&(count[None,:]<=1024)&(r>cutoff[:,None])
    tl.store(Allowed+head[:,None]*S+i[None,:],opened.to(tl.int32),mask=hm[:,None]&(i[None,:]<S))
    bits=tl.sum(tl.where(opened,1<<lane[:,None],0),0);union=bits!=0
    rank=tl.cumsum(union.to(tl.int32),0)-1
    size=tl.load(Lengths+kh*S+i,mask=i<S,other=0);pages=tl.where(union,(size+15)//16,0)
    before=tl.cumsum(pages,0)-pages;dst=(seq*NG+group)*TILE+rank
    tl.store(Slots+dst,i,mask=union);tl.store(Masks+dst,bits,mask=union);tl.store(Offsets+dst,before,mask=union)
    tl.store(TileCount+seq*NG+group,tl.sum(union.to(tl.int32),0));tl.store(TilePages+seq*NG+group,tl.sum(pages,0))


@triton.jit
def expand_tiles(Slots,Masks,Offsets,TileCount,GroupPageOffsets,TotalPages,Lengths,FixedOffsets,
                 Descriptors,DescriptorMasks,ContextLens,StateLen,LocalLen,NewK,NewV,ArenaK,ArenaV,
                 S:tl.constexpr,D:tl.constexpr,TILE:tl.constexpr,NG:tl.constexpr,PAGECAP:tl.constexpr,
                 FIXED_BEGIN:tl.constexpr,LOCAL_OFFSET:tl.constexpr,LOCAL_CAP:tl.constexpr,LOCAL_LIMIT:tl.constexpr,SINK:tl.constexpr,
                 SUBGROUPS:tl.constexpr):
    seq=tl.program_id(0);kh=seq//SUBGROUPS;group=tl.program_id(1);lane=tl.arange(0,64)
    count=tl.load(TileCount+seq*NG+group);base=tl.load(GroupPageOffsets+seq*NG+group)
    for rank in range(count):
        src=(seq*NG+group)*TILE+rank
        slot=tl.load(Slots+src);bits=tl.load(Masks+src);offset=base+tl.load(Offsets+src)
        size=tl.load(Lengths+kh*S+slot);pages=tl.cdiv(size,16)
        fixed=tl.load(FixedOffsets+kh*(S+1)+slot)+FIXED_BEGIN
        valid=lane<pages;tail=tl.minimum(16,size-lane*16)
        packed=(tail<<24)|(fixed+lane*16)
        tl.store(Descriptors+seq*PAGECAP+offset+lane,packed,mask=valid)
        tl.store(DescriptorMasks+seq*PAGECAP+offset+lane,bits,mask=valid)
    if group==0:
        local=tl.load(LocalLen);active=tl.load(StateLen)
        pages=tl.load(TotalPages+seq)
        tl.store(ContextLens+seq,active+tl.minimum(local,LOCAL_LIMIT)+1+SINK+pages*16)
        if seq%SUBGROUPS==0:
            dim=tl.arange(0,D);key=tl.load(NewK+kh*D+dim);value=tl.load(NewV+kh*D+dim)
            tl.store(ArenaK+(LOCAL_OFFSET+kh*LOCAL_CAP+local)*D+dim,key)
            tl.store(ArenaV+(LOCAL_OFFSET+kh*LOCAL_CAP+local)*D+dim,value)


@triton.jit
def attention(Q,ArenaK,ArenaV,Counts,Allowed,Descriptors,DescriptorMasks,Fixed,ContextLens,TotalPages,StateLen,LocalLen,
               Partial,Maximum,Denominator,S:tl.constexpr,G:tl.constexpr,D:tl.constexpr,SCALE:tl.constexpr,
               PAGECAP:tl.constexpr,FIXED_STRIDE:tl.constexpr,ARENA_CAP:tl.constexpr,COARSE_OFFSET:tl.constexpr,
               LOCAL_OFFSET:tl.constexpr,LOCAL_CAP:tl.constexpr,LOCAL_LIMIT:tl.constexpr,SINK_OFFSET:tl.constexpr,
               SINK:tl.constexpr,N:tl.constexpr,SPLITS:tl.constexpr,PRECISE:tl.constexpr,SUB:tl.constexpr):
    seq=tl.program_id(0);kh=seq//(G//SUB);split=tl.program_id(1);lane=tl.arange(0,16);head=seq*SUB+lane;hm=lane<SUB
    dim=tl.arange(0,D);token=tl.arange(0,N)
    q=tl.load(Q+head[:,None]*D+dim[None,:],mask=hm[:,None],other=0).to(tl.bfloat16)
    total=tl.load(ContextLens+seq);pages=tl.load(TotalPages+seq);remote=pages*16
    prefix=total-remote;local=tl.load(LocalLen);old=tl.minimum(local,LOCAL_LIMIT);active=tl.load(StateLen)
    tiles=tl.cdiv(total,SPLITS*N);begin=split*tiles
    if begin*N>=total:return
    mx=tl.full((16,),-float('inf'),tl.float32);den=tl.zeros((16,),tl.float32);acc=tl.zeros((16,D),tl.float32)
    for tile in range(begin,tl.minimum((split+1)*tiles,tl.cdiv(total,N))):
        pos=tile*N+token;valid=pos<total
        is_old=pos<old;is_current=pos==old
        sink_idx=pos-old-1;is_sink=(sink_idx>=0)&(sink_idx<SINK)
        region=pos-old-1-SINK;is_coarse=(region>=0)&(region<active)
        exact=pos-prefix;is_exact=exact>=0
        rank=tl.maximum(0,tl.minimum(exact//16,PAGECAP-1))
        desc=tl.load(Descriptors+seq*PAGECAP+rank,mask=valid&is_exact,other=0)
        bits=tl.load(DescriptorMasks+seq*PAGECAP+rank,mask=valid&is_exact,other=0)
        fixedpos=(desc&0x00ffffff)+exact%16
        exact_ok=valid&is_exact&(exact%16<(desc>>24))&(fixedpos<FIXED_STRIDE)
        physical=tl.load(Fixed+kh*FIXED_STRIDE+tl.maximum(fixedpos,0),mask=exact_ok,other=-1)
        physical=tl.where(is_coarse,COARSE_OFFSET+kh*S+region,physical)
        physical=tl.where(is_sink,SINK_OFFSET+kh*SINK+sink_idx,physical)
        physical=tl.where(is_current,LOCAL_OFFSET+kh*LOCAL_CAP+local,physical)
        physical=tl.where(is_old,LOCAL_OFFSET+kh*LOCAL_CAP+pos,physical)
        ok=valid&(~is_exact|exact_ok)&(physical>=0)&(physical<ARENA_CAP)
        safe_physical=tl.maximum(0,tl.minimum(physical,ARENA_CAP-1))
        key=tl.load(ArenaK+safe_physical[None,:]*D+dim[:,None],mask=ok[None,:],other=0,cache_modifier='.cg').to(tl.bfloat16)
        value=tl.load(ArenaV+safe_physical[:,None]*D+dim[None,:],mask=ok[:,None],other=0,cache_modifier='.cg').to(tl.bfloat16)
        safe_region=tl.maximum(0,tl.minimum(region,S-1))
        count=tl.load(Counts+kh*S+safe_region,mask=is_coarse,other=1.).to(tl.float32)
        opened=tl.load(Allowed+head[:,None]*S+safe_region[None,:],mask=hm[:,None]&is_coarse[None,:],other=0)!=0
        member=hm[:,None]&ok[None,:]&tl.where(is_coarse[None,:],~opened&(count[None,:]>0),
                tl.where(is_exact[None,:],(bits[None,:]&(1<<lane[:,None]))!=0,True))
        score=tl.dot(q,key,out_dtype=tl.float32)*(SCALE*1.4426950408889634)
        score+=tl.where(is_coarse,tl.log2(tl.maximum(count,1.)),0.)[None,:]
        score=tl.where(member,score,-float('inf'))
        nm=tl.maximum(mx,tl.max(score,1));safe=tl.where(nm!=-float('inf'),nm,0.)
        adjust=tl.exp2(mx-safe);w=tl.where(member,tl.exp2(score-safe[:,None]),0.)
        den=den*adjust+tl.sum(w,1);acc=acc*adjust[:,None]
        acc=tl.dot(w.to(tl.bfloat16),value,acc=acc)
        if PRECISE:acc=tl.dot((w-w.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16),value,acc=acc)
        mx=nm
    row=head*SPLITS+split
    tl.store(Partial+row[:,None]*D+dim[None,:],acc,mask=hm[:,None])
    tl.store(Maximum+row,mx,mask=hm);tl.store(Denominator+row,den,mask=hm)


def threshold_attention_flat(pool,q,k,v,out,multiplier,*,tile=64,splits=64,rule='region',**_):
    if rule!='region':raise ValueError('flat frontier currently supports equal-region thresholds')
    cache=prepare_cache(pool);S=pool.state_capacity;KV=pool.kv_heads;D=q.size(-1);H=q.size(2);G=H//KV;NG=triton.cdiv(S,tile)
    sub=int(os.environ.get('LOD_AMD_THRESHOLD_GQA_GROUP',str(G)))
    if G%sub:raise ValueError('GQA subgroup must divide the model group size')
    sequences=H//sub
    spec=(S,tile,splits,sub);b=getattr(pool,'_threshold_flat_scratch',None);st=pool.state;page=st['page_cache'];pc=page['page_indices'].size(2)
    if b is None or b['spec']!=spec:
        def f(*shape):return torch.empty(shape,device=q.device,dtype=torch.float32)
        def i(*shape):return torch.empty(shape,device=q.device,dtype=torch.int32)
        b=dict(spec=spec,mask_only=True,route=f(H,S),part_lse=f(H,NG),part_count=i(KV,NG),allowed=i(H,S),saved=i(1),
               slots=i(sequences,NG,tile),masks=i(sequences,NG,tile),offsets=i(sequences,NG,tile),tile_count=i(sequences,NG),tile_pages=i(sequences,NG),
               group_offsets=i(sequences,NG),group_page_offsets=i(sequences,NG),count=i(sequences),pages=i(sequences),
               descriptors=i(sequences,pc),descriptor_masks=i(sequences,pc),lens=i(sequences),partial=f(H,splits,D),mx=f(H,splits),den=f(H,splits),
               cache_index=torch.zeros(1,device=q.device,dtype=torch.int32))
        pool._threshold_flat_scratch=b
    pool._threshold_scratch=b;scale=float(pool.engine.scaling);localcap=st['recent_k'].size(2);sink=st['sink_k'].size(2);limit=int(pool.decode_local_limit)
    scores_lse[(KV,NG)](q,cache['mean_k'],st['counts'],pool.state_lens,pool.local_lens,b['route'],b['part_lse'],b['part_count'],b['saved'],b['route'],
                        S,G,D,scale,tile,NG,num_warps=4)
    select_tiles[(sequences,NG)](b['route'],b['part_lse'],b['part_count'],st['counts'],page['slot_lengths'],b['allowed'],b['slots'],b['masks'],
        b['offsets'],b['tile_count'],b['tile_pages'],S,G,float(multiplier),tile,NG,triton.next_power_of_2(NG),sub,num_warps=4)
    _scan_groups[(sequences,)](b['tile_count'],b['tile_pages'],b['group_offsets'],b['group_page_offsets'],b['count'],b['pages'],NG,num_warps=1)
    expand_tiles[(sequences,NG)](b['slots'],b['masks'],b['offsets'],b['tile_count'],b['group_page_offsets'],b['pages'],page['slot_lengths'],
        cache['offsets'],b['descriptors'],b['descriptor_masks'],b['lens'],pool.state_lens,pool.local_lens,k,v,cache['arena_k'],cache['arena_v'],
        S,D,tile,NG,pc,cache['fixed_begin'],cache['local_offset'],localcap,limit,sink,G//sub,num_warps=1)
    n=int(os.environ.get('LOD_AMD_THRESHOLD_ENTRY_BLOCK','128'))
    attention[(sequences,splits)](q,cache['arena_k'],cache['arena_v'],st['counts'],b['allowed'],b['descriptors'],b['descriptor_masks'],cache['fixed'],
        b['lens'],b['pages'],pool.state_lens,pool.local_lens,b['partial'],b['mx'],b['den'],S,G,D,scale,pc,cache['fixed'].size(-1),
        cache['arena_k'].numel()//D,cache['coarse_offset'],cache['local_offset'],localcap,limit,cache['sink_offset'],sink,n,splits,
        os.environ.get('LOD_AMD_THRESHOLD_PRECISE','1')=='1',sub,num_warps=int(os.environ.get('LOD_AMD_THRESHOLD_ENTRY_WARPS','2')),num_stages=1)
    reduce_page1_segments_advance_local[(sequences,sub)](out,b['partial'],b['mx'],b['den'],b['lens'],b['cache_index'],pool.local_lens,
        sub*D,D,NUM_QUERY_HEADS=sub,KV_HEADS=sequences,TILE_SIZE=n,HEAD_SIZE=D,NUM_SEGMENTS=splits,ADVANCE_LOCAL=True,num_warps=2)
    return out
