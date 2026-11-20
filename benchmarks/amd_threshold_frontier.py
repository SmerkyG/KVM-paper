"""Three-stage AMD threshold attention using the persistent centroid leaf index.

1. Shared GQA proxy scores and tile normalizers.
2. Threshold + grouped exact/coarse/local attention, without route packing.
3. Final partial reduction and KV append.

No selected-region cap, sorting or host count reads are introduced. Qwen
already maintains the persistent means and leaf index. Other opt-in serving
paths explicitly add index maintenance at cache catch-up boundaries; full-model
experiments include that cost. This module makes no end-to-end work bound.
"""
import os
import torch
import triton
import triton.language as tl


@triton.jit
def _fixture_indices(Lengths,Offsets,Roots,Directory,Pages,Fixed,Owners,S:tl.constexpr,ROOTS:tl.constexpr,
                     PAGECAP:tl.constexpr,LEAFCAP:tl.constexpr):
    kh=tl.program_id(0);slot=tl.program_id(1);lane=tl.arange(0,256)
    size=tl.load(Lengths+kh*S+slot);base=tl.load(Offsets+kh*(S+1)+slot)
    for start in range(0,size,256):
        ordinal=start+lane;page_ord=ordinal//16
        root=tl.load(Roots+(kh*S+slot)*ROOTS+page_ord//64,mask=(ordinal<size)&(page_ord//64<ROOTS),other=-1)
        page=tl.load(Directory+(kh*PAGECAP+tl.maximum(root,0))*64+page_ord%64,
                     mask=(ordinal<size)&(root>=0)&(root<PAGECAP),other=-1)
        idx=tl.load(Pages+(kh*PAGECAP+tl.maximum(page,0))*16+ordinal%16,
                    mask=(ordinal<size)&(page>=0)&(page<PAGECAP),other=-1)
        physical=tl.where((idx>=0)&(idx<LEAFCAP),kh*LEAFCAP+idx,-1)
        tl.store(Fixed+kh*LEAFCAP+base+ordinal,physical,mask=ordinal<size)
        tl.store(Owners+kh*LEAFCAP+base+ordinal,slot,mask=ordinal<size)


def prepare_cache(pool):
    """Reuse serving storage; reconstruct it only for standalone test fixtures."""
    cached=getattr(pool,'_threshold_frontier_cache',None)
    if cached is not None:return cached
    page=pool.state['page_cache'];S=pool.state_capacity;KV=pool.kv_heads;D=pool.state['state_k'].size(-1)
    if 'unified_page1_k' in page:
        rows = pool.state['state_k'].size(0) * KV
        offset=page['unified_page1_coarse_offset'];size=rows*S
        cached=dict(mean_k=page['unified_page1_k'][offset:offset+size].view(rows,S,D),
                    mean_v=page['unified_page1_v'][offset:offset+size].view(rows,S,D),
                    arena_k=page['unified_page1_k'],arena_v=page['unified_page1_v'],
                    fixed=page['unified_page1_fixed_indices'],offsets=page['unified_page1_fixed_slot_offsets'],
                    owners=page['unified_page1_fixed_leaf_owners'],leaf_capacity=page['leaf_k'].size(2),
                    fixed_begin=int(pool.engine.local_len)+S+pool.state['sink_k'].size(2),
                    local_offset=page['unified_page1_local_offset'],sink_offset=page['unified_page1_sink_offset'],
                    coarse_offset=offset)
    else:
        # This branch is fixture setup, before graph timing. Real serving
        # pools must use their normal amortized update implementation above.
        from lod_attention.kernels.paged_decode_kernels import materialize_page1_coarse_means
        mean_k=torch.empty_like(pool.state['state_k']);mean_v=torch.empty_like(pool.state['state_v'])
        bias=torch.empty(pool.state['counts'].shape[:-1],device=mean_k.device,dtype=torch.float16)
        KV *= mean_k.size(0)
        materialize_page1_coarse_means(pool.state['state_k'],pool.state['state_v'],pool.state['counts'],mean_k,mean_v,bias)
        lengths=page['slot_lengths'].reshape(KV,S)
        offsets=torch.cat((torch.zeros((KV,1),device=lengths.device,dtype=torch.int32),lengths.cumsum(-1,dtype=torch.int32)),1)
        leafcap=page['leaf_k'].size(2);fixed=torch.empty((KV,leafcap),device=lengths.device,dtype=torch.int32)
        owners=torch.empty_like(fixed)
        _fixture_indices[(KV,S)](lengths,offsets,page['slot_pages'],page['overflow_page_values'],page['page_indices'],fixed,owners,
                                S,page['slot_pages'].size(3),page['page_indices'].size(2),leafcap,num_warps=4)
        local_offset=KV*leafcap;sink_offset=local_offset+pool.state['recent_k'].numel()//D
        coarse_offset=sink_offset+pool.state['sink_k'].numel()//D
        arena_k=torch.cat([page['leaf_k'].reshape(-1,D),pool.state['recent_k'].reshape(-1,D),pool.state['sink_k'].reshape(-1,D),mean_k.reshape(-1,D)])
        arena_v=torch.cat([page['leaf_v'].reshape(-1,D),pool.state['recent_v'].reshape(-1,D),pool.state['sink_v'].reshape(-1,D),mean_v.reshape(-1,D)])
        for key,arena in [('recent_k',arena_k),('recent_v',arena_v)]:
            pool.state[key]=arena[local_offset:sink_offset].view_as(pool.state[key])
        cached=dict(mean_k=arena_k[coarse_offset:].view_as(mean_k),mean_v=arena_v[coarse_offset:].view_as(mean_v),
                    arena_k=arena_k,arena_v=arena_v,fixed=fixed,offsets=offsets,fixed_begin=0,
                    owners=owners,leaf_capacity=leafcap,
                    local_offset=local_offset,sink_offset=sink_offset,coarse_offset=coarse_offset)
    pool._threshold_frontier_cache=cached
    return cached


@triton.jit
def scores_lse(Q,MeanK,Counts,StateLen,LocalLen,Route,PartLSE,PartCount,Saved,Native,
               S:tl.constexpr,G:tl.constexpr,D:tl.constexpr,SCALE:tl.constexpr,TILE:tl.constexpr,NG:tl.constexpr,STORE_NATIVE:tl.constexpr=False,
               NewK=None,NewV=None,LocalK=None,LocalV=None,LOCALCAP:tl.constexpr=0,APPEND:tl.constexpr=False,
               Active=None,ACTIVECAP:tl.constexpr=0,REMOTEBLOCKS:tl.constexpr=0,BLOCKN:tl.constexpr=128,
               LOCALLIMIT:tl.constexpr=0,SINK:tl.constexpr=0,CLEAR_ACTIVE:tl.constexpr=False,
               BATCH_KV_HEADS:tl.constexpr=0, CacheIndices=None):
    kh=tl.program_id(0);group=tl.program_id(1);lane=tl.arange(0,16);head=kh*G+lane;hm=lane<G
    row=kh//BATCH_KV_HEADS if BATCH_KV_HEADS else 0
    cache_row = tl.load(CacheIndices + row).to(tl.int32) if CacheIndices is not None else row
    cache_kh = cache_row * BATCH_KV_HEADS + kh % BATCH_KV_HEADS if BATCH_KV_HEADS else kh
    i=group*TILE+tl.arange(0,TILE);dim=tl.arange(0,D)
    active=tl.load(StateLen+cache_row);count=tl.load(Counts+cache_kh*S+i,mask=(i<S)&(i<active),other=0).to(tl.float32)
    valid=(i<S)&(i<active)&(count>0)
    q=tl.load(Q+head[:,None]*D+dim[None,:],mask=hm[:,None],other=0).to(tl.bfloat16)
    key=tl.load(MeanK+(cache_kh*S+i[:,None])*D+dim[None,:],mask=valid[:,None],other=0).to(tl.bfloat16)
    inv=tl.rsqrt(tl.maximum(tl.sum(q.to(tl.float32)*q.to(tl.float32),1)/D,1.e-12))
    score=tl.dot(q,tl.trans(key),out_dtype=tl.float32)*SCALE
    if STORE_NATIVE:
        native=tl.where(hm[:,None]&valid[None,:],score+tl.log(tl.maximum(count,1.))[None,:],-float('inf'))
        tl.store(Native+head[:,None]*S+i[None,:],native,mask=hm[:,None]&(i[None,:]<S))
    score=tl.where(hm[:,None]&valid[None,:],score*inv[:,None]+tl.log(tl.maximum(count,1.))[None,:],-float('inf'))
    tl.store(Route+head[:,None]*S+i[None,:],score,mask=hm[:,None]&(i[None,:]<S))
    mx=tl.max(score,1);safe=tl.where(mx!=-float('inf'),mx,0.)
    z=tl.sum(tl.where(valid[None,:],tl.exp(score-safe[:,None]),0.),1)
    tl.store(PartLSE+head*NG+group,tl.where(z>0,safe+tl.log(z),-float('inf')),mask=hm)
    tl.store(PartCount+kh*NG+group,tl.sum(valid.to(tl.int32),0))
    if (kh%BATCH_KV_HEADS==0 if BATCH_KV_HEADS else kh==0) and group==0:
        tl.store(Saved+row,tl.load(LocalLen+cache_row))
    if APPEND:
        if group==0:
            n=tl.load(LocalLen+cache_row);nk=tl.load(NewK+kh*D+dim);nv=tl.load(NewV+kh*D+dim)
            tl.store(LocalK+(cache_kh*LOCALCAP+n)*D+dim,nk);tl.store(LocalV+(cache_kh*LOCALCAP+n)*D+dim,nv)
    if CLEAR_ACTIVE:
        local=tl.minimum(tl.load(LocalLen),LOCALLIMIT)
        for block_start in range(group*TILE,ACTIVECAP,NG*TILE):
            block=block_start+tl.arange(0,TILE)
            is_local=(block>=REMOTEBLOCKS)&((block-REMOTEBLOCKS)*BLOCKN<local+1+SINK)
            tl.store(Active+kh*ACTIVECAP+block,tl.where(is_local,(1<<G)-1,0),mask=block<ACTIVECAP)


@triton.jit
def route_cutoff(PartLSE, PartCount, Cutoffs, NG: tl.constexpr, NGB: tl.constexpr,
                 G: tl.constexpr, C: tl.constexpr):
    head = tl.program_id(0)
    g = tl.arange(0, NGB)
    part = tl.load(PartLSE + head * NG + g, mask=g < NG, other=-float('inf'))
    maximum = tl.max(part, 0)
    safe = tl.where(maximum != -float('inf'), maximum, 0.)
    mass = tl.sum(tl.exp(part - safe), 0)
    count = tl.sum(tl.load(PartCount + (head // G) * NG + g, mask=g < NG, other=0), 0)
    tl.store(Cutoffs + head, tl.where(count > 0, safe + tl.log(mass) + tl.log(C / count), float('inf')))


@triton.jit
def frontier_attention(Q,MeanK,MeanV,Counts,Lengths,Route,PartLSE,PartCount,Allowed,Fixed,Offsets,ArenaK,ArenaV,
                       LocalK,LocalV,SinkK,SinkV,NewK,NewV,Saved,Out,LSE,
                       S:tl.constexpr,G:tl.constexpr,D:tl.constexpr,C:tl.constexpr,SCALE:tl.constexpr,TILE:tl.constexpr,
                       NG:tl.constexpr,NGB:tl.constexpr,N:tl.constexpr,SPLITS:tl.constexpr,FIXED_STRIDE:tl.constexpr,
                       FIXED_BEGIN:tl.constexpr,ARENA_CAP:tl.constexpr,LOCAL_CAP:tl.constexpr,LOCAL_LIMIT:tl.constexpr,
                       SINK:tl.constexpr,PRECISE:tl.constexpr):
    kh=tl.program_id(0);group=tl.program_id(1);split=tl.program_id(2)
    lane=tl.arange(0,16);head=kh*G+lane;hm=lane<G;dim=tl.arange(0,D);j=tl.arange(0,TILE);i=group*TILE+j
    q=tl.load(Q+head[:,None]*D+dim[None,:],mask=hm[:,None],other=0).to(tl.bfloat16)
    g=tl.arange(0,NGB)
    part=tl.load(PartLSE+head[:,None]*NG+g[None,:],mask=hm[:,None]&(g[None,:]<NG),other=-float('inf'))
    pm=tl.max(part,1);safe_pm=tl.where(pm!=-float('inf'),pm,0.)
    pz=tl.sum(tl.exp(part-safe_pm[:,None]),1)
    nonempty=tl.sum(tl.load(PartCount+kh*NG+g,mask=g<NG,other=0),0)
    cutoff=tl.where(nonempty>0,safe_pm+tl.log(pz)+tl.log(C/nonempty),float('inf'))
    route=tl.load(Route+head[:,None]*S+i[None,:],mask=hm[:,None]&(i[None,:]<S),other=-float('inf'))
    count=tl.load(Counts+kh*S+i,mask=i<S,other=0).to(tl.float32)
    valid=(i<S)&(count>0)&(tl.max(route,0)!=-float('inf'))
    opened=hm[:,None]&valid[None,:]&(count[None,:]<=1024)&(route>cutoff[:,None])
    if split==0:tl.store(Allowed+head[:,None]*S+i[None,:],opened.to(tl.int32),mask=hm[:,None]&(i[None,:]<S))
    mx=tl.full((16,),-float('inf'),tl.float32);den=tl.zeros((16,),tl.float32);acc=tl.zeros((16,D),tl.float32)
    if split==0:
        key=tl.load(MeanK+(kh*S+i[:,None])*D+dim[None,:],mask=valid[:,None],other=0).to(tl.bfloat16)
        value=tl.load(MeanV+(kh*S+i[:,None])*D+dim[None,:],mask=valid[:,None],other=0).to(tl.bfloat16)
        closed=hm[:,None]&valid[None,:]&~opened
        score=(tl.dot(q,tl.trans(key),out_dtype=tl.float32)*SCALE+tl.log(tl.maximum(count,1.))[None,:])*1.4426950408889634
        score=tl.where(closed,score,-float('inf'));mx=tl.max(score,1);safe=tl.where(mx!=-float('inf'),mx,0.)
        w=tl.where(closed,tl.exp2(score-safe[:,None]),0.);den=tl.sum(w,1)
        acc=tl.dot(w.to(tl.bfloat16),value,out_dtype=tl.float32)
        if PRECISE:acc+=tl.dot((w-w.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16),value,out_dtype=tl.float32)
    union=tl.sum(opened.to(tl.int32),0)>0;left=tl.sum(union.to(tl.int32),0);token=tl.arange(0,N)
    while left>0:
        selected=tl.max(tl.where(union,j,-1),0);slot=group*TILE+selected
        member=tl.sum((opened&(j[None,:]==selected)).to(tl.int32),1)>0
        length=tl.load(Lengths+kh*S+slot);offset=tl.load(Offsets+kh*(S+1)+slot)
        for start in range(split*N,length,SPLITS*N):
            ordinal=start+token;where=FIXED_BEGIN+offset+ordinal
            physical=tl.load(Fixed+kh*FIXED_STRIDE+where,mask=(ordinal<length)&(where<FIXED_STRIDE),other=-1)
            ok=(ordinal<length)&(physical>=0)&(physical<ARENA_CAP)
            safe_physical=tl.maximum(physical,0)
            key=tl.load(ArenaK+safe_physical[None,:]*D+dim[:,None],mask=ok[None,:],other=0).to(tl.bfloat16)
            value=tl.load(ArenaV+safe_physical[:,None]*D+dim[None,:],mask=ok[:,None],other=0).to(tl.bfloat16)
            active=member[:,None]&ok[None,:]
            score=tl.where(active,tl.dot(q,key,out_dtype=tl.float32)*(SCALE*1.4426950408889634),-float('inf'))
            nm=tl.maximum(mx,tl.max(score,1));safe=tl.where(nm!=-float('inf'),nm,0.)
            adjust=tl.exp2(mx-safe);w=tl.where(active,tl.exp2(score-safe[:,None]),0.)
            acc=acc*adjust[:,None]+tl.dot(w.to(tl.bfloat16),value,out_dtype=tl.float32)
            if PRECISE:acc+=tl.dot((w-w.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16),value,out_dtype=tl.float32)
            den=den*adjust+tl.sum(w,1);mx=nm
        union&=j!=selected;left-=1
    if group==0:
        local=tl.load(Saved)
        for start in range(split*N,LOCAL_LIMIT+1+SINK,SPLITS*N):
            loc=start+token;old=(loc<local)&(loc<LOCAL_LIMIT);current=loc==local
            sink=(loc>=LOCAL_LIMIT+1)&(loc<LOCAL_LIMIT+1+SINK)
            key=tl.load(LocalK+(kh*LOCAL_CAP+loc[:,None])*D+dim[None,:],mask=old[:,None],other=0).to(tl.bfloat16)
            value=tl.load(LocalV+(kh*LOCAL_CAP+loc[:,None])*D+dim[None,:],mask=old[:,None],other=0).to(tl.bfloat16)
            nk=tl.load(NewK+kh*D+dim).to(tl.bfloat16);nv=tl.load(NewV+kh*D+dim).to(tl.bfloat16)
            key=tl.where(current[:,None],nk[None,:],key);value=tl.where(current[:,None],nv[None,:],value)
            if SINK:
                sk=tl.load(SinkK+(kh*SINK+(loc-LOCAL_LIMIT-1)[:,None])*D+dim[None,:],mask=sink[:,None],other=0).to(tl.bfloat16)
                sv=tl.load(SinkV+(kh*SINK+(loc-LOCAL_LIMIT-1)[:,None])*D+dim[None,:],mask=sink[:,None],other=0).to(tl.bfloat16)
                key=tl.where(sink[:,None],sk,key);value=tl.where(sink[:,None],sv,value)
            active=hm[:,None]&(old|current|sink)[None,:]
            score=tl.where(active,tl.dot(q,tl.trans(key),out_dtype=tl.float32)*(SCALE*1.4426950408889634),-float('inf'))
            nm=tl.maximum(mx,tl.max(score,1));safe=tl.where(nm!=-float('inf'),nm,0.)
            adjust=tl.exp2(mx-safe);w=tl.where(active,tl.exp2(score-safe[:,None]),0.)
            acc=acc*adjust[:,None]+tl.dot(w.to(tl.bfloat16),value,out_dtype=tl.float32)
            if PRECISE:acc+=tl.dot((w-w.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16),value,out_dtype=tl.float32)
            den=den*adjust+tl.sum(w,1);mx=nm
    row=head*(NG*SPLITS)+group*SPLITS+split
    tl.store(Out+row[:,None]*D+dim[None,:],acc/tl.maximum(den[:,None],1.e-30),mask=hm[:,None])
    tl.store(LSE+row,tl.where(den>0,(mx+tl.log2(den))*0.6931471805599453,-float('inf')),mask=hm)


def threshold_attention_frontier(pool,q,k,v,out,multiplier,*,tile=32,splits=2,rule='region',**_):
    if rule!='region':raise ValueError('grouped frontier currently implements the equal-region threshold')
    if pool.settings.levels!=2 or pool.dtype!=torch.bfloat16 or q.shape[:2]!=(1,1):raise ValueError('requires batch-one BF16 two-tier')
    S=pool.state_capacity;KV=pool.kv_heads;D=q.size(-1);H=q.size(2);G=H//KV;NG=triton.cdiv(S,tile)
    cache=prepare_cache(pool);spec=(S,H,D,tile,splits);b=getattr(pool,'_threshold_frontier_scratch',None)
    if b is None or b['spec']!=spec:
        def f(*shape):return torch.empty(shape,device=q.device,dtype=torch.float32)
        def i(*shape):return torch.empty(shape,device=q.device,dtype=torch.int32)
        b=dict(spec=spec,mask_only=True,route=f(H,S),part_lse=f(H,NG),part_count=i(KV,NG),allowed=i(H,S),
               saved=i(1),co=f(H,NG*splits,D),cl=f(H,NG*splits))
        pool._threshold_frontier_scratch=b
    pool._threshold_scratch=b;st=pool.state;page=st['page_cache'];scale=float(pool.engine.scaling)
    scores_lse[(KV,NG)](q,cache['mean_k'],st['counts'],pool.state_lens,pool.local_lens,b['route'],b['part_lse'],
                         b['part_count'],b['saved'],b['route'],S,G,D,scale,tile,NG,num_warps=4)
    frontier_attention[(KV,NG,splits)](q,cache['mean_k'],cache['mean_v'],st['counts'],page['slot_lengths'],b['route'],
        b['part_lse'],b['part_count'],b['allowed'],cache['fixed'],cache['offsets'],cache['arena_k'],cache['arena_v'],
        st['recent_k'],st['recent_v'],st['sink_k'],st['sink_v'],k,v,b['saved'],b['co'],b['cl'],
        S,G,D,float(multiplier),scale,tile,NG,triton.next_power_of_2(NG),int(os.environ.get('LOD_AMD_THRESHOLD_ENTRY_BLOCK','64')),
        splits,cache['fixed'].size(-1),cache['fixed_begin'],cache['arena_k'].numel()//D,st['recent_k'].size(2),
        int(pool.decode_local_limit),st['sink_k'].size(2),os.environ.get('LOD_AMD_THRESHOLD_PRECISE','1')=='1',
        num_warps=int(os.environ.get('LOD_AMD_THRESHOLD_ENTRY_WARPS','4')),num_stages=1)
    from benchmarks.amd_threshold_merge import merge_split_d
    bd=int(os.environ.get('LOD_AMD_THRESHOLD_MERGE_BLOCK','32')) or 32
    merge_split_d[(H,triton.cdiv(D,bd))](b['co'],b['cl'],b['co'],b['cl'],b['co'],b['cl'],k,v,
        st['recent_k'],st['recent_v'],pool.local_lens,b['saved'],out,D,G,KV,st['recent_k'].size(2),NG*splits,0,0,
        triton.next_power_of_2(NG*splits),bd,num_warps=4)
    return out
