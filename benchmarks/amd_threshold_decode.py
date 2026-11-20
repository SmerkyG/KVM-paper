"""AMD two-tier threshold decode from shared GQA KV.

The router uses normalized-query, size-weighted region logits. The region
rule uses a threshold relative to mean proxy mass. The entry rule uses proxy
mass per KV entry and enforces a deterministic work limit. Persistent workers
consume the compacted region list. Serving batches address persistent cache
rows through a device-resident map.
"""
import math
import os
import torch
import triton
import triton.language as tl

@triton.jit
def _scores(q, state_k, counts, state_lens, local_lens, native, route, saved_len,
            S:tl.constexpr, KV:tl.constexpr, G:tl.constexpr, D:tl.constexpr,
            SCALE:tl.constexpr, TILE:tl.constexpr, BATCH_KV_HEADS:tl.constexpr=0, CacheIndices=None):
    kh=tl.program_id(0); block=tl.program_id(1)
    request = kh // BATCH_KV_HEADS if BATCH_KV_HEADS else 0
    cache_row = tl.load(CacheIndices + request).to(tl.int32) if CacheIndices is not None else request
    cache_kh = cache_row * BATCH_KV_HEADS + kh % BATCH_KV_HEADS if BATCH_KV_HEADS else kh
    heads=kh*G+tl.arange(0,16); hm=tl.arange(0,16)<G
    idx=block*TILE+tl.arange(0,TILE); dim=tl.arange(0,D)
    active=tl.load(state_lens+cache_row); valid=(idx<active)&(idx<S)
    cnt=tl.load(counts+cache_kh*S+idx,mask=valid,other=0).to(tl.float32)
    valid=valid&(cnt>0); cnt=tl.where(valid,cnt,1.)
    qq=tl.load(q+heads[:,None]*D+dim[None,:],mask=hm[:,None],other=0).to(tl.bfloat16)
    kk=tl.load(state_k+(cache_kh*S+idx[:,None])*D+dim[None,:],mask=valid[:,None],other=0)
    mean=(kk.to(tl.float32)/cnt[:,None]).to(tl.bfloat16)
    dot=tl.dot(qq,tl.trans(mean),out_dtype=tl.float32)*SCALE
    inv=tl.rsqrt(tl.maximum(tl.sum(qq.to(tl.float32)*qq.to(tl.float32),1)/D,1.e-12))
    logcnt=tl.log(cnt)
    sc=tl.where(hm[:,None]&valid[None,:],dot+logcnt[None,:],-float('inf'))
    rr=tl.where(hm[:,None]&valid[None,:],dot*inv[:,None]+logcnt[None,:],-float('inf'))
    tl.store(native+heads[:,None]*S+idx[None,:],sc,mask=hm[:,None]&(idx[None,:]<S))
    tl.store(route+heads[:,None]*S+idx[None,:],rr,mask=hm[:,None]&(idx[None,:]<S))
    if (kh % BATCH_KV_HEADS == 0 if BATCH_KV_HEADS else kh == 0) and block == 0:
        tl.store(saved_len+request,tl.load(local_lens+cache_row))

@triton.jit
def _threshold(route,slot_len,threshold,allowed,selected_counts,S:tl.constexpr,G:tl.constexpr,C:tl.constexpr,PER_ENTRY:tl.constexpr,ENTRY_BOUND:tl.constexpr,WORK_BOUND:tl.constexpr,BLOCK:tl.constexpr, BATCH_KV_HEADS:tl.constexpr=0, CacheIndices=None):
    head=tl.program_id(0);i=tl.arange(0,BLOCK)
    v=tl.load(route+head*S+i,mask=i<S,other=-float('inf'))
    valid=v!=-float('inf');n=tl.sum(valid.to(tl.float32),0)
    m=tl.max(v,0);safe=tl.where(n>0,m,0.)
    z=tl.sum(tl.where(valid,tl.exp(v-safe),0.),0)
    if PER_ENTRY:
        kh = head // G
        request = kh // BATCH_KV_HEADS if BATCH_KV_HEADS else 0
        cache_row = tl.load(CacheIndices + request).to(tl.int32) if CacheIndices is not None else request
        cache_kh = cache_row * BATCH_KV_HEADS + kh % BATCH_KV_HEADS if BATCH_KV_HEADS else kh
        sizes=tl.load(slot_len+cache_kh*S+i,mask=i<S,other=0).to(tl.int64)
        remote=tl.sum(tl.where(valid,sizes,0),0)
        cutoff=tl.where(remote>0,safe+tl.log(z)-tl.log(ENTRY_BOUND*1.)-0.5*tl.log(remote.to(tl.float32))+1.e-6,float('inf'))
        candidates=valid&(sizes>0)&(v-tl.log(tl.maximum(sizes.to(tl.float32),1.))>cutoff)
        total=tl.sum(tl.where(candidates,sizes,0),0)
        if total*total < WORK_BOUND*WORK_BOUND*remote:
            tl.store(allowed+head*S+i,candidates.to(tl.int32),mask=i<S)
        else:
            # Deterministically retain a prefix of candidates within the work
            # limit. A fixed 32-lane scan and serial carry use linear work.
            before=tl.full((),0,tl.int64);lane=tl.arange(0,32)
            for start in range(0,S,32):
                j=start+lane
                score=tl.load(route+head*S+j,mask=j<S,other=-float('inf'))
                count=tl.load(slot_len+cache_kh*S+j,mask=j<S,other=0).to(tl.int64)
                candidate=(count>0)&(score-tl.log(tl.maximum(count.to(tl.float32),1.))>cutoff)
                add=tl.where(candidate,count,0);prefix=before+tl.cumsum(add,0)
                accept=candidate&(prefix*prefix<WORK_BOUND*WORK_BOUND*remote)
                tl.store(allowed+head*S+j,accept.to(tl.int32),mask=j<S)
                before+=tl.sum(add,0)
    else:
        cutoff=tl.where(n>0,safe+tl.log(z)+tl.log(C/n),float('inf'))
    tl.store(threshold+head,cutoff);tl.store(selected_counts+head,0)

@triton.jit
def _coarse_select_gqa(native,route,threshold,allowed,counts,state_v,slot_len,selected,tile_offsets,tile_counts,tile_pages,part_o,part_lse,
                       S:tl.constexpr,G:tl.constexpr,D:tl.constexpr,TILE:tl.constexpr,NG:tl.constexpr,MAX_LEAF:tl.constexpr,PER_ENTRY:tl.constexpr,PACK:tl.constexpr=True,UNION:tl.constexpr=False, BATCH_KV_HEADS:tl.constexpr=0, CacheIndices=None):
    kh=tl.program_id(0);group=tl.program_id(1)
    request = kh // BATCH_KV_HEADS if BATCH_KV_HEADS else 0
    cache_row = tl.load(CacheIndices + request).to(tl.int32) if CacheIndices is not None else request
    cache_kh = cache_row * BATCH_KV_HEADS + kh % BATCH_KV_HEADS if BATCH_KV_HEADS else kh
    h=kh*G+tl.arange(0,16);hm=tl.arange(0,16)<G
    i=group*TILE+tl.arange(0,TILE);d=tl.arange(0,D)
    score=tl.load(native+h[:,None]*S+i[None,:],mask=hm[:,None]&(i[None,:]<S),other=-float('inf'))
    r=tl.load(route+h[:,None]*S+i[None,:],mask=hm[:,None]&(i[None,:]<S),other=-float('inf'))
    cnt=tl.load(counts+cache_kh*S+i,mask=i<S,other=0).to(tl.float32)
    valid=(i<S)&(cnt>0);cut=tl.load(threshold+h,mask=hm,other=float('inf'))
    if PER_ENTRY:
        opened=hm[:,None]&valid[None,:]&(tl.load(allowed+h[:,None]*S+i[None,:],mask=hm[:,None]&(i[None,:]<S),other=0)!=0)
    else:
        opened=hm[:,None]&valid[None,:]&(cnt[None,:]<=MAX_LEAF)&(r>cut[:,None])
    if UNION:
        bits=tl.sum(tl.where(opened,1<<tl.arange(0,16)[:,None],0),axis=0)
        tl.store(tile_offsets+kh*S+i,bits,mask=i<S)
    if PACK:
        prefix=tl.cumsum(opened.to(tl.int32),axis=1)-1;num=tl.sum(opened.to(tl.int32),1)
        lengths=tl.load(slot_len+cache_kh*S+i,mask=i<S,other=0)
        pages=tl.where(opened,(lengths[None,:]+15)//16,0)
        page_prefix=tl.cumsum(pages,axis=1)-pages
        tl.store(tile_counts+h*NG+group,num,mask=hm)
        tl.store(tile_pages+h*NG+group,tl.sum(pages,1),mask=hm)
        tl.store(selected+h[:,None]*NG*TILE+group*TILE+prefix,i[None,:],mask=opened)
        tl.store(tile_offsets+h[:,None]*NG*TILE+group*TILE+prefix,page_prefix,mask=opened)
    closed=hm[:,None]&valid[None,:]&~opened
    scores=tl.where(closed,score,-float('inf'));mx=tl.max(scores,1);safe=tl.where(mx!=-float('inf'),mx,0.)
    w=tl.where(closed,tl.exp(scores-safe[:,None]),0.);z=tl.sum(w,1)
    val=tl.load(state_v+(cache_kh*S+i[:,None])*D+d[None,:],mask=valid[:,None],other=0).to(tl.float32)
    mean=(val/tl.maximum(cnt[:,None],1.)).to(tl.bfloat16)
    result=tl.dot(w.to(tl.bfloat16),mean,out_dtype=tl.float32)/tl.maximum(z[:,None],1.e-30)
    row=h*NG+group
    tl.store(part_o+row[:,None]*D+d[None,:],result,mask=hm[:,None])
    tl.store(part_lse+row,tl.where(z>0,safe+tl.log(z),-float('inf')),mask=hm)

@triton.jit
def _scan_groups(tile_counts,tile_pages,group_offsets,group_page_offsets,selected_counts,total,NG:tl.constexpr):
    head=tl.program_id(0);lane=tl.arange(0,32)
    count_base=tl.full((),0,tl.int32);page_base=tl.full((),0,tl.int32)
    # Fixed-size scans plus a serial carry give linear work as NG grows.
    for start in range(0,NG,32):
        g=start+lane;counts=tl.load(tile_counts+head*NG+g,mask=g<NG,other=0)
        pages=tl.load(tile_pages+head*NG+g,mask=g<NG,other=0)
        tl.store(group_offsets+head*NG+g,count_base+tl.cumsum(counts,0)-counts,mask=g<NG)
        tl.store(group_page_offsets+head*NG+g,page_base+tl.cumsum(pages,0)-pages,mask=g<NG)
        count_base+=tl.sum(counts,0);page_base+=tl.sum(pages,0)
    tl.store(selected_counts+head,count_base);tl.store(total+head,page_base)

@triton.jit
def _pack_selected(tiles,tile_offsets,tile_counts,group_offsets,group_page_offsets,selected,offsets,S:tl.constexpr,NG:tl.constexpr,TILE:tl.constexpr):
    head=tl.program_id(0);group=tl.program_id(1);off=tl.arange(0,TILE)
    before=tl.load(group_offsets+head*NG+group)
    before_pages=tl.load(group_page_offsets+head*NG+group)
    number=tl.load(tile_counts+head*NG+group)
    slots=tl.load(tiles+head*NG*TILE+group*TILE+off,mask=off<number,other=0)
    local_offsets=tl.load(tile_offsets+head*NG*TILE+group*TILE+off,mask=off<number,other=0)
    tl.store(selected+head*S+before+off,slots,mask=off<number)
    tl.store(offsets+head*S+before+off,before_pages+local_offsets,mask=off<number)

@triton.jit
def _expand_pages(selected,selected_counts,slot_len,slot_pages,dirs,offsets,descriptors,
                  S:tl.constexpr,G:tl.constexpr,ROOTS:tl.constexpr,PAGECAP:tl.constexpr,WORKERS:tl.constexpr, BATCH_KV_HEADS:tl.constexpr=0, CacheIndices=None):
    head=tl.program_id(0);worker=tl.program_id(1);kh=head//G; n=tl.load(selected_counts+head)
    request = kh // BATCH_KV_HEADS if BATCH_KV_HEADS else 0
    cache_row = tl.load(CacheIndices + request).to(tl.int32) if CacheIndices is not None else request
    cache_kh = cache_row * BATCH_KV_HEADS + kh % BATCH_KV_HEADS if BATCH_KV_HEADS else kh
    for i in range(worker,n,WORKERS):
        slot=tl.load(selected+head*S+i);cnt=tl.load(slot_len+cache_kh*S+slot)
        np=(cnt+15)//16;offset=tl.load(offsets+head*S+i)
        for chunk in range(0,tl.cdiv(np,64)):
            ordinal=chunk*64+tl.arange(0,64);valid=ordinal<np
            root=tl.load(slot_pages+(cache_kh*S+slot)*ROOTS+chunk,mask=chunk<ROOTS,other=-1)
            root_ok=valid&(root>=0)&(root<PAGECAP)
            page=tl.load(dirs+(cache_kh*PAGECAP+tl.maximum(root,0))*64+ordinal%64,mask=root_ok,other=-1)
            page_ok=root_ok&(page>=0)&(page<PAGECAP)
            count=tl.minimum(16,cnt-ordinal*16);packed=tl.where(page_ok,page*32+count,-1)
            tl.store(descriptors+head*PAGECAP+offset+ordinal,packed,mask=valid)

@triton.jit
def _leaves(q,descriptors,total,page_indices,leaf_k,leaf_v,part_o,part_lse,
            KV:tl.constexpr,G:tl.constexpr,D:tl.constexpr,PAGECAP:tl.constexpr,LEAFCAP:tl.constexpr,
            SCALE:tl.constexpr,SPLITS:tl.constexpr, BATCH_KV_HEADS:tl.constexpr=0, CacheIndices=None):
    head=tl.program_id(0);split=tl.program_id(1);kh=head//G
    request = kh // BATCH_KV_HEADS if BATCH_KV_HEADS else 0
    cache_row = tl.load(CacheIndices + request).to(tl.int32) if CacheIndices is not None else request
    cache_kh = cache_row * BATCH_KV_HEADS + kh % BATCH_KV_HEADS if BATCH_KV_HEADS else kh
    d=tl.arange(0,D);t=tl.arange(0,16);qv=tl.load(q+head*D+d).to(tl.float32)
    n=tl.load(total+head);mx=tl.full((),-float('inf'),tl.float32)
    den=tl.zeros((),tl.float32);acc=tl.zeros((D,),tl.float32)
    for ii in range(split,n,SPLITS):
        desc=tl.load(descriptors+head*PAGECAP+ii); page=desc//32; count=desc%32
        page_ok=(page>=0)&(page<PAGECAP);valid=page_ok&(t<count)
        idx=tl.load(page_indices+(cache_kh*PAGECAP+tl.maximum(page,0))*16+t,mask=valid,other=-1)
        mask=valid&(idx>=0)&(idx<LEAFCAP)
        key=tl.load(leaf_k+(cache_kh*LEAFCAP+tl.maximum(idx,0))[:,None]*D+d[None,:],mask=mask[:,None],other=0).to(tl.float32)
        val=tl.load(leaf_v+(cache_kh*LEAFCAP+tl.maximum(idx,0))[:,None]*D+d[None,:],mask=mask[:,None],other=0).to(tl.float32)
        sc=tl.where(mask,tl.sum(key*qv[None,:],1)*SCALE,-float('inf'))
        pm=tl.max(sc,0);nm=tl.maximum(mx,pm);safe=tl.where(nm!=-float('inf'),nm,0.)
        factor=tl.where(mx!=-float('inf'),tl.exp(mx-safe),0.);w=tl.where(mask,tl.exp(sc-safe),0.)
        den=den*factor+tl.sum(w,0);acc=acc*factor+tl.sum(w[:,None]*val,0);mx=nm
    off=head*SPLITS+split
    tl.store(part_o+off*D+d,acc/tl.maximum(den,1.e-30));tl.store(part_lse+off,tl.where(den>0,mx+tl.log(den),-float('inf')))

@triton.jit
def _local(q,local_k,local_v,sink_k,sink_v,new_k,new_v,saved_len,out,lse,
           KV:tl.constexpr,G:tl.constexpr,D:tl.constexpr,LOCAL_CAP:tl.constexpr,
           LOCAL_LIMIT:tl.constexpr,SINK:tl.constexpr,SCALE:tl.constexpr,TILE:tl.constexpr,NG:tl.constexpr,
           BATCH_KV_HEADS:tl.constexpr=0, CacheIndices=None):
    head=tl.program_id(0);group=tl.program_id(1);kh=head//G
    row=kh//BATCH_KV_HEADS if BATCH_KV_HEADS else 0
    cache_row = tl.load(CacheIndices + row).to(tl.int32) if CacheIndices is not None else row
    cache_kh = cache_row * BATCH_KV_HEADS + kh % BATCH_KV_HEADS if BATCH_KV_HEADS else kh
    d=tl.arange(0,D);i=group*TILE+tl.arange(0,TILE);n=tl.load(saved_len+row)
    old=(i<n)&(i<LOCAL_LIMIT); current=(i==n); sink=(i>=LOCAL_LIMIT+1)&(i<LOCAL_LIMIT+1+SINK)
    qv=tl.load(q+head*D+d).to(tl.float32)
    key=tl.load(local_k+(cache_kh*LOCAL_CAP+i[:,None])*D+d[None,:],mask=old[:,None],other=0).to(tl.float32)
    val=tl.load(local_v+(cache_kh*LOCAL_CAP+i[:,None])*D+d[None,:],mask=old[:,None],other=0).to(tl.float32)
    nk=tl.load(new_k+kh*D+d).to(tl.float32);nv=tl.load(new_v+kh*D+d).to(tl.float32)
    key=tl.where(current[:,None],nk[None,:],key);val=tl.where(current[:,None],nv[None,:],val)
    if SINK:
        sk=tl.load(sink_k+(cache_kh*SINK+tl.maximum(i-LOCAL_LIMIT-1,0)[:,None])*D+d[None,:],mask=sink[:,None],other=0).to(tl.float32)
        sv=tl.load(sink_v+(cache_kh*SINK+tl.maximum(i-LOCAL_LIMIT-1,0)[:,None])*D+d[None,:],mask=sink[:,None],other=0).to(tl.float32)
        key=tl.where(sink[:,None],sk,key);val=tl.where(sink[:,None],sv,val)
    valid=old|current|sink;sc=tl.where(valid,tl.sum(key*qv[None,:],1)*SCALE,-float('inf'))
    mx=tl.max(sc,0);safe=tl.where(mx!=-float('inf'),mx,0.);w=tl.where(valid,tl.exp(sc-safe),0.);z=tl.sum(w,0)
    value=tl.sum(w[:,None]*val,0)/tl.maximum(z,1.e-30);off=head*NG+group
    tl.store(out+off*D+d,value);tl.store(lse+off,tl.where(z>0,safe+tl.log(z),-float('inf')))

@triton.jit
def _merge_append(co,cl,eo,el,lo,ll,new_k,new_v,local_k,local_v,local_lens,saved_len,out,
                  D:tl.constexpr,G:tl.constexpr,KV:tl.constexpr,LOCAL_CAP:tl.constexpr,
                  NC:tl.constexpr,NE:tl.constexpr,NL:tl.constexpr,BLOCK:tl.constexpr):
    head=tl.program_id(0);kh=head//G; x=tl.arange(0,BLOCK);d=tl.arange(0,D)
    ci=x<NC;ei=(x>=NC)&(x<NC+NE);li=(x>=NC+NE)&(x<NC+NE+NL)
    l=tl.load(cl+head*NC+x,mask=ci,other=-float('inf'))
    l=tl.where(ei,tl.load(el+head*NE+tl.maximum(x-NC,0),mask=ei,other=-float('inf')),l)
    l=tl.where(li,tl.load(ll+head*NL+tl.maximum(x-NC-NE,0),mask=li,other=-float('inf')),l)
    mx=tl.max(l,0);w=tl.exp(l-mx);z=tl.sum(w,0)
    v=tl.load(co+(head*NC+x[:,None])*D+d[None,:],mask=ci[:,None],other=0)
    ev=tl.load(eo+(head*NE+tl.maximum(x-NC,0)[:,None])*D+d[None,:],mask=ei[:,None],other=0)
    lv=tl.load(lo+(head*NL+tl.maximum(x-NC-NE,0)[:,None])*D+d[None,:],mask=li[:,None],other=0)
    value=tl.sum(w[:,None]*(v+ev+lv),axis=0)/z
    tl.store(out+head*D+d,value)
    if head%G==0:
        n=tl.load(saved_len);nk=tl.load(new_k+kh*D+d);nv=tl.load(new_v+kh*D+d)
        tl.store(local_k+(kh*LOCAL_CAP+n)*D+d,nk);tl.store(local_v+(kh*LOCAL_CAP+n)*D+d,nv)
    if head==0:tl.store(local_lens,tl.load(saved_len)+1)


def threshold_attention(pool,q,k,v,out,multiplier,*,tile=64,splits=64,rule='region',entry_bound=16,work_bound=None,cache_indices=None):
    if os.environ.get('LOD_AMD_THRESHOLD_CAPTURE_QUERY') == '1':
        pool._amd_threshold_probe = (q.clone(), k.clone(), v.clone(), pool.local_lens.clone(), pool.state_lens.clone())
    implementation=os.environ.get('LOD_AMD_THRESHOLD_IMPLEMENTATION','legacy')
    if cache_indices is not None and implementation not in ('legacy', 'materialized'):
        raise ValueError('batched AMD two-tier decode requires the legacy or materialized implementation')
    if implementation in ('frontier','flat','tiled','ranked','cached','balanced','active','shared','materialized','table'):
        if (pool.settings.levels != 2 or pool.dtype != torch.bfloat16
                or q.ndim != 4 or q.size(1) != 1 or (implementation != 'materialized' and q.size(0) != 1)
                or pool.state['page_cache'].get('leaf_quant_bits',0) != 0):
            raise ValueError('optimized AMD threshold workers require batch-one two-tier BF16 KV')
        if q.size(2) % pool.kv_heads or (q.size(-1),q.size(2)//pool.kv_heads) not in ((128,8),(256,6)):
            raise ValueError('optimized AMD threshold workers require the tested Qwen3.8 or K2 head geometry')
    if implementation == 'table':
        from benchmarks.amd_threshold_fused_table import threshold_attention_table
        return threshold_attention_table(pool,q,k,v,out,multiplier,tile=tile,splits=splits,rule=rule)
    if implementation == 'materialized':
        from benchmarks.amd_threshold_materialized import threshold_attention_materialized
        return threshold_attention_materialized(
            pool,q,k,v,out,multiplier,tile=tile,splits=splits,rule=rule,cache_indices=cache_indices,
            optimized=cache_indices is not None and os.environ.get('LOD_AMD_THRESHOLD_BATCH_FAST','1')=='1')
    if implementation == 'shared':
        from benchmarks.amd_threshold_shared import threshold_attention_shared
        return threshold_attention_shared(pool,q,k,v,out,multiplier,tile=tile,splits=splits,rule=rule)
    if implementation == 'active':
        from benchmarks.amd_threshold_active_blocks import threshold_attention_active
        return threshold_attention_active(pool,q,k,v,out,multiplier,tile=tile,splits=splits,rule=rule)
    if implementation == 'balanced':
        from benchmarks.amd_threshold_balanced import threshold_attention_balanced
        return threshold_attention_balanced(pool,q,k,v,out,multiplier,tile=tile,splits=splits,rule=rule)
    if implementation == 'cached':
        from benchmarks.amd_threshold_cached import threshold_attention_cached
        return threshold_attention_cached(pool,q,k,v,out,multiplier,tile=tile,splits=splits,rule=rule)
    if implementation == 'ranked':
        from benchmarks.amd_threshold_ranked import threshold_attention_ranked
        return threshold_attention_ranked(pool,q,k,v,out,multiplier,tile=tile,splits=splits,rule=rule)
    if implementation == 'tiled':
        from benchmarks.amd_threshold_tiled import threshold_attention_tiled
        return threshold_attention_tiled(pool,q,k,v,out,multiplier,tile=tile,splits=splits,rule=rule)
    if implementation == 'flat':
        from benchmarks.amd_threshold_flat_frontier import threshold_attention_flat
        return threshold_attention_flat(pool,q,k,v,out,multiplier,tile=tile,splits=splits,rule=rule)
    if implementation == 'frontier':
        from benchmarks.amd_threshold_frontier import threshold_attention_frontier
        return threshold_attention_frontier(pool,q,k,v,out,multiplier,tile=tile,splits=splits,rule=rule)
    if implementation != 'legacy':
        from benchmarks.amd_threshold_decode_fast import threshold_attention_fast
        return threshold_attention_fast(pool,q,k,v,out,multiplier,tile=tile,splits=splits,rule=rule,
                                        entry_bound=entry_bound,work_bound=work_bound,implementation=implementation)
    if rule not in ('region','entry'):raise ValueError('unknown threshold rule')
    if not isinstance(entry_bound,int) or entry_bound<1:raise ValueError('entry bound must be a positive integer')
    if work_bound is None:work_bound=entry_bound
    if not isinstance(work_bound,int) or work_bound<1:raise ValueError('work bound must be a positive integer')
    if q.ndim != 4 or q.shape[1] != 1 or pool.settings.levels != 2:
        raise ValueError('threshold decode requires single-token two-tier BF16 queries')
    batch = q.size(0)
    if (cache_indices is None and batch != pool.state_lens.numel()
            or cache_indices is not None and cache_indices.shape != (batch,)):
        raise ValueError('threshold decode requires one cache index per query row')
    page = pool.state['page_cache']
    S, D = pool.state_capacity, q.size(-1)
    H, KV = batch * q.size(2), batch * pool.kv_heads
    G = H // KV
    batch_heads = pool.kv_heads if batch > 1 or cache_indices is not None else 0
    if H != KV*G or (D,G) not in ((128,8),(256,6)):
        raise ValueError('experimental AMD threshold path requires Qwen3.8 or K2 geometry')
    if not page.get('paged_page_directory') or pool.dtype!=torch.bfloat16:raise ValueError('requires BF16 directory cache')
    localcap=int(pool.state['recent_k'].shape[2]);local_limit=int(pool.decode_local_limit)
    sink=int(pool.state['sink_k'].shape[2]);groups=triton.cdiv(S,tile);localgroups=triton.cdiv(local_limit+1+sink,tile)
    spec=(S,tile,splits,localgroups,batch)
    scratch = vars(pool).setdefault('_threshold_legacy_scratch_by_shape', {})
    if spec not in scratch:
        dev=q.device;f=lambda *shape:torch.empty(shape,device=dev,dtype=torch.float32)
        scratch[spec]=dict(spec=spec,native=f(H,S),route=f(H,S),threshold=f(H),
          allowed=torch.empty((H,S),device=dev,dtype=torch.int32),
          count=torch.empty(H,device=dev,dtype=torch.int32),selected=torch.empty((H,S),device=dev,dtype=torch.int32),
          tiles=torch.empty((H,groups*tile),device=dev,dtype=torch.int32),tile_counts=torch.empty((H,groups),device=dev,dtype=torch.int32),
          tile_offsets=torch.empty((H,groups*tile),device=dev,dtype=torch.int32),tile_pages=torch.empty((H,groups),device=dev,dtype=torch.int32),
          group_offsets=torch.empty((H,groups),device=dev,dtype=torch.int32),group_page_offsets=torch.empty((H,groups),device=dev,dtype=torch.int32),
          offsets=torch.empty((H,S),device=dev,dtype=torch.int32),total_pages=torch.empty(H,device=dev,dtype=torch.int32),
          descriptors=torch.empty((H,int(page['page_indices'].shape[2])),device=dev,dtype=torch.int32),
          saved=torch.empty((batch,),device=dev,dtype=torch.int32),co=f(H,groups,D),cl=f(H,groups),
          eo=f(H,splits,D),el=f(H,splits),lo=f(H,localgroups,D),ll=f(H,localgroups))
    b=pool._threshold_scratch=scratch[spec]; st=pool.state; scale=float(pool.engine.scaling)
    mapping = dict(BATCH_KV_HEADS=batch_heads, CacheIndices=cache_indices)
    _scores[(KV,groups)](q,st['state_k'],st['counts'],pool.state_lens,pool.local_lens,b['native'],b['route'],b['saved'],S,KV,G,D,scale,tile,num_warps=4,**mapping)
    _threshold[(H,)](b['route'],page['slot_lengths'],b['threshold'],b['allowed'],b['count'],S,G,float(multiplier),rule=='entry',entry_bound,work_bound,triton.next_power_of_2(S),num_warps=4,**mapping)
    _coarse_select_gqa[(KV,groups)](b['native'],b['route'],b['threshold'],b['allowed'],st['counts'],st['state_v'],page['slot_lengths'],b['tiles'],b['tile_offsets'],b['tile_counts'],b['tile_pages'],b['co'],b['cl'],S,G,D,tile,groups,1024,rule=='entry',num_warps=4,**mapping)
    _scan_groups[(H,)](b['tile_counts'],b['tile_pages'],b['group_offsets'],b['group_page_offsets'],b['count'],b['total_pages'],groups,num_warps=1)
    _pack_selected[(H,groups)](b['tiles'],b['tile_offsets'],b['tile_counts'],b['group_offsets'],b['group_page_offsets'],b['selected'],b['offsets'],S,groups,tile,num_warps=1)
    pagecap=int(page['page_indices'].shape[2]); roots=int(page['slot_pages'].shape[3])
    _expand_pages[(H,splits)](b['selected'],b['count'],page['slot_lengths'],page['slot_pages'],page['overflow_page_values'],b['offsets'],b['descriptors'],S,G,roots,pagecap,splits,num_warps=2,**mapping)
    _leaves[(H,splits)](q,b['descriptors'],b['total_pages'],page['page_indices'],page['leaf_k'],page['leaf_v'],b['eo'],b['el'],KV,G,D,pagecap,int(page['leaf_k'].shape[2]),scale,splits,num_warps=4,**mapping)
    if batch_heads:
        from benchmarks.amd_threshold_three_tier import local_gqa
        from benchmarks.amd_threshold_merge import merge_split_d
        local_gqa[(KV,localgroups)](q,st['recent_k'],st['recent_v'],st['sink_k'],st['sink_v'],k,v,
            b['saved'],b['lo'],b['ll'],G,D,pool.kv_heads,localcap,local_limit,sink,scale,tile,localgroups,
            CacheIndices=cache_indices,num_warps=4)
        bd = int(os.environ.get('LOD_AMD_THRESHOLD_MERGE_BLOCK', '32'))
        merge_split_d[(H,triton.cdiv(D,bd))](b['co'],b['cl'],b['eo'],b['el'],b['lo'],b['ll'],k,v,
            st['recent_k'],st['recent_v'],pool.local_lens,b['saved'],out,D,G,KV,localcap,groups,splits,
            localgroups,triton.next_power_of_2(groups+splits+localgroups),bd,num_warps=4,**mapping)
    else:
        _local[(H,localgroups)](q,st['recent_k'],st['recent_v'],st['sink_k'],st['sink_v'],k,v,b['saved'],b['lo'],b['ll'],KV,G,D,localcap,local_limit,sink,scale,tile,localgroups,num_warps=4)
        _merge_append[(H,)](b['co'],b['cl'],b['eo'],b['el'],b['lo'],b['ll'],k,v,st['recent_k'],st['recent_v'],pool.local_lens,b['saved'],out,D,G,KV,localcap,groups,splits,localgroups,triton.next_power_of_2(groups+splits+localgroups),num_warps=4)
    return out


def selected_region_mask(pool):
    """Read diagnostic routing outside measured attention calls."""
    b=pool._threshold_scratch
    if b.get('mask_only'):return b['allowed']!=0
    H,S=b['selected'].shape
    valid=torch.arange(S,device=b['selected'].device)[None,:]<b['count'][:,None]
    mask=torch.zeros((H,S),device=b['selected'].device,dtype=torch.int32)
    mask.scatter_add_(1,b['selected'].clamp(0,S-1).long(),valid.to(torch.int32))
    return mask!=0
