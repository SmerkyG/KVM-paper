"""Opt-in uncapped threshold decode with fused scans and compact exact work.

Reuse the existing native/normalized scores and coarse attention arithmetic.
Fuse threshold, prefix packing, entry offsets, and page offsets in one kernel.
Compact exact workers traverse actual entries, avoiding page-padding work.
The combined variant includes local, current, and sink entries in those workers.
No host-side selected counts or fixed selected-region cap are used.
"""
import math
import os
import torch
import triton
import triton.language as tl
from benchmarks.h100_threshold_decode import _scores, _coarse_select_gqa, _expand_pages, _leaves, _local, _merge_append


def merge_threshold(b, k, v, st, local_lens, out, d, g, kv, localcap, nc, ne, nl):
    """Optionally share the AMD register-bounded merge on CUDA as well."""
    block = int(os.environ.get('LOD_H100_THRESHOLD_MERGE_BLOCK', '0'))
    args = (b['co'], b['cl'], b['eo'], b['el'], b['lo'], b['ll'], k, v,
            st['recent_k'], st['recent_v'], local_lens, b['saved'], out,
            d, g, kv, localcap, nc, ne, nl, triton.next_power_of_2(nc + ne + nl))
    if block:
        if block not in (16, 32, 64, 128):
            raise ValueError('merge dimension block must be 0, 16, 32, 64 or 128')
        from benchmarks.amd_threshold_merge import merge_split_d
        merge_split_d[(kv * g, triton.cdiv(d, block))](*args, block, num_warps=4)
    else:
        _merge_append[(kv * g,)](*args, num_warps=4)


@triton.jit
def _threshold_pack(Route, Lengths, StateCounts, Threshold, Allowed, Selected, Counts,
                    PageOffsets, TotalPages, EntryEnds, TotalEntries,
                    S:tl.constexpr,G:tl.constexpr,C:tl.constexpr,ENTRY:tl.constexpr,
                    BOUND:tl.constexpr,WORK:tl.constexpr,BLOCK:tl.constexpr,PAGES:tl.constexpr):
    head=tl.program_id(0);idx=tl.arange(0,BLOCK)
    score=tl.load(Route+head*S+idx,mask=idx<S,other=-float('inf'))
    valid=score!=-float('inf');n=tl.sum(valid.to(tl.float32),0)
    mx=tl.max(score,0);safe=tl.where(n>0,mx,0.)
    z=tl.sum(tl.where(valid,tl.exp(score-safe),0.),0)
    size=tl.load(Lengths+(head//G)*S+idx,mask=idx<S,other=0).to(tl.int64)
    if ENTRY:
        remote=tl.sum(tl.where(valid,size,0),0)
        cutoff=tl.where(remote>0,safe+tl.log(z)-tl.log(BOUND*1.)-0.5*tl.log(remote.to(tl.float32))+1.e-6,float('inf'))
        candidate=valid&(size>0)&(score-tl.log(tl.maximum(size.to(tl.float32),1.))>cutoff)
        prefix=tl.cumsum(tl.where(candidate,size,0),0)
        opened=candidate&(prefix*prefix<WORK*WORK*remote)
    else:
        cutoff=tl.where(n>0,safe+tl.log(z)+tl.log(C/n),float('inf'))
        state_count=tl.load(StateCounts+(head//G)*S+idx,mask=idx<S,other=0).to(tl.float32)
        opened=valid&(state_count>0)&(state_count<=1024)&(score>cutoff)
    tl.store(Threshold+head,cutoff)
    tl.store(Allowed+head*S+idx,opened.to(tl.int32),mask=idx<S)
    rank=tl.cumsum(opened.to(tl.int32),0)-1
    tl.store(Selected+head*S+rank,idx,mask=opened)
    count=tl.sum(opened.to(tl.int32),0);tl.store(Counts+head,count)
    entries=tl.where(opened,size.to(tl.int32),0)
    ends=tl.cumsum(entries,0)
    tl.store(EntryEnds+head*S+rank,ends,mask=opened)
    tl.store(TotalEntries+head,tl.sum(entries,0))
    pages=tl.where(opened,(size.to(tl.int32)+15)//16,0)
    tl.store(TotalPages+head,tl.sum(pages,0))
    if PAGES:
        offsets=tl.cumsum(pages,0)-pages
        tl.store(PageOffsets+head*S+rank,offsets,mask=opened)


@triton.jit
def _compact_leaves(Q,Selected,Counts,EntryEnds,TotalEntries,Roots,Directory,Indices,LeafK,LeafV,
                    LocalK,LocalV,SinkK,SinkV,NewK,NewV,Saved,Out,LSE,
                    S:tl.constexpr,G:tl.constexpr,D:tl.constexpr,ROOTS:tl.constexpr,PAGECAP:tl.constexpr,
                    LEAFCAP:tl.constexpr,SPLITS:tl.constexpr,SCALE:tl.constexpr,N:tl.constexpr,
                    LOCAL_CAP:tl.constexpr,LOCAL_LIMIT:tl.constexpr,SINK:tl.constexpr,COMBINED:tl.constexpr):
    head=tl.program_id(0);split=tl.program_id(1);kh=head//G
    dim=tl.arange(0,D);lane=tl.arange(0,N)
    q=tl.load(Q+head*D+dim).to(tl.float32)
    count=tl.load(Counts+head);remote=tl.load(TotalEntries+head)
    total=remote
    if COMBINED:
        local=tl.minimum(tl.load(Saved),LOCAL_LIMIT)
        total+=local+1+SINK
    mx=tl.full((),-float('inf'),tl.float32);den=tl.full((),0.,tl.float32);acc=tl.zeros((D,),tl.float32)
    for start in range(split*N,total,SPLITS*N):
        entry=start+lane;is_remote=entry<remote
        low=tl.zeros((N,),tl.int32);high=tl.full((N,),count,tl.int32)
        # Binary-search compact region entry ends. Mask nonremote lanes so
        # local/sink entries never read uninitialized route-list slots.
        for _ in range(S.bit_length()):
            mid=(low+high)//2
            end=tl.load(EntryEnds+head*S+mid,mask=is_remote&(mid<count),other=2147483647)
            advance=is_remote&(mid<count)&(end<=entry)
            low=tl.where(advance,mid+1,low);high=tl.where(advance,high,mid)
        rank=low
        slot=tl.load(Selected+head*S+rank,mask=is_remote&(rank<count),other=0)
        before=tl.load(EntryEnds+head*S+rank-1,mask=is_remote&(rank>0)&(rank<count),other=0)
        ordinal=entry-before;page_ord=ordinal//16
        root=tl.load(Roots+(kh*S+slot)*ROOTS+page_ord//64,
                     mask=is_remote&(rank<count)&(page_ord//64<ROOTS),other=-1)
        root_ok=is_remote&(rank<count)&(root>=0)&(root<PAGECAP)
        page=tl.load(Directory+(kh*PAGECAP+tl.maximum(root,0))*64+page_ord%64,mask=root_ok,other=-1)
        page_ok=root_ok&(page>=0)&(page<PAGECAP)
        idx=tl.load(Indices+(kh*PAGECAP+tl.maximum(page,0))*16+ordinal%16,mask=page_ok,other=-1)
        valid=page_ok&(idx>=0)&(idx<LEAFCAP)
        key=tl.load(LeafK+(kh*LEAFCAP+tl.maximum(idx,0)[:,None])*D+dim[None,:],mask=valid[:,None],other=0).to(tl.float32)
        val=tl.load(LeafV+(kh*LEAFCAP+tl.maximum(idx,0)[:,None])*D+dim[None,:],mask=valid[:,None],other=0).to(tl.float32)
        if COMBINED:
            local_idx=entry-remote
            old=(local_idx>=0)&(local_idx<local)
            current=local_idx==local
            sink_idx=local_idx-local-1
            sink=(sink_idx>=0)&(sink_idx<SINK)
            lk=tl.load(LocalK+(kh*LOCAL_CAP+local_idx[:,None])*D+dim[None,:],mask=old[:,None],other=0).to(tl.float32)
            lv=tl.load(LocalV+(kh*LOCAL_CAP+local_idx[:,None])*D+dim[None,:],mask=old[:,None],other=0).to(tl.float32)
            nk=tl.load(NewK+kh*D+dim).to(tl.float32);nv=tl.load(NewV+kh*D+dim).to(tl.float32)
            key=tl.where(old[:,None],lk,key);val=tl.where(old[:,None],lv,val)
            key=tl.where(current[:,None],nk[None,:],key);val=tl.where(current[:,None],nv[None,:],val)
            if SINK:
                sk=tl.load(SinkK+(kh*SINK+sink_idx[:,None])*D+dim[None,:],mask=sink[:,None],other=0).to(tl.float32)
                sv=tl.load(SinkV+(kh*SINK+sink_idx[:,None])*D+dim[None,:],mask=sink[:,None],other=0).to(tl.float32)
                key=tl.where(sink[:,None],sk,key);val=tl.where(sink[:,None],sv,val)
            valid|=old|current|sink
        sc=tl.where(valid,tl.sum(key*q[None,:],1)*SCALE,-float('inf'))
        nm=tl.maximum(mx,tl.max(sc,0));safe=tl.where(nm!=-float('inf'),nm,0.)
        adjust=tl.where(mx!=-float('inf'),tl.exp(mx-safe),0.)
        w=tl.where(valid,tl.exp(sc-safe),0.)
        acc=acc*adjust+tl.sum(w[:,None]*val,0);den=den*adjust+tl.sum(w,0);mx=nm
    row=head*SPLITS+split
    tl.store(Out+row*D+dim,acc/tl.maximum(den,1.e-30))
    tl.store(LSE+row,tl.where(den>0,mx+tl.log(den),-float('inf')))


def threshold_attention_fast(pool,q,k,v,out,multiplier,*,tile=64,splits=64,rule='region',
                             entry_bound=16,work_bound=None,implementation='combined'):
    if implementation not in ('packed','indexed','combined'):
        raise ValueError('unknown optimized threshold decode implementation')
    if work_bound is None:work_bound=entry_bound
    if rule not in ('region','entry') or multiplier<=0 or not isinstance(entry_bound,int) or entry_bound<1 or not isinstance(work_bound,int) or work_bound<1:
        raise ValueError('invalid threshold configuration')
    if q.shape!=(1,1,32,128) or pool.kv_heads!=8 or pool.settings.levels!=2 or pool.dtype!=torch.bfloat16:
        raise ValueError('optimized threshold decode requires batch-one BF16 Llama GQA two-tier')
    page=pool.state['page_cache']
    if not page.get('paged_page_directory'):raise ValueError('requires directory cache')
    S=pool.state_capacity;H=32;KV=8;G=4;D=128
    localcap=int(pool.state['recent_k'].size(2));limit=int(pool.decode_local_limit);sink=int(pool.state['sink_k'].size(2))
    groups=triton.cdiv(S,tile);combined=implementation=='combined'
    localgroups=0 if combined else triton.cdiv(limit+1+sink,tile)
    spec=('fast',S,tile,splits,localgroups)
    b=getattr(pool,'_threshold_fast_scratch',None)
    if b is None or b['spec']!=spec:
        def f(*shape):return torch.empty(shape,device=q.device,dtype=torch.float32)
        def i(*shape):return torch.empty(shape,device=q.device,dtype=torch.int32)
        b=dict(spec=spec,native=f(H,S),route=f(H,S),threshold=f(H),allowed=i(H,S),
               selected=i(H,S),count=i(H),offsets=i(H,S),entry_ends=i(H,S),total_entries=i(H),
               total_pages=i(H),saved=i(1),co=f(H,groups,D),cl=f(H,groups),eo=f(H,splits,D),el=f(H,splits),
               lo=f(H,max(1,localgroups),D),ll=f(H,max(1,localgroups)))
        if implementation=='packed':b['descriptors']=i(H,int(page['page_indices'].size(2)))
        pool._threshold_fast_scratch=b
    elif implementation=='packed' and 'descriptors' not in b:
        b['descriptors']=torch.empty((H,int(page['page_indices'].size(2))),device=q.device,dtype=torch.int32)
    pool._threshold_scratch=b
    st=pool.state;scale=float(pool.engine.scaling)
    profiling=os.environ.get('LOD_H100_DECODE_PROFILE')=='1'
    if profiling:
        first=torch.cuda.Event(enable_timing=True);first.record()
    def mark(name):
        nonlocal first
        if profiling:
            last=torch.cuda.Event(enable_timing=True);last.record()
            pool._threshold_decode_events.setdefault(name,[]).append((first,last));first=last
    if profiling and not hasattr(pool,'_threshold_decode_events'):pool._threshold_decode_events={}
    _scores[(KV,groups)](q,st['state_k'],st['counts'],pool.state_lens,pool.local_lens,b['native'],b['route'],b['saved'],S,KV,G,D,scale,tile,num_warps=4)
    mark('scores')
    _threshold_pack[(H,)](b['route'],page['slot_lengths'],st['counts'],b['threshold'],b['allowed'],b['selected'],b['count'],
        b['offsets'],b['total_pages'],b['entry_ends'],b['total_entries'],S,G,float(multiplier),rule=='entry',
        entry_bound,work_bound,triton.next_power_of_2(S),implementation=='packed',num_warps=4)
    mark('select_pack')
    _coarse_select_gqa[(KV,groups)](b['native'],b['route'],b['threshold'],b['allowed'],st['counts'],st['state_v'],
        page['slot_lengths'],b['selected'],b['offsets'],b['count'],b['total_pages'],b['co'],b['cl'],
        S,G,D,tile,groups,1024,True,False,num_warps=4)
    mark('coarse')
    pagecap=int(page['page_indices'].size(2));roots=int(page['slot_pages'].size(3))
    if implementation=='packed':
        _expand_pages[(H,splits)](b['selected'],b['count'],page['slot_lengths'],page['slot_pages'],page['overflow_page_values'],
            b['offsets'],b['descriptors'],S,G,roots,pagecap,splits,num_warps=2)
        _leaves[(H,splits)](q,b['descriptors'],b['total_pages'],page['page_indices'],page['leaf_k'],page['leaf_v'],b['eo'],b['el'],
            KV,G,D,pagecap,int(page['leaf_k'].size(2)),scale,splits,num_warps=4)
    else:
        block=int(os.environ.get('LOD_H100_THRESHOLD_ENTRY_BLOCK','32'))
        _compact_leaves[(H,splits)](q,b['selected'],b['count'],b['entry_ends'],b['total_entries'],page['slot_pages'],
            page['overflow_page_values'],page['page_indices'],page['leaf_k'],page['leaf_v'],st['recent_k'],st['recent_v'],
            st['sink_k'],st['sink_v'],k,v,b['saved'],b['eo'],b['el'],S,G,D,roots,pagecap,int(page['leaf_k'].size(2)),
            splits,scale,block,localcap,limit,sink,combined,num_warps=4)
    mark('exact')
    if not combined:
        _local[(H,localgroups)](q,st['recent_k'],st['recent_v'],st['sink_k'],st['sink_v'],k,v,b['saved'],b['lo'],b['ll'],
            KV,G,D,localcap,limit,sink,scale,tile,localgroups,num_warps=4)
    mark('local')
    merge_threshold(b,k,v,st,pool.local_lens,out,D,G,KV,localcap,groups,splits,localgroups)
    mark('merge_append')
    return out
