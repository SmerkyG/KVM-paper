"""Experimental GQA-shared exact-leaf worker for AMD threshold routing."""
import torch
import triton
import triton.language as tl
from benchmarks.amd_threshold_decode import _scores, _threshold, _scan_groups, _local, _merge_append


@triton.jit
def _coarse_union(native,route,threshold,counts,state_v,slot_len,tiles,tile_masks,tile_offsets,tile_counts,tile_pages,part_o,part_lse,
                  S:tl.constexpr,G:tl.constexpr,D:tl.constexpr,TILE:tl.constexpr,NG:tl.constexpr,MAX_LEAF:tl.constexpr):
    kh=tl.program_id(0);group=tl.program_id(1)
    lane=tl.arange(0,16);h=kh*G+lane;hm=lane<G
    i=group*TILE+tl.arange(0,TILE);d=tl.arange(0,D)
    score=tl.load(native+h[:,None]*S+i[None,:],mask=hm[:,None]&(i[None,:]<S),other=-float('inf'))
    r=tl.load(route+h[:,None]*S+i[None,:],mask=hm[:,None]&(i[None,:]<S),other=-float('inf'))
    cnt=tl.load(counts+kh*S+i,mask=i<S,other=0).to(tl.float32)
    valid=(i<S)&(cnt>0);cut=tl.load(threshold+h,mask=hm,other=float('inf'))
    opened=hm[:,None]&valid[None,:]&(cnt[None,:]<=MAX_LEAF)&(r>cut[:,None])
    bits=tl.sum(tl.where(opened,1<<lane[:,None],0),axis=0)
    union=bits!=0;prefix=tl.cumsum(union.to(tl.int32),axis=0)-1
    sizes=tl.load(slot_len+kh*S+i,mask=i<S,other=0)
    pages=tl.where(union,(sizes+15)//16,0);page_prefix=tl.cumsum(pages,axis=0)-pages
    tl.store(tile_counts+kh*NG+group,tl.sum(union.to(tl.int32),0))
    tl.store(tile_pages+kh*NG+group,tl.sum(pages,0))
    dst=kh*NG*TILE+group*TILE+prefix
    tl.store(tiles+dst,i,mask=union);tl.store(tile_masks+dst,bits,mask=union);tl.store(tile_offsets+dst,page_prefix,mask=union)
    closed=hm[:,None]&valid[None,:]&~opened
    scores=tl.where(closed,score,-float('inf'));mx=tl.max(scores,1);safe=tl.where(mx!=-float('inf'),mx,0.)
    w=tl.where(closed,tl.exp(scores-safe[:,None]),0.);z=tl.sum(w,1)
    val=tl.load(state_v+(kh*S+i[:,None])*D+d[None,:],mask=valid[:,None],other=0).to(tl.float32)
    mean=(val/tl.maximum(cnt[:,None],1.)).to(tl.bfloat16)
    result=tl.dot(w.to(tl.bfloat16),mean,out_dtype=tl.float32)/tl.maximum(z[:,None],1.e-30)
    row=h*NG+group
    tl.store(part_o+row[:,None]*D+d[None,:],result,mask=hm[:,None])
    tl.store(part_lse+row,tl.where(z>0,safe+tl.log(z),-float('inf')),mask=hm)


@triton.jit
def _pack(tiles,masks,tile_offsets,tile_counts,group_offsets,group_page_offsets,selected,selected_masks,offsets,
          S:tl.constexpr,NG:tl.constexpr,TILE:tl.constexpr):
    kh=tl.program_id(0);group=tl.program_id(1);i=tl.arange(0,TILE)
    before=tl.load(group_offsets+kh*NG+group);before_pages=tl.load(group_page_offsets+kh*NG+group)
    n=tl.load(tile_counts+kh*NG+group);src=kh*NG*TILE+group*TILE+i;dst=kh*S+before+i
    slot=tl.load(tiles+src,mask=i<n,other=0);mask=tl.load(masks+src,mask=i<n,other=0);off=tl.load(tile_offsets+src,mask=i<n,other=0)
    tl.store(selected+dst,slot,mask=i<n);tl.store(selected_masks+dst,mask,mask=i<n);tl.store(offsets+dst,before_pages+off,mask=i<n)


@triton.jit
def _expand(selected,masks,counts,slot_len,roots,dirs,offsets,desc,desc_masks,
            S:tl.constexpr,ROOTS:tl.constexpr,PAGECAP:tl.constexpr,WORKERS:tl.constexpr):
    kh=tl.program_id(0);worker=tl.program_id(1);n=tl.load(counts+kh);ordinal=tl.arange(0,64)
    for i in range(worker,n,WORKERS):
        slot=tl.load(selected+kh*S+i);bits=tl.load(masks+kh*S+i);cnt=tl.load(slot_len+kh*S+slot)
        np=(cnt+15)//16;valid=ordinal<np
        root=tl.load(roots+(kh*S+slot)*ROOTS+ordinal//64,mask=valid,other=-1)
        root_ok=valid&(root>=0)&(root<PAGECAP)
        page=tl.load(dirs+(kh*PAGECAP+tl.maximum(root,0))*64+ordinal%64,mask=root_ok,other=-1)
        page_ok=root_ok&(page>=0)&(page<PAGECAP)
        count=tl.minimum(16,cnt-ordinal*16);packed=tl.where(page_ok,page*32+count,-1)
        offset=tl.load(offsets+kh*S+i)
        tl.store(desc+kh*PAGECAP+offset+ordinal,packed,mask=valid)
        tl.store(desc_masks+kh*PAGECAP+offset+ordinal,bits,mask=valid)


@triton.jit
def _leaves_gqa(q,descriptors,masks,total,page_indices,leaf_k,leaf_v,part_o,part_lse,
                G:tl.constexpr,D:tl.constexpr,PAGECAP:tl.constexpr,LEAFCAP:tl.constexpr,SCALE:tl.constexpr,SPLITS:tl.constexpr):
    kh=tl.program_id(0);split=tl.program_id(1);lane=tl.arange(0,16);h=kh*G+lane;hm=lane<G
    d=tl.arange(0,D);t=tl.arange(0,16)
    qq=tl.load(q+h[:,None]*D+d[None,:],mask=hm[:,None],other=0).to(tl.bfloat16)
    n=tl.load(total+kh);mx=tl.full((16,),-float('inf'),tl.float32)
    den=tl.zeros((16,),tl.float32);acc=tl.zeros((16,D),tl.float32)
    for ii in range(split,n,SPLITS):
        desc=tl.load(descriptors+kh*PAGECAP+ii);bits=tl.load(masks+kh*PAGECAP+ii)
        page=desc//32;count=desc%32;page_ok=(page>=0)&(page<PAGECAP)
        valid=page_ok&(t<count);idx=tl.load(page_indices+(kh*PAGECAP+tl.maximum(page,0))*16+t,mask=valid,other=-1)
        valid=valid&(idx>=0)&(idx<LEAFCAP)
        key=tl.load(leaf_k+(kh*LEAFCAP+tl.maximum(idx,0))[:,None]*D+d[None,:],mask=valid[:,None],other=0).to(tl.bfloat16)
        val=tl.load(leaf_v+(kh*LEAFCAP+tl.maximum(idx,0))[:,None]*D+d[None,:],mask=valid[:,None],other=0).to(tl.bfloat16)
        member=hm&((bits&(1<<lane))!=0);valid2=member[:,None]&valid[None,:]
        sc=tl.where(valid2,tl.dot(qq,tl.trans(key),out_dtype=tl.float32)*SCALE,-float('inf'))
        pm=tl.max(sc,1);nm=tl.maximum(mx,pm);safe=tl.where(nm!=-float('inf'),nm,0.)
        factor=tl.where(mx!=-float('inf'),tl.exp(mx-safe),0.);w=tl.where(valid2,tl.exp(sc-safe[:,None]),0.)
        den=den*factor+tl.sum(w,1);acc=acc*factor[:,None]+tl.dot(w.to(tl.bfloat16),val,out_dtype=tl.float32);mx=nm
    off=h*SPLITS+split
    tl.store(part_o+off[:,None]*D+d[None,:],acc/tl.maximum(den[:,None],1.e-30),mask=hm[:,None])
    tl.store(part_lse+off,tl.where(den>0,mx+tl.log(den),-float('inf')),mask=hm)


def threshold_attention(pool,q,k,v,out,multiplier,*,tile=64,splits=64):
    if q.ndim!=4 or q.shape[:2]!=(1,1) or pool.settings.levels!=2 or pool.dtype!=torch.bfloat16:
        raise ValueError('AMD union experiment requires batch-one two-tier BF16')
    page=pool.state['page_cache'];S=pool.state_capacity;D=int(q.shape[-1]);H=int(q.shape[2]);KV=pool.kv_heads;G=H//KV
    if H!=KV*G or (D,G) not in ((128,8),(256,6)) or not page.get('paged_page_directory'):
        raise ValueError('requires Qwen3.8 or K2 directory cache')
    localcap=int(pool.state['recent_k'].shape[2]);local_limit=int(pool.decode_local_limit)
    sink=int(pool.state['sink_k'].shape[2]);groups=triton.cdiv(S,tile);localgroups=triton.cdiv(local_limit+1+sink,tile)
    spec=(S,tile,splits,localgroups)
    if not hasattr(pool,'_threshold_union_scratch') or pool._threshold_union_scratch['spec']!=spec:
        dev=q.device;f=lambda *shape:torch.empty(shape,device=dev,dtype=torch.float32);z=lambda *shape:torch.empty(shape,device=dev,dtype=torch.int32)
        pool._threshold_union_scratch=dict(spec=spec,native=f(H,S),route=f(H,S),threshold=f(H),head_count=z(H),
          count=z(KV),selected=z(KV,S),selected_masks=z(KV,S),tiles=z(KV,groups*tile),tile_masks=z(KV,groups*tile),
          tile_counts=z(KV,groups),tile_offsets=z(KV,groups*tile),tile_pages=z(KV,groups),offsets=z(KV,S),total_pages=z(KV),
          group_offsets=z(KV,groups),group_page_offsets=z(KV,groups),
          descriptors=z(KV,int(page['page_indices'].shape[2])),descriptor_masks=z(KV,int(page['page_indices'].shape[2])),
          saved=z(),co=f(H,groups,D),cl=f(H,groups),eo=f(H,splits,D),el=f(H,splits),lo=f(H,localgroups,D),ll=f(H,localgroups))
    b=pool._threshold_union_scratch;st=pool.state;scale=float(pool.engine.scaling)
    _scores[(KV,groups)](q,st['state_k'],st['counts'],pool.state_lens,pool.local_lens,b['native'],b['route'],b['saved'],S,KV,G,D,scale,tile,num_warps=4)
    _threshold[(H,)](b['route'],page['slot_lengths'],b['threshold'],b['head_count'],b['head_count'],S,G,float(multiplier),False,16,16,triton.next_power_of_2(S),num_warps=4)
    _coarse_union[(KV,groups)](b['native'],b['route'],b['threshold'],st['counts'],st['state_v'],page['slot_lengths'],b['tiles'],b['tile_masks'],b['tile_offsets'],b['tile_counts'],b['tile_pages'],b['co'],b['cl'],S,G,D,tile,groups,1024,num_warps=4)
    _scan_groups[(KV,)](b['tile_counts'],b['tile_pages'],b['group_offsets'],b['group_page_offsets'],b['count'],b['total_pages'],groups,num_warps=1)
    _pack[(KV,groups)](b['tiles'],b['tile_masks'],b['tile_offsets'],b['tile_counts'],b['group_offsets'],b['group_page_offsets'],b['selected'],b['selected_masks'],b['offsets'],S,groups,tile,num_warps=1)
    pagecap=int(page['page_indices'].shape[2]);roots=int(page['slot_pages'].shape[3])
    _expand[(KV,splits)](b['selected'],b['selected_masks'],b['count'],page['slot_lengths'],page['slot_pages'],page['overflow_page_values'],b['offsets'],b['descriptors'],b['descriptor_masks'],S,roots,pagecap,splits,num_warps=2)
    _leaves_gqa[(KV,splits)](q,b['descriptors'],b['descriptor_masks'],b['total_pages'],page['page_indices'],page['leaf_k'],page['leaf_v'],b['eo'],b['el'],G,D,pagecap,int(page['leaf_k'].shape[2]),scale,splits,num_warps=4)
    _local[(H,localgroups)](q,st['recent_k'],st['recent_v'],st['sink_k'],st['sink_v'],k,v,b['saved'],b['lo'],b['ll'],KV,G,D,localcap,local_limit,sink,scale,tile,localgroups,num_warps=4)
    _merge_append[(H,)](b['co'],b['cl'],b['eo'],b['el'],b['lo'],b['ll'],k,v,st['recent_k'],st['recent_v'],pool.local_lens,b['saved'],out,D,G,KV,localcap,groups,splits,localgroups,triton.next_power_of_2(groups+splits+localgroups),num_warps=4)
    return out
