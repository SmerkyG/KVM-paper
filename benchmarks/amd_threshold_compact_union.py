"""GQA-shared MFMA exact attention with directly indexed compact page unions."""
import triton
import triton.language as tl


@triton.jit
def union_scan(Bits,Lengths,Selected,Masks,Ends,Counts,Pages,S:tl.constexpr,BLOCK:tl.constexpr):
    kh=tl.program_id(0);i=tl.arange(0,BLOCK)
    bits=tl.load(Bits+kh*S+i,mask=i<S,other=0)
    size=tl.load(Lengths+kh*S+i,mask=i<S,other=0)
    valid=(bits!=0)&(size>0);rank=tl.cumsum(valid.to(tl.int32),0)-1
    pages=tl.where(valid,(size+15)//16,0);ends=tl.cumsum(pages,0)
    tl.store(Selected+kh*S+rank,i,mask=valid)
    tl.store(Masks+kh*S+rank,bits,mask=valid)
    tl.store(Ends+kh*S+rank,ends,mask=valid)
    tl.store(Counts+kh,tl.sum(valid.to(tl.int32),0));tl.store(Pages+kh,tl.sum(pages,0))


@triton.jit
def union_leaves(Q,Selected,Masks,Ends,Counts,Pages,Lengths,Roots,Directory,Indices,LeafK,LeafV,
                 LocalK,LocalV,SinkK,SinkV,NewK,NewV,Saved,Out,LSE,
                 S:tl.constexpr,G:tl.constexpr,D:tl.constexpr,ROOTS:tl.constexpr,PAGECAP:tl.constexpr,LEAFCAP:tl.constexpr,
                 SPLITS:tl.constexpr,SCALE:tl.constexpr,N:tl.constexpr,LOCAL_CAP:tl.constexpr,LOCAL_LIMIT:tl.constexpr,
                 SINK:tl.constexpr,COMBINED:tl.constexpr):
    kh=tl.program_id(0);split=tl.program_id(1)
    lane=tl.arange(0,16);head=kh*G+lane;hm=lane<G;dim=tl.arange(0,D)
    p=tl.arange(0,N//16);t=tl.arange(0,16)
    query=tl.load(Q+head[:,None]*D+dim[None,:],mask=hm[:,None],other=0).to(tl.bfloat16)
    count=tl.load(Counts+kh);remote=tl.load(Pages+kh);total=remote
    if COMBINED:total+=tl.cdiv(LOCAL_LIMIT+1+SINK,16)
    mx=tl.full((16,),-float('inf'),tl.float32);den=tl.zeros((16,),tl.float32);acc=tl.zeros((16,D),tl.float32)
    for start in range(split*(N//16),total,SPLITS*(N//16)):
        pagepos=start+p;is_remote=pagepos<remote
        low=tl.zeros((N//16,),tl.int32);high=tl.full((N//16,),count,tl.int32)
        for _ in range(S.bit_length()):
            mid=(low+high)//2;search=is_remote&(low<high)
            end=tl.load(Ends+kh*S+mid,mask=search&(mid<count),other=2147483647)
            advance=search&(mid<count)&(end<=pagepos)
            low=tl.where(advance,mid+1,low);high=tl.where(advance,high,mid)
        rank=low
        slot=tl.load(Selected+kh*S+rank,mask=is_remote&(rank<count),other=0)
        bits=tl.load(Masks+kh*S+rank,mask=is_remote&(rank<count),other=0)
        before=tl.load(Ends+kh*S+rank-1,mask=is_remote&(rank>0)&(rank<count),other=0)
        ordinal=pagepos-before
        size=tl.load(Lengths+kh*S+slot,mask=is_remote&(rank<count),other=0)
        root=tl.load(Roots+(kh*S+slot)*ROOTS+ordinal//64,
                     mask=is_remote&(rank<count)&(ordinal//64<ROOTS),other=-1)
        root_ok=is_remote&(root>=0)&(root<PAGECAP)
        page=tl.load(Directory+(kh*PAGECAP+tl.maximum(root,0))*64+ordinal%64,mask=root_ok,other=-1)
        page_ok=root_ok&(page>=0)&(page<PAGECAP)
        idx=tl.load(Indices+(kh*PAGECAP+tl.maximum(page,0)[:,None])*16+t[None,:],
                    mask=page_ok[:,None]&(ordinal[:,None]*16+t[None,:]<size[:,None]),other=-1)
        idx=tl.reshape(idx,(N,));valid=(idx>=0)&(idx<LEAFCAP)
        key=tl.load(LeafK+(kh*LEAFCAP+tl.maximum(idx,0)[:,None])*D+dim[None,:],mask=valid[:,None],other=0).to(tl.bfloat16)
        value=tl.load(LeafV+(kh*LEAFCAP+tl.maximum(idx,0)[:,None])*D+dim[None,:],mask=valid[:,None],other=0).to(tl.bfloat16)
        entry_bits=tl.reshape(tl.broadcast_to(bits[:,None],(N//16,16)),(N,))
        if COMBINED:
            if start+N//16>remote:
                local=tl.load(Saved)
                loc=tl.reshape((pagepos-remote)[:,None]*16+t[None,:],(N,))
                old=(loc>=0)&(loc<local)&(loc<LOCAL_LIMIT);current=loc==local
                sink=(loc>=LOCAL_LIMIT+1)&(loc<LOCAL_LIMIT+1+SINK)
                lk=tl.load(LocalK+(kh*LOCAL_CAP+loc[:,None])*D+dim[None,:],mask=old[:,None],other=0).to(tl.bfloat16)
                lv=tl.load(LocalV+(kh*LOCAL_CAP+loc[:,None])*D+dim[None,:],mask=old[:,None],other=0).to(tl.bfloat16)
                nk=tl.load(NewK+kh*D+dim).to(tl.bfloat16);nv=tl.load(NewV+kh*D+dim).to(tl.bfloat16)
                key=tl.where(old[:,None],lk,key);value=tl.where(old[:,None],lv,value)
                key=tl.where(current[:,None],nk[None,:],key);value=tl.where(current[:,None],nv[None,:],value)
                if SINK:
                    sk=tl.load(SinkK+(kh*SINK+(loc-LOCAL_LIMIT-1)[:,None])*D+dim[None,:],mask=sink[:,None],other=0).to(tl.bfloat16)
                    sv=tl.load(SinkV+(kh*SINK+(loc-LOCAL_LIMIT-1)[:,None])*D+dim[None,:],mask=sink[:,None],other=0).to(tl.bfloat16)
                    key=tl.where(sink[:,None],sk,key);value=tl.where(sink[:,None],sv,value)
                all_heads=old|current|sink
                entry_bits=tl.where(all_heads,(1<<G)-1,entry_bits);valid|=all_heads
        active=hm[:,None]&valid[None,:]&((entry_bits[None,:]&(1<<lane[:,None]))!=0)
        score=tl.where(active,tl.dot(query,tl.trans(key),out_dtype=tl.float32)*SCALE,-float('inf'))
        next_m=tl.maximum(mx,tl.max(score,1));safe=tl.where(next_m!=-float('inf'),next_m,0.)
        adjust=tl.where(mx!=-float('inf'),tl.exp(mx-safe),0.)
        w=tl.where(active,tl.exp(score-safe[:,None]),0.)
        den=den*adjust+tl.sum(w,1);acc=acc*adjust[:,None]+tl.dot(w.to(tl.bfloat16),value,out_dtype=tl.float32);mx=next_m
    row=head*SPLITS+split
    tl.store(Out+row[:,None]*D+dim[None,:],acc/tl.maximum(den[:,None],1.e-30),mask=hm[:,None])
    tl.store(LSE+row,tl.where(den>0,mx+tl.log(den),-float('inf')),mask=hm)
