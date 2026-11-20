"""Tile threshold scores once, then pack through linear device prefix scans.

The default uses no atomics; the optional hybrid reserves expert ranges per
query tile. Neither path sorts, reads counts on the host, or caps selections.
Prefixes give each selected query/region pair a unique output position.
"""
import triton
import triton.language as tl
import torch


@triton.jit
def _tiles(Scores,Inv,LogZ,Lengths,NRegions,Bits,QCounts,QEntries,Expert,HeadCounts,
           T:tl.constexpr,S:tl.constexpr,CAP:tl.constexpr,G:tl.constexpr,
           RG:tl.constexpr,QG:tl.constexpr,C:tl.constexpr,M:tl.constexpr,N:tl.constexpr,ATOMIC:tl.constexpr):
    head=tl.program_id(0);qg=tl.program_id(1);rg=tl.program_id(2)
    q=qg*M+tl.arange(0,M);r=rg*N+tl.arange(0,N)
    row=head.to(tl.int64)*T+q
    inv=tl.load(Inv+row,mask=q<T,other=0.)
    z=tl.load(LogZ+row,mask=q<T,other=0.)
    nonempty=tl.load(NRegions+head//G).to(tl.float32)
    cutoff=z+tl.log(C/tl.maximum(nonempty,1.))
    size=tl.load(Lengths+head//G*CAP+r,mask=r<S,other=0).to(tl.int32)
    dot=tl.load(Scores+row[:,None]*S+r[None,:],mask=(q[:,None]<T)&(r[None,:]<S),other=0.)
    opened=(q[:,None]<T)&(r[None,:]<S)&(size[None,:]>0)&(size[None,:]<=1024)&(
        dot*inv[:,None]+tl.log(tl.maximum(size[None,:].to(tl.float32),1.))>cutoff[:,None])
    hit=opened.to(tl.int32)
    tl.store(QCounts+row*RG+rg,tl.sum(hit,1),mask=q<T)
    tl.store(QEntries+row*RG+rg,tl.sum(tl.where(opened,size[None,:],0),1),mask=q<T)
    per_region=tl.sum(hit,0)
    if ATOMIC:
        base=tl.atomic_add(HeadCounts+head*S+r,per_region,mask=(r<S)&(per_region>0),sem='relaxed')
        tl.store(Expert+(head*QG+qg)*S+r,base,mask=r<S)
    else:
        tl.store(Expert+(head*QG+qg)*S+r,per_region,mask=r<S)
    bit=tl.arange(0,8)
    packed=tl.sum(tl.reshape(hit,(M,N//8,8))*(1<<bit[None,None,:]),2).to(tl.uint8)
    byte=rg*(N//8)+tl.arange(0,N//8)
    tl.store(Bits+row[:,None]*tl.cdiv(S,8)+byte[None,:],packed,
             mask=(q[:,None]<T)&(byte[None,:]<tl.cdiv(S,8)))


@triton.jit
def _query_prefix(QCounts,QEntries,Counts,Entries,ROWS:tl.constexpr,RG:tl.constexpr,M:tl.constexpr,R:tl.constexpr):
    q=tl.program_id(0).to(tl.int64)*M+tl.arange(0,M)
    r=tl.arange(0,R)
    count=tl.load(QCounts+q[:,None]*RG+r[None,:],mask=(q[:,None]<ROWS)&(r[None,:]<RG),other=0)
    entries=tl.load(QEntries+q[:,None]*RG+r[None,:],mask=(q[:,None]<ROWS)&(r[None,:]<RG),other=0)
    tl.store(QCounts+q[:,None]*RG+r[None,:],tl.cumsum(count,1)-count,
             mask=(q[:,None]<ROWS)&(r[None,:]<RG))
    tl.store(Counts+q,tl.sum(count,1),mask=q<ROWS)
    tl.store(Entries+q,tl.sum(entries,1),mask=q<ROWS)


@triton.jit
def _expert_prefix(Expert,HeadCounts,S:tl.constexpr,QG:tl.constexpr,Q:tl.constexpr,N:tl.constexpr):
    head=tl.program_id(0);r=tl.program_id(1)*N+tl.arange(0,N)
    q=tl.arange(0,Q)
    pos=(head*QG+q[:,None])*S+r[None,:]
    valid=(q[:,None]<QG)&(r[None,:]<S)
    count=tl.load(Expert+pos,mask=valid,other=0)
    tl.store(Expert+pos,tl.cumsum(count,0)-count,mask=valid)
    tl.store(HeadCounts+head*S+r,tl.sum(count,0),mask=r<S)


@triton.jit
def _pack(Bits,QPrefix,Expert,Counts,Slots,Offsets,
          T:tl.constexpr,S:tl.constexpr,WIDTH:tl.constexpr,RG:tl.constexpr,QG:tl.constexpr,
          M:tl.constexpr,N:tl.constexpr,SKIP_EMPTY:tl.constexpr):
    head=tl.program_id(0);qg=tl.program_id(1);rg=tl.program_id(2)
    q=qg*M+tl.arange(0,M);r=rg*N+tl.arange(0,N)
    row=head.to(tl.int64)*T+q
    qp=tl.load(QPrefix+row*RG+rg,mask=q<T,other=0)
    nonempty=tl.full((),True,tl.int1)
    if SKIP_EMPTY:
        nxt=tl.load(QPrefix+row*RG+rg+1,mask=(q<T)&(rg+1<RG),other=0)
        total=tl.load(Counts+row,mask=q<T,other=0)
        after=tl.where(rg+1<RG,nxt,total)
        nonempty=tl.sum(after-qp,0)>0
    if nonempty:
        bits=tl.load(Bits+row[:,None]*tl.cdiv(S,8)+r[None,:]//8,
                     mask=(q[:,None]<T)&(r[None,:]<S),other=0).to(tl.int32)
        opened=(q[:,None]<T)&(r[None,:]<S)&(((bits>>(r[None,:]%8))&1)!=0)
        ep=tl.load(Expert+(head*QG+qg)*S+r,mask=r<S,other=0)
        rank=qp[:,None]+tl.cumsum(opened.to(tl.int32),1)-1
        offset=ep[None,:]+tl.cumsum(opened.to(tl.int32),0)-1
        tl.store(Slots+row[:,None]*WIDTH+rank,r[None,:],mask=opened)
        tl.store(Offsets+row[:,None]*WIDTH+rank,offset,mask=opened)


def select(scores,inv,logz,lengths,nregions,slots,counts,entries,head_counts,offsets,bits,
           *,t,s,cap,g,multiplier,buffers,m=32,n=128,warps=4,q=None,atomic_experts=False,skip_empty=True):
    if m not in (8,16,32,64) or n not in (64,128,256):raise ValueError('unsupported scan selection tile')
    h=scores.size(0);rg=triton.cdiv(s,n);qg=triton.cdiv(t,m)
    def ws(name,shape):
        key='scan_'+name;numel=1
        for size in shape:numel*=size
        old=buffers.get(key)
        if old is None or old.numel()<numel:
            old=torch.empty(numel,dtype=torch.int32,device=scores.device);buffers[key]=old
        return old[:numel].view(shape)
    qc=ws('query_counts',(h*t,rg));qe=ws('query_entries',(h*t,rg));expert=ws('expert_counts',(h,qg,s))
    grid=(h,qg,rg)
    if atomic_experts:head_counts.zero_()
    _tiles[grid](scores,inv,logz,lengths,nregions,bits,qc,qe,expert,head_counts,t,s,cap,g,rg,qg,float(multiplier),m,n,atomic_experts,num_warps=warps)
    _query_prefix[(triton.cdiv(h*t,16),)](qc,qe,counts,entries,h*t,rg,16,triton.next_power_of_2(rg),num_warps=4)
    if not atomic_experts:
        _expert_prefix[(h,triton.cdiv(s,32))](expert,head_counts,s,qg,triton.next_power_of_2(qg),32,num_warps=4)
    _pack[grid](bits,qc,expert,counts,slots,offsets,t,s,slots.size(-1),rg,qg,m,n,skip_empty,num_warps=warps)
