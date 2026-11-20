"""Finish exact threshold selection from a producer's conservative prefix mask.

Since prefix partition mass never exceeds final partition mass, every final
qualifying region appears in this mask. There is no selection/work-count cap.
The final comparison uses the original FP32 score and final routing normalizer.
"""
import os
import torch
import triton
import triton.language as tl


@triton.jit
def _full(Scores,Inv,LogZ,Lengths,LogSize,Shift,Bits,Slots,Counts,Entries,HeadCounts,Offsets,
          T:tl.constexpr,S:tl.constexpr,CAP:tl.constexpr,G:tl.constexpr,WIDTH:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);head=row//T;kh=head//G;r=tl.arange(0,B)
    byte=tl.load(Bits+row*tl.cdiv(S,8)+r//8,mask=r<S,other=0).to(tl.int32)
    candidate=(r<S)&(((byte>>(r%8))&1)!=0)
    raw=tl.load(Scores+row*S+r,mask=candidate,other=0.)
    log=tl.load(LogSize+kh*S+r,mask=candidate,other=-float('inf'))
    inv=tl.load(Inv+row);cutoff=tl.load(LogZ+row)+tl.load(Shift+kh)
    opened=candidate&(raw*inv+log>cutoff)
    rank=tl.cumsum(opened.to(tl.int32),0)-1
    tl.store(Slots+row*WIDTH+rank,r,mask=opened)
    offset=tl.atomic_add(HeadCounts+head*S+r,1,mask=opened,sem='relaxed')
    tl.store(Offsets+row*WIDTH+rank,offset,mask=opened)
    size=tl.load(Lengths+kh*CAP+r,mask=opened,other=0).to(tl.int32)
    tl.store(Counts+row,tl.sum(opened.to(tl.int32),0));tl.store(Entries+row,tl.sum(size,0))
    packed=tl.sum(tl.reshape(opened.to(tl.int32),(B//8,8))*(1<<tl.arange(0,8)[None,:]),1).to(tl.uint8)
    b=tl.arange(0,B//8)
    tl.store(Bits+row*tl.cdiv(S,8)+b,packed,mask=b<tl.cdiv(S,8))


@triton.jit
def _words(Bits,IDs,Counts,ROWS:tl.constexpr,S:tl.constexpr,W:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);word=tl.arange(0,B)
    byte=word[:,None]*4+tl.arange(0,4)[None,:]
    data=tl.load(Bits+row*tl.cdiv(S,8)+byte,mask=byte<tl.cdiv(S,8),other=0).to(tl.uint32)
    mask=tl.sum(data<<(8*tl.arange(0,4)[None,:]),1).to(tl.uint32)
    live=(word<W)&(mask!=0);rank=tl.cumsum(live.to(tl.int32),0)-1
    tl.store(IDs+row*W+rank,word,mask=live)
    tl.store(Counts+row,tl.sum(live.to(tl.int32),0))


@triton.jit
def _select(Scores,Inv,LogZ,Lengths,NRegions,Bits,IDs,WordCounts,Slots,Counts,Entries,HeadCounts,Offsets,
            T:tl.constexpr,S:tl.constexpr,CAP:tl.constexpr,G:tl.constexpr,WIDTH:tl.constexpr,W:tl.constexpr,C:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);head=row//T;kh=head//G
    inv=tl.load(Inv+row);cutoff=tl.load(LogZ+row)+tl.log(C/tl.maximum(tl.load(NRegions+kh).to(tl.float32),1.))
    count=tl.load(WordCounts+row);group=tl.arange(0,4);lane=tl.arange(0,32);byte=tl.arange(0,4)
    selected=tl.full((),0,tl.int32);entries=tl.full((),0,tl.int32)
    for start in range(tl.cdiv(count,4)):
        pos=start*4+group
        word=tl.load(IDs+row*W+pos,mask=pos<count,other=0)
        b=word[:,None]*4+byte[None,:]
        data=tl.load(Bits+row*tl.cdiv(S,8)+b,mask=(pos[:,None]<count)&(b<tl.cdiv(S,8)),other=0).to(tl.uint32)
        mask=tl.sum(data<<(8*byte[None,:]),1).to(tl.uint32)
        region=word[:,None]*32+lane[None,:]
        candidate=(pos[:,None]<count)&(region<S)&(((mask[:,None]>>lane[None,:])&1)!=0)
        raw=tl.load(Scores+row*S+region,mask=candidate,other=0.)
        size=tl.load(Lengths+kh*CAP+region,mask=candidate,other=0).to(tl.int32)
        opened=candidate&(size>0)&(size<=1024)&(raw*inv+tl.log(tl.maximum(size.to(tl.float32),1.))>cutoff)
        hit=tl.reshape(opened.to(tl.int32),(128,));r=tl.reshape(region,(128,));sz=tl.reshape(size,(128,))
        rank=selected+tl.cumsum(hit,0)-1
        active=hit!=0
        tl.store(Slots+row*WIDTH+rank,r,mask=active)
        offset=tl.atomic_add(HeadCounts+head*S+r,1,mask=active,sem='relaxed')
        tl.store(Offsets+row*WIDTH+rank,offset,mask=active)
        selected+=tl.sum(hit,0);entries+=tl.sum(tl.where(active,sz,0),0)
        packed=tl.sum(tl.reshape(opened.to(tl.int32),(4,4,8))*(1<<tl.arange(0,8)[None,None,:]),2).to(tl.uint8)
        tl.store(Bits+row*tl.cdiv(S,8)+b,packed,mask=(pos[:,None]<count)&(b<tl.cdiv(S,8)))
    tl.store(Counts+row,selected);tl.store(Entries+row,entries)


def select(scores,inv,logz,lengths,nregions,slots,counts,entries,head_counts,offsets,bits,
           *,t,s,cap,g,multiplier,buffers):
    h=scores.size(0);w=triton.cdiv(s,32)
    def ws(name,shape,dtype=torch.int32):
        key='prefix_'+name;numel=1
        for v in shape:numel*=v
        old=buffers.get(key)
        if old is None or old.numel()<numel:
            old=torch.empty(numel,device=scores.device,dtype=dtype);buffers[key]=old
        return old[:numel].view(shape)
    if os.environ.get('LOD_H100_FAST_PREFIX_FULL')=='1':
        from benchmarks.h100_threshold_prefill_bias import _prepare
        kv=h//g;log=ws('log_size',(kv,s),torch.float32);shift=ws('shift',(kv,),torch.float32)
        _prepare[(kv,triton.cdiv(s,256))](lengths,nregions,log,shift,s,cap,float(multiplier),256,num_warps=4)
        _full[(h*t,)](scores,inv,logz,lengths,log,shift,bits,slots,counts,entries,head_counts,offsets,
            t,s,cap,g,slots.size(-1),triton.next_power_of_2(max(s,8)),num_warps=8)
        return
    ids=ws('word_ids',(h*t,w));wc=ws('word_counts',(h*t,))
    _words[(h*t,)](bits,ids,wc,h*t,s,w,triton.next_power_of_2(w),num_warps=4)
    _select[(h*t,)](scores,inv,logz,lengths,nregions,bits,ids,wc,slots,counts,entries,head_counts,offsets,
        t,s,cap,g,slots.size(-1),w,float(multiplier),num_warps=4)
