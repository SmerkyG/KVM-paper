"""Device-only variable-route dispatch into the existing paged H100 leaf kernel.

The temporary output capacity is a memory budget, not a routing cap. Whole
queries beyond it use exact per-query attention instead of dropping regions.
"""
from __future__ import annotations
import math
import os
import torch
import triton
import triton.language as tl
from lod_attention.kernels.paged_prefill import _paged_leaf_attention_kernel
from lod_attention.kernels._paged_common import _lookup_page_id
from benchmarks.h100_threshold_prefill_fused import _direct_leaf_attention


@triton.jit
def _pack(Slots, Counts, Prefix, FlatSlots, QueryRows, LocalOffsets, HeadCounts, Kept, Overflow,
          QLEN: tl.constexpr, S: tl.constexpr, WIDTH: tl.constexpr, CAPACITY: tl.constexpr, BLOCK: tl.constexpr):
    row=tl.program_id(0)
    head=row//QLEN
    count=tl.load(Counts+row)
    base=tl.load(Prefix+row).to(tl.int64)
    fits=base+count<=CAPACITY
    tl.store(Overflow+row,tl.where(fits,0,count))
    if fits:
        tl.atomic_max(Kept,base+count,sem='relaxed')
        t=tl.arange(0,BLOCK)
        for start in range(tl.cdiv(count,BLOCK)):
            rank=start*BLOCK+t
            valid=rank<count
            slot=tl.load(Slots+row*WIDTH+rank,mask=valid,other=0).to(tl.int32)
            offset=tl.atomic_add(HeadCounts+head*S+slot,1,mask=valid,sem='relaxed')
            tl.store(FlatSlots+base+rank,slot,mask=valid)
            tl.store(QueryRows+base+rank,row,mask=valid)
            tl.store(LocalOffsets+base+rank,offset,mask=valid)


@triton.jit
def _scatter(FlatSlots, QueryRows, LocalOffsets, HeadOffsets, Starts, BlockStarts, Order, BlockExpert, Kept,
             QLEN: tl.constexpr, S: tl.constexpr, G: tl.constexpr, M: tl.constexpr, CAPACITY: tl.constexpr, BLOCK: tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    n=tl.load(Kept)
    valid=(i<n)&(i<CAPACITY)
    row=tl.load(QueryRows+i,mask=valid,other=0).to(tl.int32)
    head=row//QLEN
    slot=tl.load(FlatSlots+i,mask=valid,other=0).to(tl.int32)
    expert=(head//G)*S+slot
    offset=tl.load(LocalOffsets+i,mask=valid,other=0)+tl.load(HeadOffsets+head*S+slot,mask=valid,other=0)
    start=tl.load(Starts+expert,mask=valid,other=0)
    tl.store(Order+start+offset,i,mask=valid)
    begins=valid&(offset%M==0)
    b=tl.load(BlockStarts+expert,mask=begins,other=0)+offset//M
    tl.store(BlockExpert+b,expert,mask=begins)


@triton.jit
def _reduce(RouteOut, RouteLSE, Counts, Prefix, FallbackOut, FallbackLSE, Out, LSE,
             D: tl.constexpr, CAPACITY: tl.constexpr, BLOCK: tl.constexpr):
    row=tl.program_id(0)
    d=tl.arange(0,D)
    t=tl.arange(0,BLOCK)
    count=tl.load(Counts+row)
    base=tl.load(Prefix+row).to(tl.int64)
    if base+count<=CAPACITY:
        mx=tl.full((),-float('inf'),tl.float32)
        den=tl.full((),0.,tl.float32)
        num=tl.zeros((D,),tl.float32)
        for start in range(tl.cdiv(count,BLOCK)):
            rank=start*BLOCK+t
            score=tl.load(RouteLSE+base+rank,mask=rank<count,other=-float('inf'))
            val=tl.load(RouteOut+(base+rank[:,None])*D+d[None,:],mask=rank[:,None]<count,other=0).to(tl.float32)
            nxt=tl.maximum(mx,tl.max(score,0))
            safe=tl.where(nxt != -float('inf'),nxt,0.)
            adjust=tl.where(mx != -float('inf'),tl.exp(mx-safe),0.)
            w=tl.where(rank<count,tl.exp(score-safe),0.)
            den=den*adjust+tl.sum(w,0)
            num=num*adjust+tl.sum(w[:,None]*val,0)
            mx=nxt
        result=num/tl.maximum(den,1.e-30)
        logmass=tl.where(den>0,mx+tl.log(den),-float('inf'))
    else:
        result=tl.load(FallbackOut+row*D+d).to(tl.float32)
        logmass=tl.load(FallbackLSE+row)
    tl.store(Out+row*D+d,result)
    tl.store(LSE+row,logmass)


def attend(engine,q,slots,counts,cache,*,state_len,workspace):
    h,n,d=int(q.size(1)),int(q.size(2)),int(q.size(-1))
    g=engine.num_key_value_groups;kv=int(cache['leaf_k'].size(1));width=int(slots.size(-1))
    average_capacity=int(os.environ.get('LOD_H100_FAST_ROUTE_CAPACITY','128'))
    if average_capacity<1:raise ValueError('route workspace capacity must be positive')
    capacity=max(1,h*n*min(average_capacity,width))
    prefix=workspace('ragged_prefix',(h*n+1,),torch.int64);prefix[:1].zero_()
    torch.cumsum(counts.reshape(-1),0,dtype=torch.int64,out=prefix[1:])
    flat_slots=workspace('ragged_slots',(capacity,),torch.int32)
    query_rows=workspace('ragged_queries',(capacity,),torch.int32)
    local_offsets=workspace('ragged_offsets',(capacity,),torch.int32)
    head_counts=workspace('ragged_head_counts',(h*state_len,),torch.int32);head_counts.zero_()
    kept=workspace('ragged_kept',(1,),torch.int64);kept.zero_()
    overflow=workspace('ragged_overflow',(h,n),torch.int32)
    _pack[(h*n,)](slots,counts,prefix,flat_slots,query_rows,local_offsets,head_counts,kept,overflow,
                  n,state_len,width,capacity,64,num_warps=4)
    grouped=head_counts.view(kv,g,state_len)
    expert_counts=workspace('ragged_expert_counts',(kv*state_len,),torch.int32)
    torch.sum(grouped,dim=1,dtype=torch.int32,out=expert_counts.view(kv,state_len))
    head_offsets=workspace('ragged_head_offsets',(kv,g,state_len),torch.int32)
    torch.cumsum(grouped,dim=1,dtype=torch.int32,out=head_offsets)
    torch.sub(head_offsets,grouped,out=head_offsets)
    cumulative=workspace('ragged_expert_prefix',(kv*state_len+1,),torch.int32);cumulative[:1].zero_()
    torch.cumsum(expert_counts,0,dtype=torch.int32,out=cumulative[1:])
    block_m,block_n=engine.leaf_block_m,engine.leaf_block_n
    blocks=workspace('ragged_blocks',(kv*state_len,),torch.int32)
    torch.add(expert_counts,block_m-1,out=blocks);torch.div(blocks,block_m,rounding_mode='floor',out=blocks)
    cumulative_blocks=workspace('ragged_block_prefix',(kv*state_len+1,),torch.int32);cumulative_blocks[:1].zero_()
    torch.cumsum(blocks,0,dtype=torch.int32,out=cumulative_blocks[1:])
    max_blocks=triton.cdiv(capacity,block_m)+kv*state_len
    block_expert=workspace('ragged_block_expert',(max_blocks,),torch.int32)
    order=workspace('ragged_order',(capacity,),torch.int32)
    _scatter[(triton.cdiv(capacity,256),)](flat_slots,query_rows,local_offsets,head_offsets,cumulative[:-1],
        cumulative_blocks,order,block_expert,kept,n,state_len,g,block_m,capacity,256,num_warps=4)
    route_out=workspace('ragged_route_out',(capacity,d),q.dtype)
    route_lse=workspace('ragged_route_lse',(capacity,),torch.float32)
    roots,directory,pages=cache['slot_pages'],cache['overflow_page_values'],cache['page_indices']
    leaf_k,leaf_v=cache['leaf_k'],cache['leaf_v'];lengths=cache['slot_lengths']
    _paged_leaf_attention_kernel[max_blocks,](q,q,query_rows,order,block_expert,cumulative_blocks[:-1],
        leaf_k,leaf_v,pages,leaf_k,leaf_v,leaf_k,leaf_v,leaf_k,leaf_v,leaf_k,leaf_v,leaf_k,leaf_v,lengths,
        roots,cache['overflow_page_keys'],directory,cache['overflow_used'],lengths,expert_counts,cumulative,
        route_out,route_lse,0,cumulative_blocks[-1:],kv*state_len,state_len,
        PAGE_CAPACITY=int(pages.size(2)),LEAF_CAPACITY=int(leaf_k.size(2)),STATE_CAPACITY=int(roots.size(2)),
        INLINE_PAGES_PER_SLOT=int(roots.size(3)),HASH_CAPACITY=int(directory.size(2)),HASH_PROBES=engine._page_lookup_probes(cache),
        HEAD_DIM=d,VALUE_DIM=d,PAGE_SIZE=16,ROUTE_COUNT=1,SCALE_LOG2=float(engine.scaling)*math.log2(math.e),
        BLOCK_M=block_m,BLOCK_N=block_n,SPLIT_N=1,PARTIAL_OUTPUT=False,INT8_MMA=False,INT8_PV_MMA=False,
        QUANT_BITS=0,QUANT_GROUP_SIZE=1,QUANT_TOKEN_GROUP_SIZE=1,QUANTIZED_SUMMARIES=False,INDEXED=True,
        PROGRAMS_POINTER=True,SEARCH_BLOCKS=False,SEARCH_STEPS=1,RAGGED=True,num_warps=engine.leaf_num_warps)
    fallback_out=workspace('ragged_fallback_out',(1,h,n,d),q.dtype)
    fallback_lse=workspace('ragged_fallback_lse',(1,h,n),torch.float32)
    _direct_leaf_attention[(h*n,)](q,slots,overflow,lengths,roots,cache['overflow_page_keys'],directory,
        cache['overflow_used'],pages,leaf_k,leaf_v,fallback_out,fallback_lse,n,width,state_len,int(roots.size(2)),g,
        int(roots.size(3)),int(directory.size(2)),int(pages.size(2)),int(leaf_k.size(2)),engine._page_lookup_probes(cache),
        d,float(engine.scaling),128,width.bit_length(),num_warps=4)
    out=workspace('ragged_combined',(1,h,n,d),q.dtype)
    lse=workspace('ragged_combined_lse',(1,h,n),torch.float32)
    _reduce[(h*n,)](route_out,route_lse,counts,prefix,fallback_out,fallback_lse,out,lse,d,capacity,32,num_warps=4)
    return out,lse,overflow
