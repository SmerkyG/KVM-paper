"""Exact BF16 row maximum and first index via a packed unsigned reduction."""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _maximum(Scores,Values,Indices,ROWS:tl.constexpr,S:tl.constexpr,M:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)*M+tl.arange(0,M);r=tl.arange(0,B)
    valid=(row[:,None]<ROWS)&(r[None,:]<S)
    raw=tl.load(Scores+row[:,None]*S+r[None,:],mask=valid,other=0)
    bits=raw.to(tl.uint16,bitcast=True).to(tl.uint32)
    # torch.max treats signed zeros as equal and returns the first index.
    bits=tl.where((bits&32767)==0,0,bits)
    ordered=tl.where((bits&32768)!=0,(~bits)&65535,bits^32768)
    # All NaNs outrank infinity; ties between NaN payloads use first index.
    ordered=tl.where((bits&32767)>32640,65535,ordered)
    packed=tl.where(valid,(ordered<<16)|(65535-r[None,:]),0).to(tl.uint32)
    best=tl.max(packed,1);index=65535-(best&65535)
    value=tl.load(Scores+row*S+index,mask=row<ROWS,other=0)
    tl.store(Values+row,value,mask=row<ROWS)
    tl.store(Indices+row,index.to(tl.int64),mask=row<ROWS)


def maximum(scores):
    if scores.dtype!=torch.bfloat16 or not scores.is_cuda or not scores.is_contiguous()or not 0<scores.size(-1)<=65535:
        raise ValueError('packed state maximum requires contiguous CUDA BF16 rows with at most 65535 columns')
    shape=scores.shape[:-1];rows=math.prod(shape);s=scores.size(-1)
    values=torch.empty(shape,device=scores.device,dtype=scores.dtype)
    indices=torch.empty(shape,device=scores.device,dtype=torch.long)
    _maximum[(triton.cdiv(rows,4),)](scores,values,indices,rows,s,4,triton.next_power_of_2(s),num_warps=4)
    return values,indices
