"""Fused D128 BF16 key normalization with separately rounded FP32 operations."""
import torch
import triton
import triton.language as tl


@triton.jit
def _add(a,b):
    return tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[a,b],dtype=tl.float32,is_pure=True,pack=1)


@triton.jit
def _normalize(X,Y,T,H,S0,S1,S2,M:tl.constexpr,STYLE:tl.constexpr,MeanOut=None,RootOut=None,InvOut=None):
    bh=tl.program_id(0).to(tl.int64);q=tl.program_id(1)*M+tl.arange(0,M)
    base=bh//H*S0+bh%H*S1+q[:,None]*S2
    if STYLE==3:
        g=tl.arange(0,32)
        offset=g
        step:tl.constexpr=32
        x0=tl.load(X+base+offset[None,:],mask=q[:,None]<T,other=0).to(tl.float32)
        x1=tl.load(X+base+offset[None,:]+step,mask=q[:,None]<T,other=0).to(tl.float32)
        x2=tl.load(X+base+offset[None,:]+2*step,mask=q[:,None]<T,other=0).to(tl.float32)
        x3=tl.load(X+base+offset[None,:]+3*step,mask=q[:,None]<T,other=0).to(tl.float32)
        part=_add(_add(_add(x0*x0,x1*x1),x2*x2),x3*x3)
        # PyTorch 2.5 D128 means use four strided accumulators per lane,
        # then shuffle offsets 1,2,4,8,16. Pin each FP32 addition rather
        # than permitting the ordinary reduction to choose another tree.
        a,b=tl.split(tl.reshape(part,(M,16,2)));part=_add(a,b)
        a,b=tl.split(tl.reshape(part,(M,8,2)));part=_add(a,b)
        a,b=tl.split(tl.reshape(part,(M,4,2)));part=_add(a,b)
        a,b=tl.split(tl.reshape(part,(M,2,2)));part=_add(a,b)
        a,b=tl.split(tl.reshape(part,(M,1,2)));part=_add(a,b)
        mean=tl.reshape(part,(M,))/128
    else:
        d=tl.arange(0,128)
        x=tl.load(X+base+d[None,:],mask=q[:,None]<T,other=0).to(tl.float32)
        mean=tl.sum(x*x,1)/128
    safe=tl.maximum(mean,1.e-12)
    root=tl.inline_asm_elementwise('sqrt.rn.f32 $0, $1;',constraints='=f,f',args=[safe],dtype=tl.float32,is_pure=True,pack=1)
    inv=tl.inline_asm_elementwise('div.rn.f32 $0, 0f3F800000, $1;',constraints='=f,f',args=[root],dtype=tl.float32,is_pure=True,pack=1)
    row=bh*T+q
    if MeanOut is not None:
        tl.store(MeanOut+row,mean,mask=q<T);tl.store(RootOut+row,root,mask=q<T);tl.store(InvOut+row,inv,mask=q<T)
    if STYLE==3:
        tl.store(Y+row[:,None]*128+offset[None,:],x0*inv[:,None],mask=q[:,None]<T)
        tl.store(Y+row[:,None]*128+offset[None,:]+step,x1*inv[:,None],mask=q[:,None]<T)
        tl.store(Y+row[:,None]*128+offset[None,:]+2*step,x2*inv[:,None],mask=q[:,None]<T)
        tl.store(Y+row[:,None]*128+offset[None,:]+3*step,x3*inv[:,None],mask=q[:,None]<T)
    else:
        tl.store(Y+row[:,None]*128+d[None,:],x*inv[:,None],mask=q[:,None]<T)


def normalize(x,style=3,diagnostics=None):
    if x.dtype!=torch.bfloat16 or not x.is_cuda or x.ndim!=4 or x.size(-1)!=128 or x.stride(-1)!=1:
        raise ValueError('fused key normalization requires CUDA BF16 BHT128 inputs')
    if style not in (0,3):raise ValueError('normalization style must be PyTorch-compatible (3) or the diagnostic control (0)')
    if x.numel()//128<16:raise ValueError('small normalization fields must use the PyTorch fallback')
    out=torch.empty(x.shape,device=x.device,dtype=x.dtype)
    if diagnostics is not None:
        for k in ('mean','root','inverse'):diagnostics[k]=torch.empty(x.shape[:-1],device=x.device,dtype=torch.float32)
    _normalize[(x.size(0)*x.size(1),triton.cdiv(x.size(2),4))](x,out,x.size(2),x.size(1),x.stride(0),x.stride(1),x.stride(2),4,style,
        MeanOut=diagnostics['mean']if diagnostics is not None else None,
        RootOut=diagnostics['root']if diagnostics is not None else None,
        InvOut=diagnostics['inverse']if diagnostics is not None else None,num_warps=4)
    return out
