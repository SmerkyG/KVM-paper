"""Fuse the BF16 centroid / FP32 constituent-RMS division used by Qwen3.

This preserves PyTorch's separately rounded FP32 division and final BF16
conversion. It does not apply Llama's cosine normalization to Qwen keys.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rescale(X,R,Y,T:tl.constexpr,H:tl.constexpr,X0:tl.constexpr,X1:tl.constexpr,X2:tl.constexpr,
             R0:tl.constexpr,R1:tl.constexpr,R2:tl.constexpr,M:tl.constexpr):
    bh=tl.program_id(0).to(tl.int64);row=tl.program_id(1)*M+tl.arange(0,M)
    dim=tl.arange(0,128)
    den=tl.load(R+bh//H*R0+bh%H*R1+row*R2,mask=row<T,other=1.).to(tl.float32)
    den=tl.where(den!=den,den,tl.maximum(den,1.e-12))
    value=tl.load(X+bh//H*X0+bh%H*X1+row[:,None]*X2+dim[None,:],mask=row[:,None]<T,other=0).to(tl.float32)
    out=tl.inline_asm_elementwise('div.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[value,den[:,None]],
        dtype=tl.float32,is_pure=True,pack=1)
    tl.store(Y+(bh*T+row[:,None])*128+dim[None,:],out,mask=row[:,None]<T)


def rescale(key,radial_rms):
    if (not key.is_cuda or key.dtype!=torch.bfloat16 or key.ndim!=4 or key.size(-1)!=128
            or key.stride(-1)!=1 or radial_rms.shape!=(*key.shape[:-1],1)
            or radial_rms.device!=key.device or radial_rms.dtype not in (torch.float32,torch.bfloat16)):
        raise ValueError('fused centroid rescale requires BF16 BHT128 keys and matching per-row RMS values')
    out=torch.empty(key.shape,device=key.device,dtype=key.dtype)
    _rescale[(key.size(0)*key.size(1),triton.cdiv(key.size(2),4))](key,radial_rms,out,key.size(2),key.size(1),
        *key.stride()[:3],*radial_rms.stride()[:3],4,num_warps=4)
    return out
