"""Project closed summaries after prefix-filtered selection without subtraction."""
import os
import torch
import triton
import triton.language as tl
from lod_attention.kernels.threshold_prefill import _pv


@triton.jit
def _maximum(a,b):return tl.maximum(a,b)


@triton.jit
def _closed_online(Scores,Log,Bits,Probability,Residual,LSE,T:tl.constexpr,S:tl.constexpr,G:tl.constexpr,B:tl.constexpr):
    """Reproduce online 64-region BF16 coarse weights minus exact summaries.

    The original producer rounds unnormalized weights before each tile's PV,
    rescales old tiles in FP32, and subtracts unrounded opened-summary weights.
    Encode those effective coefficients as two BF16 parts for shared-KV BLAS.
    This preserves that arithmetic's small signed rounding residuals rather
    than assuming normalized closed-only BF16 probabilities are equivalent.
    """
    row=tl.program_id(0).to(tl.int64);kh=row//T//G;r=tl.arange(0,B)
    log=tl.load(Log+kh*S+r,mask=r<S,other=-float('inf'))
    raw=tl.load(Scores+row*S+r,mask=r<S,other=0.)
    mask=tl.load(Bits+row*tl.cdiv(S,8)+r//8,mask=r<S,other=0).to(tl.int32)
    valid=(r<S)&(log!=-float('inf'));opened=valid&(((mask>>(r%8))&1)!=0)
    native=tl.where(valid,raw+log,-float('inf'))
    tile=tl.reshape(native,(B//64,64))
    maximum=tl.associative_scan(tl.max(tile,1),0,_maximum)
    prefix=tl.reshape(tl.broadcast_to(maximum[:,None],(B//64,64)),(B,))
    safe_prefix=tl.where(prefix!=-float('inf'),prefix,0.)
    fullmax=tl.max(native,0);safe=tl.where(fullmax!=-float('inf'),fullmax,0.)
    den=tl.sum(tl.where(valid,tl.exp(native-safe),0.),0)
    full_lse=safe+tl.log(tl.maximum(den,1.e-30))
    w=tl.where(valid,tl.exp(native-safe_prefix),0.).to(Probability.dtype.element_ty).to(tl.float32)
    effective=w*tl.exp(safe_prefix-full_lse)
    removed=tl.where(opened,tl.exp(native-full_lse),0.)
    remaining=tl.maximum(1.-tl.sum(removed,0),0.)
    coeff=(effective-removed)/tl.maximum(remaining,1.e-30)
    coarse_lse=full_lse+tl.log(tl.maximum(remaining,1.e-30))
    if remaining<1.e-4:
        closed=valid&~opened
        closedmax=tl.max(tl.where(closed,native,-float('inf')),0)
        closed_safe=tl.where(closedmax!=-float('inf'),closedmax,0.)
        weight=tl.where(closed,tl.exp(native-closed_safe),0.);closed_den=tl.sum(weight,0)
        coeff=weight/tl.maximum(closed_den,1.e-30)
        coarse_lse=tl.where(closed_den>0,closed_safe+tl.log(closed_den),-float('inf'))
    hi=coeff.to(Probability.dtype.element_ty)
    tl.store(Probability+row*S+r,hi,mask=r<S)
    if Residual is not None:
        tl.store(Residual+row*S+r,coeff-hi.to(tl.float32),mask=r<S)
    tl.store(LSE+row,tl.where(den>0,coarse_lse,-float('inf')))


@triton.jit
def _closed(Scores,Lengths,Bits,Probability,LSE,T:tl.constexpr,S:tl.constexpr,CAP:tl.constexpr,G:tl.constexpr,B:tl.constexpr,Residual=None,Denominator=None,Offset=None):
    row=tl.program_id(0).to(tl.int64);kh=row//T//G;r=tl.arange(0,B)
    size=tl.load(Lengths+kh*CAP+r,mask=r<S,other=0).to(tl.float32)
    raw=tl.load(Scores+row*S+r,mask=r<S,other=0.)
    mask=tl.load(Bits+row*tl.cdiv(S,8)+r//8,mask=r<S,other=0).to(tl.int32)
    closed=(r<S)&(size>0)&(((mask>>(r%8))&1)==0)
    logit=tl.where(closed,raw+tl.log(tl.maximum(size,1.)),-float('inf'))
    maximum=tl.max(logit,0);safe=tl.where(maximum!=-float('inf'),maximum,0.)
    w=tl.where(closed,tl.exp(logit-safe),0.);den=tl.sum(w,0)
    p=w if Denominator is not None else w/tl.maximum(den,1.e-30)
    if Offset is not None:
        p=tl.where((r<S)&(size>0),p-tl.load(Offset+kh),0.)
    tl.store(Probability+row*S+r,p,mask=r<S)
    if Residual is not None:
        hi=p.to(Probability.dtype.element_ty).to(tl.float32)
        tl.store(Residual+row*S+r,p-hi,mask=r<S)
    if Denominator is not None:tl.store(Denominator+row,den)
    tl.store(LSE+row,tl.where(den>0,safe+tl.log(den),-float('inf')))


@triton.jit
def _prepare_log(Lengths,Log,S:tl.constexpr,CAP:tl.constexpr,B:tl.constexpr):
    kh=tl.program_id(0);r=tl.program_id(1)*B+tl.arange(0,B)
    size=tl.load(Lengths+kh*CAP+r,mask=r<S,other=0).to(tl.float32)
    tl.store(Log+kh*S+r,tl.where(size>0,tl.log(tl.maximum(size,1.)),-float('inf')),mask=r<S)


@triton.jit
def _closed_shared(Scores,Log,Bits,Probability,LSE,T:tl.constexpr,S:tl.constexpr,G:tl.constexpr,M:tl.constexpr,B:tl.constexpr,Residual=None,Denominator=None,Offset=None):
    head=tl.program_id(0);q=tl.program_id(1)*M+tl.arange(0,M)
    row=head.to(tl.int64)*T+q;r=tl.arange(0,B)
    log=tl.load(Log+head//G*S+r,mask=r<S,other=-float('inf'))
    raw=tl.load(Scores+row[:,None]*S+r[None,:],mask=(q[:,None]<T)&(r[None,:]<S),other=0.)
    mask=tl.load(Bits+row[:,None]*tl.cdiv(S,8)+r[None,:]//8,mask=(q[:,None]<T)&(r[None,:]<S),other=0).to(tl.int32)
    closed=(q[:,None]<T)&(r[None,:]<S)&(log[None,:]!=-float('inf'))&(((mask>>(r[None,:]%8))&1)==0)
    logit=tl.where(closed,raw+log[None,:],-float('inf'))
    maximum=tl.max(logit,1);safe=tl.where(maximum!=-float('inf'),maximum,0.)
    w=tl.where(closed,tl.exp(logit-safe[:,None]),0.);den=tl.sum(w,1)
    p=w if Denominator is not None else w/tl.maximum(den[:,None],1.e-30)
    if Offset is not None:
        p=tl.where((r[None,:]<S)&(log[None,:]!=-float('inf')),p-tl.load(Offset+head//G),0.)
    tl.store(Probability+row[:,None]*S+r[None,:],p,mask=(q[:,None]<T)&(r[None,:]<S))
    if Residual is not None:
        hi=p.to(Probability.dtype.element_ty).to(tl.float32)
        tl.store(Residual+row[:,None]*S+r[None,:],p-hi,mask=(q[:,None]<T)&(r[None,:]<S))
    if Denominator is not None:tl.store(Denominator+row,den,mask=q<T)
    tl.store(LSE+row,tl.where(den>0,safe+tl.log(den),-float('inf')),mask=q<T)


@triton.jit
def _finish_scaled(Output,Denominator,Center,T:tl.constexpr,G:tl.constexpr,D:tl.constexpr,M:tl.constexpr):
    head=tl.program_id(0);q=tl.program_id(1)*M+tl.arange(0,M);d=tl.arange(0,D)
    row=head.to(tl.int64)*T+q
    den=tl.load(Denominator+row,mask=q<T,other=0.) if Denominator is not None else tl.full((M,),1.,tl.float32)
    value=tl.load(Output+row[:,None]*D+d[None,:],mask=q[:,None]<T,other=0.)/tl.maximum(den[:,None],1.e-30)
    if Center is not None:
        center=tl.load(Center+head//G*D+d)
        value+=tl.where(den[:,None]>0,center[None,:],0.)
    tl.store(Output+row[:,None]*D+d[None,:],value,mask=q[:,None]<T)


def prepare_projection(mean_v,lengths,*,s,cap,buffers):
    """Prepare shared coarse operands once per fixed region state, before query tiling."""
    if (os.environ.get('LOD_H100_CLOSED_PRECISION')!='scaled-fp16'
            or os.environ.get('LOD_H100_CLOSED_VALUE_CENTER')=='1'
            or os.environ.get('LOD_H100_CLOSED_ARITHMETIC','direct')!='direct'):
        raise ValueError('hoisted operands require direct scaled FP16 without value centering')
    kv=mean_v.size(1);key='direct_log_sizes';old=buffers.get(key)
    if old is None or old.numel()<kv*s:
        old=torch.empty(kv*s,device=mean_v.device,dtype=torch.float32);buffers[key]=old
    log=old[:kv*s].view(kv,s)
    _prepare_log[(kv,triton.cdiv(s,256))](lengths,log,s,cap,256,num_warps=4)
    return dict(log=log,values=mean_v.to(torch.float16))


def project(scores,lengths,bits,mean_v,output,lse,*,t,s,cap,g,buffers,timing_events=None,
            prepared=None,routing=None):
    h=scores.size(0);kv=h//g;d=mean_v.size(-1);key='direct_probabilities';numel=h*t*s
    precision=os.environ.get('LOD_H100_CLOSED_PRECISION','bf16')
    if precision not in ('bf16','split-bf16','scaled-bf16','scaled-fp16','offset-bf16','offset-fp16'):raise ValueError('unsupported closed probability precision')
    scaled=precision.startswith('scaled-')
    offset_mode=precision.startswith('offset-')
    probability_dtype=torch.float16 if precision.endswith('-fp16') else mean_v.dtype
    old=buffers.get(key)
    if old is None or old.numel()<numel or old.dtype!=probability_dtype:
        old=torch.empty(numel,device=scores.device,dtype=probability_dtype);buffers[key]=old
    p=old[:numel].view(h,t,s)
    denominator=None;center=None;offset=None;values=mean_v
    if scaled:
        key='direct_denominator';old=buffers.get(key)
        if old is None or old.numel()<h*t:
            old=torch.empty(h*t,device=scores.device,dtype=torch.float32);buffers[key]=old
        denominator=old[:h*t].view(h,t)
        if os.environ.get('LOD_H100_CLOSED_VALUE_CENTER')=='1':
            # Center only values, never routing scores. The exact identity is
            # v0 + sum(w * (v-v0))/sum(w); the offset is restored in FP32.
            # This diagnostic preparation stays on the GPU and shares KV.
            size=lengths.view(kv,cap)[:,:s].float()
            vf=mean_v.view(kv,s,d).float()
            center=(vf*size[:,:,None]).sum(1)/size.sum(1).clamp_min(1)[:,None]
            values=(vf-center[:,None,:]).to(probability_dtype)
        elif prepared is not None:
            values=prepared['values']
        else:
            values=mean_v.to(probability_dtype)
    elif os.environ.get('LOD_H100_CLOSED_VALUE_CENTER')=='1':
        raise ValueError('value centering requires scaled probabilities')
    if offset_mode:
        # p=0 on opened regions. Center p across ALL nonempty summaries:
        # sum(p*v) = mean(v) + sum((p-1/R)*v). The common term is shared
        # across every query of the KV head, so this needs only one GEMM.
        valid=lengths.view(kv,cap)[:,:s]>0
        offset=valid.sum(1).clamp_min(1).float().reciprocal()
        center=torch.where(valid[:,:,None],mean_v.view(kv,s,d).float(),0.).sum(1)*offset[:,None]
        values=mean_v.to(probability_dtype)
    residual=None
    if precision=='split-bf16':
        if os.environ.get('LOD_H100_FAST_CLOSED_BLAS')!='1':
            raise ValueError('split BF16 projection requires cuBLAS')
        key='direct_probability_residual';old=buffers.get(key)
        if old is None or old.numel()<numel:
            old=torch.empty(numel,device=scores.device,dtype=mean_v.dtype);buffers[key]=old
        residual=old[:numel].view(h,t,s)
    if timing_events is not None:
        start=torch.cuda.Event(enable_timing=True);middle=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record()
    m=int(os.environ.get('LOD_H100_CLOSED_NORMALIZE_M','0'))
    arithmetic=os.environ.get('LOD_H100_CLOSED_ARITHMETIC','direct')
    if arithmetic not in ('direct','online64'):raise ValueError('unsupported closed arithmetic')
    if (scaled or offset_mode) and arithmetic!='direct':raise ValueError('centered/scaled probabilities require direct closed arithmetic')
    if m or arithmetic=='online64' or routing is not None:
        if m not in (0,1,2,4,8):raise ValueError('unsupported closed normalization group')
        if prepared is not None:
            log=prepared['log']
        else:
            key='direct_log_sizes';old=buffers.get(key)
            if old is None or old.numel()<kv*s:
                old=torch.empty(kv*s,device=scores.device,dtype=torch.float32);buffers[key]=old
            log=old[:kv*s].view(kv,s)
            _prepare_log[(kv,triton.cdiv(s,256))](lengths,log,s,cap,256,num_warps=4)
        if routing is not None:
            if precision!='scaled-fp16' or arithmetic!='direct':
                raise ValueError('fused selection requires direct scaled FP16')
            from benchmarks.h100_prefill_direct_fused import select_normalize,normalize_bitmap,pack_bitmap
            inv,route_lse,nregions,slots,number,entries,head_counts,offsets,multiplier=routing
            mode=os.environ.get('LOD_H100_DIRECT_SELECT_MODE','bitmap')
            warps=int(os.environ.get('LOD_H100_DIRECT_SELECT_WARPS','8'))
            if mode=='flat':
                select_normalize[(h*t,)](scores,inv,route_lse,lengths,log,nregions,
                    slots,number,entries,head_counts,offsets,bits,p,denominator,lse,
                    t,s,cap,g,slots.size(-1),float(multiplier),triton.next_power_of_2(max(s,8)),num_warps=warps)
            elif mode=='bitmap':
                w=triton.cdiv(s,32)
                def integer_workspace(key,n):
                    old=buffers.get(key)
                    if old is None or old.numel()<n:
                        old=torch.empty(n,device=scores.device,dtype=torch.int32);buffers[key]=old
                    return old[:n]
                ids=integer_workspace('prefix_word_ids',h*t*w);wc=integer_workspace('prefix_word_counts',h*t)
                split_words=os.environ.get('LOD_H100_DIRECT_WORDS_SPLIT','1')=='1'
                normalize_bitmap[(h*t,)](scores,inv,route_lse,lengths,log,nregions,bits,ids,wc,p,denominator,lse,
                    t,s,cap,g,float(multiplier),triton.next_power_of_2(max(s,32)),COMPACT_WORDS=not split_words,num_warps=warps)
                if timing_events is not None:
                    pack_start=torch.cuda.Event(enable_timing=True);pack_end=torch.cuda.Event(enable_timing=True)
                    pack_start.record()
                if split_words:
                    from benchmarks.h100_threshold_prefill_prefix import _words
                    _words[(h*t,)](bits,ids,wc,h*t,s,w,triton.next_power_of_2(w),num_warps=4)
                pack_bitmap[(h*t,)](lengths,bits,ids,wc,slots,number,entries,head_counts,offsets,
                    t,s,cap,g,slots.size(-1),w,num_warps=4)
                if timing_events is not None:
                    pack_end.record();timing_events.setdefault('direct_bitmap_pack',[]).append((pack_start,pack_end))
            else:raise ValueError('direct selection mode must be flat or bitmap')
        elif arithmetic=='online64':
            _closed_online[(h*t,)](scores,log,bits,p,residual,lse,t,s,g,triton.next_power_of_2(max(s,64)),num_warps=8)
        else:
            _closed_shared[(h,triton.cdiv(t,m))](scores,log,bits,p,lse,t,s,g,m,triton.next_power_of_2(max(s,8)),Residual=residual,Denominator=denominator,Offset=offset,num_warps=8)
    else:
        _closed[(h*t,)](scores,lengths,bits,p,lse,t,s,cap,g,triton.next_power_of_2(max(s,8)),Residual=residual,Denominator=denominator,Offset=offset,num_warps=8)
    if timing_events is not None:middle.record()
    if os.environ.get('LOD_H100_FAST_CLOSED_BLAS')=='1':
        from benchmarks.h100_blas import pv_fp32
        pv_fp32(p.view(kv,g*t,s),values.view(kv,s,d),output.view(kv,g*t,d))
        if residual is not None:
            # Recover the low probability bits with a second tensor-core GEMM.
            # Both products accumulate into the same FP32 output, without a KV
            # copy or another output-sized merge. This remains experimental.
            pv_fp32(residual.view(kv,g*t,s),mean_v.view(kv,s,d),output.view(kv,g*t,d),accumulate=True)
    else:
        _pv[(h,triton.cdiv(t,32),triton.cdiv(d,64))](p,values,output,t,s,h,kv,g,d,32,64,64,num_warps=4)
    if scaled or offset_mode:
        _finish_scaled[(h,triton.cdiv(t,16))](output,denominator,center,t,g,d,16,num_warps=4)
    if timing_events is not None:
        end.record();timing_events.setdefault('direct_closed_normalize',[]).append((start,middle))
        timing_events.setdefault('direct_closed_pv',[]).append((middle,end))
