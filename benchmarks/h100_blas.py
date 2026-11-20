"""BF16 shared-GQA QK with FP32 output on the host's older PyTorch release.

PyTorch 2.5 lacks bmm(out_dtype=...). Use its current cuBLAS handle, stream and
workspace, with cuBLAS's ordinary strided-batched GEMM interface. No KV copy or
new CUDA context/stream is created. The wrapper is forward-only and graph-safe.
"""
import ctypes
import functools

import torch

_NATIVE_OUTPUT_DTYPE = 'out_dtype' in (torch.bmm.__doc__ or '')


@functools.lru_cache(None)
def _gemm():
    lib = ctypes.CDLL('libcublas.so.12')
    function = lib.cublasGemmStridedBatchedEx
    v, i, stride = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
    function.argtypes = [v, i, i, i, i, i, v, v, i, i, stride,
                        v, i, i, stride, v, v, i, i, stride, i, i, i]
    function.restype = i
    return function


def qk_fp32(q, mean_k, scores):
    if (q.dtype != torch.bfloat16 or mean_k.dtype != q.dtype or scores.dtype != torch.float32
            or not q.is_cuda or q.device != mean_k.device or q.device != scores.device
            or not all(t.is_contiguous() for t in (q, mean_k, scores))):
        raise ValueError('cuBLAS QK requires contiguous same-device BF16 inputs and FP32 output')
    batches, rows, dim = q.shape
    if mean_k.shape[0] != batches or mean_k.shape[2] != dim or scores.shape != (batches, rows, mean_k.shape[1]):
        raise ValueError('cuBLAS QK dimensions differ')
    states = mean_k.size(1)
    if _NATIVE_OUTPUT_DTYPE:
        return torch.bmm(q, mean_k.transpose(1, 2), out=scores, out_dtype=torch.float32)
    one, zero = ctypes.c_float(1.), ctypes.c_float(0.)
    # CUDA 12.4 headers: OP_T=1, OP_N=0, R_16BF=14, R_32F=0,
    # COMPUTE_32F=68 and GEMM_DEFAULT_TENSOR_OP=99. Column-major C is
    # the transpose of the desired row-major QK product.
    status = _gemm()(torch.cuda.current_blas_handle(), 1, 0, states, rows, dim,
        ctypes.byref(one), mean_k.data_ptr(), 14, dim, states * dim,
        q.data_ptr(), 14, dim, rows * dim, ctypes.byref(zero),
        scores.data_ptr(), 0, states, rows * states, batches, 68, 99)
    if status:
        raise RuntimeError(f'cuBLAS BF16-to-FP32 GEMM failed: status={status}')
    return scores


def pv_fp32(probability, mean_v, output, *, accumulate=False):
    """Shared-KV BF16/FP16 weight/value product with FP32 accumulation/output."""
    if (probability.dtype not in (torch.bfloat16,torch.float16) or mean_v.dtype!=probability.dtype or output.dtype!=torch.float32
            or not probability.is_cuda or probability.device!=mean_v.device or probability.device!=output.device
            or not all(t.is_contiguous()for t in (probability,mean_v,output))):
        raise ValueError('cuBLAS PV requires contiguous BF16 or FP16 inputs and FP32 output')
    batches,rows,states=probability.shape;dim=mean_v.size(-1)
    if mean_v.shape!=(batches,states,dim) or output.shape!=(batches,rows,dim):
        raise ValueError('cuBLAS PV dimensions differ')
    if _NATIVE_OUTPUT_DTYPE and not accumulate:
        return torch.bmm(probability,mean_v,out=output,out_dtype=torch.float32)
    one,beta=ctypes.c_float(1.),ctypes.c_float(float(accumulate))
    input_type=14 if probability.dtype==torch.bfloat16 else 2
    # Column-major C^T = V^T P^T; both input views use OP_N.
    status=_gemm()(torch.cuda.current_blas_handle(),0,0,dim,rows,states,
        ctypes.byref(one),mean_v.data_ptr(),input_type,dim,states*dim,
        probability.data_ptr(),input_type,states,rows*states,ctypes.byref(beta),
        output.data_ptr(),0,dim,rows*dim,batches,68,99)
    if status:raise RuntimeError(f'cuBLAS BF16-to-FP32 PV failed: status={status}')
    return output
