"""Exact, streaming local prefill for NoPE MLA: Q=512 and K=V=latent.

Keep QK, causal masking, online softmax and PV in one kernel. No score matrix,
per-head KV expansion, dummy RoPE tail or changed attention approximation is
needed. The cache remains a single shared 512-channel latent record.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .paged_prefill import _workspace_tensor


@gluon.jit(do_not_specialize=["QUERY_LEN", "KEY_LEN", "QUERY_OFFSET"])
def _latent_local_prefill_gluon(
    Q, KV, Out, Lse,
    Q_BATCH_STRIDE, Q_HEAD_STRIDE, Q_TOKEN_STRIDE,
    KV_BATCH_STRIDE, KV_TOKEN_STRIDE,
    OUT_BATCH_STRIDE, OUT_HEAD_STRIDE, OUT_TOKEN_STRIDE,
    QUERY_LEN, KEY_LEN, QUERY_OFFSET,
    QUERY_HEADS: gl.constexpr, SCALE: gl.constexpr,
    RETURN_LSE: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    WARPS_M: gl.constexpr, WARPS_N: gl.constexpr,
    MFMA_DIM: gl.constexpr,
):
    """Explicit CDNA3 MFMA layouts for the exact shared-latent field."""
    batch_head = gl.program_id(0)
    batch = batch_head // QUERY_HEADS
    head = batch_head % QUERY_HEADS
    load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[1, 64],
        warps_per_cta=[WARPS_M * WARPS_N, 1], order=[1, 0])
    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[MFMA_DIM, MFMA_DIM, 16 if MFMA_DIM == 16 else 8], transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N])
    operand_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8)
    operand_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=8)
    rows = gl.program_id(1) * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, load_layout))
    dims = gl.arange(0, 512, layout=gl.SliceLayout(0, load_layout))
    query = gl.load(Q + batch * Q_BATCH_STRIDE + head * Q_HEAD_STRIDE
                    + rows[:, None] * Q_TOKEN_STRIDE + dims[None, :],
                    mask=rows[:, None] < QUERY_LEN, other=0.0)
    query = gl.convert_layout(query, operand_a)
    score_rows = gl.program_id(1) * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfma))
    score_cols = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma))
    maximum = gl.full((BLOCK_M,), -float("inf"), gl.float32, gl.SliceLayout(1, mfma))
    denominator = gl.zeros((BLOCK_M,), gl.float32, gl.SliceLayout(1, mfma))
    accumulator = gl.zeros((BLOCK_M, 512), gl.float32, mfma)
    cols = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, load_layout))
    end = gl.minimum(KEY_LEN, QUERY_OFFSET + (gl.program_id(1) + 1) * BLOCK_M)
    for begin in range(0, end, BLOCK_N):
        positions = begin + cols
        latent = gl.load(KV + batch * KV_BATCH_STRIDE
                         + positions[:, None] * KV_TOKEN_STRIDE + dims[None, :],
                         mask=positions[:, None] < KEY_LEN, other=0.0)
        key = gl.convert_layout(latent.permute(1, 0), operand_b)
        logits = gl.amd.cdna3.mfma(query, key, gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma))
        logits *= SCALE * 1.4426950408889634
        valid = (score_rows[:, None] < QUERY_LEN) & (begin + score_cols[None, :] < KEY_LEN)
        valid &= begin + score_cols[None, :] <= QUERY_OFFSET + score_rows[:, None]
        logits = gl.where(valid, logits, -float("inf"))
        next_maximum = gl.maximum(maximum, gl.max(logits, axis=1))
        safe_maximum = gl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        correction = gl.exp2(maximum - safe_maximum)
        probability = gl.exp2(logits - safe_maximum[:, None])
        denominator = denominator * correction + gl.sum(probability, axis=1)
        accumulator *= correction[:, None]
        prob_operand = gl.convert_layout(probability.to(latent.dtype), operand_a)
        value = gl.convert_layout(latent, operand_b)
        accumulator = gl.amd.cdna3.mfma(prob_operand, value, accumulator)
        maximum = next_maximum
    result = accumulator / denominator[:, None]
    result = gl.convert_layout(result.to(Out.dtype.element_ty), load_layout)
    gl.store(Out + batch * OUT_BATCH_STRIDE + head * OUT_HEAD_STRIDE
             + rows[:, None] * OUT_TOKEN_STRIDE + dims[None, :],
             result, mask=rows[:, None] < QUERY_LEN)
    if RETURN_LSE:
        gl.store(Lse + batch_head * QUERY_LEN + score_rows,
                 (maximum + gl.log2(denominator)) * 0.6931471805599453,
                 mask=score_rows < QUERY_LEN)


@triton.jit(do_not_specialize=["QUERY_LEN", "KEY_LEN", "QUERY_OFFSET"])
def _latent_local_prefill_kernel(
    Q, KV, Out, Lse,
    Q_BATCH_STRIDE, Q_HEAD_STRIDE, Q_TOKEN_STRIDE,
    KV_BATCH_STRIDE, KV_TOKEN_STRIDE,
    OUT_BATCH_STRIDE, OUT_HEAD_STRIDE, OUT_TOKEN_STRIDE,
    QUERY_LEN, KEY_LEN, QUERY_OFFSET,
    QUERY_HEADS: tl.constexpr, SCALE: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    batch_head = tl.program_id(0).to(tl.int64)
    batch = batch_head // QUERY_HEADS
    head = batch_head % QUERY_HEADS
    rows = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, 512)
    query = tl.load(
        Q + batch * Q_BATCH_STRIDE + head * Q_HEAD_STRIDE
        + rows[:, None] * Q_TOKEN_STRIDE + dims[None, :],
        mask=rows[:, None] < QUERY_LEN, other=0.0,
    )
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, 512), tl.float32)
    end = tl.minimum(KEY_LEN, QUERY_OFFSET + (tl.program_id(1) + 1) * BLOCK_M)
    for begin in tl.range(0, end, BLOCK_N, num_stages=1):
        positions = begin + cols
        latent = tl.load(
            KV + batch * KV_BATCH_STRIDE
            + positions[:, None] * KV_TOKEN_STRIDE + dims[None, :],
            mask=positions[:, None] < KEY_LEN, other=0.0,
        )
        logits = tl.dot(query, tl.trans(latent)) * (SCALE * 1.4426950408889634)
        visible = (rows[:, None] < QUERY_LEN) & (positions[None, :] < KEY_LEN)
        visible &= positions[None, :] <= QUERY_OFFSET + rows[:, None]
        logits = tl.where(visible, logits, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(logits, axis=1))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        correction = tl.exp2(maximum - safe_maximum)
        probability = tl.exp2(logits - safe_maximum[:, None])
        denominator = denominator * correction + tl.sum(probability, axis=1)
        accumulator *= correction[:, None]
        # K and V are the same record. Reuse this loaded tile for both dots.
        accumulator = tl.dot(probability.to(latent.dtype), latent, accumulator)
        maximum = next_maximum
    result = accumulator / denominator[:, None]
    tl.store(
        Out + batch * OUT_BATCH_STRIDE + head * OUT_HEAD_STRIDE
        + rows[:, None] * OUT_TOKEN_STRIDE + dims[None, :],
        result, mask=rows[:, None] < QUERY_LEN,
    )
    if RETURN_LSE:
        tl.store(
            Lse + batch_head * QUERY_LEN + rows,
            (maximum + tl.log2(denominator)) * 0.6931471805599453,
            mask=rows < QUERY_LEN,
        )


def latent_local_prefill_attention(
    q: torch.Tensor, latent: torch.Tensor, *, query_offset: int, scale: float,
    output_buffer: torch.Tensor | None = None, return_lse: bool = True,
    buffers: dict[str, torch.Tensor] | None = None,
    block_m: int = 64, block_n: int = 64, num_warps: int = 4,
    matrix_instr_nonkdim: int = 16,
    gluon_warps: tuple[int, int] | None = (2, 2),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal suffix attention over a single shared NoPE MLA latent head."""
    if q.ndim != 4 or latent.ndim != 4 or q.size(-1) != 512:
        raise ValueError("latent local attention requires rank-four Q/K of width 512")
    batch, heads, supplied_len, _ = q.shape
    if tuple(latent.shape[:2]) != (batch, 1) or latent.size(-1) != 512:
        raise ValueError("latent local attention requires one shared KV head")
    key_len = int(latent.size(2))
    target_len = key_len - query_offset
    if not 0 <= query_offset < key_len:
        raise ValueError("query offset must leave a nonempty causal suffix")
    if supplied_len == key_len:
        actual_q = q[..., query_offset:, :]
    elif supplied_len == target_len:
        actual_q = q
    else:
        raise ValueError("query length must match the full field or its suffix")
    if not q.is_cuda or not latent.is_cuda or q.device != latent.device:
        raise ValueError("latent local attention requires matching GPU tensors")
    if q.dtype not in (torch.bfloat16, torch.float16) or latent.dtype != q.dtype:
        raise ValueError("latent local attention requires matching BF16/FP16 tensors")
    if q.stride(-1) != 1 or latent.stride(-1) != 1:
        raise ValueError("latent channels must be contiguous")
    shape = (batch, heads, target_len, 512)
    if output_buffer is not None:
        if (tuple(output_buffer.shape) != shape or output_buffer.device != q.device
                or output_buffer.dtype != q.dtype or output_buffer.stride(-1) != 1):
            raise ValueError("latent local output buffer has incompatible geometry")
        output = output_buffer
    else:
        # An exact front may stay live while a later local field is evaluated
        # on another stream. Never overwrite that front with the local scratch.
        name = "latent_local_output" if return_lse else "latent_exact_front_output"
        output = _workspace_tensor(buffers, name, shape,
                                   dtype=q.dtype, device=q.device)
    lse = (_workspace_tensor(buffers, "latent_local_lse", shape[:3],
                             dtype=torch.float32, device=q.device) if return_lse
           else torch.empty(0, dtype=torch.float32, device=q.device))
    kernel = _latent_local_prefill_kernel if gluon_warps is None else _latent_local_prefill_gluon
    options = (dict(matrix_instr_nonkdim=matrix_instr_nonkdim) if gluon_warps is None
               else dict(WARPS_M=gluon_warps[0], WARPS_N=gluon_warps[1],
                         MFMA_DIM=matrix_instr_nonkdim))
    kernel[(batch * heads, triton.cdiv(target_len, block_m))](
        actual_q, latent, output, lse,
        actual_q.stride(0), actual_q.stride(1), actual_q.stride(2),
        latent.stride(0), latent.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        target_len, key_len, query_offset,
        QUERY_HEADS=heads, SCALE=float(scale), RETURN_LSE=return_lse,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=num_warps, num_stages=1,
        **options,
    )
    return output, lse
