"""Absorbed-MLA dense and compact-LoD decode kernels for gfx942 using Gluon.

The kernels consume Kimi's cached ``[latent_512, direct_key_64]`` rows without
expanding per-head keys or values.  The compact-LoD entry point additionally
understands the release two-tier page-descriptor representation and per-row
log-mass bias. Stage one uses CDNA3 MFMA and ordinary buffer loads; head-tiled
LoD uses the parallel FP32 split-output reducer below.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


# LoD's effective sequence is bounded by the centroid schedule plus eight
# opened regions, and is far shorter than dense history.  On MI325X, 64 splits
# is the fixed backing capacity. Live head-tiled launches use 32 splits at
# physical B1 or 16 at B4+, selected below without allocating new buffers.
KIMI_GLUON_LOD_SPLITS = 64


def kimi_lod_decode_splits(batch: int, *, head_tiled_metadata: bool) -> int:
    """Keep short compact sequences from being oversplit at live B4+.

    With all 96 heads, 16 splits give 384/768 workgroups at B4/B8, already
    enough to occupy gfx942. Keeping 64 there wastes partial K tiles and
    quadruples intermediate outputs. This is geometry only, not a work cap.
    Physical B1 benefits from 32 even at 24K compact rows. Keep 64 for the
    not-yet-tuned B2/B3 and legacy per-GQA layouts.
    """
    if head_tiled_metadata:
        if batch >= 4:
            return 16
        if batch == 1:
            return 32
    return KIMI_GLUON_LOD_SPLITS


@triton.jit
def _reduce_head_tiled_mla_splits_kernel(
    partial,
    partial_lse,
    output,
    final_lse,
    sequence_lengths,
    cache_indices,
    local_lens,
    dcp_global_lens,
    PARTIAL_BATCH_STRIDE: tl.constexpr,
    PARTIAL_HEAD_STRIDE: tl.constexpr,
    PARTIAL_SPLIT_STRIDE: tl.constexpr,
    LSE_BATCH_STRIDE: tl.constexpr,
    LSE_HEAD_STRIDE: tl.constexpr,
    LSE_SPLIT_STRIDE: tl.constexpr,
    OUTPUT_BATCH_STRIDE: tl.constexpr,
    OUTPUT_HEAD_STRIDE: tl.constexpr,
    FINAL_LSE_BATCH_STRIDE: tl.constexpr,
    FINAL_LSE_HEAD_STRIDE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_TILES: tl.constexpr,
    HAS_FINAL_LSE: tl.constexpr,
    ADVANCE_DCP_LENGTHS: tl.constexpr,
    DCP_RANK: tl.constexpr,
    DCP_WORLD_SIZE: tl.constexpr,
    DCP_INTERLEAVE_SIZE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    value_tile = tl.program_id(2)
    metadata_row = batch * HEAD_TILES + head // 16
    sequence_length = tl.load(sequence_lengths + metadata_row).to(tl.int32)
    # Stage one assigns ceil(length/splits) rows to every split and leaves the
    # empty suffix untouched. Re-derive the number of initialized splits.
    split_length = tl.maximum(1, tl.cdiv(sequence_length, NUM_SPLITS))
    active_splits = tl.cdiv(sequence_length, split_length)
    splits = tl.arange(0, BLOCK_S)
    dimension = value_tile * BLOCK_D + tl.arange(0, BLOCK_D)
    split_lse = tl.load(
        partial_lse + batch * LSE_BATCH_STRIDE + head * LSE_HEAD_STRIDE
        + splits * LSE_SPLIT_STRIDE,
        mask=(splits < NUM_SPLITS) & (splits < active_splits),
        other=-float("inf"),
    ).to(tl.float32)
    maximum = tl.max(split_lse, axis=0)
    # Empty split outputs are deliberately unwritten by stage one. Never
    # load their values, including active splits whose attention is all masked.
    valid = (splits < NUM_SPLITS) & (splits < active_splits) & (split_lse != -float("inf"))
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp(split_lse - safe_maximum)
    denominator = tl.sum(weights, axis=0)
    split_value = tl.load(
        partial + batch * PARTIAL_BATCH_STRIDE + head * PARTIAL_HEAD_STRIDE
        + splits[:, None] * PARTIAL_SPLIT_STRIDE + dimension[None, :],
        mask=valid[:, None] & (dimension[None, :] < HEAD_DIM), other=0.0,
    ).to(tl.float32)
    accumulator = tl.sum(split_value * weights[:, None], axis=0)
    output_value = tl.where(denominator > 0.0, accumulator / denominator, 0.0)
    tl.store(
        output
        + batch * OUTPUT_BATCH_STRIDE
        + head * OUTPUT_HEAD_STRIDE
        + dimension,
        output_value,
        mask=dimension < HEAD_DIM,
    )
    if HAS_FINAL_LSE & (value_tile == 0):
        tl.store(
            final_lse
            + batch * FINAL_LSE_BATCH_STRIDE
            + head * FINAL_LSE_HEAD_STRIDE,
            maximum + tl.log(denominator),
        )
    # This reducer is the last rank-local attention kernel.  One head can
    # therefore advance ownership metadata without another launch; all route,
    # descriptor, and attention programs have already consumed the old length.
    if ADVANCE_DCP_LENGTHS & (head == 0) & (value_tile == 0):
        cache_row = tl.load(cache_indices + batch).to(tl.int64)
        global_length = tl.load(dcp_global_lens + cache_row).to(tl.int64)
        owner = (
            (global_length // DCP_INTERLEAVE_SIZE) % DCP_WORLD_SIZE
        ) == DCP_RANK
        local_length = tl.load(local_lens + cache_row)
        tl.store(
            local_lens + cache_row,
            local_length + owner.to(local_length.dtype),
        )
        tl.store(dcp_global_lens + cache_row, global_length + 1)


@gluon.jit
def _absorbed_mla_stage1_gfx942(
    Q,
    KV,
    Bias,
    PageTable,
    FixedIndices,
    CacheIndices,
    SeqLens,
    ExactTokenCounts,
    LocalLens,
    DCPGlobalLens,
    OpenedStamps,
    SequenceEpochs,
    Partial,
    PartialLSE,
    stride_q_b,
    stride_q_h,
    stride_kv_n,
    stride_page_b,
    stride_fixed_b,
    stride_opened_b,
    stride_partial_b,
    stride_partial_h,
    stride_partial_s,
    stride_lse_b,
    stride_lse_h,
    stride_lse_s,
    scale,
    NHEAD: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    COMPACT_LOD: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    LOCAL_LIMIT: gl.constexpr,
    INCLUDE_NEW: gl.constexpr,
    HEAD_TILED_METADATA: gl.constexpr,
    STATE_CAPACITY: gl.constexpr,
    SINK_LEN: gl.constexpr,
    MASK_OPENED_COARSE: gl.constexpr,
    DCP_ROW_MASKED_NEW: gl.constexpr,
    DCP_RANK: gl.constexpr,
    DCP_WORLD_SIZE: gl.constexpr,
    DCP_INTERLEAVE_SIZE: gl.constexpr,
):
    batch = gl.program_id(0)
    split = gl.program_id(1)
    head_tile = gl.program_id(2)
    metadata_row = (
        batch * gl.cdiv(NHEAD, 16) + head_tile
        if HEAD_TILED_METADATA
        else batch
    )

    BLOCK_H: gl.constexpr = 16
    D_LATENT: gl.constexpr = 512
    D_DIRECT: gl.constexpr = 64

    q_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[1, 64],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    q_direct_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0, 1), (0, 2), (0, 4)),
        lane_bases=((0, 8), (0, 16), (0, 32), (1, 0), (2, 0), (4, 0)),
        warp_bases=((8, 0), (0, 0)),
        block_bases=[],
        shape=[BLOCK_H, D_DIRECT],
    )
    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3,
        # CDNA3's BF16 instruction is v_mfma_f32_16x16x16_bf16.
        # The K=32 form used by AITER's gfx950 kernel is a CDNA4 shape.
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    mfma_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma, k_width=8
    )
    mfma_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma, k_width=8
    )
    kv_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((1, 0), (2, 0), (4, 0), (0, 8), (0, 4), (0, 16), (0, 32)),
        lane_bases=((8, 0), (16, 0), (32, 0), (64, 0), (128, 0), (256, 0)),
        warp_bases=((0, 1), (0, 2)),
        block_bases=[],
        shape=[D_LATENT, BLOCK_N],
    )
    direct_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((1, 0), (2, 0), (4, 0), (0, 32)),
        lane_bases=((8, 0), (16, 0), (32, 0), (0, 4), (0, 8), (0, 16)),
        warp_bases=((0, 1), (0, 2)),
        block_bases=[],
        shape=[D_DIRECT, BLOCK_N],
    )
    page_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0,),),
        lane_bases=((1,), (2,), (4,), (8,), (16,), (32,)),
        warp_bases=((0,), (0,)),
        block_bases=[],
        shape=[BLOCK_N],
    )
    value_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0, 1), (0, 2), (0, 4), (0, 32), (64, 0), (128, 0), (256, 0)),
        lane_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 16)),
        warp_bases=((16, 0), (32, 0)),
        block_bases=[],
        shape=[D_LATENT, BLOCK_N],
    )

    dtype = Q.type.element_ty
    head_base = head_tile * BLOCK_H
    heads = head_base + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, q_layout)
    )
    d_latent_q = gl.arange(0, D_LATENT, layout=gl.SliceLayout(0, q_layout))
    q_offsets = (
        batch * stride_q_b
        + heads[:, None] * stride_q_h
        + d_latent_q[None, :]
    ).to(gl.int32)
    q = gl.amd.cdna3.buffer_load(
        Q,
        q_offsets,
        mask=(heads < NHEAD)[:, None],
        other=0.0,
    )
    q = gl.convert_layout(q, mfma_a)

    heads_direct = head_base + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, q_direct_layout)
    )
    d_direct_q = gl.arange(
        0, D_DIRECT, layout=gl.SliceLayout(0, q_direct_layout)
    )
    q_direct_offsets = (
        batch * stride_q_b
        + heads_direct[:, None] * stride_q_h
        + D_LATENT
        + d_direct_q[None, :]
    ).to(gl.int32)
    q_direct = gl.amd.cdna3.buffer_load(
        Q,
        q_direct_offsets,
        mask=(heads_direct < NHEAD)[:, None],
        other=0.0,
    )
    q_direct = gl.convert_layout(q_direct, mfma_a)

    seq_len = gl.load(SeqLens + metadata_row)
    split_len = gl.cdiv(seq_len, NUM_SPLITS)
    split_start = split * split_len
    split_end = gl.minimum(split_start + split_len, seq_len)
    num_tiles = gl.cdiv(split_end - split_start, BLOCK_N)

    e_max = (
        gl.zeros([BLOCK_H], dtype=gl.float32, layout=gl.SliceLayout(1, mfma))
        - float("inf")
    )
    e_sum = gl.zeros(
        [BLOCK_H], dtype=gl.float32, layout=gl.SliceLayout(1, mfma)
    )
    acc = gl.zeros([BLOCK_H, D_LATENT], dtype=gl.float32, layout=mfma)
    LOG2E: gl.constexpr = 1.4426950408889634

    for tile in range(num_tiles):
        token_offsets = (
            split_start
            + tile * BLOCK_N
            + gl.arange(0, BLOCK_N, layout=page_layout)
        )
        valid = token_offsets < split_end
        if COMPACT_LOD:
            cache_batch = gl.load(CacheIndices + batch).to(gl.int32)
            exact_count = gl.load(ExactTokenCounts + metadata_row).to(gl.int32)
            prefix_len = seq_len - exact_count
            is_exact = token_offsets >= prefix_len
            local_count = gl.minimum(
                gl.load(LocalLens + cache_batch).to(gl.int32), LOCAL_LIMIT
            )
            if DCP_ROW_MASKED_NEW:
                global_length = gl.load(DCPGlobalLens + cache_batch).to(gl.int32)
                row_includes_new = (
                    (global_length // DCP_INTERLEAVE_SIZE) % DCP_WORLD_SIZE
                ) == DCP_RANK
                local_count += row_includes_new.to(gl.int32)
            else:
                local_count += INCLUDE_NEW
            fixed_prefix = gl.where(
                token_offsets < local_count,
                token_offsets,
                LOCAL_LIMIT + token_offsets - local_count,
            )
            fixed_prefix = gl.maximum(
                0, gl.minimum(fixed_prefix, stride_fixed_b - 1)
            )
            prefix_pages = gl.amd.cdna3.buffer_load(
                FixedIndices,
                cache_batch * stride_fixed_b + fixed_prefix,
                mask=valid & ~is_exact,
                other=0,
            ).to(gl.int32)

            exact_token = token_offsets - prefix_len
            descriptor_rank = exact_token // 16
            within_page = exact_token % 16
            descriptor_valid = descriptor_rank >= 0
            safe_descriptor = gl.maximum(0, descriptor_rank)
            packed = gl.amd.cdna3.buffer_load(
                PageTable,
                metadata_row * stride_page_b + safe_descriptor,
                mask=valid & is_exact & descriptor_valid,
                other=0,
            ).to(gl.int32)
            page_base = packed & 0x00FFFFFF
            valid_lanes = packed >> 24
            fixed_exact = page_base + within_page
            exact_valid = (
                valid
                & is_exact
                & descriptor_valid
                & (within_page < valid_lanes)
                & (fixed_exact >= 0)
                & (fixed_exact < stride_fixed_b)
            )
            safe_exact = gl.maximum(
                0, gl.minimum(fixed_exact, stride_fixed_b - 1)
            )
            exact_pages = gl.amd.cdna3.buffer_load(
                FixedIndices,
                cache_batch * stride_fixed_b + safe_exact,
                mask=exact_valid,
                other=0,
            ).to(gl.int32)
            pages = gl.where(is_exact, exact_pages, prefix_pages)
            valid &= ~is_exact | exact_valid
        else:
            block_ranks = token_offsets // PAGE_SIZE
            within_blocks = token_offsets % PAGE_SIZE
            block_offsets = (
                batch * stride_page_b + block_ranks
            ).to(gl.int32)
            blocks = gl.amd.cdna3.buffer_load(
                PageTable, block_offsets, mask=valid, other=0
            ).to(gl.int32)
            pages = blocks * PAGE_SIZE + within_blocks

        d_latent = gl.arange(
            0, D_LATENT, layout=gl.SliceLayout(1, kv_layout)
        )
        pages_kv = gl.convert_layout(pages, gl.SliceLayout(0, kv_layout))
        valid_kv = gl.convert_layout(valid, gl.SliceLayout(0, kv_layout))
        kv_offsets = (
            d_latent[:, None] + pages_kv[None, :] * stride_kv_n
        ).to(gl.int32)
        latent = gl.amd.cdna3.buffer_load(
            KV, kv_offsets, mask=valid_kv[None, :], other=0.0
        )
        latent_dot = gl.convert_layout(latent, mfma_b)
        qk = gl.amd.cdna3.mfma(
            q,
            latent_dot.to(dtype),
            gl.zeros([BLOCK_H, BLOCK_N], dtype=gl.float32, layout=mfma),
        )

        d_direct = gl.arange(
            0, D_DIRECT, layout=gl.SliceLayout(1, direct_layout)
        )
        pages_direct = gl.convert_layout(
            pages, gl.SliceLayout(0, direct_layout)
        )
        valid_direct = gl.convert_layout(
            valid, gl.SliceLayout(0, direct_layout)
        )
        direct_offsets = (
            D_LATENT
            + d_direct[:, None]
            + pages_direct[None, :] * stride_kv_n
        ).to(gl.int32)
        direct = gl.amd.cdna3.buffer_load(
            KV,
            direct_offsets,
            mask=valid_direct[None, :],
            other=0.0,
        )
        qk = gl.amd.cdna3.mfma(
            q_direct, gl.convert_layout(direct, mfma_b).to(dtype), qk
        )
        qk *= scale
        if HAS_BIAS:
            bias = gl.amd.cdna3.buffer_load(
                Bias, pages, mask=valid, other=float("-inf")
            ).to(gl.float32)
            if MASK_OPENED_COARSE:
                # Each 16-head tile has its own globally selected centroid
                # union.  K/V rows remain shared; only the coarse replacement
                # mask is tile-local.  Prefix positions after the compacted
                # local field map to the fixed sink/coarse suffix.
                fixed_prefix_for_mask = gl.where(
                    token_offsets < local_count,
                    token_offsets,
                    LOCAL_LIMIT + token_offsets - local_count,
                )
                coarse_slot = fixed_prefix_for_mask - LOCAL_LIMIT - SINK_LEN
                is_coarse = (
                    ~is_exact
                    & (coarse_slot >= 0)
                    & (coarse_slot < STATE_CAPACITY)
                )
                safe_coarse_slot = gl.maximum(
                    0, gl.minimum(coarse_slot, STATE_CAPACITY - 1)
                )
                stamp = gl.amd.cdna3.buffer_load(
                    OpenedStamps,
                    metadata_row * stride_opened_b + safe_coarse_slot,
                    mask=valid & is_coarse,
                    other=0,
                ).to(gl.int32)
                epoch = gl.load(SequenceEpochs + metadata_row).to(gl.int32)
                bias = gl.where(is_coarse & (stamp == epoch), float("-inf"), bias)
            qk += gl.convert_layout(bias, gl.SliceLayout(0, mfma))[None, :]
        valid_scores = gl.convert_layout(valid, gl.SliceLayout(0, mfma))
        qk = gl.where(valid_scores[None, :], qk, float("-inf"))

        next_max = gl.maximum(gl.max(qk, 1), e_max)
        # A split/tile can contain only replaced coarse entries or padding.
        # Preserve its zero mass rather than evaluating -inf - -inf. This
        # guard does not hide NaN inputs/addressing errors: only -inf changes.
        safe_max = gl.where(next_max == -float("inf"), 0.0, next_max)
        rescale = gl.exp2((e_max - safe_max) * LOG2E)
        probs = gl.exp2((qk - safe_max[:, None]) * LOG2E)
        e_sum = e_sum * rescale + gl.sum(probs, 1)
        e_max = next_max
        acc *= rescale[:, None]

        latent_v = gl.convert_layout(latent, value_layout)
        latent_v = gl.permute(latent_v, [1, 0])
        latent_v = gl.convert_layout(latent_v, mfma_b)
        probs = gl.convert_layout(probs.to(dtype), mfma_a)
        acc = gl.amd.cdna3.mfma(probs, latent_v.to(dtype), acc)

    out_heads = head_base + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, mfma)
    )
    out_dim = gl.arange(0, D_LATENT, layout=gl.SliceLayout(0, mfma))
    out_offsets = (
        batch * stride_partial_b
        + out_heads[:, None] * stride_partial_h
        + split * stride_partial_s
        + out_dim[None, :]
    ).to(gl.int32)
    gl.amd.cdna3.buffer_store(
        (acc / gl.where(e_sum == 0, 1.0, e_sum)[:, None]).to(dtype),
        Partial,
        out_offsets,
        mask=(out_heads < NHEAD)[:, None],
    )

    lse_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[64],
        warps_per_cta=[4],
        order=[0],
    )
    lse_heads = head_base + gl.arange(0, BLOCK_H, layout=lse_layout)
    lse = gl.convert_layout(e_max + gl.log(e_sum), lse_layout)
    lse_offsets = (
        batch * stride_lse_b
        + lse_heads * stride_lse_h
        + split * stride_lse_s
    ).to(gl.int32)
    gl.amd.cdna3.buffer_store(
        lse,
        PartialLSE,
        lse_offsets,
        mask=lse_heads < NHEAD,
    )


def absorbed_mla_decode_gfx942(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    *,
    num_splits: int = 128,
    partial: torch.Tensor | None = None,
    partial_lse: torch.Tensor | None = None,
    final_lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run dense absorbed MLA over a standard vLLM paged KV cache.

    ``kv`` may be either a flat ``[N,576]`` cache (page size one) or vLLM's
    ordinary contiguous ``[num_blocks,page_size,576]`` cache.  Optional
    scratch tensors make the entry point allocation-free under CUDA graphs.
    """
    from aiter.ops.triton.gluon.mla_gluon import _mla_softmax_reducev_kernel

    batch, nhead, qk_dim = q.shape
    if qk_dim != 576 or kv.shape[-1] != 576 or out.shape != (batch, nhead, 512):
        raise ValueError("expected q=[B,H,576], kv=[...,576], out=[B,H,512]")
    if nhead < 1:
        raise ValueError("q must contain at least one attention head")
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise ValueError("this gfx942 specialization requires BF16 q and kv")
    if page_table.ndim != 2 or page_table.shape[0] != batch:
        raise ValueError("page_table must have shape [B,max_blocks]")
    if seq_lens.shape != (batch,) or num_splits < 1:
        raise ValueError("seq_lens must have shape [B] and num_splits must be positive")
    if kv.ndim == 2:
        page_size = 1
        flat_kv = kv
    elif kv.ndim == 3:
        page_size = int(kv.size(1))
        if not kv.is_contiguous():
            raise ValueError("paged kv must be contiguous")
        flat_kv = kv.view(-1, kv.size(-1))
    else:
        raise ValueError("kv must have shape [N,576] or [blocks,page_size,576]")
    expected_partial = (batch, nhead, num_splits, 512)
    expected_lse = (batch, nhead, num_splits)
    if partial is None:
        partial = torch.empty(expected_partial, dtype=q.dtype, device=q.device)
    elif tuple(partial.shape) != expected_partial or partial.dtype != q.dtype:
        raise ValueError("partial output has the wrong shape or dtype")
    if partial_lse is None:
        partial_lse = torch.empty(expected_lse, dtype=torch.float32, device=q.device)
    elif tuple(partial_lse.shape) != expected_lse or partial_lse.dtype != torch.float32:
        raise ValueError("partial LSE has the wrong shape or dtype")
    if final_lse is not None and (
        tuple(final_lse.shape) != (batch, nhead)
        or final_lse.dtype != torch.float32
    ):
        raise ValueError("final_lse must be fp32 with q's [B,H] shape")
    _absorbed_mla_stage1_gfx942[(batch, num_splits, triton.cdiv(nhead, 16))](
        q,
        flat_kv,
        q,  # Unused bias pointer when HAS_BIAS=False.
        page_table,
        page_table,
        seq_lens,
        seq_lens,
        seq_lens,
        seq_lens,
        seq_lens,
        seq_lens,
        seq_lens,
        partial,
        partial_lse,
        q.stride(0),
        q.stride(1),
        flat_kv.stride(0),
        page_table.stride(0),
        page_table.stride(0),
        0,  # Unused opened-stamp stride when MASK_OPENED_COARSE=False.
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        partial_lse.stride(0),
        partial_lse.stride(1),
        partial_lse.stride(2),
        scale,
        NHEAD=nhead,
        NUM_SPLITS=num_splits,
        BLOCK_N=64,
        PAGE_SIZE=page_size,
        COMPACT_LOD=False,
        HAS_BIAS=False,
        LOCAL_LIMIT=0,
        INCLUDE_NEW=False,
        HEAD_TILED_METADATA=False,
        STATE_CAPACITY=1,
        SINK_LEN=0,
        MASK_OPENED_COARSE=False,
        DCP_ROW_MASKED_NEW=False,
        DCP_RANK=0,
        DCP_WORLD_SIZE=1,
        DCP_INTERLEAVE_SIZE=1,
        num_warps=4,
    )
    _mla_softmax_reducev_kernel[(batch, nhead, 1)](
        partial,
        partial_lse,
        out,
        final_lse,
        seq_lens,
        partial.stride(0),
        0,
        partial.stride(1),
        partial.stride(2),
        partial_lse.stride(0),
        0,
        partial_lse.stride(1),
        partial_lse.stride(2),
        out.stride(0),
        0,
        out.stride(1),
        final_lse.stride(0) if final_lse is not None else 0,
        0,
        final_lse.stride(1) if final_lse is not None else 0,
        NUM_KV_SPLITS=num_splits,
        HEAD_DIM_CKV=512,
        HAS_FINAL_LSE=final_lse is not None,
        USE_2D_VIEW=True,
        BLOCK_S=min(64, triton.next_power_of_2(num_splits)),
        BLOCK_N=64,
        num_warps=8,
    )
    return out


def absorbed_mla_lod_decode_gfx942(
    q: torch.Tensor,
    kv: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    page_descriptors: torch.Tensor,
    fixed_indices: torch.Tensor,
    cache_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    exact_token_counts: torch.Tensor,
    local_lens: torch.Tensor,
    scale: float,
    *,
    local_limit: int,
    opened_stamps: torch.Tensor | None = None,
    sequence_epochs: torch.Tensor | None = None,
    state_capacity: int = 0,
    sink_len: int = 0,
    head_tiled_metadata: bool = False,
    include_new: bool = True,
    dcp_global_lens: torch.Tensor | None = None,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    dcp_interleave_size: int = 1,
    num_splits: int = KIMI_GLUON_LOD_SPLITS,
    partial: torch.Tensor | None = None,
    partial_lse: torch.Tensor | None = None,
    final_lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run dense attention over a compact two-tier LoD effective sequence.

    The prefix is addressed directly through ``fixed_indices``.  Its exact
    suffix is a packed list of 16-token leaf pages, matching the production
    two-tier compact-descriptor format.  ``bias`` stores zero for exact rows
    and ``log(count)`` for coarse centroid rows.
    """
    from aiter.ops.triton.gluon.mla_gluon import _mla_softmax_reducev_kernel

    batch, nhead, qk_dim = q.shape
    if qk_dim != 576 or kv.shape[-1] != 576 or out.shape != (batch, nhead, 512):
        raise ValueError("expected q=[B,H,576], kv=[N,576], out=[B,H,512]")
    if nhead < 1 or q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise ValueError("gfx942 absorbed MLA requires BF16 and at least one head")
    if bias.shape != (kv.size(0),):
        raise ValueError("bias must provide one natural-log bias per KV row")
    metadata_rows = batch * triton.cdiv(nhead, 16) if head_tiled_metadata else batch
    if page_descriptors.ndim != 2 or page_descriptors.size(0) != metadata_rows:
        raise ValueError(
            "page_descriptors must have one row per request/head tile: "
            f"got shape={tuple(page_descriptors.shape)}, batch={batch}, "
            f"heads={nhead}, head_tiled_metadata={head_tiled_metadata}, "
            f"expected_rows={metadata_rows}"
        )
    if fixed_indices.ndim != 2:
        raise ValueError("fixed_indices must have shape [cache_rows,capacity]")
    if cache_indices.shape != (batch,):
        raise ValueError("cache_indices must have one entry per request")
    if seq_lens.shape != (metadata_rows,) or exact_token_counts.shape != (
        metadata_rows,
    ):
        raise ValueError("decode metadata must have one entry per request/head tile")
    if local_lens.ndim != 1 or num_splits < 1 or local_limit < 0:
        raise ValueError("invalid local lengths, split count, or local limit")
    mask_opened_coarse = opened_stamps is not None or sequence_epochs is not None
    if mask_opened_coarse:
        if opened_stamps is None or sequence_epochs is None:
            raise ValueError("opened stamps and epochs must be supplied together")
        if opened_stamps.shape != (metadata_rows, state_capacity):
            raise ValueError("opened stamps have the wrong tiled shape")
        if sequence_epochs.shape != (metadata_rows,) or state_capacity <= 0:
            raise ValueError("invalid opened-centroid metadata")
    else:
        # Non-null placeholders keep one stable Gluon ABI for dense and LoD
        # callers; these pointers are never dereferenced in this mode.
        opened_stamps = page_descriptors
        sequence_epochs = seq_lens
    dcp_row_masked_new = dcp_global_lens is not None
    if dcp_row_masked_new:
        if dcp_global_lens.ndim != 1:
            raise ValueError("DCP global lengths must be a vector")
        if not 0 <= dcp_rank < dcp_world_size or dcp_interleave_size <= 0:
            raise ValueError("invalid DCP ownership geometry")
    else:
        dcp_global_lens = local_lens

    expected_partial = (batch, nhead, num_splits, 512)
    expected_lse = (batch, nhead, num_splits)
    if partial is None:
        partial = torch.empty(
            expected_partial, dtype=q.dtype, device=q.device
        )
    elif tuple(partial.shape) != expected_partial or partial.dtype != q.dtype:
        raise ValueError("partial output has the wrong shape or dtype")
    if partial_lse is None:
        partial_lse = torch.empty(
            expected_lse, dtype=torch.float32, device=q.device
        )
    elif tuple(partial_lse.shape) != expected_lse or partial_lse.dtype != torch.float32:
        raise ValueError("partial LSE has the wrong shape or dtype")
    if final_lse is not None and (
        tuple(final_lse.shape) != (batch, nhead)
        or final_lse.dtype != torch.float32
    ):
        raise ValueError("final_lse must be fp32 with q's [B,H] shape")
    _absorbed_mla_stage1_gfx942[(batch, num_splits, triton.cdiv(nhead, 16))](
        q,
        kv,
        bias,
        page_descriptors,
        fixed_indices,
        cache_indices,
        seq_lens,
        exact_token_counts,
        local_lens,
        dcp_global_lens,
        opened_stamps,
        sequence_epochs,
        partial,
        partial_lse,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        page_descriptors.stride(0),
        fixed_indices.stride(0),
        opened_stamps.stride(0),
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        partial_lse.stride(0),
        partial_lse.stride(1),
        partial_lse.stride(2),
        scale,
        NHEAD=nhead,
        NUM_SPLITS=num_splits,
        BLOCK_N=64,
        PAGE_SIZE=1,
        COMPACT_LOD=True,
        HAS_BIAS=True,
        LOCAL_LIMIT=local_limit,
        INCLUDE_NEW=include_new,
        HEAD_TILED_METADATA=head_tiled_metadata,
        STATE_CAPACITY=max(1, state_capacity),
        SINK_LEN=sink_len,
        MASK_OPENED_COARSE=mask_opened_coarse,
        DCP_ROW_MASKED_NEW=dcp_row_masked_new,
        DCP_RANK=dcp_rank,
        DCP_WORLD_SIZE=dcp_world_size,
        DCP_INTERLEAVE_SIZE=dcp_interleave_size,
        num_warps=4,
    )
    if head_tiled_metadata:
        _reduce_head_tiled_mla_splits_kernel[(batch, nhead, 2)](
            partial,
            partial_lse,
            out,
            final_lse if final_lse is not None else out,
            seq_lens,
            cache_indices,
            local_lens,
            dcp_global_lens,
            partial.stride(0),
            partial.stride(1),
            partial.stride(2),
            partial_lse.stride(0),
            partial_lse.stride(1),
            partial_lse.stride(2),
            out.stride(0),
            out.stride(1),
            final_lse.stride(0) if final_lse is not None else 0,
            final_lse.stride(1) if final_lse is not None else 0,
            NUM_SPLITS=num_splits,
            HEAD_DIM=512,
            HEAD_TILES=triton.cdiv(nhead, 16),
            HAS_FINAL_LSE=final_lse is not None,
            ADVANCE_DCP_LENGTHS=dcp_row_masked_new,
            DCP_RANK=dcp_rank,
            DCP_WORLD_SIZE=dcp_world_size,
            DCP_INTERLEAVE_SIZE=dcp_interleave_size,
            BLOCK_S=triton.next_power_of_2(num_splits),
            BLOCK_D=256,
            num_warps=4,
        )
    else:
        _mla_softmax_reducev_kernel[(batch, nhead, 1)](
            partial,
            partial_lse,
            out,
            final_lse,
            seq_lens,
            partial.stride(0),
            0,
            partial.stride(1),
            partial.stride(2),
            partial_lse.stride(0),
            0,
            partial_lse.stride(1),
            partial_lse.stride(2),
            out.stride(0),
            0,
            out.stride(1),
            final_lse.stride(0) if final_lse is not None else 0,
            0,
            final_lse.stride(1) if final_lse is not None else 0,
            NUM_KV_SPLITS=num_splits,
            HEAD_DIM_CKV=512,
            HAS_FINAL_LSE=final_lse is not None,
            USE_2D_VIEW=True,
            BLOCK_S=min(64, triton.next_power_of_2(num_splits)),
            BLOCK_N=64,
            num_warps=8,
        )
    return out
