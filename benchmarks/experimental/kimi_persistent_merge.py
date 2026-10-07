"""Single-launch split attention/last-arriver merge experiment for gfx942.

Clone the production Gluon stage-one source, retaining all attention math
and descriptors. Only the epilogue changes. Workspace is owned by one ordered
stream and counters reset after every call; overlapping streams must never
share it. No spin-wait, host split planning, or production monkeypatch.
"""

import functools
import linecache
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@functools.lru_cache(maxsize=1)
def fused_stage():
    from lod_attention.kernels.kimi_gluon_decode import _absorbed_mla_stage1_gfx942
    original = _absorbed_mla_stage1_gfx942.src
    signature, body = original.split("):", 1)
    source = signature.replace("_absorbed_mla_stage1_gfx942", "_persistent_mla_stage") + """
    FusedOut,
    FusedLSE,
    Counters,
    stride_out_b,
    stride_out_h,
    stride_final_lse_b,
    stride_final_lse_h,
):""" + body + """
    # Every producer thread finishes its global stores before one thread
    # performs the device-scope release/acquire atomic. The last arrival
    # observes all previous partial stores; it never waits for unscheduled CTAs.
    gl.barrier()
    arrived = gl.atomic_add(Counters + metadata_row, 1, sem="acq_rel", scope="gpu")
    if arrived == NUM_SPLITS - 1:
        merge_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1, 4], threads_per_warp=[1, 4, 16],
            warps_per_cta=[1, 4, 1], order=[2, 1, 0])
        l_layout: gl.constexpr = gl.SliceLayout(2, merge_layout)
        merge_s = gl.arange(0, NUM_SPLITS, layout=gl.SliceLayout(1, l_layout))
        merge_h = head_base + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(0, l_layout))
        ml = gl.load(PartialLSE + batch * stride_lse_b
                     + merge_h[None, :] * stride_lse_h + merge_s[:, None] * stride_lse_s,
                     mask=merge_h[None, :] < NHEAD, other=-float("inf"))
        mm = gl.max(ml, 0)
        safe_mm = gl.where(mm == -float("inf"), 0.0, mm)
        mw = gl.exp(ml - safe_mm[None, :])
        md = gl.sum(mw, 0)
        mh = gl.arange(0, BLOCK_H, layout=gl.SliceLayout(0, l_layout)) + head_base
        gl.store(FusedLSE + batch * stride_final_lse_b + mh * stride_final_lse_h,
                 mm + gl.log(md), mask=mh < NHEAD)
        hd_layout: gl.constexpr = gl.SliceLayout(0, merge_layout)
        hh = gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, hd_layout)) + head_base
        dd = gl.arange(0, 64, layout=gl.SliceLayout(0, hd_layout))
        value_offsets = batch * stride_partial_b + merge_h[None, :] * stride_partial_h + merge_s[:, None] * stride_partial_s
        value_mask = (merge_h[None, :] < NHEAD) & (ml != -float("inf"))
        output_denominator = gl.convert_layout(gl.where(md == 0.0, 1.0, md), gl.SliceLayout(1, hd_layout))
        for step in range(8):
            dim = step * 64 + dd
            dim_full = gl.convert_layout(dim, gl.SliceLayout(0, gl.SliceLayout(1, merge_layout)))
            pv = gl.load(Partial + value_offsets[:, :, None] + dim_full[None, None, :],
                         mask=value_mask[:, :, None], other=0.0).to(gl.float32)
            ov = gl.sum(pv * mw[:, :, None], 0) / output_denominator[:, None]
            merge_out_dim = gl.convert_layout(dim, gl.SliceLayout(0, hd_layout))
            gl.store(FusedOut + batch * stride_out_b + hh[:, None] * stride_out_h + merge_out_dim[None, :],
                     ov.to(dtype), mask=hh[:, None] < NHEAD)
        gl.barrier()
        gl.atomic_xchg(Counters + metadata_row, 0, sem="release", scope="gpu")
"""
    filename = "<lod-experiment-persistent-mla>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = {"gl": gl}
    exec(compile(source, filename, "exec"), namespace)
    return gluon.jit(namespace["_persistent_mla_stage"])


def persistent_lod(q, kv, bias, out, descriptors, fixed, cache_indices,
                   lengths, exact_counts, local_lens, partial, partial_lse,
                   final_lse, counters, scale, *, local_limit, splits,
                   opened_stamps=None, sequence_epochs=None, state_capacity=0,
                   sink_len=0):
    if splits not in (8, 16, 32, 64):
        raise ValueError("experimental merge requires a power-of-two split count")
    b, h = q.shape[:2]
    head_tiles = triton.cdiv(h, 16)
    if counters.shape != (b * head_tiles,) or counters.dtype != torch.int32:
        raise ValueError("counter shape/dtype does not match the metadata rows")
    has_stamps = opened_stamps is not None
    kernel = fused_stage()[(b, splits, head_tiles)](
        q, kv, bias, descriptors, fixed, cache_indices, lengths, exact_counts,
        local_lens, local_lens, opened_stamps if has_stamps else descriptors,
        sequence_epochs if has_stamps else lengths, partial, partial_lse,
        q.stride(0), q.stride(1), kv.stride(0), descriptors.stride(0), fixed.stride(0),
        opened_stamps.stride(0) if has_stamps else descriptors.stride(0),
        partial.stride(0), partial.stride(1), partial.stride(2),
        partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2), scale,
        NHEAD=h, NUM_SPLITS=splits, BLOCK_N=64, PAGE_SIZE=1, COMPACT_LOD=True,
        HAS_BIAS=True, LOCAL_LIMIT=local_limit, INCLUDE_NEW=False,
        HEAD_TILED_METADATA=True, STATE_CAPACITY=max(1, state_capacity),
        SINK_LEN=sink_len, MASK_OPENED_COARSE=has_stamps, DCP_ROW_MASKED_NEW=False,
        DCP_RANK=0, DCP_WORLD_SIZE=1, DCP_INTERLEAVE_SIZE=1,
        FusedOut=out, FusedLSE=final_lse, Counters=counters,
        stride_out_b=out.stride(0), stride_out_h=out.stride(1),
        stride_final_lse_b=final_lse.stride(0), stride_final_lse_h=final_lse.stride(1),
        num_warps=4)
    persistent_lod.last_kernel = kernel
    return out, final_lse
