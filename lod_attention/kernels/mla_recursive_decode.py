"""Absorbed-MLA coarse/local baseline for three-tier decode.

Reuse the fast 16-head Gluon decoder over a fixed prefix, with no leaf-list
construction. Page refinement and disjoint coarse replacement are shared with
the other model families. Persistent records remain latent512 [+ direct64].
"""

import triton
import triton.language as tl

from .kimi_gluon_decode import (
    KIMI_GLUON_LOD_SPLITS, absorbed_mla_lod_decode_gfx942, kimi_lod_decode_splits,
)


@triton.jit
def _prepare_prefix(cache_indices, local_lens, state_lens, global_lens,
                    new_k, local_k, context_lens, exact_counts,
                    NK_B: tl.constexpr, LK_B: tl.constexpr, LK_T: tl.constexpr,
                    HEAD_DIM: tl.constexpr, HEAD_TILES: tl.constexpr,
                    STATE_LEN: tl.constexpr, LOCAL_LIMIT: tl.constexpr,
                    SINK_LEN: tl.constexpr, INCLUDE_NEW: tl.constexpr,
                    USE_STATE_LENS: tl.constexpr, DCP: tl.constexpr,
                    RANK: tl.constexpr, WORLD: tl.constexpr, INTERLEAVE: tl.constexpr,
                    BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    cache_row = tl.load(cache_indices + row).to(tl.int64)
    local_len = tl.minimum(tl.load(local_lens + cache_row), LOCAL_LIMIT)
    state_len = STATE_LEN
    if USE_STATE_LENS:
        state_len = tl.minimum(tl.load(state_lens + cache_row), STATE_LEN)
    owns_new = INCLUDE_NEW
    if DCP:
        global_len = tl.load(global_lens + cache_row)
        owns_new = owns_new & ((global_len // INTERLEAVE) % WORLD == RANK)
    tile = tl.arange(0, triton.next_power_of_2(HEAD_TILES))
    tl.store(context_lens + row * HEAD_TILES + tile,
             local_len + owns_new + SINK_LEN + state_len, mask=tile < HEAD_TILES)
    tl.store(exact_counts + row * HEAD_TILES + tile, 0, mask=tile < HEAD_TILES)
    if owns_new:
        channel = tl.arange(0, BLOCK_D)
        record = tl.load(new_k + row * NK_B + channel, mask=channel < HEAD_DIM, other=0)
        # K and V alias the same latent record; copy it once, including K3's
        # direct-key tail. Virtual head tiles must never write separate rows.
        tl.store(local_k + cache_row * LK_B + local_len * LK_T + channel,
                 record, mask=channel < HEAD_DIM)


def recursive_mla_baseline(q, local_k, new_k, cache_indices, local_lens,
                           state_lens, buffers, arena, bias, fixed_indices, *,
                           state_len, local_limit, sink_len, scale, include_new,
                           dcp_global_lens=None, dcp_rank=0, dcp_world_size=1,
                           dcp_interleave_size=1):
    """Write output/LSE into fixed scratch; safe for CUDA graph replay."""
    batch, heads, _, dim = q.shape
    tiles = triton.cdiv(heads, 16)
    sequences = batch * tiles
    lengths = buffers["gqa_union_hip_context_lens"][:sequences]
    exact_counts = buffers["gqa_union_token_counts"][:sequences]
    _prepare_prefix[(batch,)](
        cache_indices, local_lens, state_lens if state_lens is not None else local_lens,
        dcp_global_lens if dcp_global_lens is not None else local_lens,
        new_k, local_k, lengths, exact_counts,
        NK_B=new_k.stride(0), LK_B=local_k.stride(0), LK_T=local_k.stride(2),
        HEAD_DIM=dim, HEAD_TILES=tiles, STATE_LEN=state_len, LOCAL_LIMIT=local_limit,
        SINK_LEN=sink_len, INCLUDE_NEW=include_new,
        USE_STATE_LENS=state_lens is not None, DCP=dcp_global_lens is not None,
        RANK=dcp_rank, WORLD=dcp_world_size, INTERLEAVE=dcp_interleave_size,
        BLOCK_D=triton.next_power_of_2(dim), num_warps=4,
    )
    splits = kimi_lod_decode_splits(batch, head_tiled_metadata=True)
    partial = buffers["kimi_gluon_partial"][:sequences].view(
        batch, heads, KIMI_GLUON_LOD_SPLITS, 512)[:, :, :splits]
    partial_lse = buffers["kimi_gluon_partial_lse"][:sequences].view(
        batch, heads, KIMI_GLUON_LOD_SPLITS)[:, :, :splits]
    absorbed_mla_lod_decode_gfx942(
        q[:, :, 0], arena, bias, buffers["coarse_out"].view(batch, heads, 512),
        buffers["gqa_union_token_indices"][:sequences],
        fixed_indices.view(fixed_indices.size(0), -1), cache_indices,
        lengths, exact_counts, local_lens, scale,
        local_limit=local_limit, sink_len=sink_len, head_tiled_metadata=True,
        include_new=include_new, dcp_global_lens=dcp_global_lens,
        dcp_rank=dcp_rank, dcp_world_size=dcp_world_size,
        dcp_interleave_size=dcp_interleave_size, num_splits=splits,
        partial=partial, partial_lse=partial_lse,
        final_lse=buffers["coarse_lse"].view(batch, heads),
        opened_slots=buffers["route_top_slots"], state_capacity=state_len,
    )
