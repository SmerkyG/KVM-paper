"""Isolated full-score emission probe on trained K3 prefill records.

This is deliberately benchmark-only. It copies the two modified AITER source
files into a temporary CK tree, never overwrites installed AITER, and uses a
separate JIT module. Its FP32 score matrix is much larger than tile maxima.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import torch
import triton
import triton.language as tl

from benchmarks.kimi_k3_leaf_replay import timed
from benchmarks.kimi_k3_update_graph import reconstruct_centroids
from lod_attention.kernels import aiter_mla_prefill_attention as attention
from lod_attention.kernels import kimi_route_tile_refine as refine
from lod_attention.kernels._paged_common import (
    _pack_route_score_index, _unpack_route_score_index,
)
from lod_attention.kernels.aiter_prefill_attention import _workspace_tensor


SCORE_STORE = r"""
                    auto score_tile = s;
                    constexpr auto score_spans = decltype(score_tile)::get_distributed_spans();
                    const index_t score_keys = integer_divide_ceil(
                        variant_params.route_seqlen_k, index_t{128}) * 128;
                    auto* scores = reinterpret_cast<float*>(variant_params.route_block_max_ptr);
                    sweep_tile_span(score_spans[number<0>{}], [&](auto idx0) {
                        sweep_tile_span(score_spans[number<1>{}], [&](auto idx1) {
                            const auto xy = get_x_indices_from_distributed_indices(
                                score_tile.get_tile_distribution(), make_tuple(idx0, idx1));
                            const index_t row = q_origin.at(number<0>{}) + xy.at(number<0>{});
                            const index_t column = kv_load_start + i_total_loops * kN0 +
                                                     xy.at(number<1>{});
                            if(row < variant_params.route_seqlen_q && column < score_keys)
                            {
                                const long_index_t head = static_cast<long_index_t>(
                                    block_indices.batch_idx) * variant_params.route_num_heads +
                                    block_indices.qo_head_idx;
                                scores[(head * variant_params.route_seqlen_q + row) *
                                       score_keys + column] = type_convert<float>(
                                    score_tile(make_tuple(idx0, idx1)));
                            }
                        });
                    });
"""


@contextlib.contextmanager
def isolated_score_sources(key_major=False):
    import aiter.jit.core as core

    original_root = core.CK_3RDPARTY_DIR
    with tempfile.TemporaryDirectory(prefix="lod-kimi-scores-") as temporary:
        root = Path(temporary)
        ck = root / "ck"
        # Include search must see the private tree before installed CK.
        shutil.copytree(Path(original_root) / "include", ck / "include", symlinks=True)
        (ck / "library").symlink_to(Path(original_root) / "library", target_is_directory=True)
        pipeline = ck / "include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp"
        text = pipeline.read_text()
        begin = text.index("                    static_assert(kN0 == 128,")
        end = text.index("\n                }\n            }\n            auto m_local", begin)
        # Ordinary binaries built while this private tree is active must still
        # execute the original candidate emitter with its original allocation.
        store = SCORE_STORE
        if key_major:
            store = store.replace("(head * variant_params.route_seqlen_q + row) *\n                                       score_keys + column",
                                  "(head * score_keys + column) * variant_params.route_seqlen_q + row")
        text = (text[:begin] + "\n#if CK_TILE_LOD_FULL_SCORES\n" + store
                + "\n#else\n" + text[begin:end] + "\n#endif\n" + text[end:])
        pipeline.write_text(text)
        wrapper = Path(core.AITER_CSRC_DIR) / "py_itfs_ck/mha_fwd_kernels.cu"
        text = wrapper.read_text()
        old = "{batch_size, num_heads, route_blocks, 2 * CK_TILE_FMHA_ROUTE_TOPK, seqlen_q}"
        if text.count(old) != 2:
            raise RuntimeError("installed AITER score allocation changed")
        layout = ("{batch_size, num_heads, route_blocks * 128, seqlen_q}" if key_major
                  else "{batch_size, num_heads, seqlen_q, route_blocks * 128}")
        text = text.replace(old, layout)
        private_wrapper = root / wrapper.name
        private_wrapper.write_text(text)
        build_args = core.get_args_of_build("module_mha_fwd")
        sources = [str(private_wrapper) if Path(source).name == wrapper.name else source
                   for source in build_args["srcs"]]
        if str(private_wrapper) not in sources:
            raise RuntimeError("installed AITER source list does not include the wrapper")
        core.CK_3RDPARTY_DIR = str(ck)
        try:
            yield sources
        finally:
            core.CK_3RDPARTY_DIR = original_root


def fullscore_factory(sources, key_major=False):
    from aiter.jit.core import compile_ops, get_args_of_build
    from aiter.ops.mha import cmdGenFunc_mha_fwd

    def build(*args, **kwargs):
        generated = cmdGenFunc_mha_fwd(*args, **kwargs)
        generated["md_name"] += ("_lod_kimi_d192v128_fullscores_keymajor_probe1" if key_major
                                 else "_lod_kimi_d192v128_fullscores_probe1")
        generated["blob_gen_cmd"] = [
            command.replace("--receipt 100", "--receipt 104").replace(
                " --output_dir", " --optdim 192 --output_dir")
            for command in generated["blob_gen_cmd"]]
        generated["srcs"] = sources
        flags = get_args_of_build("module_mha_fwd")["flags_extra_hip"]
        generated["flags_extra_hip"] = [flag for flag in flags if
                                           not flag.startswith("-DCK_TILE_FMHA_ROUTE_")] + [
            "-DCK_TILE_FMHA_ROUTE_QUERY_NORMALIZE=0",
            "-DCK_TILE_FMHA_ROUTE_TOPK=8",
            "-DCK_TILE_FMHA_ROUTE_GLOBAL_TOPK=0",
            "-DCK_TILE_FMHA_ROUTE_TILE_MAX_ONLY=1",
            "-DCK_TILE_LOD_FULL_SCORES=1",
        ]
        return generated

    @compile_ops("module_mha_fwd", fc_name="mha_fwd", gen_func=build)
    def fullscore_mha(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        dropout_p: float, softmax_scale: float, is_causal: bool,
        window_size_left: int, window_size_right: int, sink_size: int,
        return_softmax_lse: bool, return_dropout_randval: bool,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        q_descale: Optional[torch.Tensor] = None, k_descale: Optional[torch.Tensor] = None,
        v_descale: Optional[torch.Tensor] = None, sink_ptr: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    return fullscore_mha


@triton.jit
def _select_full_scores(scores, output, Q, STRIDE, STATES,
                        BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
                        KEY_MAJOR: tl.constexpr = False):
    head = tl.program_id(0)
    query = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    key = tl.arange(0, BLOCK_N)
    if KEY_MAJOR:
        offset = (head * STRIDE + key[None, :]) * Q + query[:, None]
    else:
        offset = (head * Q + query[:, None]) * STRIDE + key[None, :]
    score = tl.load(scores + offset,
                    mask=(query[:, None] < Q) & (key[None, :] < STATES),
                    other=-float("inf"))
    best = tl.topk(_pack_route_score_index(score, key[None, :]), 8, dim=1)
    value, index = _unpack_route_score_index(best)
    rank = tl.arange(0, 8)
    tl.store(output + (head * 16 + rank[None, :]) * Q + query[:, None],
             value, mask=query[:, None] < Q)
    tl.store(output + (head * 16 + 8 + rank[None, :]) * Q + query[:, None],
             index.to(tl.float32), mask=query[:, None] < Q)


def select_full_scores(scores, q, k, counts, *, state_len, scale, buffers,
                       key_major=False, **kwargs):
    batch, heads, rows, columns = scores.shape
    queries, stride = (columns, rows) if key_major else (rows, columns)
    output = _workspace_tensor(buffers, "score_output_candidates", (batch, heads, 1, 16, queries),
                               dtype=torch.float32, device=scores.device)
    _select_full_scores[(batch * heads, triton.cdiv(queries, 4))](
        scores, output, queries, stride, state_len,
        BLOCK_N=triton.next_power_of_2(stride), BLOCK_M=4, num_warps=4,
        KEY_MAJOR=key_major)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-major", action="store_true")
    args = parser.parse_args()
    os.environ["LOD_KIMI_TILE_REFINE"] = "1"
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    sums, counts, _ = reconstruct_centroids(payload)
    q = payload["q"].cuda().contiguous()
    keys, weights = sums[None, None].cuda().bfloat16(), counts[None, None].cuda()
    lengths = weights[..., 0].int()
    uk, uv = payload["w_uk_t"].cuda(), payload["w_uv"].cuda()
    buffers = {}

    def run():
        slots, coarse, _, _ = attention.aiter_kimi_expanded_prefill_route_coarse_attention(
            q, keys, keys[..., :512], weights, uk, uv,
            state_len=sums.size(0), scale=payload["scale"],
            normalize_route_query=False, slot_lengths=lengths,
            max_open_leaf_tokens=1024, buffers=buffers)
        torch.cuda.current_stream().wait_stream(coarse.ready_stream)
        return slots, coarse.output_0, coarse.lse_0, coarse.selected_route_scores

    def check(candidate, reference):
        torch.testing.assert_close(candidate[1], reference[1], atol=0, rtol=0)
        torch.testing.assert_close(candidate[2], reference[2], atol=0, rtol=0)
        same = candidate[0].eq(reference[0])
        mismatch_rows = candidate[0].sort(-1).values.ne(
            reference[0].sort(-1).values).any(-1)
        # CK and the ordinary Triton rescore use slightly different reduction
        # orders. Report near-tie changes rather than claiming bitwise routing.
        torch.testing.assert_close(candidate[3][same], reference[3][same], atol=1e-5, rtol=1e-5)
        stats = {"coarse_output_lse_bitwise": True,
                 "route_element_mismatches": int((~same).sum()),
                 "route_set_mismatch_queries": int(mismatch_rows.sum()),
                 "query_head_rows": mismatch_rows.numel(),
                 "matching_route_scores_within_1e5": True}
        if int(mismatch_rows.sum()) > mismatch_rows.numel() * 0.0001:
            raise RuntimeError(f"too many full-score route differences: {stats}")
        return stats

    original_factory = attention._specialized_kimi_coarse_mha_fwd
    original_refine = refine.refine_kimi_centroid_tiles
    result = {"scope": "trained-geometry GPU stage, not model latency",
              "query_shape": list(q.shape), "state_len": sums.size(0),
              "score_workspace_bytes": q.size(0) * q.size(1) * q.size(2) *
                                       triton.cdiv(sums.size(0), 128) * 128 * 4}
    result["score_layout"] = "BHNQ" if args.key_major else "BHQN"
    with torch.inference_mode(), isolated_score_sources(args.key_major) as sources:
        kernel = fullscore_factory(sources, args.key_major)
        result["ordinary_before"] = timed(run)
        reference = tuple(t.clone() for t in run())
        try:
            attention._specialized_kimi_coarse_mha_fwd = lambda *args, **kwargs: kernel
            def select(*args_, **kwargs):
                return select_full_scores(*args_, **kwargs, key_major=args.key_major)
            refine.refine_kimi_centroid_tiles = select
            result["full_score_output"] = timed(run)
            result["ordinary_input_check"] = check(run(), reference)
            q.mul_(-0.75)
            keys.mul_(1.25)
            candidate = tuple(t.clone() for t in run())
        finally:
            attention._specialized_kimi_coarse_mha_fwd = original_factory
            refine.refine_kimi_centroid_tiles = original_refine
        result["fresh_input_check"] = check(candidate, run())
        # Restore the original inputs before the final timing control.
        q.copy_(payload["q"])
        keys.copy_(sums[None, None])
        result["ordinary_after"] = timed(run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
