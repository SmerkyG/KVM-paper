"""Benchmark 64-key route subtiles inside unchanged 128-key CK attention.

Only a private copy of CK and its C++ wrapper is edited. Each existing 128-key
score tile emits one maximum per half, so exact top-eight refinement rescores
512 rather than 1024 centroid scores per query. Coarse softmax/PV is unchanged.
The private specialization is also callable by the opt-in fixture/model
benchmark. It is not enabled by default.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch

from benchmarks.kimi_k3_leaf_replay import timed
from benchmarks.kimi_k3_update_graph import reconstruct_centroids
from lod_attention.kernels import aiter_mla_prefill_attention as attention
from lod_attention.kernels import kimi_route_tile_refine as refine


SCORE_EMITTER = r'''
                    static_assert(kN0 == 128, "subtile routing retains the native key tile");
                    const index_t route_block_capacity = integer_divide_ceil(
                        variant_params.route_seqlen_k, index_t{64});
                    static_for<0, 2, 1>{}([&](auto half) {
                        auto half_s = s;
                        constexpr auto spans = decltype(half_s)::get_distributed_spans();
                        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                                const auto xy = get_x_indices_from_distributed_indices(
                                    half_s.get_tile_distribution(), make_tuple(idx0, idx1));
                                if(xy.at(number<1>{}) / 64 != half)
                                    half_s(make_tuple(idx0, idx1)) =
                                        -numeric<SMPLComputeDataType>::infinity();
                            });
                        });
                        auto maximum = block_tile_reduce<SMPLComputeDataType>(
                            half_s, sequence<1>{}, f_max,
                            -numeric<SMPLComputeDataType>::infinity());
                        block_tile_reduce_sync(maximum, f_max, bool_constant<false>{});
                        auto* route_ptr = reinterpret_cast<float*>(variant_params.route_block_max_ptr) +
                            ((static_cast<long_index_t>(block_indices.batch_idx) *
                                  variant_params.route_num_heads + block_indices.qo_head_idx) *
                                 route_block_capacity + i_total_loops * 2 + half) *
                                    variant_params.route_seqlen_q;
                        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                                const auto xy = get_x_indices_from_distributed_indices(
                                    half_s.get_tile_distribution(), make_tuple(idx0, idx1));
                                const index_t row = q_origin.at(number<0>{}) + xy.at(number<0>{});
                                if(row < variant_params.route_seqlen_q && xy.at(number<1>{}) == 0)
                                    route_ptr[row] = type_convert<float>(maximum(make_tuple(idx0)));
                            });
                        });
                    });
'''


@contextlib.contextmanager
def isolated_subtile_sources(score_only=False, reuse_max=False, tile_n=64):
    import aiter.jit.core as core

    original_root = core.CK_3RDPARTY_DIR
    if tile_n not in (32, 64) or (tile_n == 32 and not score_only):
        raise ValueError("32/64-key subgroup experiments require score-only for 32")
    score_emitter = SCORE_EMITTER
    if tile_n == 32:
        score_emitter = score_emitter.replace('index_t{64}', 'index_t{32}').replace(
            'static_for<0, 2, 1>', 'static_for<0, 4, 1>').replace(
            '/ 64 != half', '/ 32 != half').replace(
            'i_total_loops * 2 + half', 'i_total_loops * 4 + half')
    with tempfile.TemporaryDirectory(prefix="lod-kimi-subtile-") as temporary:
        root = Path(temporary)
        ck = root / "ck"
        shutil.copytree(Path(original_root) / "include", ck / "include", symlinks=True)
        (ck / "library").symlink_to(Path(original_root) / "library", target_is_directory=True)
        header = ck / "include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp"
        text = header.read_text()
        begin = text.index('                    static_assert(kN0 == 128,')
        end = text.index('\n                }\n            }\n            auto m_local', begin)
        emitter = text[begin:end]
        changes = (
            ('variant_params.route_seqlen_k, index_t{128});',
             'variant_params.route_seqlen_k, index_t{64});'),
            ('static_for<0, CK_TILE_FMHA_ROUTE_EMIT_TOPK, 1>{}', 'static_for<0, 2, 1>{}'),
            ('RouteArgmaxPacket{route_s(i_j_idx), global_column};',
             'RouteArgmaxPacket{tile_idx.at(number<1>{}) / 64 == route_rank ? '
             'route_s(i_j_idx) : -numeric<SMPLComputeDataType>::infinity(), global_column};'),
            ('route_rank + output_kind * CK_TILE_FMHA_ROUTE_TOPK;',
             'output_kind * CK_TILE_FMHA_ROUTE_TOPK;'),
            ('i_total_loops) *', '(i_total_loops * 2 + route_rank)) *'),
        )
        for old, new in changes:
            # The global-top-k branch is not compiled for this experiment.
            expected = 2 if old.startswith('route_rank + output_kind') else 1
            if emitter.count(old) != expected:
                raise RuntimeError(f"installed CK subtile emitter changed: {old}")
            emitter = emitter.replace(old, new)
        if score_only:
            emitter = score_emitter
        text = (text[:begin] + '\n#if CK_TILE_LOD_SUBTILE64\n' + emitter
                + '\n#else\n' + text[begin:end] + '\n#endif\n' + text[end:])
        if reuse_max:
            if not score_only:
                raise ValueError("coarse maximum reuse requires the score-only emitter")
            first = text.index('            // The receipt-104 LoD specialization')
            last = text.index('            const auto m_old = m;', first)
            original = text[first:last]
            fused = score_emitter.replace(
                f'                    static_for<0, {128 // tile_n}, 1>{{}}',
                '''                    using MaximumTile = decltype(block_tile_reduce<SMPLComputeDataType>(
                        s, sequence<1>{}, f_max, -numeric<SMPLComputeDataType>::infinity()));
                    MaximumTile combined;
                    set_tile(combined, -numeric<SMPLComputeDataType>::infinity());
                    static_for<0, 2, 1>{}''').replace(
                '                        block_tile_reduce_sync(maximum, f_max, bool_constant<false>{});',
                '''                        block_tile_reduce_sync(maximum, f_max, bool_constant<false>{});
                        tile_elementwise_inout(
                            [](auto& e0, auto e1) { e0 = max(e0, e1); }, combined, maximum);''')
            if tile_n == 32:
                fused = fused.replace('static_for<0, 2, 1>', 'static_for<0, 4, 1>')
            replacement = '''
#if CK_TILE_LOD_SUBTILE64_REUSE_MAX
            auto m_local = [&]() {
                if constexpr(!kHasLogitsSoftCap &&
                             BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                {
                    if(variant_params.route_block_max_ptr != nullptr)
                    {
''' + fused + '''
                        return combined;
                    }
                }
                auto fallback = block_tile_reduce<SMPLComputeDataType>(
                    s, sequence<1>{}, f_max, -numeric<SMPLComputeDataType>::infinity());
                block_tile_reduce_sync(fallback, f_max, bool_constant<false>{});
                return fallback;
            }();
#else
''' + original + '\n#endif\n'
            text = text[:first] + replacement + text[last:]
        header.write_text(text)
        wrapper = Path(core.AITER_CSRC_DIR) / 'py_itfs_ck/mha_fwd_kernels.cu'
        text = wrapper.read_text()
        old = '(head_size_q == 128 || head_size_q == 192) ? 128 : 64;'
        if text.count(old) != 1:
            raise RuntimeError("installed CK route allocation changed")
        # This wrapper is used only by the distinct subtile specialization.
        text = text.replace(old, f'{tile_n};')
        if score_only:
            old = '{batch_size, num_heads, route_blocks, 2 * CK_TILE_FMHA_ROUTE_TOPK, seqlen_q}'
            if text.count(old) != 2:
                raise RuntimeError("installed CK candidate allocation changed")
            text = text.replace(old, '{batch_size, num_heads, route_blocks, 1, seqlen_q}')
        private_wrapper = root / wrapper.name
        private_wrapper.write_text(text)
        sources = [str(private_wrapper) if Path(source).name == wrapper.name else source
                   for source in core.get_args_of_build('module_mha_fwd')['srcs']]
        core.CK_3RDPARTY_DIR = str(ck)
        try:
            yield sources
        finally:
            core.CK_3RDPARTY_DIR = original_root


def subtile_operator_name(score_only=False, reuse_max=False, tile_n=64, query_tile=128):
    """AITER's torch guard caches by Python name, not generated module name."""
    return (f"lod_kimi_subtile_mha_n{tile_n}_q{query_tile}"
            f"_score{int(score_only)}_reuse{int(reuse_max)}_v13")


def subtile_factory(sources, score_only=False, reuse_max=False, tile_n=64, query_tile=128):
    # Reuse the existing typed schema/generator, changing only private sources,
    # an explicit macro and a distinct module name. Ordinary builds remain safe.
    from aiter.jit.core import compile_ops, get_args_of_build
    from aiter.ops.mha import cmdGenFunc_mha_fwd

    def build(*args, **kwargs):
        generated = cmdGenFunc_mha_fwd(*args, **kwargs)
        generated['md_name'] += (f'_lod_kimi_d192v128_asyncbias_subtile{tile_n}_reusemax_probe{2 if tile_n == 32 else 1}_v13'
                                 if reuse_max else f'_lod_kimi_d192v128_asyncbias_subtile{tile_n}_score_probe1_v13'
                                 if score_only else '_lod_kimi_d192v128_asyncbias_subtile64_probe1_v13')
        generated['blob_gen_cmd'] = [command.replace('--receipt 100', '--receipt 104').replace(
            ' --output_dir', ' --optdim 192 --output_dir')
            for command in generated['blob_gen_cmd']]
        if query_tile != 128:
            generated['md_name'] = generated['md_name'].replace('_v13', f'_q{query_tile}_v13')
            wrapper = Path(__file__).with_name('_kimi_coarse_codegen.py')
            commands = []
            for command in generated['blob_gen_cmd']:
                generator, remainder = command.split(' -d fwd', 1)
                commands.append(f'{shlex.quote(str(wrapper))} --base-generator '
                                f'{shlex.quote(generator)} --query-tile {query_tile} '
                                f'--key-step 32 -d fwd{remainder}')
            generated['blob_gen_cmd'] = commands
        generated['srcs'] = sources
        flags = get_args_of_build('module_mha_fwd')['flags_extra_hip']
        generated['flags_extra_hip'] = [flag for flag in flags
                                      if not flag.startswith('-DCK_TILE_FMHA_ROUTE_')] + [
            '-DCK_TILE_FMHA_ROUTE_QUERY_NORMALIZE=0', '-DCK_TILE_FMHA_ROUTE_TOPK=8',
            '-DCK_TILE_FMHA_ROUTE_GLOBAL_TOPK=0', '-DCK_TILE_FMHA_ROUTE_TILE_MAX_ONLY=1',
            '-DCK_TILE_LOD_SUBTILE64=1',
            f'-DCK_TILE_LOD_SUBTILE64_REUSE_MAX={int(reuse_max)}']
        return generated

    def subtile_mha(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        dropout_p: float, softmax_scale: float, is_causal: bool,
        window_size_left: int, window_size_right: int, sink_size: int,
        return_softmax_lse: bool, return_dropout_randval: bool,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        q_descale: Optional[torch.Tensor] = None,
        k_descale: Optional[torch.Tensor] = None,
        v_descale: Optional[torch.Tensor] = None,
        sink_ptr: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    # Distinct generated .so names alone are insufficient: torch_compile_guard
    # silently reuses a registered op when the Python name already exists.
    # In particular, a same-process q128/q64 comparison would otherwise call
    # q128 twice. Include every compile-time variant in the registration name.
    subtile_mha.__name__ = subtile_operator_name(score_only, reuse_max, tile_n, query_tile)
    return compile_ops('module_mha_fwd', fc_name='mha_fwd', gen_func=build)(subtile_mha)


@lru_cache(maxsize=6)
def serving_subtile_factory(score_only=False, reuse_max=False, query_tile=128):
    """Initialize the opt-in fixture candidate once, then call the loaded op.

    The private source tree is required only during the initial JIT build. It
    is cleaned up after that call, and subsequent calls/replays do no source
    copying, allocation or loader reconfiguration. Installed AITER is untouched.
    """
    loaded = None

    def call(*args, **kwargs):
        nonlocal loaded
        if loaded is None:
            with isolated_subtile_sources(score_only, reuse_max) as sources:
                function = subtile_factory(sources, score_only, reuse_max,
                                           query_tile=query_tile)
                output = function(*args, **kwargs)
                loaded = function
                return output
        return loaded(*args, **kwargs)

    return call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--score-only', action='store_true',
                        help='emit only subgroup maxima, without unused key IDs/candidate channels')
    parser.add_argument('--reuse-coarse-max', action='store_true',
                        help='reuse max of the two subgroup maxima for coarse softmax')
    parser.add_argument('--subtile-n', type=int, choices=(32, 64), default=64,
                        help='native attention tile remains 128 keys')
    parser.add_argument('--diagnostic-maxima', action='store_true',
                        help='check one query/head subgroup score vector before timing')
    parser.add_argument('--report-near-ties', action='store_true',
                        help='diagnose changed routes; require top-eight score deficit <=1e-5')
    parser.add_argument('--query-tile', type=int, choices=(64, 128), default=128)
    parser.add_argument('--allow-coarse-roundoff', action='store_true',
                        help='record changed coarse/LSE arithmetic; require relative L2 <=0.001')
    args = parser.parse_args()
    if args.reuse_coarse_max and not args.score_only:
        parser.error('--reuse-coarse-max requires --score-only')
    if args.subtile_n == 32 and not args.score_only:
        parser.error('--subtile-n 32 requires --score-only')
    os.environ['LOD_KIMI_TILE_REFINE'] = '1'
    os.environ['LOD_KIMI_CHUNK_TILE_PACK'] = '1'
    os.environ['LOD_KIMI_TILE_PACK_QUERY_BLOCK'] = '1024'
    payload = torch.load(args.input, map_location='cpu', weights_only=True)
    sums, counts, _ = reconstruct_centroids(payload)
    q = payload['q'].cuda().contiguous()
    keys, weights = sums[None, None].cuda().bfloat16(), counts[None, None].cuda()
    uk, uv = payload['w_uk_t'].cuda(), payload['w_uv'].cuda()
    lengths = weights[..., 0].int()
    buffers = {}

    def run():
        slots, coarse, _, _ = attention.aiter_kimi_expanded_prefill_route_coarse_attention(
            q, keys, keys[..., :512], weights, uk, uv, state_len=sums.size(0),
            scale=payload['scale'], normalize_route_query=False, slot_lengths=lengths,
            max_open_leaf_tokens=1024, buffers=buffers)
        torch.cuda.current_stream().wait_stream(coarse.ready_stream)
        return slots, coarse.output_0, coarse.lse_0, coarse.selected_route_scores

    factory, original_refine = attention._specialized_kimi_coarse_mha_fwd, refine.refine_kimi_centroid_tiles
    result = {'scope': 'trained coarse/route GPU stage, not model latency',
              'source': str(args.input), 'state_len': sums.size(0), 'query_shape': list(q.shape),
              'all_controls_chunk1024': True, 'score_only': args.score_only,
              'reuse_coarse_max': args.reuse_coarse_max, 'subtile_n': args.subtile_n,
              'query_tile': args.query_tile}

    def compare(actual, reference, name):
        if args.allow_coarse_roundoff:
            difference = actual[1].float() - reference[1].float()
            relative = (difference.square().sum()
                        / reference[1].float().square().sum().clamp_min(1e-20)).sqrt().item()
            if relative > 0.001 or not actual[1].isfinite().all():
                raise AssertionError('coarse query geometry exceeds roundoff tolerance')
            torch.testing.assert_close(actual[2], reference[2], atol=1e-4, rtol=0)
            coarse = {'relative_l2': relative, 'maximum_absolute_error': difference.abs().max().item(),
                      'lse_maximum_absolute_error': (actual[2] - reference[2]).abs().max().item()}
        else:
            torch.testing.assert_close(actual[1], reference[1], atol=0, rtol=0)
            torch.testing.assert_close(actual[2], reference[2], atol=0, rtol=0)
            coarse = 'bitwise equal'
        if not args.report_near_ties:
            torch.testing.assert_close(actual[0], reference[0], atol=0, rtol=0)
            torch.testing.assert_close(actual[3], reference[3], atol=0, rtol=0)
            if args.allow_coarse_roundoff:
                result[name] = {'changed_query_heads': 0, 'coarse_output_lse': coarse}
            return
        # This diagnostic does not relax coarse attention. For changed route
        # sets only, recompute every centroid score for that query/head in
        # FP32 and check both selected sets against its true eighth score.
        changed = (actual[0] != reference[0]).any(-1).nonzero()
        details = []
        expanded = buffers['kimi_expanded_coarse_k'].view(q.size(0), -1, q.size(1), 192)
        log_counts = buffers['kimi_coarse_log_counts'].view(q.size(0), -1)
        for batch, head, row in changed.tolist():
            scores = (expanded[batch, :sums.size(0), head].float()
                      @ q[batch, head, row].float()) * payload['scale']
            scores += log_counts[batch, :sums.size(0)].float()
            eighth = scores.topk(8).values[-1]
            deficits = [(eighth - scores[chosen[batch, head, row]].min()).clamp_min(0)
                        for chosen in (actual[0], reference[0])]
            if any(deficit.item() > 1e-5 for deficit in deficits):
                raise AssertionError('changed route set is not an FP32 top-eight near-tie')
            details.append({'batch': batch, 'head': head, 'query': row,
                            'candidate_deficit': deficits[0].item(),
                            'control_deficit': deficits[1].item()})
        equal_slots = actual[0] == reference[0]
        torch.testing.assert_close(actual[3][equal_slots], reference[3][equal_slots],
                                   atol=1e-5, rtol=0)
        result[name] = {'changed_query_heads': len(details), 'details': details,
                        'coarse_output_lse': coarse}

    with torch.inference_mode():
        result['ordinary_before'] = timed(run)
        expected = tuple(t.clone() for t in run())
        original_q, original_keys = q.clone(), keys.clone()
        with isolated_subtile_sources(args.score_only, args.reuse_coarse_max, args.subtile_n) as sources:
            candidate = subtile_factory(sources, args.score_only, args.reuse_coarse_max,
                                        args.subtile_n, args.query_tile)
            diagnostic_done = False
            try:
                attention._specialized_kimi_coarse_mha_fwd = lambda *_a, **_k: candidate
                def refine64(*call_args, **kwargs):
                    nonlocal diagnostic_done
                    if args.diagnostic_maxima and not diagnostic_done:
                        candidates, queries, expanded, log_counts = call_args
                        raw = (expanded[0, :, 0].float() @ queries[0, 0, 0].float())
                        raw = raw * payload['scale'] + log_counts[0].float()
                        padded = torch.nn.functional.pad(
                            raw, (0, candidates.size(2) * args.subtile_n - raw.numel()),
                            value=-float('inf'))
                        expected = padded.view(-1, args.subtile_n).amax(-1) / torch.log(
                            torch.tensor(2.0, device=raw.device))
                        actual = candidates[0, 0, :, 0, 0]
                        print('KIMI_SUBGROUP_MAXIMA ' + json.dumps({
                            'expected': expected.tolist(), 'actual': actual.tolist(),
                            'maximum_absolute_error': (actual - expected).abs().max().item()}),
                            flush=True)
                        diagnostic_done = True
                    kwargs['tile_n'] = args.subtile_n
                    return original_refine(*call_args, **kwargs)

                refine.refine_kimi_centroid_tiles = refine64
                result[f'subtile{args.subtile_n}'] = timed(run)
                compare(run(), expected, 'original_comparison')
                q.mul_(-0.75)
                keys.mul_(1.25)
                fresh = tuple(t.clone() for t in run())
            finally:
                attention._specialized_kimi_coarse_mha_fwd = factory
                refine.refine_kimi_centroid_tiles = original_refine
        compare(fresh, run(), 'changed_input_comparison')
        q.copy_(original_q)
        keys.copy_(original_keys)
        result['ordinary_after'] = timed(run)
    result['fresh_routes_scores_coarse_lse'] = (
        'see explicit comparison diagnostics' if args.report_near_ties or args.allow_coarse_roundoff
        else 'matched exactly')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
