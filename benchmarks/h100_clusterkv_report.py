"""Publish the matched H100 ClusterKV timings and first-response RULER scores."""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics

from benchmarks.h100_threshold_next_report import TASKS, score_files


THREE_TIER_NAME = 'LoD three-tier, c=16 threshold decode'
THREE_TIER_KV = {'retro-lod-bf16': 'BF16', 'retro-lod-int4': 'INT4'}
ADAPTIVE_PAGE = 'lod-three-c16-page05-bf16'
ADAPTIVE_NAME = 'LoD three-tier, region c=16, adaptive page c=0.5'


def check_three_tier_settings(data, *, adaptive=False):
    threshold = data['decode_threshold']
    assert {key: threshold[key] for key in ('implementation', 'rule', 'multiplier', 'three_tier_pages',
                                             'page_threshold_multiplier')} == {
        'implementation': 'three-tier', 'rule': 'region', 'multiplier': 16.0,
        'three_tier_pages': None if adaptive else 2,
        'page_threshold_multiplier': 0.5 if adaptive else None,
    }
    assert data['dtype'] == 'bfloat16' and data['prefill_threshold'] is None
    if adaptive:
        assert data['method'] == 'retro-lod-bf16'
        expected = {'page_threshold_rule': 'page', 'page_query': 'normalized', 'page_residual': 'region'}
        assert {key: threshold[key] for key in expected} == expected
    return THREE_TIER_KV[data['method']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--timing', nargs=6, required=True, metavar='FILE')
    parser.add_argument('--optimized-lod', nargs=8, metavar='FILE',
                        help='Batch-one then batch-eight timings, each in two-tier BF16, fixed BF16, '
                             'fixed INT4 and adaptive BF16 order')
    parser.add_argument('--optimized-lod-b1-controls', nargs=4, metavar='FILE',
                        help='Batch-one controls in the same LoD method order as --optimized-lod')
    parser.add_argument('--graph-ab', nargs=4, metavar='FILE',
                        help='Matched dense eager/graph followed by LoD eager/graph')
    parser.add_argument('--graph-panel', nargs=7, metavar='FILE',
                        help='ClusterKV graph, device eager, author dense graph, FP16 dense graph, '
                             'BF16 dense graph, LoD graph, and ClusterKV same-cache check')
    parser.add_argument('--retro-graph', nargs=2, metavar='FILE',
                        help='Matched GPU-only RetroInfer graph timing and same-cache check')
    parser.add_argument('--three-tier-graph', nargs=2, metavar='FILE',
                        help='Matched batch-one three-tier c=16 BF16 and INT4 graph timings')
    parser.add_argument('--three-tier-fixture', metavar='FILE',
                        help='Three-tier graph check with distinct requests and cache lengths')
    parser.add_argument('--adaptive-page-fixture', metavar='FILE',
                        help='Adaptive page graph check with distinct requests and cache lengths')
    parser.add_argument('--batch-eight-precision-audit', nargs=2, metavar='FILE',
                        help='Matched fixed-two-page batch-eight BF16 and INT4 timings')
    parser.add_argument('--batch-eight-adaptive-retime', metavar='FILE',
                        help='Optimized batch-eight timing with the same adaptive page rule')
    parser.add_argument('--adaptive-page-speedup', nargs=6, metavar='FILE',
                        help='Adaptive control, optimized timing, graph check, fixture and two multikey-3 runs')
    parser.add_argument('--adaptive-default-ruler', metavar='FILE',
                        help='Complete RULER panel using the adaptive H100 default')
    parser.add_argument('--adaptive-dense-prefill', metavar='FILE',
                        help='Complete adaptive BF16 RULER panel with dense prefill')
    parser.add_argument('--adaptive-default-checks', nargs=2, metavar='FILE',
                        help='Batch-one then batch-eight graph checks of the adaptive H100 default')
    parser.add_argument('--adaptive-page-profile', metavar='FILE', help='Kernel summary for optimized adaptive pages')
    parser.add_argument('--adaptive-page-repeat', metavar='FILE', help='Repeat of the optimized adaptive timing')
    parser.add_argument('--batch-eight-page-profiles', nargs=3, metavar='FILE',
                        help='Kernel summaries for fixed BF16, fixed INT4 and adaptive BF16')
    parser.add_argument('--batch-eight-retro-retime', metavar='FILE',
                        help='Fresh matching RetroInfer sparse control for the precision audit')
    parser.add_argument('--batch-eight-fixed-retime', nargs=3, metavar='FILE',
                        help='Optimized fixed-two-page timing, model graph check and independent-request fixture')
    parser.add_argument('--batch-eight-fixed-repeat', metavar='FILE',
                        help='Repeat of the selected fixed-two-page timing')
    parser.add_argument('--batch-eight-two-tier-port', nargs=4, metavar='FILE',
                        help='Fresh two-tier control, optimized timing, model graph check and independent-request fixture')
    parser.add_argument('--two-tier-port-quality', nargs=2, metavar='FILE',
                        help='Matched reference and optimized multikey-3 runs')
    parser.add_argument('--two-tier-port-repeat', metavar='FILE',
                        help='Repeat of the optimized two-tier timing')
    parser.add_argument('--batch-eight-int4-retime', nargs=6, metavar='FILE',
                        help='INT4 control, optimized timing, graph check, fixture and two multikey-3 runs')
    parser.add_argument('--int4-retime-repeat', metavar='FILE', help='Repeat of the optimized INT4 timing')
    parser.add_argument('--int4-retime-profile', metavar='FILE', help='Kernel summary for the optimized INT4 path')
    parser.add_argument('--int4-default-checks', nargs=3, metavar='FILE',
                        help='Default INT4 timing, its graph check and the opt-out graph check')
    parser.add_argument('--bf16-defaults', nargs=4, metavar='FILE',
                        help='Default two-tier timing and check, then default fixed-page BF16 timing and check')
    parser.add_argument('--fixed-multikey', metavar='FILE',
                        help='Targeted batch-one RULER multikey-3 regression check for the optimized decoder')
    parser.add_argument('--fixed-multikey-controls', nargs=3, metavar='FILE',
                        help='Current reference kernel at original and matched geometry, then specialized page kernel alone')
    parser.add_argument('--fixed-multikey-selected', metavar='FILE',
                        help='Full batch-one multikey-3 rerun with the selected batch-eight kernel settings')
    parser.add_argument('--batch-eight', nargs='+', metavar='FILE',
                        help='Matched batch-eight timing files in table order, using c=16 for LoD')
    parser.add_argument('--batch-eight-checks', nargs='+', metavar='FILE',
                        help='Same-cache checks for the batch-eight sparse methods')
    parser.add_argument('--quality', help='Completed ClusterKV quality JSON')
    parser.add_argument('--quality-control', help='Completed FP16 dense quality JSON')
    parser.add_argument('--lod-dense-prefill', help='Completed BF16 LoD quality JSON with dense prefill')
    args = parser.parse_args()
    if args.optimized_lod_b1_controls and not args.optimized_lod:
        parser.error('--optimized-lod-b1-controls requires --optimized-lod')
    if (args.retro_graph or args.three_tier_graph) and not args.graph_panel:
        parser.error('--retro-graph and --three-tier-graph require --graph-panel')
    if (args.three_tier_fixture or args.adaptive_page_fixture) and not args.batch_eight:
        parser.error('both three-tier fixture options require --batch-eight')
    if args.batch_eight_precision_audit and not args.batch_eight:
        parser.error('--batch-eight-precision-audit requires --batch-eight')
    if args.batch_eight_adaptive_retime and not args.batch_eight:
        parser.error('--batch-eight-adaptive-retime requires --batch-eight')
    if args.adaptive_page_speedup and not args.batch_eight:
        parser.error('--adaptive-page-speedup requires --batch-eight')
    if (args.adaptive_page_profile or args.adaptive_page_repeat) and not args.adaptive_page_speedup:
        parser.error('--adaptive-page-profile and --adaptive-page-repeat require --adaptive-page-speedup')
    if args.batch_eight_retro_retime and not args.batch_eight_precision_audit:
        parser.error('--batch-eight-retro-retime requires --batch-eight-precision-audit')
    if args.batch_eight_fixed_retime and not args.batch_eight_precision_audit:
        parser.error('--batch-eight-fixed-retime requires --batch-eight-precision-audit')
    if args.batch_eight_fixed_repeat and not args.batch_eight_fixed_retime:
        parser.error('--batch-eight-fixed-repeat requires --batch-eight-fixed-retime')
    if args.batch_eight_two_tier_port and not args.batch_eight:
        parser.error('--batch-eight-two-tier-port requires --batch-eight')
    if args.batch_eight_int4_retime and not args.batch_eight:
        parser.error('--batch-eight-int4-retime requires --batch-eight')
    if args.int4_retime_repeat and not args.batch_eight_int4_retime:
        parser.error('--int4-retime-repeat requires --batch-eight-int4-retime')
    if args.int4_retime_profile and not args.batch_eight_int4_retime:
        parser.error('--int4-retime-profile requires --batch-eight-int4-retime')
    if args.int4_default_checks and not args.batch_eight_int4_retime:
        parser.error('--int4-default-checks requires --batch-eight-int4-retime')
    if args.bf16_defaults and not (args.batch_eight_two_tier_port and args.batch_eight_fixed_retime):
        parser.error('--bf16-defaults requires both optimized BF16 reference runs')
    if (args.two_tier_port_quality or args.two_tier_port_repeat) and not args.batch_eight_two_tier_port:
        parser.error('--two-tier-port-quality and --two-tier-port-repeat require --batch-eight-two-tier-port')
    if args.fixed_multikey and not args.batch_eight_fixed_retime:
        parser.error('--fixed-multikey requires --batch-eight-fixed-retime')
    if (args.fixed_multikey_controls or args.fixed_multikey_selected) and not args.fixed_multikey:
        parser.error('the multikey controls and selected rerun require --fixed-multikey')
    if args.adaptive_default_ruler and not args.quality:
        parser.error('--adaptive-default-ruler requires --quality')
    if args.adaptive_dense_prefill and not args.adaptive_default_ruler:
        parser.error('--adaptive-dense-prefill requires --adaptive-default-ruler')
    if args.adaptive_default_checks and not args.adaptive_page_speedup:
        parser.error('--adaptive-default-checks requires --adaptive-page-speedup')
    if args.lod_dense_prefill and not args.quality:
        parser.error('--lod-dense-prefill requires --quality')
    root = args.directory
    prior = root.parent / 'ruler-32k-panel'
    timing = [json.loads((root / f).read_text()) for f in args.timing]
    provenance = {}
    for path in (root / 'provenance').glob('*/run.json'):
        run = json.loads(path.read_text())
        if run.get('exit_code') == 0 and '--output' in run['command']:
            output = run['command'][run['command'].index('--output') + 1]
            provenance[Path(output).name] = (path.parent, run['hardware'], run['command'])
    files = (args.timing + (args.graph_ab or []) + (args.graph_panel or []) + (args.retro_graph or [])
             + (args.three_tier_graph or []) + (args.optimized_lod or [])
             + (args.optimized_lod_b1_controls or [])
             + (args.batch_eight or []) + (args.batch_eight_checks or [])
             + (args.batch_eight_precision_audit or [])
             + (args.batch_eight_fixed_retime or [])
             + (args.batch_eight_two_tier_port or [])
             + (args.batch_eight_int4_retime or [])
             + (args.int4_default_checks or [])
             + (args.bf16_defaults or [])
             + (args.adaptive_page_speedup or [])
             + (args.adaptive_default_checks or [])
             + (args.two_tier_port_quality or [])
             + (args.fixed_multikey_controls or [])
             + [f for f in (args.batch_eight_adaptive_retime, args.batch_eight_retro_retime,
                            args.batch_eight_fixed_repeat, args.fixed_multikey, args.fixed_multikey_selected,
                            args.two_tier_port_repeat, args.int4_retime_repeat, args.adaptive_page_repeat) if f]
             + [f for f in (args.three_tier_fixture, args.adaptive_page_fixture) if f]
             + [f for f in (args.quality, args.quality_control, args.lod_dense_prefill, args.adaptive_default_ruler,
                            args.adaptive_dense_prefill) if f])
    assert len({provenance[f][1] for f in files if f != args.adaptive_dense_prefill}) == 1
    if args.adaptive_dense_prefill:
        dense_gpu = provenance[args.adaptive_dense_prefill][1].split(', ')
        sparse_gpu = provenance[args.adaptive_default_ruler][1].split(', ')
        assert (dense_gpu[0], dense_gpu[2:]) == (sparse_gpu[0], sparse_gpu[2:])
    reference = timing[0]
    assert reference['prompt_tokens'] == 32768 and reference['timed_decode_steps'] == 1024
    assert reference['batch_size'] == 1 and reference['dtype'] == 'float16'
    assert {(d['method'], d['dtype']) for d in timing} == {
        ('retro-clusterkv', 'float16'), ('retro-clusterkv-full', 'float16'),
        ('retro-full', 'float16'), ('retro-full', 'bfloat16'), ('retro', 'bfloat16'),
        ('retro-lod-bf16', 'bfloat16')}
    for data in timing:
        for key in ('model', 'checkpoint', 'batch_size', 'manifest_sha256', 'prompt_token_sha256',
                    'timed_decode_steps', 'decode_warmup_steps', 'prefill_mlp_chunk'):
            assert data[key] == reference[key], key
        assert data['replay_prompt_tokens'] and not any(data[k] for k in (
            'lod_cuda_graph', 'native_cuda_graph', 'multipole_cuda_graph', 'quest_cuda_graph', 'squeezed_cuda_graph'))
        assert len([r for r in data['records'] if not r['warmup']]) == 2

    lines = [
        '# ClusterKV on H100',
        '',
        'Llama-3.1-8B-Instruct, 32,768-token ProLong prefix, '
        'native Llama 3.1 RoPE, and the same RetroInfer model runtime. '
        'Decode uses 256 untimed steps and 1,024 timed teacher-forced steps. Each row is the median of two repetitions after '
        'one warmup. The CUDA graph table uses device-resident planning for ClusterKV. '
        'The earlier eager table uses ClusterKV’s released planner, which reads GPU metadata on the CPU.',
        '',
        'ClusterKV uses its released cosine K-means, retrieval, and paged-attention kernels with '
        '400 initial centroids per KV head, a 1,024-token budget, two dense layers, 16 sink tokens, '
        'and a 320-token recent window. The 1,024-token budget covers 1,023 retrieved entries '
        'and the current token. It adds eight centroids per window update, following the released '
        'benchmark default. The paper describes four. Full KV stays '
        'on the GPU using the release’s resident path. The adapter supplies native Llama 3.1 '
        'RoPE. The CUDA port corrects centroid counts, sorts all live centroids, fixes '
        'the block-scan carry, and maps each query head’s selected indices to its shared KV head. '
        'The adapter gives the current token its own final page and indexes a completed window '
        'after its last exact attention step. '
        '[Paper](https://arxiv.org/html/2412.03213v1#S5), '
        '[author source](https://github.com/sjtu-zhao-lab/ClusterKV/tree/c7380819588771e8c5dadacdbd33942d7c53adea).',
        '',
        'Prefill includes index construction and any deferred cache completion. Model loading '
        'and initial cache allocation are outside the timing windows. The decode window '
        'contains four ClusterKV updates and one RetroInfer index rebuild. '
        'Peak allocated memory is the PyTorch allocator peak, including model '
        'weights. The CUDA extension’s own allocations are outside this memory counter.',
    ]
    if args.optimized_lod:
        lines += optimized_lod_lines(root, args.optimized_lod, reference, provenance,
                                     args.optimized_lod_b1_controls)
    lines += [
        '',
        '## Batch-one eager comparison',
        '',
        '| Decoder | Model and KV precision | Prefill (s) | Decode (ms/token) | Peak allocated (GiB) |',
        '|---|---|---:|---:|---:|',
    ]
    names = {
        'retro-clusterkv': 'ClusterKV',
        'retro-clusterkv-full': 'Dense, ClusterKV attention kernels',
        'retro-full': 'Dense, RetroInfer attention kernels',
        'retro': 'RetroInfer GPU-only',
        'retro-lod-bf16': 'LoD two-tier, c=16 threshold decode',
    }
    for file, data in zip(args.timing, timing, strict=True):
        peak = max(r['memory_bytes']['peak_allocated'] for r in data['records'] if not r['warmup']) / 2**30
        if data['method'] == 'retro-clusterkv':
            assert all(r['clusterkv_updates_total'] == 4 and r['clusterkv_centroids_end'] == 432
                       for r in data['records'])
        if data['method'] == 'retro':
            assert all(r['retro_index_updates_timed'] == 1 for r in data['records'])
        precision = 'BF16' if data['dtype'] == 'bfloat16' else 'FP16'
        lines.append(f'| [{names[data["method"]]}]({file}) | {precision} | {data["median_prefill_s"]:.3f} | '
                     f'{data["median_decode_ms_per_token"]:.3f} | {peak:.2f} |')

    clusterkv_decode = reference['median_decode_ms_per_token']
    native_fp16 = next(d for d in timing if (d['method'], d['dtype']) == ('retro-full', 'float16'))
    author_dense = next(d for d in timing if d['method'] == 'retro-clusterkv-full')
    native_slowdown = 100 * (clusterkv_decode / native_fp16['median_decode_ms_per_token'] - 1)
    author_slowdown = 100 * (clusterkv_decode / author_dense['median_decode_ms_per_token'] - 1)
    lines += ['', f'ClusterKV decode takes {native_slowdown:.1f}% more time per token than the '
              f'FP16 RetroInfer dense control and {author_slowdown:.1f}% more than its own FP16 dense path.',
              '',
              'The FP16 dense controls match ClusterKV’s precision. The BF16 dense control '
              'matches LoD’s precision. LoD uses materialized threshold decode with top-eight prefill.']
    dense_fp16 = {d['method']: d for d in timing if d['dtype'] == 'float16' and d['method'] != 'retro-clusterkv'}
    a, b = dense_fp16['retro-clusterkv-full']['records'], dense_fp16['retro-full']['records']
    assert all(x['output_token_ids'] == y['output_token_ids'] for x, y in zip(a, b, strict=True))
    _, first, second = reference['records']
    differences = sum(x != y for x, y in zip(first['output_token_ids'], second['output_token_ids'], strict=True))
    lines += ['', 'The two FP16 dense implementations chose identical next tokens at every '
              f'step in all three timing repetitions. ClusterKV’s two measured repetitions '
              f'differed on {differences} of {len(first["output_token_ids"]):,} predicted tokens '
              'with the same seed and teacher-forced input. Centroid accumulation uses FP16 atomics.']

    if args.graph_ab:
        lines += ['', '## Matched CUDA graph retime', '',
                  'Both BF16 paths use the same prompt, H100, and timing window as the eager table. '
                  'Each graph captures the full model decode step. LoD’s cache catch-up runs '
                  'outside the graph and inside the timing window.', '',
                  '| Method | Eager (ms/token) | CUDA graph (ms/token) | Latency reduction |',
                  '|---|---:|---:|---:|']
        for name, method, eager_file, graph_file in (
            ('Dense BF16', 'retro-full', *args.graph_ab[:2]),
            ('LoD two-tier BF16, c=16 threshold decode', 'retro-lod-bf16', *args.graph_ab[2:]),
        ):
            eager, graph = [json.loads((root / path).read_text()) for path in (eager_file, graph_file)]
            for data in (eager, graph):
                assert data['method'] == method and data['dtype'] == 'bfloat16'
                assert data['replay_prompt_tokens']
                assert len([r for r in data['records'] if not r['warmup']]) == 2
                for key in ('model', 'checkpoint', 'batch_size', 'prompt_token_sha256',
                            'timed_decode_steps', 'decode_warmup_steps', 'prefill_mlp_chunk'):
                    assert data[key] == reference[key], key
            assert not eager['lod_cuda_graph'] and not eager['native_cuda_graph']
            assert graph['lod_cuda_graph'] == (method == 'retro-lod-bf16')
            assert graph['native_cuda_graph'] == (method == 'retro-full')
            if method == 'retro-lod-bf16':
                assert eager['decode_threshold'] == graph['decode_threshold']
                assert all(r['lod_layer_catch_up_events_timed'] == 128
                           for data in (eager, graph) for r in data['records'])
                assert all(e['output_token_ids'] == g['output_token_ids']
                           for e, g in zip(eager['records'], graph['records'], strict=True))
            before, after = [d['median_decode_ms_per_token'] for d in (eager, graph)]
            lines.append(f'| {name} | [{before:.3f}]({eager_file}) | [{after:.3f}]({graph_file}) '
                         f'| {100 * (1 - after / before):.1f}% |')
        lines += ['', 'Both LoD paths include 128 layer cache updates per repetition and '
                  'choose identical next tokens on the teacher-forced sequence. '
                  'LoD’s eager path submits five attention kernels per layer per token, '
                  'including proxy scoring, selection, index packing, exact attention and the merge. '
                  'Whole-model graph replay avoids issuing those launches individually from Python. '
                  'The batch-one eager comparison uses eager execution for every arm.']

    if args.graph_panel:
        *graph_files, check_file = args.graph_panel
        arms = [
            ('ClusterKV', 'retro-clusterkv', 'float16', 'clusterkv_cuda_graph'),
            ('ClusterKV, device planning in eager mode', 'retro-clusterkv', 'float16', None),
            ('Dense, ClusterKV attention kernels', 'retro-clusterkv-full', 'float16', 'clusterkv_cuda_graph'),
            ('Dense, RetroInfer attention kernels', 'retro-full', 'float16', 'native_cuda_graph'),
            ('Dense, RetroInfer attention kernels', 'retro-full', 'bfloat16', 'native_cuda_graph'),
            ('LoD two-tier, c=16 threshold decode', 'retro-lod-bf16', 'bfloat16', 'lod_cuda_graph'),
        ]
        if args.retro_graph:
            retro_file, retro_check_file = args.retro_graph
            graph_files.insert(-1, retro_file)
            arms.insert(-1, ('RetroInfer GPU-only sparse decode', 'retro', 'bfloat16', 'native_cuda_graph'))
            retro_check = json.loads((root / retro_check_file).read_text())
            assert retro_check['native_cuda_graph'] and retro_check['method'] == 'retro'
            for key in ('model', 'checkpoint', 'batch_size', 'manifest_sha256', 'prompt_token_sha256',
                        'timed_decode_steps', 'decode_warmup_steps', 'prefill_mlp_chunk'):
                assert retro_check[key] == reference[key], key
            retro_record, = retro_check['records']
            assert retro_record['retro_index_updates_timed'] == 1
            retro_checks = retro_record['graph_same_cache_checks']
            assert {'1023', '1024', '1025'} <= retro_checks.keys()
            assert all(r['max_abs'] == 0 and r['graph_argmax'] == r['eager_argmax']
                       for r in retro_checks.values())
        if args.three_tier_graph:
            graph_files.extend(args.three_tier_graph)
            arms.extend((THREE_TIER_NAME, method, 'bfloat16', 'lod_cuda_graph') for method in THREE_TIER_KV)
        if args.optimized_lod:
            graph_files.append(args.optimized_lod[3])
            arms.append((ADAPTIVE_NAME, 'retro-lod-bf16', 'bfloat16', 'lod_cuda_graph'))
        check = json.loads((root / check_file).read_text())
        assert check['clusterkv_cuda_graph'] and check['prompt_token_sha256'] == reference['prompt_token_sha256']
        checks = check['records'][-1]['graph_same_cache_checks']
        assert set(map(str, (318, 319, 320, 638, 639, 640))) <= checks.keys()
        assert all(r['max_abs'] == 0 and r['graph_argmax'] == r['eager_argmax'] for r in checks.values())
        lines += ['', '## Whole-model CUDA graph comparison', '',
                  'Same 32K prompt, H100, batch size of one, and 1,024-token timed window as the eager table. '
                  'The new ClusterKV path computes split offsets and the current KV page on the GPU '
                  'using the position, prompt length, and retrieval budget. It calls the same '
                  'centroid selector and split-KV attention kernels with persistent metadata. '
                  'It uses fixed split counts for dense and sparse layers. '
                  'The full-model graph covers KV append, retrieval, attention, and next-token '
                  'selection. Window clustering runs between replays and is included in timed decode. '
                  'Each ClusterKV repetition contains four updates. Initial workspace setup '
                  'and initial graph capture are outside timed decode.', '',
                  '| Method | Precision | Execution | Prefill (s) | Decode (ms/token) |',
                  '|---|---|---|---:|---:|']
        for file, (name, method, dtype, graph_flag) in zip(graph_files, arms, strict=True):
            data = json.loads((root / file).read_text())
            assert (data['method'], data['dtype']) == (method, dtype)
            for key in ('model', 'checkpoint', 'batch_size', 'manifest_sha256', 'prompt_token_sha256',
                        'timed_decode_steps', 'decode_warmup_steps', 'prefill_mlp_chunk'):
                assert data[key] == reference[key], key
            enabled = {flag for flag in ('clusterkv_cuda_graph', 'native_cuda_graph', 'lod_cuda_graph') if data[flag]}
            assert enabled == ({graph_flag} if graph_flag else set())
            assert data['replay_prompt_tokens']
            assert len([r for r in data['records'] if not r['warmup']]) == 2
            if method.startswith('retro-clusterkv'):
                assert data['clusterkv_device_metadata']
            if method == 'retro-clusterkv':
                assert all(r['clusterkv_updates_timed'] == 4 for r in data['records'])
            if method == 'retro':
                assert all(r['retro_index_updates_timed'] == 1 for r in data['records'])
            if method in THREE_TIER_KV:
                assert all(r['lod_layer_catch_up_events_timed'] == 128 for r in data['records'])
            precision = 'BF16' if dtype == 'bfloat16' else 'FP16'
            if name in (THREE_TIER_NAME, ADAPTIVE_NAME):
                kv = check_three_tier_settings(data, adaptive=name == ADAPTIVE_NAME)
                assert data['prefill_includes_deferred_cache_completion'] and len(data['records']) == 3
                if kv == 'INT4':
                    precision = 'BF16 weights / INT4 KV'
            execution = 'CUDA graph' if graph_flag else 'Eager'
            lines.append(f'| [{name}]({file}) | {precision} | {execution} | {data["median_prefill_s"]:.3f} '
                         f'| {data["median_decode_ms_per_token"]:.3f} |')
        if args.retro_graph:
            lines += ['', 'RetroInfer uses dense prefill and its GPU-only sparse decoder '
                      'with 1.8% exact cluster retrieval and 23.2% cluster estimation. '
                      'Every timed repetition includes one index rebuild and the ensuing graph recapture. '
                      f'The [same-cache check]({retro_check_file}) matches eager logits at '
                      f'{len(retro_checks)} sampled steps around and between updates.']
        graphed = json.loads((root / graph_files[0]).read_text())
        release_tokens = reference['records'][1]['output_token_ids']
        graph_tokens = graphed['records'][1]['output_token_ids']
        disagreements = sum(a != b for a, b in zip(release_tokens, graph_tokens, strict=True))
        lines += ['', f'The first measured graph repetition differs from the earlier released-planner '
                  f'repetition on {disagreements} of {len(graph_tokens):,} predicted tokens. '
                  'The split geometry differs, and prefill centroid sums use FP16 atomics.']
        lines += ['', f'[Same-cache verification]({check_file}) found identical graph and '
                  f'eager logits at {len(checks)} sampled steps, including both sides of two '
                  'window boundaries. The [synthetic fixture](provenance/'
                  '1794547874683084330-clusterkv-h100-graph-fixture-v1/output.log) checks the '
                  'released planner, exact attention and retrieval indices, and passes CUDA '
                  'memcheck. The RULER table below uses the earlier released-planner run.']
        if args.optimized_lod:
            lines += ['', 'The adaptive BF16 row uses the optimized 0.5 page threshold. It includes '
                      '128 layer cache catch-ups per timed repetition. The other LoD rows '
                      'use the original batch-one settings.']
        if args.three_tier_graph:
            lines += ['', 'Fixed-page three-tier uses the same c=16 region cutoff as two-tier and opens '
                      'up to two pages per selected region. In the targeted BF16 RULER evaluation at 32K, '
                      'fixed two-page refinement scored '
                      '90.63% on multikey-3 versus 100.00% for two-tier (32 matched prompts). '
                      '[Three-tier retrieval diagnosis](../three-tier-repair/REPORT.md#why-two-pages-fail). '
                      'The full RULER table below reports two-tier LoD.']

    if args.batch_eight:
        lines += batch_eight_lines(root, args.batch_eight, args.batch_eight_checks or [], reference, provenance,
                                  args.batch_eight_two_tier_port, args.batch_eight_int4_retime,
                                  args.batch_eight_fixed_retime, args.adaptive_page_speedup,
                                  args.optimized_lod[4:] if args.optimized_lod else None)
        if args.batch_eight_two_tier_port:
            lines += two_tier_port_lines(root, args.batch_eight_two_tier_port, reference, provenance,
                                        args.two_tier_port_quality, args.two_tier_port_repeat, args.bf16_defaults)
        if args.batch_eight_int4_retime:
            lines += int4_retime_lines(root, args.batch_eight_int4_retime, reference, provenance,
                                      args.int4_retime_repeat, args.int4_default_checks)
        if args.batch_eight_precision_audit:
            matched_keys = ('model', 'checkpoint', 'manifest_sha256', 'prompt_token_sha256', 'prompt_tokens',
                            'prefill_mlp_chunk', 'timed_decode_steps', 'decode_warmup_steps')
            lines += ['', '### Three-tier BF16 and INT4 with matched page tiles', '',
                      'The original BF16 timing used two warps and a 32-page scan tile. The original '
                      'INT4 timing used one warp and a 16-page scan tile. These reruns use one warp, '
                      '16 pages per scan tile and 64 page workers in both arms. Model precision is BF16. '
                      'The timing window and cache updates match the table above.', '',
                      '| KV precision | Prefill (s) | Decode (ms/batch step) | Output tokens/s |',
                      '|---|---:|---:|---:|']
            audit_times = []
            for file, expected_method in zip(args.batch_eight_precision_audit, THREE_TIER_KV, strict=True):
                data = json.loads((root / file).read_text())
                precision = check_three_tier_settings(data)
                assert data['method'] == expected_method and data['batch_size'] == 8
                assert data['lod_cuda_graph'] and data['replay_prompt_tokens']
                assert data['prefill_includes_deferred_cache_completion']
                for key in matched_keys:
                    assert data[key] == reference[key], key
                for key, expected in (('three_tier_page_warps', 1), ('three_tier_page_block', 16),
                                      ('three_tier_workers', 64)):
                    assert data['decode_threshold'][key] == expected, key
                assert len(data['records']) == 3 and sum(r['warmup'] for r in data['records']) == 1
                assert all(r['lod_layer_catch_up_events_timed'] == 1024 for r in data['records'])
                audit_times.append(data['median_decode_ms_per_batch_step'])
                lines.append(f'| [{precision}]({file}) | {data["median_prefill_s"]:.3f} '
                             f'| {data["median_decode_ms_per_batch_step"]:.3f} '
                             f'| {data["median_decode_tokens_per_second"]:.1f} |')
            bf16, int4 = audit_times
            lines += ['', f'INT4 takes {100 * (int4 / bf16 - 1):.1f}% more decode time than BF16 with '
                      'matched page tiles. BF16’s original timing improves by about 1 ms per batch step.']
            if args.batch_eight_retro_retime:
                file = args.batch_eight_retro_retime
                data = json.loads((root / file).read_text())
                assert data['method'] == 'retro' and data['dtype'] == 'bfloat16' and data['batch_size'] == 8
                assert data['native_cuda_graph'] and data['replay_prompt_tokens']
                for key in matched_keys:
                    assert data[key] == reference[key], key
                assert len(data['records']) == 3 and sum(r['warmup'] for r in data['records']) == 1
                assert all(r['retro_index_updates_timed'] == 1 for r in data['records'])
                lines += ['', f'The [fresh RetroInfer sparse control]({file}) takes '
                          f'{data["median_decode_ms_per_batch_step"]:.3f} ms per batch step '
                          'including one rebuild and graph recapture per repetition.']
            if args.batch_eight_fixed_retime:
                timing_file, check_file, fixture_file = args.batch_eight_fixed_retime
                data, check, fixture = (json.loads((root / file).read_text())
                                        for file in (timing_file, check_file, fixture_file))
                for run in (data, check):
                    assert check_three_tier_settings(run) == 'BF16'
                    assert run['batch_size'] == 8 and run['lod_cuda_graph'] and run['replay_prompt_tokens']
                    assert run['prefill_includes_deferred_cache_completion']
                    for key in matched_keys[:-2]:
                        assert run[key] == reference[key], key
                for key in matched_keys[-2:]:
                    assert data[key] == reference[key], key
                for key in ('tile', 'three_tier_workers', 'three_tier_page_warps', 'three_tier_page_block',
                            'three_tier_route_warps', 'three_tier_fused_pack', 'three_tier_gqa_local',
                            'three_tier_trim_merge', 'materialize_cutoffs', 'fixed_page_scan'):
                    assert data['decode_threshold'][key] == check['decode_threshold'][key], key
                for key in ('LOD_H100_THREE_TIER_LOCAL_TILE', 'LOD_H100_THREE_TIER_LOCAL_APPEND'):
                    settings = [next((item for item in provenance[f][2] if item.startswith(key + '=')), None)
                                for f in (timing_file, check_file, fixture_file)]
                    assert len(set(settings)) == 1, key
                assert len(data['records']) == 3 and sum(r['warmup'] for r in data['records']) == 1
                assert all(r['lod_layer_catch_up_events_timed'] == 1024 for r in data['records'])
                samples = check['records'][0]['graph_same_cache_checks']
                assert {'256', '512'} <= samples.keys() and len(samples) == 17
                assert all(row['max_abs'] == 0 and row['graph_argmax_by_request'] == row['eager_argmax_by_request']
                           for row in samples.values())
                assert fixture['stage'] == 'complete'
                bf16_fixture, = [row for row in fixture['records'] if row['kv_bits'] == 0]
                assert bf16_fixture['batch_size'] == 8 and bf16_fixture['graph_replays'] == 3
                assert bf16_fixture['page_multiplier'] is None and bf16_fixture['max_abs'] <= .008
                ms = data['median_decode_ms_per_batch_step']
                batch_times = [json.loads((root / f).read_text()) for f in args.batch_eight]
                original_ms, = [run['median_decode_ms_per_batch_step'] for run in batch_times
                                if run['method'] == 'retro-lod-bf16'
                                and run['decode_threshold']['implementation'] == 'three-tier'
                                and run['decode_threshold']['three_tier_pages'] == 2]
                if args.batch_eight_two_tier_port:
                    two_tier_run = json.loads((root / args.batch_eight_two_tier_port[1]).read_text())
                    two_tier_ms = two_tier_run['median_decode_ms_per_batch_step']
                else:
                    two_tier_ms, = [run['median_decode_ms_per_batch_step'] for run in batch_times
                                    if run['method'] == 'retro-lod-bf16'
                                    and run['decode_threshold']['implementation'] == 'materialized']
                control = json.loads((root / args.batch_eight_precision_audit[0]).read_text())
                assert [r['output_token_ids'] for r in data['records']] == [
                    r['output_token_ids'] for r in control['records']]
                lines += ['', '### Optimized BF16 fixed two-page refinement', '',
                          'Region routing uses c=16. Each opened region contributes up to two exact pages '
                          'and its residual summary. The specialized kernel reads both pages in one tile '
                          'and uses a shorter summary scan for small regions. The local attention kernel '
                          'shares KV loads across query heads in each GQA group. Region packing uses '
                          'one linear prefix scan and copy per head. '
                          f'The timed configuration uses {data["decode_threshold"]["tile"]}-region '
                          f'tiles and {data["decode_threshold"]["three_tier_workers"]} page workers per query head.',
                          '', f'The [full-model timing]({timing_file}) is **{ms:.3f} ms per batch step** '
                          f'({data["median_decode_tokens_per_second"]:.1f} tokens/s), including 1,024 '
                          f'layer-by-request cache updates per repetition. The [model graph check]({check_file}) '
                          'matches eager logits at 17 steps. The '
                          f'[independent-request check]({fixture_file}) compares distinct caches and queries '
                          f'against the existing page decoder (maximum absolute attention-output '
                          f'difference {bf16_fixture["max_abs"]:.5f}).',
                          '', 'Predicted tokens match the BF16 control with the same page tile and warp count '
                          'across all requests and repetitions.',
                          '', f'The retime reduces fixed-two-page latency by {100*(1-ms/original_ms):.1f}% '
                          f'from the original BF16 run. Two-tier takes {two_tier_ms:.3f} ms per batch step '
                          f'with the same timing window.']
                if args.batch_eight_fixed_repeat:
                    file = args.batch_eight_fixed_repeat
                    repeat = json.loads((root / file).read_text())
                    assert check_three_tier_settings(repeat) == 'BF16'
                    for key in (*matched_keys, 'batch_size', 'lod_cuda_graph', 'replay_prompt_tokens'):
                        assert repeat[key] == data[key], key
                    assert len(repeat['records']) == 3 and sum(r['warmup'] for r in repeat['records']) == 1
                    assert all(r['lod_layer_catch_up_events_timed'] == 1024 for r in repeat['records'])
                    assert [r['output_token_ids'] for r in repeat['records']] == [
                        r['output_token_ids'] for r in data['records']]
                    lines += ['', f'The [independent timing repeat]({file}) measures '
                              f'{repeat["median_decode_ms_per_batch_step"]:.3f} ms per batch step.']
                if args.bf16_defaults:
                    lines += bf16_default_lines(root, args.bf16_defaults[2:], timing_file, two_tier=False)
                if args.fixed_multikey:
                    tok = prior / 'llama-tokenizer.json'
                    previous = root.parent / 'three-tier-repair/three-tier-repair-v1-pages2-quality.json'
                    previous_scores, previous_rows = score_files([previous], tok)
                    latest_scores, latest_rows = score_files([root / args.fixed_multikey], tok)
                    assert set(previous_rows) == set(latest_rows) and len(previous_rows) == 32
                    for key, row in latest_rows.items():
                        assert all(row[field] == previous_rows[key][field] for field in (
                            'prompt_sha256', 'effective_prompt_sha256', 'input_token_sha256', 'answers'))
                    before = previous_scores['tasks']['niah_multikey_3']['first']
                    after = latest_scores['tasks']['niah_multikey_3']['first']
                    lines += ['', f'On 32 matched batch-one RULER multikey-3 prompts, the optimized '
                              f'fixed-two-page kernel with 64-region tiles and 32 page workers '
                              f'scored {after:.2f}% versus {before:.2f}% for '
                              f'the original fixed-two-page kernel. [RULER rerun]({args.fixed_multikey}).']
                    if args.fixed_multikey_controls or args.fixed_multikey_selected:
                        lines += ['', '| Current batch-one kernel | Multikey-3 |', '|---|---:|']
                        variants = ('Reference kernel, original launch settings',
                                    'Reference kernel, 16-page tile and one warp',
                                    'Specialized page kernel, 16-page tile and one warp')
                        panels = list(zip(variants, args.fixed_multikey_controls, strict=True)) if args.fixed_multikey_controls else []
                        if args.fixed_multikey_selected:
                            panels.append(('Optimized kernel, selected batch-eight launch settings', args.fixed_multikey_selected))
                            timing_command, ruler_command = (provenance[name][2]
                                                             for name in (timing_file, args.fixed_multikey_selected))
                            for prefix in ('LOD_H100_THRESHOLD_TILE=', 'LOD_H100_THREE_TIER_WORKERS=',
                                           'LOD_H100_THREE_TIER_PAGE_WARPS=', 'LOD_H100_THREE_TIER_PAGE_BLOCK=',
                                           'LOD_H100_FIXED_PAGE_SCAN=', 'LOD_H100_THREE_TIER_LOCAL_TILE=',
                                           'LOD_H100_THREE_TIER_GQA_LOCAL=', 'LOD_H100_THREE_TIER_FUSED_PACK=',
                                           'LOD_H100_THREE_TIER_MATERIALIZE_CUTOFFS=', 'LOD_H100_THREE_TIER_TRIM_MERGE='):
                                assert {x for x in timing_command if x.startswith(prefix)} == {
                                    x for x in ruler_command if x.startswith(prefix)}, prefix
                        for label, file in panels:
                            score, rows = score_files([root / file], tok)
                            assert set(rows) == set(previous_rows)
                            for key, row in rows.items():
                                assert all(row[field] == previous_rows[key][field] for field in (
                                    'prompt_sha256', 'effective_prompt_sha256', 'input_token_sha256', 'answers'))
                            lines.append(f'| [{label}]({file}) | {score["tasks"]["niah_multikey_3"]["first"]:.2f}% |')
        if args.batch_eight_adaptive_retime and not args.adaptive_page_speedup:
            file = args.batch_eight_adaptive_retime
            data = json.loads((root / file).read_text())
            check_three_tier_settings(data, adaptive=True)
            assert data['batch_size'] == 8 and data['lod_cuda_graph'] and data['replay_prompt_tokens']
            assert data['prefill_includes_deferred_cache_completion']
            for key in ('model', 'checkpoint', 'manifest_sha256', 'prompt_token_sha256', 'prompt_tokens',
                        'prefill_mlp_chunk', 'timed_decode_steps', 'decode_warmup_steps'):
                assert data[key] == reference[key], key
            assert len(data['records']) == 3 and sum(r['warmup'] for r in data['records']) == 1
            assert all(r['lod_layer_catch_up_events_timed'] == 1024 for r in data['records'])
            lines += ['', f'The [optimized adaptive-page retime]({file}) takes '
                      f'{data["median_decode_ms_per_batch_step"]:.3f} ms per batch step '
                      f'({data["median_decode_tokens_per_second"]:.1f} tokens/s). '
                      'It uses the same RMS-normalized c=0.5 page cutoff and includes the same cache updates.']
        if args.adaptive_page_speedup:
            lines += adaptive_page_lines(root, args.adaptive_page_speedup, reference, provenance,
                                         args.adaptive_page_profile, args.adaptive_page_repeat,
                                         args.adaptive_default_checks)
        if args.batch_eight_page_profiles:
            lines += ['', 'A separate 256-step CUDA trace shows mean GPU time per layer for each '
                      'page-refinement kernel call. The trace records the original adaptive path and '
                      'the matched fixed-page launches.', '',
                      '| Page refinement | Mean GPU time (µs/layer) |', '|---|---:|']
            mean_page_times = []
            for file, (label, kernel) in zip(args.batch_eight_page_profiles, (
                ('Fixed two pages, BF16', '_query_major_residual_page_attention_kernel'),
                ('Fixed two pages, INT4', '_query_major_residual_page_attention_kernel'),
                ('Adaptive pages, BF16', '_pages'),
            ), strict=True):
                rows = (root / file).read_text().splitlines()
                header = next(i for i, line in enumerate(rows) if line.startswith('Time (%),'))
                row, = (row for row in csv.DictReader(rows[header:]) if row['Name'] == kernel)
                assert int(row['Instances']) == 8192
                mean_us = float(row['Avg (ns)']) / 1000
                mean_page_times.append(mean_us)
                lines.append(f'| [{label}]({file}) | {mean_us:.1f} |')
            if args.int4_retime_profile:
                file = args.int4_retime_profile
                rows = (root / file).read_text().splitlines()
                header = next(i for i, line in enumerate(rows) if line.startswith('Time (%),'))
                by_name = {row['Name']: row for row in csv.DictReader(rows[header:])}
                page = by_name['_query_major_residual_page_attention_kernel']
                local = by_name['local_gqa']
                assert int(page['Instances']) == int(local['Instances']) == 8192
                lines.append(f'| [Optimized fixed two pages, INT4]({file}) | {float(page["Avg (ns)"])/1000:.1f} |')
            page_overhead = (mean_page_times[2] - mean_page_times[0]) * 32 / 1000
            lines += ['', f'The longer adaptive page kernel accounts for {page_overhead:.2f} ms per batch step '
                      'across 32 layers in the trace. Region scoring, packing, and local attention '
                      'have similar kernel times across all three traces.']
        if args.three_tier_fixture:
            fixture = json.loads((root / args.three_tier_fixture).read_text())
            assert fixture['stage'] == 'complete' and {r['kv_bits'] for r in fixture['records']} == {0, 4}
            assert all(r['batch_size'] == 8 and r['graph_replays'] == 3 and r['max_abs'] == 0
                       for r in fixture['records'])
            lines += ['', 'Three-tier also matched eight independent single-request runs on distinct '
                      'queries, cache contents, and cache lengths in both precisions. '
                      f'[Independent-request check]({args.three_tier_fixture}).']
        if args.adaptive_page_fixture:
            fixture = json.loads((root / args.adaptive_page_fixture).read_text())
            record, = fixture['records']
            expected = dict(batch_size=8, kv_bits=0, graph_replays=3, max_abs=0, page_multiplier=0.5,
                            page_rule='page', page_query='normalized', page_residual='region')
            assert fixture['stage'] == 'complete' and {key: record[key] for key in expected} == expected
            lines += ['', 'Adaptive page routing also matched eight independent single-request runs '
                      'with distinct queries, cache contents, and cache lengths. '
                      f'[Adaptive-page independent-request check]({args.adaptive_page_fixture}).']

    if args.quality:
        tok = prior / 'llama-tokenizer.json'
        repaired = root.parent / 'three-tier-repair'
        paths = {'ClusterKV FP16': [root / args.quality]}
        if args.quality_control:
            paths['Dense FP16'] = [root / args.quality_control]
        paths.update({
            'Dense BF16': [prior / 'ruler-32k-dense-full.json', prior / 'ruler-32k-dense-missing8.json'],
            'RetroInfer sparse BF16': [prior / 'ruler-llama-retroinfer-32k-selected.json',
                                       prior / 'ruler-llama-retroinfer-32k-missing5.json'],
            'LoD BF16, sparse prefill': [
                repaired / 'three-tier-repair-v1-two-tier-control.json',
                repaired / 'three-tier-repair-mq32-v1-two-tier-control.json',
                repaired / 'three-tier-repair-twotier-full11-v2-two-tier-control.json'],
        })
        dense_prefill_label = 'LoD BF16, dense prefill'
        if args.lod_dense_prefill or args.adaptive_dense_prefill:
            first_tokens = {}
            for file in paths['Dense BF16']:
                for row in json.loads(file.read_text())['records']:
                    first_tokens[row['task'], row['source_row']] = row['output_token_ids'][0]
        if args.lod_dense_prefill:
            paths[dense_prefill_label] = [root / args.lod_dense_prefill]
            dense_prefill = json.loads((root / args.lod_dense_prefill).read_text())
            assert dense_prefill['prefill_attention'] == 'dense with LoD cache construction'
            assert dense_prefill['model_precision'] == dense_prefill['kv_precision'] == 'bf16'
            assert dense_prefill['mode'] == 'two-tier' and dense_prefill['graph_decode']
            assert dense_prefill['model_revision'] == Path(reference['checkpoint']).name
            assert dense_prefill['prefill_threshold'] is None
            for file in paths['LoD BF16, sparse prefill']:
                sparse_prefill = json.loads(file.read_text())
                for key in ('model_revision', 'task_token_limits'):
                    assert dense_prefill[key] == sparse_prefill[key], (key, file)
                routing = sparse_prefill['threshold_routing']
                assert dense_prefill['threshold_routing'] == {'page_threshold': None, **routing}, file
            assert all(row['output_token_ids'][0] == first_tokens[row['task'], row['source_row']]
                       for row in dense_prefill['records'])
            verified_tasks = {r['task'] for r in dense_prefill['records'] if r['graph_check'] is not None}
            assert verified_tasks == set(TASKS)
            assert all(c['max_abs'] == 0 and c['graph_argmax'] == c['eager_argmax']
                       for r in dense_prefill['records'] if (c := r['graph_check']) is not None)
        adaptive_label = 'LoD adaptive three-tier, sparse prefill'
        adaptive_dense_label = 'LoD adaptive three-tier, dense prefill'
        if args.adaptive_default_ruler:
            adaptive_file = root / args.adaptive_default_ruler
            adaptive = json.loads(adaptive_file.read_text())
            assert adaptive['model_revision'] == Path(reference['checkpoint']).name
            assert adaptive['mode'] == 'three-tier-bf16' and adaptive['graph_decode']
            assert adaptive['model_precision'] == adaptive['kv_precision'] == 'bf16'
            assert adaptive['prefill_attention'] == 'LoD sparse' and adaptive['prefill_threshold'] is None
            assert adaptive['expected'] == dict.fromkeys(TASKS, 32)
            route = adaptive['threshold_routing']
            assert (route['implementation'], route['rule'], route['mass_multiplier']) == ('three-tier', 'region', 16)
            page = route['page_threshold']
            assert page['tuned_defaults'] and (page['multiplier'], page['query'], page['workers'], page['warps']) == (
                0.5, 'normalized', 24, 2)
            assert page['kernel'] == dict(fast_bf16=True, exact_pages=2, reuse_directory=True, short_scan=8,
                                           fast_residual_logit=True)
            assert not any(flag.startswith(('LOD_H100_ADAPTIVE_', 'LOD_H100_THREE_TIER_', 'LOD_H100_PAGE_QUERY=',
                                            'LOD_H100_PAGE_THRESHOLD_WARPS=', 'LOD_H100_THRESHOLD_TILE='))
                           for flag in provenance[args.adaptive_default_ruler][2])
            graph_checks = {r['task']: r['graph_check'] for r in adaptive['records'] if r['graph_check'] is not None}
            assert graph_checks.keys() == set(TASKS)
            assert all(c['max_abs'] == 0 and c['graph_argmax'] == c['eager_argmax'] for c in graph_checks.values())
            paths[adaptive_label] = [adaptive_file]
        if args.adaptive_dense_prefill:
            dense_adaptive_file = root / args.adaptive_dense_prefill
            dense_adaptive = json.loads(dense_adaptive_file.read_text())
            for key in ('model_revision', 'model_precision', 'kv_precision', 'cohort_sha256', 'mode',
                        'graph_decode', 'prefill_threshold', 'threshold_routing', 'task_token_limits', 'expected'):
                assert dense_adaptive[key] == adaptive[key], key
            assert dense_adaptive['prefill_attention'] == 'dense with LoD cache construction'
            assert all(row['output_token_ids'][0] == first_tokens[row['task'], row['source_row']]
                       for row in dense_adaptive['records'])
            graph_checks = {r['task']: r['graph_check'] for r in dense_adaptive['records']
                            if r['graph_check'] is not None}
            assert graph_checks.keys() == set(TASKS)
            assert all(c['max_abs'] == 0 and c['graph_argmax'] == c['eager_argmax'] for c in graph_checks.values())
            paths[adaptive_dense_label] = [dense_adaptive_file]
        scored, paired, keys = {}, {}, None
        for label, panel_files in paths.items():
            score, samples = score_files(panel_files, tok)
            assert set(score['tasks']) == set(TASKS) and all(t['n'] == 32 for t in score['tasks'].values())
            identities = {key: {k: v for k, v in row.items() if k != 'score'} for key, row in samples.items()}
            if keys is None:
                keys = identities
            else:
                assert keys == identities, label
            scored[label] = score['tasks']
            paired[label] = samples
        quality = json.loads((root / args.quality).read_text())
        assert quality['model_revision'] == Path(reference['checkpoint']).name
        assert quality['mode'] == 'clusterkv' and not quality['graph_decode']
        settings = timing[0]['clusterkv_settings'].copy()
        settings['budget'] = settings.pop('token_budget')
        del settings['cuda_graph']
        assert quality['clusterkv_settings'] == settings
        if args.quality_control:
            dense = json.loads((root / args.quality_control).read_text())
            assert dense['mode'] == 'dense' and not dense['graph_decode']
            for key in ('cohort_sha256', 'model_revision', 'model_precision', 'kv_precision'):
                assert dense[key] == quality[key], key
        lines += ['', '## 32K RULER quality', '',
                  f'[ClusterKV generations]({args.quality}) use the '
                  '[frozen 32-example-per-task cohort](../quality-ruler-32k-frozen-13x32.json.gz). '
                  'The 416 tokenized inputs match the controls. Scores stop at the first '
                  'EOS/EOT and the overall mean weights all 13 tasks equally. ClusterKV uses FP16 '
                  'and its native dense prefill. RetroInfer uses dense prefill and GPU-only sparse '
                  'decode in BF16. LoD uses BF16 and a c=16 region threshold.', '']
        if args.adaptive_default_ruler:
            lines += [f'The [adaptive three-tier run]({args.adaptive_default_ruler}) uses sparse top-eight '
                      'prefill and the optimized H100 default for the 0.5 page threshold. Graph and eager '
                      'logits matched for the sampled step on each task.', '']
        if args.lod_dense_prefill:
            lines += [f'The [dense-prefill LoD run]({args.lod_dense_prefill}) uses dense causal '
                      'attention outputs in every prefill layer and builds the usual LoD cache from '
                      'those layers’ activations. Its first token matches dense BF16 on all 416 prompts. '
                      'The adapter also computes sparse prefill while constructing the cache and discards its '
                      'attention outputs. The comparison measures generation quality.', '']
        if args.adaptive_dense_prefill:
            lines += [f'The [adaptive three-tier dense-prefill run]({args.adaptive_dense_prefill}) uses the same '
                      'c=16 region and c=0.5 page thresholds as the sparse-prefill run. Its first token '
                      'matches dense BF16 on all 416 prompts. This quality run used a separate H100 of '
                      'the same model and driver version.', '']
        lines += ['| Task | ' + ' | '.join(paths) + ' |',
                  '|---|' + '---:|' * len(paths)]
        for task in TASKS:
            lines.append('| ' + task + ' | ' + ' | '.join(
                f'{scores[task]["first"]:.2f}%' for scores in scored.values()) + ' |')
        retrieval_tasks = [task for task in TASKS if task.startswith('niah_')]
        lines.append('| NIAH mean | ' + ' | '.join(
            f'{statistics.mean(scores[t]["first"] for t in retrieval_tasks):.2f}%' for scores in scored.values()) + ' |')
        means = {name: statistics.mean(scores[t]['first'] for t in TASKS) for name, scores in scored.items()}
        lines.append('| **13-task mean** | ' + ' | '.join(f'**{mean:.2f}%**' for mean in means.values()) + ' |')
        if args.lod_dense_prefill:
            mean = means[dense_prefill_label]
            delta_sparse = mean - means['LoD BF16, sparse prefill']
            delta_retro = mean - means['RetroInfer sparse BF16']
            dense_samples = paired[dense_prefill_label]
            sparse_ci = paired_task_mean_interval(dense_samples, paired['LoD BF16, sparse prefill'])
            retro_ci = paired_task_mean_interval(dense_samples, paired['RetroInfer sparse BF16'])
            lines += ['', f'Dense-prefill LoD changes the 13-task mean by {delta_sparse:+.2f} points '
                      f'versus sparse-prefill LoD (95% paired bootstrap interval {sparse_ci[0]:+.2f} to '
                      f'{sparse_ci[1]:+.2f}) and by {delta_retro:+.2f} points versus RetroInfer '
                      f'({retro_ci[0]:+.2f} to {retro_ci[1]:+.2f}). The bootstrap resamples matched '
                      'prompts within each task.']
        if args.adaptive_dense_prefill:
            lo, hi = paired_task_mean_interval(paired[adaptive_dense_label], paired[adaptive_label])
            delta = means[adaptive_dense_label] - means[adaptive_label]
            lines += ['', f'Dense prefill changes the adaptive three-tier mean by {delta:+.2f} points '
                          f'(95% paired interval {lo:+.2f} to {hi:+.2f}).']
        if args.adaptive_default_ruler:
            earlier_paths = [repaired / f'three-tier-repair-{name}-pagec0p5-quality.json'
                             for name in ('normpage-v1', 'normpage-full11-v3')]
            earlier, earlier_rows = score_files(earlier_paths, tok)
            candidate = paired[adaptive_label]
            assert set(earlier_rows) == set(candidate)
            for key, row in earlier_rows.items():
                assert all(row[k] == candidate[key][k] for k in
                           ('prompt_sha256', 'effective_prompt_sha256', 'input_token_sha256', 'answers'))
            earlier_mean = statistics.mean(earlier['tasks'][task]['first'] for task in TASKS)
            delta = means[adaptive_label] - earlier_mean
            lo, hi = paired_task_mean_interval(candidate, earlier_rows)
            lines += ['', f'The optimized adaptive kernel changes the 13-task mean by {delta:+.2f} points '
                          f'versus the earlier adaptive kernel at the same thresholds ({earlier_mean:.2f}%, '
                          f'95% paired bootstrap interval {lo:+.2f} to {hi:+.2f}). '
                          '[Earlier adaptive panel](../three-tier-repair/three-tier-repair-normpage-full13-summary.md).']
            summary_file = Path(args.adaptive_default_ruler).stem + '-summary.md'
            if (root / summary_file).is_file():
                lines += ['', f'[First-response and full-output comparisons]({summary_file}).']
        lines += ['', '[BF16 quality controls](../three-tier-repair/REPORT.md).']

    lines += ['', '## Reproduction', '',
              'Runs use `uv` and `cluster-run` on a single H100 80 GB. '
              '[ClusterKV source patch](clusterkv-h100.patch).', '']
    if args.graph_panel:
        lines += ['[Device-planning source patch](clusterkv-h100-graph.patch).', '']
    for run_type, label in (('clusterkv-h100-build-v9', 'H100 extension build'),
                            ('clusterkv-h100-selector-check-v9', 'Selector and attention verification')):
        (run_dir,) = (root / 'provenance').glob(f'*-{run_type}')
        command_dir = run_dir.relative_to(root).as_posix()
        lines.append(f'- {label} [command]({command_dir}/run.json) · [log]({command_dir}/output.log)')
    for file in files:
        command_dir = provenance[file][0].relative_to(root).as_posix()
        lines.append(f'- `{file}` [command]({command_dir}/run.json) · [log]({command_dir}/output.log)')
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def optimized_lod_lines(root, files, reference, provenance, controls):
    data = [json.loads((root / file).read_text()) for file in files]
    names = ('Two-tier', 'Three-tier, fixed top-two pages', 'Three-tier, fixed top-two pages',
             'Three-tier, adaptive page c=0.5')
    methods = ('retro-lod-bf16', 'retro-lod-bf16', 'retro-lod-int4', 'retro-lod-bf16')
    adaptive_launch = dict(fast_bf16=True, exact_pages=2, reuse_directory=True, short_scan=8,
                           fast_residual_logit=True)
    for offset, batch in ((0, 1), (4, 8)):
        for index, method in enumerate(methods):
            run = data[offset + index]
            assert (run['batch_size'], run['method'], run['dtype']) == (batch, method, 'bfloat16')
            for key in ('model', 'checkpoint', 'manifest_sha256', 'prompt_token_sha256', 'prompt_tokens',
                        'timed_decode_steps', 'decode_warmup_steps', 'prefill_mlp_chunk'):
                assert run[key] == reference[key], key
            assert run['lod_cuda_graph'] and run['replay_prompt_tokens']
            assert run['prefill_threshold'] is None and run['prefill_includes_deferred_cache_completion']
            assert run['cache_update_optimizations'] == dict(packed_max=True, fused_rms=False, fused_rescale=False)
            assert [record['warmup'] for record in run['records']] == [True, False, False]
            assert all(record['lod_layer_catch_up_events_timed'] == 128 * batch for record in run['records'])
            launch = run['decode_threshold']
            assert (launch['rule'], launch['multiplier'], launch['tile']) == ('region', 16, 128)
            command = provenance[files[offset + index]][2]
            if index == 0:
                assert launch['implementation'] == 'materialized' and launch['splits'] == 32
                assert 'LOD_RETRO_TWO_TIER=1' in command
                if batch == 1:
                    assert {'LOD_H100_TWO_TIER_OPTIMIZED=1', 'LOD_AMD_THRESHOLD_PRECISE=1',
                            'LOD_AMD_THRESHOLD_ENTRY_BLOCK=128', 'LOD_AMD_THRESHOLD_ENTRY_WARPS=4',
                            'LOD_AMD_THRESHOLD_MERGE_BLOCK=32'} <= set(command)
                else:
                    assert launch['two_tier_optimized']
                    assert (launch['entry_block'], launch['entry_warps'], launch['merge_block'],
                            launch['precise']) == (128, 4, 32, True)
                continue
            check_three_tier_settings(run, adaptive=index == 3)
            assert all(launch[key] for key in ('three_tier_trim_merge', 'three_tier_gqa_local',
                                               'three_tier_fused_pack', 'materialize_cutoffs'))
            assert launch['three_tier_local_tile'] == 64
            assert launch['three_tier_workers'] == (16 if index == 1 else 24)
            if index != 3:
                assert (launch['three_tier_page_block'], launch['three_tier_page_warps']) == (16, 1)
            if index == 1:
                assert launch['fixed_page_scan'] == 8
            elif index == 2:
                if batch == 1:
                    assert 'LOD_H100_INT4_FACTOR_PAGE_SCALES=1' in command
                else:
                    assert launch['factor_int4_page_scales']
            else:
                assert launch['page_threshold_warps'] == 2 and launch['adaptive_page_kernel'] == adaptive_launch

    lines = ['', '## Optimized LoD at batch one and eight', '',
             'All four variants use the c=16 region cutoff, BF16 model weights, and top-eight prefill '
             'with cache completion. Each row times full-model CUDA graph decode, including 128 '
             'layer cache catch-ups per repetition at B=1 and 1,024 at B=8. A decode step emits one '
             'token per request. B=8 uses independent '
             'cache state for eight copies of the same prompt and prefills them sequentially. '
             'The B=1 runs explicitly enable the optimized kernels and settings. The adaptive-page '
             'variant is enabled by selecting the 0.5 page threshold.', '',
             '| LoD decode | Remote KV | B=1 prefill (s) | B=1 decode (ms/step) | B=8 prefill (s) '
             '| B=8 decode (ms/step) | B=8 output tokens/s |',
             '|---|---|---:|---:|---:|---:|---:|']
    for index, name in enumerate(names):
        one, eight = data[index], data[index + 4]
        kv = 'INT4' if methods[index] == 'retro-lod-int4' else 'BF16'
        lines.append(f'| {name} | {kv} | {one["median_prefill_s"]:.3f} '
                     f'| [{one["median_decode_ms_per_batch_step"]:.3f}]({files[index]}) '
                     f'| {eight["median_prefill_s"]:.3f} '
                     f'| [{eight["median_decode_ms_per_batch_step"]:.3f}]({files[index + 4]}) '
                     f'| {eight["median_decode_tokens_per_second"]:.1f} |')
    if controls:
        lines += ['', 'Batch-one retimes of the earlier kernel settings use the same prompt, graph, '
                      'precision, and cache update schedule.', '',
                  '| LoD decode | Remote KV | Earlier B=1 settings (ms/step) | B=8 tuning at B=1 (ms/step) '
                  '| Latency change |', '|---|---|---:|---:|---:|']
        changes = []
        for index, file in enumerate(controls):
            control = json.loads((root / file).read_text())
            optimized = data[index]
            for key in ('model', 'checkpoint', 'batch_size', 'method', 'dtype', 'manifest_sha256',
                        'prompt_token_sha256', 'prompt_tokens', 'timed_decode_steps', 'decode_warmup_steps',
                        'prefill_mlp_chunk', 'prefill_threshold', 'cache_update_optimizations',
                        'replay_prompt_tokens', 'prefill_includes_deferred_cache_completion', 'lod_cuda_graph'):
                assert control[key] == optimized[key], key
            assert [record['warmup'] for record in control['records']] == [True, False, False]
            assert all(record['lod_layer_catch_up_events_timed'] == 128 for record in control['records'])
            assert control['decode_threshold']['multiplier'] == 16
            if index:
                check_three_tier_settings(control, adaptive=index == 3)
            else:
                assert control['decode_threshold']['implementation'] == 'materialized'
                assert 'LOD_RETRO_TWO_TIER=1' in provenance[file][2]
            before, after = (run['median_decode_ms_per_batch_step'] for run in (control, optimized))
            kv = 'INT4' if methods[index] == 'retro-lod-int4' else 'BF16'
            change = 100 * (after / before - 1)
            changes.append(change)
            lines.append(f'| {names[index]} | {kv} | [{before:.3f}]({file}) | {after:.3f} | {change:+.1f}% |')
        if min(changes[:3]) > 0 and changes[3] < 0:
            lines += ['', 'The earlier settings give the lowest B=1 latency for two-tier and both fixed-page '
                          'variants. The optimized adaptive-page kernel gives the lowest B=1 latency for that variant.']
    return lines


def paired_task_mean_interval(candidate, control):
    """Resample paired differences within each equally weighted task."""
    differences = {
        task: [candidate[key]['score'] - control[key]['score'] for key in sorted(candidate) if key[0] == task]
        for task in TASKS
    }
    rng = random.Random(4242)
    samples = sorted(100 * statistics.mean(
        sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for diffs in differences.values()
    ) for _ in range(10000))
    return samples[249], samples[9749]


def bf16_default_lines(root, files, optimized_file, *, two_tier):
    timing_file, check_file = files
    timing, check, optimized = (json.loads((root / file).read_text()) for file in (*files, optimized_file))
    for run in (timing, check):
        for key in ('model', 'checkpoint', 'prompt_token_sha256', 'batch_size', 'dtype', 'method',
                    'lod_cuda_graph', 'replay_prompt_tokens', 'prefill_includes_deferred_cache_completion'):
            assert run[key] == optimized[key], key
        options = run['decode_threshold']
        assert options['tile'] == 128 and options['multiplier'] == 16
        if two_tier:
            assert options['implementation'] == 'materialized' and options['two_tier_optimized']
            assert options['splits'] == 32
            if 'entry_warps' in options:
                assert (options['entry_block'], options['entry_warps'], options['merge_block'], options['precise']) == (
                    128, 4, 32, True)
        else:
            assert check_three_tier_settings(run) == 'BF16'
            assert options['fixed_page_scan'] == 8 and options['three_tier_workers'] == 16
            assert options['three_tier_page_block'] == 16
    assert len(timing['records']) == 3 and timing['timed_decode_steps'] == 1024 and timing['decode_warmup_steps'] == 256
    assert all(row['lod_layer_catch_up_events_timed'] == 1024 for row in timing['records'])
    for row, prior in zip(timing['records'], optimized['records'], strict=True):
        assert row['warmup'] == prior['warmup'] and row['output_token_ids'] == prior['output_token_ids']
    samples = check['records'][0]['graph_same_cache_checks']
    assert len(samples) == 17 and all(s['max_abs'] == 0 and s['graph_argmax_by_request'] == s['eager_argmax_by_request']
                                      for s in samples.values())
    flag = 'LOD_H100_TWO_TIER_OPTIMIZED' if two_tier else 'LOD_H100_BF16_TUNED_DEFAULTS'
    return ['', f'These settings are the H100 RetroInfer harness default for Llama-3.1-8B-Instruct '
            f'at 32K and batch eight in this mode. The [default timing]({timing_file}) measures '
            f'{timing["median_decode_ms_per_batch_step"]:.3f} ms per batch step and matches the '
            f'explicit configuration’s token predictions. Set `{flag}=0` to opt out. [Graph check]({check_file}).']


def two_tier_port_lines(root, files, reference, provenance, quality_files, repeat_file, default_files):
    before_file, after_file, check_file, fixture_file = files
    before, after, check, fixture = (json.loads((root / name).read_text()) for name in files)
    for name, run in zip(files, (before, after, check)):
        assert run['method'] == 'retro-lod-bf16' and run['dtype'] == 'bfloat16' and run['batch_size'] == 8
        assert run['lod_cuda_graph'] and run['replay_prompt_tokens'] and run['prefill_includes_deferred_cache_completion']
        assert run['prefill_threshold'] is None and 'LOD_RETRO_TWO_TIER=1' in provenance[name][2]
        threshold = run['decode_threshold']
        assert (threshold['implementation'], threshold['rule'], threshold['multiplier']) == ('materialized', 'region', 16)
        for key in ('model', 'checkpoint', 'manifest_sha256', 'prompt_token_sha256', 'prompt_tokens', 'prefill_mlp_chunk'):
            assert run[key] == reference[key], key
    assert [run['decode_threshold']['tile'] for run in (before, after, check)] == [64, 128, 128]
    assert all(run['decode_threshold']['splits'] == 32 for run in (before, after, check))
    for run in (before, after):
        assert run['timed_decode_steps'] == 1024 and run['decode_warmup_steps'] == 256
        assert len(run['records']) == 3 and sum(row['warmup'] for row in run['records']) == 1
        assert all(row['lod_layer_catch_up_events_timed'] == 1024 for row in run['records'])
    assert [row['output_token_ids'] for row in after['records']] == [row['output_token_ids'] for row in before['records']]
    samples = check['records'][0]['graph_same_cache_checks']
    assert len(samples) == 17 and {'256', '512'} <= samples.keys()
    assert all(s['max_abs'] == 0 and s['graph_argmax_by_request'] == s['eager_argmax_by_request']
               and len(s['graph_argmax_by_request']) == 8 for s in samples.values())
    assert fixture['stage'] == 'complete' and fixture['batch_size'] == 8 and fixture['reference_checked']
    assert fixture['graph_replays'] == 3 and fixture['max_abs'] <= .008
    settings = {'LOD_H100_TWO_TIER_OPTIMIZED=1', 'LOD_H100_THRESHOLD_TILE=128'}
    assert all(settings <= set(provenance[name][2]) for name in files[1:])
    assert settings.isdisjoint(provenance[before_file][2])
    latency = [d['median_decode_ms_per_batch_step'] for d in (before, after)]
    lines = [
        '', '### Optimized BF16 two-tier threshold decode', '',
        'The optimized kernel computes the global cutoff once per query head and packs indices '
        'for four selected regions at a time. Routing uses 128 regions per tile.', '',
        f'The port reduces batch-step latency by {100*(1-latency[1]/latency[0]):.1f}% at region c=16. '
        'Both runs include 1,024 layer-by-request cache updates per timed repetition and give identical '
        'predicted tokens across the timed sequence.', '',
        '| Configuration | Prefill (s) | Decode (ms/batch step) | Output tokens/s |',
        '|---|---:|---:|---:|',
        f'| [Reference]({before_file}) | {before["median_prefill_s"]:.3f} | {latency[0]:.3f} | '
        f'{before["median_decode_tokens_per_second"]:.1f} |',
        f'| [Optimized]({after_file}) | {after["median_prefill_s"]:.3f} | {latency[1]:.3f} | '
        f'{after["median_decode_tokens_per_second"]:.1f} |',
        '', 'The graph matches eager logits at 17 sampled steps. The independent-request test checks '
        'distinct cache rows, the 1,024-entry routing eligibility boundary, and KV appends. '
        f'The maximum absolute attention-output difference was {fixture["max_abs"]:.5f}. '
        f'[Model graph check]({check_file}). [Independent-request check]({fixture_file}).',
    ]
    if repeat_file:
        repeat = json.loads((root / repeat_file).read_text())
        command, replay_command = (provenance[file][2] for file in (after_file, repeat_file))
        output_arg = command.index('--output') + 1
        assert command[:output_arg] == replay_command[:output_arg]
        assert command[output_arg + 1:] == replay_command[output_arg + 1:]
        for key in ('model', 'checkpoint', 'prompt_token_sha256', 'batch_size', 'dtype', 'decode_threshold',
                    'timed_decode_steps', 'decode_warmup_steps', 'lod_cuda_graph', 'replay_prompt_tokens'):
            assert repeat[key] == after[key], key
        for row, previous in zip(repeat['records'], after['records'], strict=True):
            assert row['lod_layer_catch_up_events_timed'] == 1024 and row['warmup'] == previous['warmup']
            assert row['output_token_ids'] == previous['output_token_ids']
        lines += ['', f'The [timing repeat]({repeat_file}) measures '
                  f'{repeat["median_decode_ms_per_batch_step"]:.3f} ms per batch step.']
    if quality_files:
        runs = [json.loads((root / file).read_text()) for file in quality_files]
        assert all(d['mode'] == 'two-tier' and d['graph_decode'] and d['prefill_attention'] == 'LoD sparse'
                   and d['expected'] == {'niah_multikey_3': 32} for d in runs)
        assert runs[0]['cohort_sha256'] == runs[1]['cohort_sha256']
        assert settings.isdisjoint(provenance[quality_files[0]][2])
        assert settings <= set(provenance[quality_files[1]][2])
        tokenizer = root.parent / 'ruler-32k-panel' / 'llama-tokenizer.json'
        (left, left_rows), (right, right_rows) = (score_files([root / file], tokenizer) for file in quality_files)
        assert left['identity'] == right['identity'] and left_rows.keys() == right_rows.keys()
        for key in left_rows:
            for field in ('prompt_sha256', 'effective_prompt_sha256', 'input_token_sha256', 'answers'):
                assert left_rows[key][field] == right_rows[key][field]
        a, b = (score['tasks']['niah_multikey_3']['first'] for score in (left, right))
        identical = sum(x['output_token_ids'] == y['output_token_ids']
                        for x, y in zip(runs[0]['records'], runs[1]['records'], strict=True))
        lines += ['', f'On the 32 matching batch-one RULER multikey-3 prompts, first-response accuracy '
                  f'was {a:.2f}% with the [reference]({quality_files[0]}) and {b:.2f}% '
                  f'with the [optimized kernels]({quality_files[1]}). The complete generated token '
                  f'sequences matched on {identical} of 32 prompts.']
    if default_files:
        lines += bf16_default_lines(root, default_files[:2], after_file, two_tier=True)
    return lines


def int4_retime_lines(root, files, reference, provenance, repeat_file, default_files):
    before_file, after_file, check_file, fixture_file, ruler_before, ruler_after = files
    before, after, check, fixture, baseline_ruler, optimized_ruler = (
        json.loads((root / file).read_text()) for file in files)
    for run in (before, after, check):
        assert check_three_tier_settings(run) == 'INT4'
        assert run['batch_size'] == 8 and run['lod_cuda_graph'] and run['replay_prompt_tokens']
        assert run['prefill_includes_deferred_cache_completion']
        for key in ('model', 'checkpoint', 'manifest_sha256', 'prompt_token_sha256',
                    'prompt_tokens', 'prefill_mlp_chunk'):
            assert run[key] == reference[key], key
    assert after['decode_threshold'] == check['decode_threshold']
    launch_fields = ('LOD_H100_THRESHOLD_TILE', 'LOD_H100_THREE_TIER_WORKERS',
                     'LOD_H100_THREE_TIER_PAGE_WARPS', 'LOD_H100_THREE_TIER_PAGE_BLOCK',
                     'LOD_H100_INT4_FACTOR_PAGE_SCALES')
    launch = [{arg.partition('=')[0]: arg.partition('=')[2] for arg in provenance[name][2]
               if arg.partition('=')[0] in launch_fields} for name in (after_file, check_file, fixture_file, ruler_after)]
    assert all(options == launch[0] for options in launch[1:])
    for run in (before, after):
        assert run['timed_decode_steps'] == 1024 and run['decode_warmup_steps'] == 256
        assert len(run['records']) == 3 and sum(row['warmup'] for row in run['records']) == 1
        assert all(row['lod_layer_catch_up_events_timed'] == 1024 for row in run['records'])
    changed = 0
    before_window = 0
    for left, right in zip(before['records'], after['records'], strict=True):
        assert left['warmup'] == right['warmup']
        if left['warmup']:
            continue
        for lhs, rhs in zip(left['output_token_ids'], right['output_token_ids'], strict=True):
            assert len(lhs) == len(rhs) == 1281
            changed += sum(a != b for a, b in zip(lhs[-1024:], rhs[-1024:], strict=True))
            before_window += sum(a != b for a, b in zip(lhs[:-1024], rhs[:-1024], strict=True))
    samples = check['records'][0]['graph_same_cache_checks']
    assert len(samples) == 17 and {'256', '512'} <= samples.keys()
    assert all(s['max_abs'] == 0 and s['graph_argmax_by_request'] == s['eager_argmax_by_request']
               and len(s['graph_argmax_by_request']) == 8 for s in samples.values())
    assert fixture['stage'] == 'complete'
    result, = [row for row in fixture['records'] if row['kv_bits'] == 4]
    assert result['batch_size'] == 8 and result['graph_replays'] == 3 and result['max_abs'] <= .008
    assert baseline_ruler['cohort_sha256'] == optimized_ruler['cohort_sha256']
    for run in (baseline_ruler, optimized_ruler):
        assert run['mode'] == 'three-tier-int4' and run['expected'] == {'niah_multikey_3': 32}
        assert run['graph_decode'] and run['prefill_attention'] == 'LoD sparse'
    tokenizer = root.parent / 'ruler-32k-panel' / 'llama-tokenizer.json'
    (a, a_rows), (b, b_rows) = (score_files([root / file], tokenizer) for file in (ruler_before, ruler_after))
    assert a['identity'] == b['identity'] and a_rows.keys() == b_rows.keys()
    for key in a_rows:
        for field in ('prompt_sha256', 'effective_prompt_sha256', 'input_token_sha256', 'answers'):
            assert a_rows[key][field] == b_rows[key][field]
    t0, t1 = (run['median_decode_ms_per_batch_step'] for run in (before, after))
    accuracy = [score['tasks']['niah_multikey_3']['first'] for score in (a, b)]
    changed_answers = sum(a_rows[key]['score'] != b_rows[key]['score'] for key in a_rows)
    settings = after['decode_threshold']
    kernel = ''
    if launch[0].get('LOD_H100_INT4_FACTOR_PAGE_SCALES') == '1':
        kernel = ' Page scales are applied once to the query and after the weighted value reduction.'
    if default_files:
        kernel += (' The H100 RetroInfer harness defaults to these settings for Llama-3.1-8B-Instruct '
                   'at 32K and batch eight with three-tier INT4, region c=16, and two pages per opened region. '
                   'Set `LOD_H100_INT4_TUNED_DEFAULTS=0` to restore the previous launch defaults.')
    lines = [
        '', '### Optimized INT4 fixed two-page refinement', '',
        f'The timed configuration uses {settings["tile"]} regions per routing tile, '
        f'{settings["three_tier_workers"]} page workers per head, and '
        f'{settings["three_tier_page_block"]} pages per scan tile. '
        'The region cutoff is computed once per query head, region packing uses a fused prefix scan, '
        f'and local KV loads are shared within each GQA group.{kernel}',
        '',
        f'INT4 decode takes **{t1:.3f} ms per batch step** ({after["median_decode_tokens_per_second"]:.1f} '
        f'tokens/s), down {100*(1-t1/t0):.1f}% from {t0:.3f} ms. Both timings include 1,024 '
        f'layer-by-request cache updates per repetition. [Reference]({before_file}). [Optimized]({after_file}).',
        '', f'Next-token predictions differ at {changed} of {2*1024*8:,} timed positions and '
        f'{before_window} of {2*257*8:,} preceding positions in the measured repetitions. '
        'The optimized graph matches its eager logits at 17 sampled steps. '
        f'The maximum absolute output difference against independent reference requests was '
        f'{result["max_abs"]:.5f}. [Model graph check]({check_file}). [Independent-request check]({fixture_file}).',
        '', 'First-response accuracy on 32 matching RULER multikey-3 prompts was '
        f'{accuracy[0]:.2f}% with the [reference]({ruler_before}) and '
        f'{accuracy[1]:.2f}% with the [optimized configuration]({ruler_after}). '
        f'{changed_answers} prompts changed first-response score.',
    ]
    if repeat_file:
        repeat = json.loads((root / repeat_file).read_text())
        command, replay_command = (provenance[file][2] for file in (after_file, repeat_file))
        output_arg = command.index('--output') + 1
        assert command[:output_arg] == replay_command[:output_arg]
        assert command[output_arg + 1:] == replay_command[output_arg + 1:]
        for key in ('prompt_token_sha256', 'batch_size', 'decode_threshold', 'timed_decode_steps',
                    'decode_warmup_steps', 'lod_cuda_graph'):
            assert repeat[key] == after[key], key
        for row, previous in zip(repeat['records'], after['records'], strict=True):
            assert row['lod_layer_catch_up_events_timed'] == 1024 and row['warmup'] == previous['warmup']
            assert row['output_token_ids'] == previous['output_token_ids']
        lines += ['', f'The [timing repeat]({repeat_file}) measures '
                  f'{repeat["median_decode_ms_per_batch_step"]:.3f} ms per batch step.']
    if default_files:
        timing_file, graph_file, opt_out_file = default_files
        timing, graph, opt_out = (json.loads((root / file).read_text()) for file in default_files)
        for candidate in (timing, graph, opt_out):
            for key in ('model', 'checkpoint', 'manifest_sha256', 'prompt_token_sha256', 'batch_size',
                        'dtype', 'lod_cuda_graph', 'replay_prompt_tokens', 'prefill_includes_deferred_cache_completion'):
                assert candidate[key] == after[key], key
        for run, historical, file in ((timing, after, timing_file), (graph, check, graph_file), (opt_out, before, opt_out_file)):
            expected = {key: historical['decode_threshold'][key] for key in (
                'tile', 'three_tier_workers', 'three_tier_page_warps', 'three_tier_page_block', 'three_tier_trim_merge',
                'three_tier_gqa_local', 'three_tier_fused_pack', 'three_tier_local_tile', 'materialize_cutoffs')}
            expected['factor_int4_page_scales'] = run is not opt_out
            actual = {key: run['decode_threshold'][key] for key in expected}
            actual['three_tier_page_block'] = run['records'][0]['lod_route_geometry']['threshold_page_block']
            assert actual == expected
            command = provenance[file][2]
            assert not any(part.startswith(('LOD_H100_THRESHOLD_TILE=', 'LOD_H100_THREE_TIER_WORKERS=',
                                           'LOD_H100_THREE_TIER_PAGE_BLOCK=', 'LOD_H100_INT4_FACTOR_PAGE_SCALES='))
                           for part in command)
            assert ('LOD_H100_INT4_TUNED_DEFAULTS=0' in command) == (run is opt_out)
        assert timing['timed_decode_steps'] == 1024 and timing['decode_warmup_steps'] == 256
        for row, previous in zip(timing['records'], after['records'], strict=True):
            assert row['lod_layer_catch_up_events_timed'] == 1024 and row['warmup'] == previous['warmup']
            assert row['output_token_ids'] == previous['output_token_ids']
        for run in (graph, opt_out):
            checks = run['records'][0]['graph_same_cache_checks']
            assert checks.keys() == samples.keys()
            assert all(item['max_abs'] == 0 and item['graph_argmax_by_request'] == item['eager_argmax_by_request']
                       for item in checks.values())
        lines += ['', f'The [default launch]({timing_file}) measures '
                  f'{timing["median_decode_ms_per_batch_step"]:.3f} ms per batch step. Its predicted tokens match '
                  f'the explicit optimized run. [Default graph check]({graph_file}). [Opt-out graph check]({opt_out_file}).']
    return lines


def adaptive_page_lines(root, files, reference, provenance, profile, repeat_file, default_checks):
    control_file, timing_file, check_file, fixture_file, ruler_before, ruler_after = files
    control, timing, check, fixture, qa_before, qa_after = (
        json.loads((root / file).read_text()) for file in files)
    for run in (control, timing, check):
        check_three_tier_settings(run, adaptive=True)
        assert run['batch_size'] == 8 and run['lod_cuda_graph'] and run['replay_prompt_tokens']
        assert run['prefill_includes_deferred_cache_completion']
        for key in ('model', 'checkpoint', 'manifest_sha256', 'prompt_token_sha256', 'prompt_tokens', 'prefill_mlp_chunk'):
            assert run[key] == reference[key], key
    candidates = [timing_file, check_file, fixture_file, ruler_after]
    if repeat_file:
        candidates.append(repeat_file)
    for prefix in ('LOD_H100_THRESHOLD_TILE=', 'LOD_H100_THREE_TIER_WORKERS=', 'LOD_H100_PAGE_THRESHOLD_WARPS=',
                   'LOD_H100_ADAPTIVE_PAGE_FAST=', 'LOD_H100_ADAPTIVE_EXACT_PAGES=',
                   'LOD_H100_ADAPTIVE_REUSE_DIRECTORY=', 'LOD_H100_ADAPTIVE_SHORT_SCAN=',
                   'LOD_H100_ADAPTIVE_RESIDUAL_LOGIT='):
        options = [{word for word in provenance[file][2] if word.startswith(prefix)} for file in candidates]
        assert all(group == options[0] for group in options), prefix
    for run in (control, timing):
        assert run['timed_decode_steps'] == 1024 and run['decode_warmup_steps'] == 256
        assert len(run['records']) == 3 and sum(row['warmup'] for row in run['records']) == 1
        assert all(row['lod_layer_catch_up_events_timed'] == 1024 for row in run['records'])
    changed = sum(a != b for r, s in zip(control['records'], timing['records'], strict=True)
                  if not r['warmup'] for x, y in zip(r['output_token_ids'], s['output_token_ids'], strict=True)
                  for a, b in zip(x[-1024:], y[-1024:], strict=True))
    graph = check['records'][0]['graph_same_cache_checks']
    assert len(graph) == 17 and {'256', '512'} <= graph.keys()
    assert all(row['max_abs'] == 0 and row['graph_argmax_by_request'] == row['eager_argmax_by_request']
               for row in graph.values())
    assert fixture['stage'] == 'complete'
    result, = fixture['records']
    assert result['kv_bits'] == 0 and result['batch_size'] == 8 and result['page_multiplier'] == .5
    assert result['graph_replays'] == 3 and result['max_abs'] <= .008
    assert qa_before['cohort_sha256'] == qa_after['cohort_sha256']
    for run in (qa_before, qa_after):
        assert run['mode'] == 'three-tier-bf16' and run['expected'] == {'niah_multikey_3': 32}
        assert run['graph_decode'] and run['prefill_attention'] == 'LoD sparse'
    tokenizer = root.parent / 'ruler-32k-panel' / 'llama-tokenizer.json'
    (a, a_rows), (b, b_rows) = (score_files([root / f], tokenizer) for f in (ruler_before, ruler_after))
    assert a['identity'] == b['identity'] and a_rows.keys() == b_rows.keys() and len(a_rows) == 32
    for key in a_rows:
        for field in ('prompt_sha256', 'effective_prompt_sha256', 'input_token_sha256', 'answers'):
            assert a_rows[key][field] == b_rows[key][field]
    t0, t1 = (run['median_decode_ms_per_batch_step'] for run in (control, timing))
    settings = timing['decode_threshold']
    command = provenance[timing_file][2]
    scan = next(word.split('=', 1)[1] for word in command if word.startswith('LOD_H100_ADAPTIVE_SHORT_SCAN='))
    exact = next(word.split('=', 1)[1] for word in command if word.startswith('LOD_H100_ADAPTIVE_EXACT_PAGES='))
    scores = [score['tasks']['niah_multikey_3']['first'] for score in (a, b)]
    matching_generations = sum(x['output_token_ids'] == y['output_token_ids']
                               for x, y in zip(qa_before['records'], qa_after['records'], strict=True))
    mean_entries = [statistics.mean(head[3] for row in run['records']
                                   for layer in row['page_threshold_work']['layers'] for head in layer)
                    for run in (qa_before, qa_after)]
    launch_options = ('LOD_H100_ADAPTIVE_PAGE_FAST', 'LOD_H100_ADAPTIVE_EXACT_PAGES',
                      'LOD_H100_ADAPTIVE_REUSE_DIRECTORY', 'LOD_H100_ADAPTIVE_SHORT_SCAN',
                      'LOD_H100_ADAPTIVE_RESIDUAL_LOGIT')
    kernel_flags = ' '.join(word for word in command if word.partition('=')[0] in launch_options)
    residual_note = (' The residual logit reuses the page query-key dot products.'
                     if 'LOD_H100_ADAPTIVE_RESIDUAL_LOGIT=1' in command else '')
    lines = [
        '', '### Adaptive BF16 page threshold optimization', '',
        f'The kernel scans summaries in a tile of {scan} pages when the region fits, and uses a full scan otherwise. '
        f'It reuses the scanned page IDs and counts and processes selected entries {exact} pages at a time. '
        f'The measured launch uses {settings["three_tier_workers"]} workers per query head, '
        f'{settings["page_threshold_warps"]} warp(s) per worker, and '
        f'{settings["tile"]} regions per routing tile. The 0.5 cutoff applies to all candidate pages in the region.'
        f'{residual_note}',
        '', f'The explicit tuning run used `{kernel_flags}`.',
        '',
        f'The [optimized decoder]({timing_file}) takes **{t1:.3f} ms per batch step** '
        f'({timing["median_decode_tokens_per_second"]:.1f} tokens/s), down {100*(1-t1/t0):.1f}% '
        f'from the [retimed control]({control_file}) at {t0:.3f} ms. '
        'Both timings include 1,024 layer-by-request cache updates per repetition.',
        '', f'Next-token predictions differ at {changed} of 16,384 timed positions. Graph and eager '
        f'logits match at 17 steps. The independent-request fixture compares route membership, exact page '
        f'and entry counts, and attention output. [Graph check]({check_file}). [Fixture]({fixture_file}).',
        '', f'On 32 matched RULER multikey-3 prompts, first-response accuracy was '
        f'{scores[0]:.2f}% for the [control]({ruler_before}) and {scores[1]:.2f}% for the [optimized path]({ruler_after}). '
        f'The full generated sequence matched on {matching_generations} prompts.',
        f'At the final generated step, mean exact remote entries per layer and query head were '
        f'{mean_entries[0]:.1f} for the control and {mean_entries[1]:.1f} for the optimized path.',
    ]
    if profile:
        rows = (root / profile).read_text().splitlines()
        header = next(i for i, line in enumerate(rows) if line.startswith('Time (%),'))
        page, = (row for row in csv.DictReader(rows[header:]) if row['Name'] == '_pages')
        assert int(page['Instances']) == 8192
        lines += ['', f'The [256-step trace]({profile}) measures {float(page["Avg (ns)"])/1000:.1f} µs '
                  'per layer for adaptive page refinement.']
    if repeat_file:
        repeat = json.loads((root / repeat_file).read_text())
        for key in ('model', 'checkpoint', 'prompt_token_sha256', 'batch_size', 'timed_decode_steps',
                    'decode_warmup_steps', 'lod_cuda_graph'):
            assert repeat[key] == timing[key], key
        assert all(row['lod_layer_catch_up_events_timed'] == 1024 for row in repeat['records'])
        for row, first in zip(repeat['records'], timing['records'], strict=True):
            assert row['warmup'] == first['warmup'] and row['output_token_ids'] == first['output_token_ids']
        lines += ['', f'The [timing repeat]({repeat_file}) measures '
                  f'{repeat["median_decode_ms_per_batch_step"]:.3f} ms per batch step.']
    if default_checks:
        links = []
        for batch, file in zip((1, 8), default_checks, strict=True):
            run = json.loads((root / file).read_text())
            check_three_tier_settings(run, adaptive=True)
            launch = run['decode_threshold']
            assert (run['batch_size'], launch['tile'], launch['three_tier_workers'], launch['page_threshold_warps']) == (
                batch, 128, 24, 2)
            for key in ('prompt_token_sha256', 'model', 'checkpoint', 'lod_cuda_graph', 'replay_prompt_tokens'):
                assert run[key] == timing[key], key
            measured_kernel = settings['adaptive_page_kernel'].copy()
            assert not measured_kernel.pop('single_page_fast', False)
            assert launch['adaptive_page_kernel'] == measured_kernel
            assert not any(flag.startswith(('LOD_H100_ADAPTIVE_', 'LOD_H100_THREE_TIER_', 'LOD_H100_PAGE_QUERY=',
                                            'LOD_H100_PAGE_THRESHOLD_WARPS=', 'LOD_H100_THRESHOLD_TILE='))
                           for flag in provenance[file][2])
            record, = run['records']
            samples = record['graph_same_cache_checks']
            assert len(samples) == 17 and {'256', '512'} <= samples.keys()
            assert all(c['max_abs'] == 0 and c.get('graph_argmax_by_request', c.get('graph_argmax')) ==
                       c.get('eager_argmax_by_request', c.get('eager_argmax')) for c in samples.values())
            if batch == 8:
                assert record['output_token_ids'] == check['records'][0]['output_token_ids']
            links.append(f'[B={batch} graph check]({file})')
        lines += ['', 'Three-tier BF16 now uses the optimized kernel by default when '
                      '`LOD_H100_PAGE_THRESHOLD=0.5` selects the mean-page rule on H100. '
                      'Set `LOD_H100_ADAPTIVE_TUNED_DEFAULTS=0` to use the earlier defaults. '
                      + ' · '.join(links) + '.']
    return lines


def batch_eight_lines(root, files, check_files, reference, provenance, two_tier_port=None,
                      int4_retime=None, bf16_fixed_retime=None, adaptive_speedup=None, latest_lod=None):
    def identify(file, data):
        method = data['method']
        if method == 'retro-lod-bf16' and 'LOD_RETRO_TWO_TIER=1' in provenance[file][2]:
            threshold = data['decode_threshold']
            assert threshold and {k: threshold[k] for k in ('implementation', 'rule', 'multiplier')} == {
                'implementation': 'materialized', 'rule': 'region', 'multiplier': 16.0}
            assert data['prefill_threshold'] is None
            return 'lod-two-c16-bf16'
        if method in THREE_TIER_KV:
            adaptive = data['decode_threshold']['page_threshold_multiplier'] is not None
            check_three_tier_settings(data, adaptive=adaptive)
            if adaptive:
                return ADAPTIVE_PAGE
        return method

    configs = {
        'retro-full': ('Dense', 'BF16', 'native_cuda_graph', None),
        'retro': ('RetroInfer GPU-only sparse decode', 'BF16', 'native_cuda_graph', 'retro_index_updates_timed'),
        'lod-two-c16-bf16': ('LoD two-tier, c=16 threshold decode', 'BF16', 'lod_cuda_graph',
                            'lod_layer_catch_up_events_timed'),
        ADAPTIVE_PAGE: (ADAPTIVE_NAME, 'BF16', 'lod_cuda_graph', 'lod_layer_catch_up_events_timed'),
        'retro-multipole': ('Multipole, 512-token budget', 'BF16', 'multipole_cuda_graph',
                            'multipole_cluster_updates_timed'),
    }
    configs.update({method: (THREE_TIER_NAME, kv, 'lod_cuda_graph', 'lod_layer_catch_up_events_timed')
                    for method, kv in THREE_TIER_KV.items()})
    replacement_timings, replacement_checks = {}, {}
    if two_tier_port:
        replacement_timings['lod-two-c16-bf16'], replacement_checks['lod-two-c16-bf16'] = two_tier_port[1:3]
    if int4_retime:
        replacement_timings['retro-lod-int4'], replacement_checks['retro-lod-int4'] = int4_retime[1:3]
    if bf16_fixed_retime:
        replacement_timings['retro-lod-bf16'], replacement_checks['retro-lod-bf16'] = bf16_fixed_retime[:2]
    if adaptive_speedup:
        replacement_timings[ADAPTIVE_PAGE], replacement_checks[ADAPTIVE_PAGE] = adaptive_speedup[1:3]
    if latest_lod:
        for method, file in zip(('lod-two-c16-bf16', 'retro-lod-bf16', 'retro-lod-int4', ADAPTIVE_PAGE),
                                latest_lod, strict=True):
            replacement_timings[method] = file
    files = [replacement_timings.get(identify(file, json.loads((root / file).read_text())), file)
             for file in files]
    check_files = [replacement_checks.get(identify(file, json.loads((root / file).read_text())), file)
                   for file in check_files]
    assert set(replacement_timings.values()) <= set(files) and set(replacement_checks.values()) <= set(check_files)
    matched_keys = ('model', 'checkpoint', 'manifest_sha256', 'prompt_token_sha256',
                    'prompt_tokens', 'prefill_mlp_chunk')

    checks = {}
    for file in check_files:
        check = json.loads((root / file).read_text())
        method = identify(file, check)
        assert method not in checks
        assert check['batch_size'] == 8 and check['dtype'] == 'bfloat16'
        assert check[configs[method][2]] and check['replay_prompt_tokens']
        for key in matched_keys:
            assert check[key] == reference[key], key
        record, = check['records']
        samples = record['graph_same_cache_checks']
        assert len(samples) >= 5
        assert all(math.isfinite(row['max_abs']) and row['graph_argmax_by_request'] == row['eager_argmax_by_request']
                   and len(row['graph_argmax_by_request']) == 8 for row in samples.values())
        if method == 'retro':
            assert {'1023', '1024', '1025'} <= samples.keys()
            assert record['retro_index_updates_timed'] == 1
        if method in THREE_TIER_KV or method == ADAPTIVE_PAGE:
            assert len(samples) == 17 and {'256', '512'} <= samples.keys()
        tolerance = 5.0 if method == 'retro-multipole' else 0
        assert all(row['max_abs'] <= tolerance for row in samples.values())
        checks[method] = (file, max(row['max_abs'] for row in samples.values()))

    lines = ['', '## Batch-eight CUDA graph comparison', '',
             'Eight copies of the same 32K prompt, with independent KV state per request. '
             'The H100 runs BF16 weights, sequential per-request prefill, and a whole-model CUDA '
             'graph for decode. After 256 untimed steps, 1,024 teacher-forced batch steps are '
             'timed. Each row gives the median of two repetitions after one warmup. '
             'Decode includes periodic cache maintenance and RetroInfer graph recapture. '
             'Each batch step emits eight output tokens.', '',
             '| Method | KV precision | Prefill (s) | Decode (ms/batch step) | Output tokens/s |',
             '|---|---|---:|---:|---:|']
    seen = set()
    updates = {}
    step_times = {}
    for file in files:
        data = json.loads((root / file).read_text())
        method = identify(file, data)
        assert method not in seen
        seen.add(method)
        name, kv, flag, counter = configs[method]
        assert data['batch_size'] == 8 and data['dtype'] == 'bfloat16'
        assert data[flag] and data['replay_prompt_tokens']
        for key in (*matched_keys, 'timed_decode_steps', 'decode_warmup_steps'):
            assert data[key] == reference[key], key
        records = [r for r in data['records'] if not r['warmup']]
        assert len(records) == 2 and len(data['records']) == 3
        if counter:
            (count,) = {r[counter] for r in data['records']}
            assert count > 0 and (method != 'retro' or count == 1)
            updates[method] = count
        if method in {'lod-two-c16-bf16', ADAPTIVE_PAGE, *THREE_TIER_KV}:
            assert data['prefill_includes_deferred_cache_completion']
            assert count == 1024
        if method == 'retro-multipole':
            assert data['multipole_budget'] == 512
            assert count == 8
        step_times[method] = data['median_decode_ms_per_batch_step']
        lines.append(f'| [{name}]({file}) | {kv} | {data["median_prefill_s"]:.3f} '
                     f'| {data["median_decode_ms_per_batch_step"]:.3f} '
                     f'| {data["median_decode_tokens_per_second"]:.1f} |')
    assert seen <= checks.keys() | {'retro-full'}
    if {'retro-full', 'retro'} <= seen:
        dense, retro = step_times['retro-full'], step_times['retro']
        lines += ['', f'RetroInfer takes {100 * (1 - retro / dense):.1f}% less time per batch step '
                  'than dense at batch eight.']
    if not replacement_timings and {'lod-two-c16-bf16', *THREE_TIER_KV} <= seen:
        two_tier = step_times['lod-two-c16-bf16']
        bf16, int4 = (100 * (step_times[method] / two_tier - 1) for method in THREE_TIER_KV)
        lines += ['', f'The original fixed-two-page batch steps take {bf16:.1f}% more time with BF16 KV and '
                  f'{int4:.1f}% more time with INT4 KV compared with BF16 two-tier.']
    if ADAPTIVE_PAGE in seen:
        if not replacement_timings and {'retro-lod-bf16', 'lod-two-c16-bf16'} <= seen:
            adaptive = step_times[ADAPTIVE_PAGE]
            fixed_overhead = 100 * (adaptive / step_times['retro-lod-bf16'] - 1)
            two_tier_overhead = 100 * (adaptive / step_times['lod-two-c16-bf16'] - 1)
            lines += ['', f'In the original BF16 runs, adaptive page routing adds {fixed_overhead:.1f}% to batch-step '
                      f'latency versus fixed two-page refinement and {two_tier_overhead:.1f}% versus two-tier.']
        lines += ['', 'Adaptive page routing uses RMS-normalized queries and opens pages whose proxy '
                  'mass exceeds half the mean page mass within the selected region. On the earlier '
                  'batch-one BF16 RULER run, this configuration scored 100% on multikey-3 and '
                  '86.73% across all 13 tasks. '
                  '[Three-tier RULER results](../three-tier-repair/REPORT.md#full-ruler-panel).']
    if 'retro-multipole' in seen:
        lines += ['', 'Multipole uses its fixed-position routing query and Triton 3.1. '
                  'The other rows use Triton 3.7.1.']
    lines += ['', 'ClusterKV supports one request per decode step in this implementation. '
              'The RULER scores below use the batch-one configurations. '
              'Our QUEST and Squeezed adapters in this harness invoke each request’s attention serially.']
    if 'lod-two-c16-bf16' in seen:
        lines += ['', 'Two-tier uses materialized c=16 region threshold decode at both batch sizes. '
                  'All LoD rows use fixed top-eight prefill. '
                  'The [shared-centroid report](../llama-shared-index/REPORT.md#batch-eight-whole-model-decode) '
                  'uses optimized threshold prefill, a 30,720-token prefix and 1,536 timed steps.']
    if 'retro' in updates:
        lines += ['', 'Each RetroInfer repetition contains one index rebuild and graph recapture.']
    lod_updates = {n for method, n in updates.items() if method.startswith(('lod-', 'retro-lod-'))}
    if lod_updates:
        (count,) = lod_updates
        lines += ['', f'Each LoD repetition includes {count:,} layer-by-request cache updates.']
    if 'retro-multipole' in updates:
        lines += ['', f'Each Multipole repetition includes {updates["retro-multipole"]} online cluster updates.']
    if checks:
        links = ', '.join(f'[{configs[method][0]} {configs[method][1]}]({file})'
                          for method, (file, _) in checks.items())
        lines += ['', f'Same-cache graph and eager checks matched the next token for every request. {links}.']
        if 'retro-multipole' in checks:
            _, max_abs = checks['retro-multipole']
            lines += ['', f'Multipole’s maximum sampled absolute logit difference was {max_abs:.3f}.']
    return lines


if __name__ == '__main__':
    main()
