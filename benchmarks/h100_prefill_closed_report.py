"""Validate and summarize matched H100 threshold-prefill experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--label', default='final2')
    p.add_argument('--fast-label', default='final3')
    a = p.parse_args()
    root = a.directory
    read = lambda path: json.loads(path.read_text())
    prefix = f'prefill-closed-{a.label}'
    jobs = [j for phase in ('checks', 'controls', 'candidates')
            for j in read(root / f'{prefix}-{phase}-runs.json')['jobs']]
    jobs += read(root / 'prefill-closed-final-dense-runs.json')['jobs']
    metadata = {j['name']: read(root / 'provenance' / j['id'] / 'run.json') for j in jobs}
    if any(m['state'] != 'finished' or m.get('exit_code') != 0 for m in metadata.values()):
        raise ValueError('selected checks or model runs remain incomplete or failed')
    reference = next(iter(metadata.values()))
    evolved = {'benchmarks/h100_comparison.py', 'benchmarks/h100_threshold_prefill_fast.py',
               'benchmarks/h100_threshold_fast_check.py'}
    common = {name: sha for name, sha in reference['source_files'].items() if name not in evolved}
    if any({name: sha for name, sha in m['source_files'].items() if name not in evolved} != common
           or m['hardware'] != reference['hardware'] for m in metadata.values()):
        raise ValueError('common attention/cache source or GPU differs')
    for name, sha in common.items():
        if hashlib.sha256((root / 'source' / name).read_bytes()).hexdigest() != sha:
            raise ValueError('source hash mismatch: ' + name)
    for j in jobs:
        for name in evolved:
            path = root / 'provenance' / j['id'] / 'source' / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata[j['name']]['source_files'][name]:
                raise ValueError('per-run source hash mismatch: ' + j['name'] + '/' + name)
    fast_prefix = f'prefill-closed-{a.fast_label}'
    fixtures = [read(root / (prefix + suffix + '.json')) for suffix in ('-check', '-check-torch213')]
    if any(d['stage'] != 'complete' or len(d['records']) != 7
           or any(r['selection_mismatches'] for r in d['records']) for d in fixtures):
        raise ValueError('independent frontier checks failed')
    models = {}
    summary = {'fixture_torch_versions': [f['torch'] for f in fixtures], 'contexts': {}}
    for length in (32, 64):
        models[length] = {}
        modes = ('top8', 'dense', 'region-legacy', 'region-legacy-w4', 'entry-legacy', 'region-closed') + (('region-fast-native',) if length == 64 else ())
        for mode in modes:
            source_prefix = 'prefill-closed-final' if mode == 'dense' else fast_prefix if mode == 'region-fast-native' else prefix
            name = f'{source_prefix}-{mode}-{length}k'
            d = read(root / (name + '.json'))
            rows = [r for r in d['records'] if not r['warmup']]
            if (d['measurement_scope'] != 'full-model inference' or d['batch_size'] != 1
                    or d['prompt_tokens'] != length * 1024 or d['dtype'] != 'bfloat16'
                    or (mode != 'dense' and not d['prefill_includes_deferred_cache_completion']) or len(rows) != 3
                    or d['output_tokens'] != 2 or not d['replay_prompt_tokens']):
                raise ValueError('model measurement scope differs: ' + name)
            if mode != 'dense' and (d['decode_threshold']['rule'] != 'region' or d['decode_threshold']['multiplier'] != 16):
                raise ValueError('decode changed: ' + name)
            if mode == 'dense' and (d['method'] != 'retro-full' or d['decode_threshold'] is not None):
                raise ValueError('dense control uses sparse attention')
            if d['prefill_threshold'] is not None and d['prefill_threshold']['prefill_block_size'] != 16384:
                raise ValueError('exact-prefill schedule changed: ' + name)
            for key in ('model', 'checkpoint', 'manifest'):
                if mode != 'top8' and d[key] != models[length]['top8']['raw'][key]:
                    raise ValueError('model inputs differ: ' + key)
            models[length][mode] = dict(raw=d, file=name + '.json', ready_s=statistics.median(r['prefill_s'] for r in rows),
                before_completion_s=statistics.median(r.get('prefill_before_cache_completion_s', r['prefill_s']) for r in rows),
                completion_s=statistics.median(r.get('prefill_cache_completion_s', 0.) for r in rows),
                repetitions_s=[r['prefill_s'] for r in rows])
        vals = models[length]
        chosen = min((m for m in vals if m in ('region-closed', 'region-fast-native')), key=lambda m: vals[m]['ready_s'])
        legacy_chosen = min(('region-legacy', 'region-legacy-w4'), key=lambda m: vals[m]['ready_s'])
        summary['contexts'][str(length)] = {mode: {k: v for k, v in x.items() if k != 'raw'} for mode, x in vals.items()}
        summary['contexts'][str(length)]['chosen'] = chosen
        summary['contexts'][str(length)]['legacy_chosen'] = legacy_chosen
        summary['contexts'][str(length)]['region_reduction_pct'] = 100 * (1 - vals[chosen]['ready_s'] / vals[legacy_chosen]['ready_s'])
        summary['contexts'][str(length)]['vs_dense_pct'] = {mode:100 * (x['ready_s'] / vals['dense']['ready_s'] - 1)
                                                         for mode, x in vals.items() if mode != 'dense'}
    check = read(root / (prefix + '-region-closed-64k-check.json'))
    audit = [v for layer in check['records'][0]['prefill_closed_audit'] for v in layer]
    if not audit or any(x['mismatches_away_from_boundary'] for x in audit):
        raise ValueError('real-state frontier checks failed')
    summary['real_state_check'] = dict(query_heads=sum(x['query_heads'] for x in audit),
        selection_mismatches=sum(x['selection_mismatches'] for x in audit),
        exact_max_abs=max(x['exact_max_abs'] for x in audit), coarse_max_abs=max(x['coarse_max_abs'] for x in audit))
    fast_check = read(root / (fast_prefix + '-region-fast-native-64k-check.json'))
    fast_audit = [v for layer in fast_check['records'][0]['prefill_fast_audit'] for v in layer]
    if not fast_audit or any(x['selection_outside_tolerance'] for x in fast_audit):
        raise ValueError('real-state score-reuse checks failed')
    summary['fast_real_state_check'] = dict(query_heads=sum(x['query_heads'] for x in fast_audit),
        selection_mismatches=sum(x['selection_mismatches'] for x in fast_audit),
        max_abs_output_difference=max(x['max_abs_output_difference'] for x in fast_audit))
    lines = ['# H100 two-tier threshold prefill', '',
        'Llama-3.1-8B-Instruct, BF16 model and KV, batch one, H100 80GB HBM3, RetroInfer runtime. '
        'All experiments use uv and cluster-run with the same fixed ProLong prompt at each length. '
        'These are full-model prefill times, median of three '
        'warmed repetitions, with model loading, cache allocation and graph capture excluded. '
        'The initial exact 16,384-token chunk, subsequent prefill chunks, local/sink attention and '
        'cache-update schedule remain unchanged. Decode stays at per-region c=16 in every LoD arm; '
        'the dense control uses native dense attention and its own cache. '
        'Prefill uses eager GPU launches in every arm; CUDA graphs apply only to the separately '
        'timed decode transition.', '',
        '| Prefill routing | 32K (s) | 64K (s) |', '|---|---:|---:|']
    for title, mode in [('Dense', 'dense'), ('Fixed top-eight', 'top8'), ('Per-region c=16, previous fused implementation', 'legacy_chosen'),
                        ('Per-region c=16, optimized', 'chosen'),
                        ('Entry-16, previous optimized implementation', 'entry-legacy')]:
        selected = {n: summary['contexts'][str(n)][mode] if mode in ('chosen', 'legacy_chosen') else mode for n in (32, 64)}
        cells = [f'[{models[n][selected[n]]["ready_s"]:.3f}]({models[n][selected[n]]["file"]})' for n in (32, 64)]
        lines.append(f'| {title} | {cells[0]} | {cells[1]} |')
    lines += ['', 'The optimized per-region path remains ' + ' / '.join(
        f'{summary["contexts"][str(n)]["vs_dense_pct"][summary["contexts"][str(n)]["chosen"]]:.1f}%' for n in (32, 64)) +
        ' slower than dense at 32K / 64K. The dense controls were rerun on the same GPU instance, '
        'checkpoint, precision, prompt and prefill timing scope; these are not the earlier dense '
        'numbers from a different H100 instance.', '',
        'The per-region reduction is ' + ' / '.join(
        f'{summary["contexts"][str(n)]["region_reduction_pct"]:.1f}%' for n in (32, 64)) +
        ' at 32K / 64K. Entry-16 retains its previous implementation: the new grouped, packed-output '
        'and per-query-indexed pilots did not establish a speed improvement. The two routing rules '
        'select different frontiers; the per-region improvement does not apply to entry-16.', '',
        'The previous fused control uses a 4,096-query internal tile. Both one-warp and four-warp '
        'route reduction were measured; the table uses the faster result at each length.', '',
        '## Cache accounting', '',
        'Before stopping the LoD prefill timer, each layer explicitly completes any deferred catch-up '
        'that the harness would otherwise perform at the start of graph setup. This cost is included '
        'in the table and separately recorded as `prefill_cache_completion_s`. The graph setup '
        'timer contains graph preparation and capture. Two output tokens are used to exercise the '
        'transition to decode; that one decode step is not a steady-state decode benchmark.', '',
        '| Context | Top-eight final catch-up | Previous c=16 final catch-up | New c=16 final catch-up |',
        '|---|---:|---:|---:|']
    for n in (32, 64):
        cells = [f'{1000 * models[n][m]["completion_s"]:.3f} ms' for m in ('top8', summary['contexts'][str(n)]['legacy_chosen'], summary['contexts'][str(n)]['chosen'])]
        lines.append(f'| {n}K | ' + ' | '.join(cells) + ' |')
    r = summary['real_state_check']
    lines += ['', '## Implementation and checks', '',
        'The 32K path prepares BF16 means once per remote chunk, computes shared-GQA QK into FP32 '
        'scores, reads those scores once to normalize routing and pack all selected regions, and '
        'computes the closed-summary field directly. Query/head/expert counters are fused into '
        'selection. Exact attention uses persistent region-grouped workers; dispatch visits actual '
        'route counts, and the reducer has a 128-route fast case plus an uncapped fallback. '
        'Selections remain uncapped, including the existing 1,024-entry eligibility limit for '
        'per-region c=16. There is no selected-count CPU readback or GQA KV expansion.', '',
        'The 32K configuration forms closed summaries directly. At 64K the faster configuration '
        'uses the fused coarse producer, retains its native scores and routing normalizer, and '
        'uses a full-row parallel prefix scan instead of a serial block loop for selection. It '
        'subtracts opened summaries in the final merge. A numerical fallback recomputes the '
        'closed field when subtraction would lose precision. It uses the same persistent exact '
        'workers and count-aware dispatch. Routing normalization uses FP32 saved dot products '
        'and query RMS; the older fused control rounds the normalized query to BF16 before '
        'its routing matmul, so this is not a claim of bitwise-equivalent old/new selections.', '',
        f'The seven independent closed-summary/native-leaf cases pass on PyTorch {fixtures[0]["torch"]} and '
        f'{fixtures[1]["torch"]}, including uneven and empty regions, entry-work guards, more than '
        '128 selected regions, internal query-tile remainders and CUDA-graph replay. '
        f'Real model checks at 64K cover {r["query_heads"]:,} sampled query/head frontiers: '
        f'{r["selection_mismatches"]} selection differences; maximum exact-output difference '
        f'{r["exact_max_abs"]:.6f} against the independent scalar kernel and coarse-output '
        f'difference {r["coarse_max_abs"]:.6f} against PyTorch. These are numerical checks; '
        'the full RULER panel has not been rerun. Serving defaults are unchanged.', '',
        f'The 64K score-reuse path additionally checks {summary["fast_real_state_check"]["query_heads"]:,} '
        f'sampled query/head frontiers, with {summary["fast_real_state_check"]["selection_mismatches"]} '
        'selection differences and maximum final-output difference '
        f'{summary["fast_real_state_check"]["max_abs_output_difference"]:.6f} against its reference. '
        'The full-row selector also passes three uncapped synthetic cases, including concentrated '
        'coarse mass and precision fallback. The native CUDA regression suite passes 21 tests.', '',
        'Full-model measurements use PyTorch 2.5.1 and Triton 3.7.1. Its compiled RetroInfer '
        'dependencies remain in their existing environment. The separate PyTorch 2.13 check uses '
        'the native FP32-output bmm interface; the older environment uses its existing cuBLAS '
        'handle for the same BF16-input, FP32-output operation. No full-model PyTorch-version '
        'speedup is claimed.', '',
        '## Reproduction and limits', '',
        'Set `LOD_H100_PREFILL_RULE=region`, `LOD_H100_PREFILL_MULTIPLIER=16`, '
        '`LOD_H100_PREFILL_IMPLEMENTATION=closed`, `LOD_H100_CLOSED_MATMUL=blas`, '
        '`LOD_H100_CLOSED_METADATA=atomic`, `LOD_H100_CLOSED_SELECT_QM=1`, '
        '`LOD_H100_CLOSED_REDUCE=wide`, `LOD_H100_CLOSED_LEAF_M=64`, '
        '`LOD_H100_CLOSED_LEAF_N=32`, and `LOD_H100_CLOSED_LEAF_WARPS=4`. '
        'The measured internal query tile is 4,096 and the persistent worker count is 264. '
        'The internal query tile bounds scratch; it does not change the exact-prefill chunk size.', '',
        'For the 64K score-reuse configuration, use the same leaf/worker settings with '
        '`LOD_H100_PREFILL_IMPLEMENTATION=fast`, `LOD_H100_FAST_LEAF=native`, '
        '`LOD_H100_FAST_SAVED_CORRECTION=1`, `LOD_H100_FAST_FULL_SELECT=1`, and '
        '`LOD_H100_FAST_TILE=4096`. The exact commands '
        'in the selected run manifests are authoritative.', '',
        'Routing selection and packing use linear prefix scans in region count. Shared score '
        'production still visits every query/region pair, and the cache path still performs '
        'ownership assignment, clustering and periodic maintenance. This report makes no new '
        'subquadratic claim for complete prefill or amortized cache updates. The deterministic '
        'entry-work bound belongs to the entry rule and is not a coverage or retrieval guarantee.', '',
        'Raw pilots and failures remain alongside the selected runs. Exclusions include the first '
        'unsupported old-PyTorch bmm attempt, missing pytest in an early test environment, a '
        'cancelled broad three-tier compile, invalid inherited entry-only options in the first '
        'region controls, and a grouped-reducer address-overflow pilot fixed by 64-bit addressing. '
        'Smaller exact-attention tiles, grouped-query selection/reduction and packed outputs '
        'did not beat the selected configurations. Commands, GPU identity, logs and '
        'source hashes are in `provenance/`; `source/` contains the latest selected snapshot. '
        'Per-run copies preserve the three evolving benchmark modules. All selected runs share '
        'identical cache and leaf-kernel sources; the harness change between the final rounds '
        'only records the new selection option in result metadata.', '']
    (root / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    (root / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
