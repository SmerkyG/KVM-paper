"""Report matched H100 entry-16 prefill runs capped at eight or sixteen regions."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--label', default='cap16')
    args = parser.parse_args()
    root = args.directory
    prefix = f'threshold-fast-{args.label}'

    def read(name):
        return json.loads((root / (name + '.json')).read_text())

    manifest = read(prefix + '-runs')
    provenance = [json.loads((root / 'provenance' / j['id'] / 'run.json').read_text())
                  for j in manifest['jobs']]
    if any(p['state'] != 'finished' or p['exit_code'] != 0 for p in provenance):
        raise RuntimeError('all comparison and validation runs must finish successfully')
    if any(p['source_files'] != provenance[0]['source_files'] for p in provenance):
        raise RuntimeError('comparison source snapshots differ')
    if any(p['hardware'] != provenance[0]['hardware'] for p in provenance):
        raise RuntimeError('comparison GPU identities differ')
    timings = {}
    for mode in ('top8', 'cap8', 'cap16'):
        timings[mode] = {}
        for length in (32, 64):
            result = read(f'{prefix}-{mode}-{length}k')
            if len([r for r in result['records'] if not r['warmup']]) != 3:
                raise RuntimeError('expected three measured repetitions')
            timings[mode][length] = result['median_prefill_s']

    lines = ['# Entry-16 prefill capped at sixteen regions', '',
             'BF16 Llama-3.1-8B-Instruct, batch 1, H100 80GB. Full-model prefill includes cache updates; initialization and decode graph capture are excluded. Median of three measured repetitions after one warmup, with identical replayed prompts. Every GPU experiment ran with uv through the logged cluster-run adapter.', '',
             'The two capped variants apply the same entry-16 threshold and strict opened-entry work guard, then retain the first eight or sixteen qualifying regions in region order. They do not choose the highest-scoring eight or sixteen. Remaining regions stay coarse. The exact initial prefill chunk remains 16,384 tokens, internal tiles remain 8,192 queries, and the cache update schedule is unchanged. Both use the same score-reuse producer and grouped exact-attention kernel.', '',
             '| Configuration | 32K prefill | 64K prefill |', '|---|---:|---:|']
    for mode, label in [('top8', 'Original top-8 routing'), ('cap8', 'Entry-16 capped at 8 regions'),
                        ('cap16', 'Entry-16 capped at 16 regions')]:
        cells = [f'[{timings[mode][length]:.3f} s]({prefix}-{mode}-{length}k.json)' for length in (32, 64)]
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    increases = [100 * (timings['cap16'][length] / timings['cap8'][length] - 1) for length in (32, 64)]
    gaps = [100 * (timings['cap16'][length] / timings['top8'][length] - 1) for length in (32, 64)]
    lines += ['', f'Raising the cap from eight to sixteen changes latency by {increases[0]:+.1f}% at 32K and {increases[1]:+.1f}% at 64K. The sixteen-region version is {gaps[0]:.1f}% slower than original top-8 at 32K and {gaps[1]:.1f}% slower at 64K.', '',
              '| 64K audited configuration | Mean opened regions/head | Mean opened entries/head | Query heads reaching the cap | Maximum opened regions |',
              '|---|---:|---:|---:|---:|']
    audits = {}
    for mode, cap in [('cap8', 8), ('cap16', 16)]:
        result = read(f'{prefix}-{mode}-64k-audit')
        checks = [c for layer in result['records'][0]['prefill_fast_audit'] for c in (layer or [])]
        full = sum(c['full_tile_query_heads'] for c in checks)
        audit = {
            'sampled_query_heads': sum(c['query_heads'] for c in checks),
            'selection_mismatches': sum(c['selection_mismatches'] for c in checks),
            'selection_outside_tolerance': sum(c['selection_outside_tolerance'] for c in checks),
            'max_output_error': max(c['max_abs_output_difference'] for c in checks),
            'full_tile_query_heads': full,
            'mean_opened_regions': sum(c['full_tile_opened_regions'] for c in checks) / full,
            'mean_opened_entries': sum(c['full_tile_opened_entries'] for c in checks) / full,
            'fraction_reaching_cap': sum(c['full_tile_queries_at_cap'] for c in checks) / full,
            'max_opened_regions': max(c['full_tile_max_regions'] for c in checks),
            'max_opened_entries': max(c['full_tile_max_entries'] for c in checks),
            'numerical_fallbacks': sum(c['full_tile_numerical_fallbacks'] for c in checks),
        }
        if audit['max_opened_regions'] > cap or audit['selection_outside_tolerance']:
            raise RuntimeError('invalid cap or routing selection')
        audits[mode] = audit
        lines.append(f'| Cap {cap} | {audit["mean_opened_regions"]:.2f} | {audit["mean_opened_entries"]:.2f} | {audit["fraction_reaching_cap"]:.2%} | {audit["max_opened_regions"]} |')
    lines += ['', 'Counts cover the first 8,192-query internal tile of each remote chunk across all layers, in each configuration’s own evolving model state. They exclude the second internal tile. No query-wide CPU synchronization is introduced into timed attention.', '',
              '| 64K correctness check | Sampled query heads | Changed region decisions | Outside 2e-5 boundary tolerance | Maximum attention output difference |',
              '|---|---:|---:|---:|---:|']
    for mode, cap in [('cap8', 8), ('cap16', 16)]:
        c = audits[mode]
        lines.append(f'| Cap {cap} | {c["sampled_query_heads"]:,} | {c["selection_mismatches"]} | {c["selection_outside_tolerance"]} | {c["max_output_error"]:.6f} |')
    lines += ['', 'The independent reference checks selection, the strict entry-work bound, and attention output using the new score-reuse arithmetic. Synthetic checks also cover uneven sizes, both caps, a tighter work guard, and concentrated mass that exercises numerical fallback. These are implementation checks; no RULER quality panel was run for cap sixteen. A cap changes the retrieval frontier and does not inherit the uncapped method’s quality.', '',
              'The prior uncapped entry-16 timings and detailed pipeline scope remain in [the producer comparison](REPORT.md). This follow-up reruns cap eight and original top-eight controls with the same source snapshot as cap sixteen.', '',
              'Reproduce with the earlier fast settings and `LOD_H100_PREFILL_MAX_REGIONS=16`. These remain opt-in experiments; serving defaults are unchanged. Commands, GPU identity, source hashes, validation, and raw timings are retained alongside this report.']
    (root / 'CAP16.md').write_text('\n'.join(lines) + '\n')
    (root / 'CAP16_SUMMARY.json').write_text(json.dumps({'timings': timings, 'audits': audits}, indent=2) + '\n')


if __name__ == '__main__':
    main()
