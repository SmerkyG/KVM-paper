"""Summarize matched H100 entry-threshold prefill experiments.

Inputs are collected from logged cluster runs; this command does not run GPU
experiments. It rejects incomplete or mismatched multikey comparison cohorts.
"""
import argparse
import json
import os
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def cohort(data):
    return {(r['task'], r['index'], r['input_token_sha256']) for r in data['records']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--reference-quality', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory
    lines = [
        '# H100 two-tier entry-16 prefill optimization', '',
        'Llama-3.1-8B-Instruct, BF16, batch 1. LoD keeps the existing 16,384-token '
        'prefill chunks, including the exact first chunk. The 512-query value in '
        'the run configurations is an internal routing tile, not a change to '
        'the prefill chunk size. The strict entry-16 limit is unchanged.', '',
        'Timings measure full model prefill and the corresponding cache work '
        'in the RetroInfer runtime. Cache initialization and graph capture are '
        'timed separately and excluded. The dense reference uses dense prefill '
        'and its own cache; its cache work differs from LoD. Each cell is the '
        'median of two warmed runs, in seconds; this small sample does not '
        'establish small performance differences.', '',
        '| Method | 32K prefill (s) | 64K prefill (s) |',
        '|---|---:|---:|',
    ]
    names = [
        ('Dense', 'dense', 'dense'),
        ('Fixed top-8 LoD', 'top8', 'top8'),
        ('Entry-16, previous GPU-to-CPU count readback', 'control', 'control'),
        ('Entry-16, reuse routing data', 'reuse-entry-16-tile512', None),
        ('Entry-16, device-only padded buffers', 'nosync-entry-16-tile512', None),
        ('Entry-16, fused packing into padded buffers', 'pack-entry-16-tile512', None),
        ('Entry-16, paired-query KV reuse', 'pair64-entry-16-tile512', None),
        ('Entry-16, device-only packed buffers', 'packed-entry-16-tile512', None),
        ('Entry-16, packed buffers and routing-data reuse', 'packedreuse-entry-16-tile512', 'packedreuse'),
        ('Entry-16, packed and streamed region expansion', 'packedstream-entry-16-tile512', None),
        ('Entry-16, packed and wider selection', 'packed128-entry-16-tile512', 'packed128'),
    ]
    for title, short, long in names:
        cells = []
        for prefix, n in ((short, 32768), (long, 65536)):
            if not prefix:
                cells.append('—')
                continue
            path = directory / f'threshold-prefill-next-{prefix}-{n}-timing.json'
            if not path.exists():
                cells.append('pending')
                continue
            data = read(path)
            measured = [r['prefill_s'] for r in data['records'] if not r['warmup']]
            if (len(measured) != 2 or data.get('prompt_tokens') != n
                    or data.get('batch_size') != 1 or data.get('dtype') != 'bfloat16'
                    or data.get('measurement_scope') != 'full-model inference'):
                raise ValueError(f'incomplete timing: {path}')
            cells.append(f'[{statistics.median(measured):.3f}]({path.name})')
        lines.append(f'| {title} | {cells[0]} | {cells[1]} |')

    lines += ['', 'The packed path computes offsets on the GPU and stores selected '
              'regions and entries contiguously. It removes the per-tile host '
              'readback of route counts; it still reserves buffers from the '
              'worst-case work bound. The other device-only variants were tested '
              'separately and are not implicitly combined with it. These paths '
              'are experimental, controlled by environment variables, and do '
              'not change serving defaults.', '']
    lines += ['For the fastest tested option, use '
              '`LOD_H100_PREFILL_PACKED=1`, `LOD_H100_PREFILL_REUSE_MEAN=1`, '
              'and `LOD_H100_PREFILL_SELECT_N=128` together with the fused '
              'entry-16 prefill settings. The packed path implies no host '
              'count readback.', '']
    profile = directory / 'threshold-prefill-next-packedreuse-32768-profile.json'
    if profile.exists():
        record = read(profile)['records'][-1]
        stages = record['prefill_profile']
        if len(stages) != 32:
            raise ValueError(f'expected one profile per model layer: {profile}')
        names = ('route_logits', 'select_and_coarse', 'compact', 'expand_indices', 'exact_leaves', 'merge')
        values = {name: sum(stage.get(name, 0) for stage in stages) for name in names}
        lines += ['A separate warmed diagnostic explicitly synchronizes the GPU '
                  'between stages. Across all 32 layers it measured '
                  + ', '.join(f'{name.replace("_", " ")} {values[name]:.3f}s' for name in names)
                  + f' ([profile]({profile.name})). These are diagnostic stage '
                  'timings, not a replacement for the end-to-end timings above.', '']
    lines += ['## Correctness and retrieval', '']
    sample = directory / 'threshold-prefill-next-packedreuse-entry-16-ruler32k-verify.json'
    if sample.exists():
        data = read(sample)
        checks = [v for r in data['records'] for v in r.get('prefill_fused_validation', [])]
        lines.append('On three 32K RULER examples, checks compared '
                     f"{sum(c['query_heads'] for c in checks):,} query-head frontiers with "
                     f"the reference: {sum(c['selection_mismatches'] for c in checks)} "
                     'selection mismatches. The largest attention output absolute '
                     f"difference was {max(c['max_abs'] for c in checks):.5f}. "
                     'Reused routing logits were checked bit-for-bit against the '
                     f'[original routing computation]({sample.name}).')
        lines.append('')
    ref = read(args.reference_quality)
    lines += ['| 32K multikey retrieval | Multikey 2 | Multikey 3 |',
              '|---|---:|---:|',
              f"| [Previous entry-16]({args.reference_quality.name}) | {ref['task_scores']['niah_multikey_2']:.1f}% | {ref['task_scores']['niah_multikey_3']:.1f}% |"]
    for title, variant in [('Routing-data reuse', 'reuse'), ('Packed and reuse', 'packedreuse'),
                           ('Packed, reuse, wider selection', 'packed128')]:
        path = directory / f'threshold-prefill-next-{variant}-entry-16-ruler32k-audit8.json'
        if not path.exists():
            lines.append(f'| {title} | pending | pending |')
            continue
        data = read(path)
        if data.get('stage') != 'complete' or cohort(data) != cohort(ref):
            raise ValueError(f'incomplete or mismatched quality cohort: {path}')
        s = data['task_scores']
        lines.append(f"| [{title}]({path.name}) | {s['niah_multikey_2']:.1f}% | {s['niah_multikey_3']:.1f}% |")
    lines += ['', 'Each score covers eight matched examples per task, not the full '
              'RULER panel. The strict bound concerns logically opened remote '
              'KV entries per query head; it does not establish a subquadratic '
              'bound for complete prefill or amortized cache updates. Run '
              'commands, source hashes, and logs are in the provenance '
              'directory alongside these results.', '']
    # The reference can be outside the result directory; keep its link usable.
    lines = [line.replace(f']({args.reference_quality.name})',
                          f']({Path(os.path.relpath(args.reference_quality, args.output.parent))})')
             for line in lines]
    args.output.write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
