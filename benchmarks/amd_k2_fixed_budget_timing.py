"""Audit and report paired K2 region-routing serving timings."""

import argparse
import shutil
import statistics
import time
from pathlib import Path

from benchmarks.amd_threshold_prefill_quality_panel import job_state, write
from benchmarks.amd_threshold_ruler_panel import MODELS, digest, read


CONTEXT = '65536'
DECODE_STEPS = 1024


def measurements(data, settings, manifest, *, mode='two-tier', prefill_chunk=16384, steps=DECODE_STEPS):
    if (data['checkpoint'] != MODELS['k2'] or data['mode'] != mode
            or data['measure'] != 'speed' or data['batch_size'] != 1
            or data['tensor_parallel_size'] != 1 or data['decode_tokens'] != steps + 1
            or data['scheduler_chunk_tokens'] != prefill_chunk or set(data['measurements']) != {CONTEXT}
            or data['benchmark_identity']['source']['source_sha256'] != manifest['source_sha256']):
        raise ValueError('unexpected benchmark identity or run geometry')
    for field, prefix in (('prefill_threshold_options', 'LOD_AMD_PREFILL_'),
                          ('decode_threshold_options', 'LOD_AMD_THRESHOLD_')):
        if data[field] != {k: v for k, v in settings.items() if k.startswith(prefix)}:
            raise ValueError(f'mismatched {field}')
    output = data['measurements'][CONTEXT]
    prefill = output['prefill_timings_seconds']
    decode = output['decode_timings_seconds']
    if (len(prefill) != manifest['measured_repeats'] or len(decode) != len(prefill)
            or any(t <= 0 for t in prefill + decode) or len(output['prompts']) != 1):
        raise ValueError('incomplete timing repeats or prompt metadata')
    return {
        'prefill_samples_s': prefill,
        'decode_samples_ms': [1000 * time_s / steps for time_s in decode],
        'prefill_s': statistics.median(prefill),
        'decode_ms': 1000 * statistics.median(decode) / steps,
        'prompt': output['prompts'][0],
        'greedy_output_identical_across_repeats': output['greedy_output_identical'],
        'output_token_sha256': output['output_token_sha256'],
    }


def collect(root, study='fixed-budget'):
    manifest = read(root / 'jobs.json')
    source = Path(manifest['source'])
    settings = read(root / 'settings.json')
    if study == 'threshold':
        quality_paths = {Path(path): sha for path, sha in manifest['quality_sha256'].items()}
        multipliers = manifest.get('multipliers', (4, 8, 32, 64))
        expected = [key for c in multipliers for key in (f'p{c}-d16', f'p16-d{c}')]
        control = 'p16-d16'
        source_manifest = source.parent / 'source-manifest.json'
    else:
        quality_paths = {root.parent / 'REPORT.json': manifest['quality_sha256']}
        expected = [f'{phase}-top{k}' for phase in ('prefill', 'decode') for k in (4, 8, 16)]
        control = 'threshold'
        source_manifest = root.parent / 'source-manifest.json'
    inputs = {**quality_paths, root / 'settings.json': manifest['settings_sha256'],
              root / 'run_pair.py': manifest['runner_sha256']}
    quality = {}
    for path, sha in inputs.items():
        if digest(path) != sha:
            raise ValueError(f'changed study input: {path}')
        if path in quality_paths:
            report = read(path)
            if report['stage'] != 'complete':
                raise ValueError(f'incomplete quality study: {path}')
            for key, score in report['means'].items():
                if key in quality and quality[key] != score:
                    raise ValueError(f'conflicting quality scores for {key}')
                quality[key] = score
    frozen = read(source_manifest)
    for name, sha in frozen['source_files'].items():
        if digest(source / name) != sha:
            raise ValueError(f'changed frozen source: {name}')
    if len(manifest['jobs']) != len(expected) or {j['variant'] for j in manifest['jobs']} != set(expected):
        raise ValueError('missing or duplicate pairs')
    rows = {}
    matched_prompt = None
    provenance = root / 'provenance'
    provenance.mkdir(exist_ok=True)
    spool = Path.home() / '.cluster-run/agent'
    for job in manifest['jobs']:
        fields, candidate = job_state(job)
        if not fields or fields[7:9] != ['finished', '0']:
            raise ValueError(f'job incomplete or failed: {job["id"]}')
        baseline = read(Path(job['control']))
        variant = job['variant']
        a = measurements(candidate, settings[variant], manifest)
        b = measurements(baseline, settings[job.get('control_variant', control)], manifest)
        if a['prompt'] != b['prompt'] or (matched_prompt is not None and a['prompt'] != matched_prompt):
            raise ValueError(f'prompt mismatch: {job["id"]}')
        if candidate['benchmark_identity']['runtime'] != baseline['benchmark_identity']['runtime']:
            raise ValueError(f'runtime mismatch: {job["id"]}')
        matched_prompt = a['prompt']
        rows[variant] = {
            'candidate': a,
            'control': b,
            'prefill_change_percent': 100 * (a['prefill_s'] / b['prefill_s'] - 1),
            'decode_change_percent': 100 * (a['decode_ms'] / b['decode_ms'] - 1),
            'ruler_mean': quality[variant],
            'ruler_delta_points': quality[variant] - quality[control],
            'node': fields[2], 'gpu': fields[3],
        }
        shutil.copy2(spool / 'finished' / job['id'], provenance / (job['id'] + '.tsv'))
        shutil.copy2(spool / 'logs' / (job['id'] + '.log'), provenance / (job['id'] + '.log'))
    if study == 'threshold':
        title = 'K2 region mass threshold timings at 64K'
        recipe = 'The threshold multiplier changes in one phase and the other stays at c=16.'
        quality_link = '[Quality](../REPORT.md)'
        if 'multipliers' not in manifest:
            quality_link += (' · [c=4 quality](../../k2-threshold-c4-v1/REPORT.md)'
                             ' · [c=64 quality](../../k2-threshold-c64-v1/REPORT.md)')
        if 'timing_note' in manifest:
            recipe += ' ' + manifest['timing_note']
        headings = [f'{phase} c={c}' for c in multipliers for phase in ('Prefill', 'Decode')]
    else:
        title = 'K2 normalized fixed region budget timings at 64K'
        recipe = ('A fixed budget replaces c=16 in one phase. The other stays at c=16. '
                  'These timings characterize the selectors from the matched quality panel.')
        quality_link = '[Quality](../REPORT.md)'
        headings = [f'{phase.capitalize()} top-{k}' for phase in ('prefill', 'decode') for k in (4, 8, 16)]
    lines = [
        f'# {title}', '',
        f'K2 Horizon FP8, two-tier BF16 KV, batch one. {recipe} '
        'Each pair uses the same MI325X and ProLong prompt, one warmup and three measured repeats '
        'per setting. Decode includes 1,024 steps and periodic cache updates. The continuation '
        'is generated greedily and can differ by method.', '',
        'Prefill is time from scheduling to the first token. Work deferred into the first '
        'decode step counts toward decode.', '',
        f'The full c=16 quality control scored {quality[control]:.2f}% on the '
        'matched 32-example-per-task RULER panel.', '',
        '| Routing rule | RULER mean | Prefill, candidate / control (s) | Prefill change | Decode, candidate / control (ms/token) | Decode change |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for variant, heading in zip(expected, headings, strict=True):
        row = rows[variant]
        a, b = row['candidate'], row['control']
        lines.append(
            f'| {heading} | {row["ruler_mean"]:.2f}% | '
            f'{a["prefill_s"]:.2f} / {b["prefill_s"]:.2f} | {row["prefill_change_percent"]:+.1f}% | '
            f'{a["decode_ms"]:.2f} / {b["decode_ms"]:.2f} | {row["decode_change_percent"]:+.1f}% |'
        )
    lines += ['', f'{quality_link} · [Samples](REPORT.json) · [Timing commands](jobs.json)', '']
    write(root / 'REPORT.json', {'stage': 'complete', 'quality_control': quality[control], 'results': rows})
    (root / 'REPORT.md').write_text('\n'.join(lines))
    print(root / 'REPORT.md', flush=True)


def status(root):
    jobs = read(root / 'jobs.json')['jobs']
    complete = results = 0
    failures = []
    for job in jobs:
        fields, data = job_state(job)
        results += bool(data) + Path(job['control']).exists()
        if fields:
            if fields[7:9] == ['finished', '0'] and data and Path(job['control']).exists():
                complete += 1
            else:
                failures.append(job['id'])
    progress = {'pairs': len(jobs), 'complete': complete, 'settings_measured': results, 'failures': failures}
    write(root / 'status.json', progress)
    print(progress, flush=True)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('status', 'collect', 'watch'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--study', choices=('fixed-budget', 'threshold'), default='fixed-budget')
    args = parser.parse_args()
    root = args.output.resolve()
    if args.action == 'status':
        status(root)
    elif args.action == 'collect':
        collect(root, args.study)
    else:
        while True:
            progress = status(root)
            if progress['failures']:
                raise RuntimeError(f'failed timing pairs: {progress["failures"]}')
            if progress['pairs'] and progress['complete'] == progress['pairs']:
                collect(root, args.study)
                return
            time.sleep(30)


if __name__ == '__main__':
    main()
