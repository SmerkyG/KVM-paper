"""Collect the coupled prefill block and three-tier page-budget studies."""

import argparse
import shutil
import statistics
import time
from pathlib import Path

from benchmarks.amd_k2_fixed_budget_timing import measurements
from benchmarks.amd_k2_threshold_ablation import validate_requests, verified_cohort
from benchmarks.amd_threshold_prefill_quality_panel import job_state, write
from benchmarks.amd_threshold_ruler_panel import MODELS, digest, read


PAGE_TASKS = ('niah_multikey_2', 'niah_multikey_3')
DOCS = (14, 19, 20, 23, 24, 25, 27, 28)


def join_latency(measurement):
    return statistics.median(p + d / 1000 for p, d in zip(
        measurement['prefill_samples_s'], measurement['decode_samples_ms'], strict=True))


def matches_threshold_options(data, settings):
    return all(data[field] == {k: v for k, v in settings.items() if k.startswith(prefix)}
               for field, prefix in (
                   ('prefill_threshold_options', 'LOD_AMD_PREFILL_'),
                   ('decode_threshold_options', 'LOD_AMD_THRESHOLD_')))


def memory_metrics(data, options):
    (stats,) = data['worker_prefill_memory']
    if stats['configured_prefill_block_tokens'] != int(options['LOD_BENCH_PREFILL_BLOCK']):
        raise ValueError('wrong worker prefill block')
    if stats['configured_prefill_lookback_tokens'] != 256:
        raise ValueError('wrong prefill lookback')
    if not 0 < stats['peak_allocated_bytes'] <= stats['peak_reserved_bytes']:
        raise ValueError('invalid memory high-water mark')
    return stats


def prefill_report(jobs, settings, manifest):
    expected = {('speed', k) for k in set(settings) - {'16k'}} | {('quality', k) for k in settings}
    if {(j['measure'], j['variant']) for j in jobs} != expected:
        raise ValueError('incomplete prefill runs')
    speed, quality = {}, {}
    prompt = quality_prompts = None
    lines = [
        '# K2 coupled prefill block ablation at 64K', '',
        'K2 Horizon FP8, two-tier BF16 KV, batch one, region c=16 in both phases. '
        'The scheduler chunk, length of the first exact chunk and prefill cache catch-up '
        'move together. The subsequent exact lookback is 256 tokens and the internal '
        'query tile limit is 4,096. Each speed pair uses one GPU and the same prompt with three '
        'warmed repeats. Two-token latency includes time through the first decode token, '
        'including any deferred cache construction and one decode step.', '',
        '| Block vs 16K | First token, candidate / control (s) | Two tokens, candidate / control (s) | Peak PyTorch reserved, candidate / control (GiB) |',
        '|---|---:|---:|---:|',
    ]
    for job in jobs:
        variant = job['variant']
        data = read(Path(job['output']))
        options = settings[variant]
        chunk = int(options['LOD_BENCH_PREFILL_BLOCK'])
        peak = memory_metrics(data, options)
        if job['measure'] == 'speed':
            control = read(Path(job['control']))
            if data['benchmark_identity']['runtime'] != control['benchmark_identity']['runtime']:
                raise ValueError(f'mismatched speed runtime: {variant}')
            a = measurements(data, options, manifest, prefill_chunk=chunk, steps=1)
            b = measurements(control, settings['16k'], manifest, steps=1)
            if a['prompt'] != b['prompt'] or (prompt is not None and a['prompt'] != prompt):
                raise ValueError('mismatched speed prompt')
            prompt = a['prompt']
            bp = memory_metrics(control, settings['16k'])
            ready_a, ready_b = join_latency(a), join_latency(b)
            speed[variant] = {'candidate': a, 'control': b, 'two_token_candidate_s': ready_a,
                              'two_token_control_s': ready_b, 'memory': peak, 'control_memory': bp}
            lines.append(f'| {variant} | {a["prefill_s"]:.3f} / {b["prefill_s"]:.3f} | '
                         f'{ready_a:.3f} / {ready_b:.3f} | '
                         f'{peak["peak_reserved_bytes"]/2**30:.2f} / {bp["peak_reserved_bytes"]/2**30:.2f} |')
        else:
            if (data['checkpoint'] != MODELS['k2'] or data['measure'] != 'quality'
                    or data['mode'] != 'two-tier' or data['batch_size'] != 1
                    or data['scheduler_chunk_tokens'] != chunk
                    or data['benchmark_identity']['source']['source_sha256'] != manifest['source_sha256']
                    or not matches_threshold_options(data, options)):
                raise ValueError(f'mismatched prompt-loss settings: {variant}')
            results = data['measurements']
            if results['prediction_tokens'] != 8 * 65535:
                raise ValueError('wrong number of prompt tokens')
            docs = [{k: v for k, v in row.items() if k not in ('loss', 'perplexity')}
                    for row in results['samples']]
            if tuple(doc['dataset_index'] for doc in docs) != DOCS:
                raise ValueError(f'wrong document indices: {variant}')
            if quality_prompts is not None and docs != quality_prompts:
                raise ValueError('prompt-loss documents differ')
            quality_prompts = docs
            quality[variant] = {**results, 'memory': peak}
    lines += ['', 'Memory is the worker’s PyTorch allocation high-water mark, including setup '
              'and warmup. It excludes memory managed outside PyTorch.', '',
              '| Block | Prompt loss | Loss delta vs 16K | Perplexity |', '|---|---:|---:|---:|']
    for variant in settings:
        loss = quality[variant]['loss']
        lines.append(f'| {variant} | {loss:.6f} | {loss-quality["16k"]["loss"]:+.6f} | '
                     f'{quality[variant]["perplexity"]:.6f} |')
    if (set(settings) == {'8k', '16k', '32k'}
            and any(job.get('previous_attempt') for job in jobs)):
        lines += ['', 'The first launch initialized ROCm before vLLM forked its worker. The '
                  'completed runs read the same block size directly from the environment. '
                  '[First attempts](../k2-prefill-block-ablation-v1/jobs.json).']
    return lines, {'speed': speed, 'quality': quality}


def page_work(rows, variant, task, settings):
    entries, pages = [], []
    fixed_pages = int(settings['env'].get('LOD_AMD_THRESHOLD_PAGES', 0))
    for row in rows.values():
        routing = row['last_decode_routing']
        if len(routing) != 64:
            raise ValueError(f'incomplete routing counters: {variant}/{task}')
        for layer in routing:
            entries.extend(layer['exact_entries'])
            if variant != 'two-tier':
                counts = layer['exact_pages']
                if any(x > 16 * p or x < p for x, p in zip(layer['exact_entries'], counts, strict=True)):
                    raise ValueError('entry and page counters disagree')
                if fixed_pages and any(p > fixed_pages * r for p, r in zip(counts, layer['regions'], strict=True)):
                    raise ValueError(f'fixed page budget exceeded: {variant}/{task}')
                pages.extend(counts)
    if len(entries) != 32 * 64 * 64 or not any(entries):
        raise ValueError(f'invalid exact-entry counts: {variant}/{task}')
    return {
        "exact_entries_per_head": statistics.mean(entries),
        "exact_pages_per_head": statistics.mean(pages) if pages else None,
    }


def page_report(jobs, settings, manifest):
    expected_quality = {(v, task) for v in settings for task in PAGE_TASKS}
    actual_quality = {(j['variant'], j['task']) for j in jobs if j['measure'] == 'quality'}
    if actual_quality != expected_quality or {j['variant'] for j in jobs if j['measure'] == 'speed'} != set(settings) - {'two-tier'}:
        raise ValueError('incomplete page runs')
    cohorts, tokens = verified_cohort(Path(manifest['prior']))
    speed, retrieval = {}, {}
    prompt = None
    for j in jobs:
        variant = j['variant']
        data = read(Path(j['output']))
        opts = settings[variant]
        if j['measure'] == 'speed':
            baseline = read(Path(j['control']))
            if data['benchmark_identity']['runtime'] != baseline['benchmark_identity']['runtime']:
                raise ValueError(f'mismatched speed runtime: {variant}')
            a = measurements(data, opts['env'], manifest, mode=opts['mode'])
            b = measurements(baseline, settings['two-tier']['env'], manifest)
            if a['prompt'] != b['prompt'] or (prompt is not None and a['prompt'] != prompt):
                raise ValueError('mismatched speed prompt')
            prompt = a['prompt']
            speed[variant] = {'candidate': a, 'control': b, 'decode_change_percent':
                              100 * (a['decode_ms']/b['decode_ms'] - 1)}
            continue
        task = j['task']
        rows = validate_requests(data, task, cohorts[task], tokens[task], opts['env'],
                                 manifest['source_sha256'], mode=opts['mode'])
        retrieval.setdefault(variant, {})[task] = {
            'score': 100 * statistics.mean(r['score'] for r in rows.values()),
            **page_work(rows, variant, task, opts),
        }
    lines = [
        '# K2 three-tier page-budget diagnostic at 64K', '',
        'All runs use native top-eight prefill plus c=16 region routing in decode. '
        'The retrieval diagnostic uses 32 matched requests per task and BF16 KV. '
        'Opened-work counts average across query heads, layers, and the last decode step '
        'of each request. The two-tier row opens regions directly.', '',
        '| Decode refinement | Multikey-2 | Multikey-3 | Exact pages / head | Exact remote entries / head |',
        '|---|---:|---:|---:|---:|',
    ]
    for variant in settings:
        a, b = (retrieval[variant][t] for t in PAGE_TASKS)
        p = '—' if a['exact_pages_per_head'] is None else f'{statistics.mean((a["exact_pages_per_head"], b["exact_pages_per_head"])):.1f}'
        e = statistics.mean((a['exact_entries_per_head'], b['exact_entries_per_head']))
        lines.append(f'| {variant} | {a["score"]:.2f}% | {b["score"]:.2f}% | {p} | {e:.1f} |')
    lines += ['', 'Same-GPU speed pairs use one matched 64K prompt, one warmup and three repeats. '
              'Decode includes 1,024 timed steps and cache catch-up. The timing runs use the '
              'unmodified serving work for each arm, with fixed-page and two-tier diagnostic counters disabled.', '',
              '| Refinement | Decode, candidate / two-tier (ms/token) | Decode change |', '|---|---:|---:|']
    for variant in (key for key in settings if key != 'two-tier'):
        row = speed[variant]
        lines.append(f'| {variant} | {row["candidate"]["decode_ms"]:.2f} / {row["control"]["decode_ms"]:.2f} | '
                     f'{row["decode_change_percent"]:+.1f}% |')
    return lines, {'speed': speed, 'retrieval': retrieval}


def collect(root, study, manifest):
    if digest(root / 'settings.json') != manifest['settings_sha256']:
        raise ValueError('changed study settings')
    source_manifest = read(root / 'source-manifest.json')
    source = Path(source_manifest['source'])
    for file, sha in source_manifest['source_files'].items():
        if digest(source / file) != sha:
            raise ValueError(f'changed frozen source: {file}')
    settings = read(root / 'settings.json')
    jobs = manifest['jobs']
    if study == 'page' and 'control_study' in manifest:
        prior = Path(manifest['control_study'])
        if digest(prior / 'REPORT.json') != manifest['control_report_sha256']:
            raise ValueError('changed page control report')
        prior_manifest = read(prior / 'jobs.json')
        if (prior_manifest['source_sha256'] != manifest['source_sha256']
                or prior_manifest['prior'] != manifest['prior']):
            raise ValueError('mismatched page control source or cohort')
        if digest(prior / 'settings.json') != prior_manifest['settings_sha256']:
            raise ValueError('changed page control settings')
        prior_settings = read(prior / 'settings.json')
        if any(settings[variant] != options for variant, options in prior_settings.items()):
            raise ValueError('mismatched page control settings')
        jobs = [*prior_manifest['jobs'], *jobs]
        order = ('two-tier', 'fixed-1', 'fixed-2', 'fixed-4', 'adaptive-0.5')
        settings = {variant: settings[variant] for variant in order}
    report = prefill_report if study == 'prefill' else page_report
    lines, results = report(jobs, settings, {**manifest, 'measured_repeats': 3})
    dest = root / 'provenance'
    dest.mkdir(exist_ok=True)
    spool = Path.home() / '.cluster-run/agent'
    for job in jobs:
        shutil.copy2(spool / 'finished' / job['id'], dest / (job['id'] + '.tsv'))
        shutil.copy2(spool / 'logs' / (job['id'] + '.log'), dest / (job['id'] + '.log'))
    lines += ['', '[Settings](settings.json) · [Commands](jobs.json) · [Samples](REPORT.json)', '']
    if study == 'page' and 'control_study' in manifest:
        lines += [f'[Other refinement runs](../{prior.name}/REPORT.md)', '']
    write(root / 'REPORT.json', {'stage': 'complete', **results})
    (root / 'REPORT.md').write_text('\n'.join(lines))
    print(root / 'REPORT.md', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', choices=('prefill', 'page'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    while True:
        manifest = read(root / 'jobs.json')
        jobs = manifest['jobs']
        complete, failures = 0, []
        for job in jobs:
            fields, data = job_state(job)
            if not fields:
                continue
            if fields[7:9] == ['finished', '0'] and data:
                complete += 1
            else:
                failures.append(job['id'])
        progress = {'jobs': len(jobs), 'complete': complete, 'failures': failures}
        write(root / 'status.json', progress)
        print(progress, flush=True)
        if failures:
            raise RuntimeError(f'failed runs: {failures}')
        if jobs and complete == len(jobs):
            collect(root, args.study, manifest)
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
