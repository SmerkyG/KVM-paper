"""Audit and report three-tier threshold retrieval and amortized decode tests."""
import argparse
import json
from pathlib import Path
import shutil
import statistics

from benchmarks.amd_threshold_ruler_panel import digest, read, write
from benchmarks.amd_threshold_ruler import score


def status(root):
    for phase in sorted(root.glob('*/jobs.json')):
        done = failed = canceled = records = 0
        superseded = phase.parent / 'superseded.json'
        stopped = {j['id'] for j in read(superseded)['jobs']} if superseded.exists() else set()
        for job in read(phase)['jobs']:
            final = Path.home() / '.cluster-run/agent/finished' / job['id']
            if final.exists():
                fields = final.read_text().strip().split('\t')
                done += fields[8] == '0'
                canceled += fields[8] != '0' and job['id'] in stopped
                failed += fields[8] != '0' and job['id'] not in stopped
            output = Path(job['output']) if 'output' in job else None
            if output is not None and output.exists():
                records += len(read(output).get('records', []))
        print(phase.parent.name, json.dumps(dict(jobs=len(read(phase)['jobs']), complete=done, failed=failed, superseded_canceled=canceled, records=records)))


def collect(root, serving, probe_phase, check_phase):
    provenance = []
    checked_sources = set()

    def source_check(source):
        source = Path(source)
        if str(source) in checked_sources:
            return
        manifest = read(source.parent / 'source-manifest.json')
        for name, sha in manifest['source_files'].items():
            if digest(source / name) != sha:
                raise ValueError(f'changed frozen source: {source}/{name}')
        checked_sources.add(str(source))

    def finished(job, phase):
        final = Path.home() / '.cluster-run/agent/finished' / job['id']
        if not final.exists():
            raise ValueError(f'job still running: {job["id"]}')
        fields = final.read_text().strip().split('\t')
        if fields[7:9] != ['finished', '0']:
            raise ValueError(f'failed job: {job["id"]}: {fields[7:9]}')
        dest = root / 'provenance'
        dest.mkdir(exist_ok=True)
        shutil.copy2(final, dest / (job['id'] + '.tsv'))
        log = Path.home() / '.cluster-run/agent/logs' / (job['id'] + '.log')
        shutil.copy2(log, dest / log.name)
        provenance.append({**job, 'phase': phase, 'node': fields[2], 'gpus': fields[3],
            'started': fields[5], 'ended': fields[6], 'status': fields[7], 'exit_code': fields[8]})

    check = read(root / check_phase / 'jobs.json')
    source_check(check['source'])
    for job in check['jobs']:
        finished(job, check_phase)
    reference = read(Path(check['jobs'][0].get('output', str(root / check_phase / 'frontier.json'))))
    if reference['stage'] != 'complete' or len(reference['records']) != 24 or any(r['selection_mismatches'] for r in reference['records']):
        raise ValueError('incomplete independent frontier checks')
    page_log = root / 'provenance' / (check['jobs'][1]['id'] + '.log')
    if '66 passed' not in page_log.read_text():
        raise ValueError('existing page-opening tests did not pass')
    phase = root / serving
    jobs = read(phase / 'jobs.json')
    source_check(jobs['source'])
    if 'reused_controls_source' in jobs:
        source_check(jobs['reused_controls_source'])
    settings = read(phase / 'settings.json')
    manifest = read(phase / 'cohort-manifest.json')
    tokens = read(Path(manifest['token_audit']))['models']
    tasks = manifest['tasks']
    quality = {}
    speed = {}
    for job in jobs['jobs']:
        finished(job, serving)
        data = read(Path(job['output']))
        if data['stage'] != 'complete':
            raise ValueError(f'incomplete output: {job["id"]}')
        key = (job['model'], job['mode'])
        if job['phase'] == 'quality':
            k = (*key, job['task'], job['variant'])
            if k in quality:
                raise ValueError(f'duplicate quality run: {k}')
            quality[k] = data
        else:
            if key in speed or len(data['jobs']) != 3 or any(j['exit_code'] for j in data['jobs']):
                raise ValueError(f'invalid serial speed runs: {key}')
            speed[key] = data
    expected_quality = {(m, p, t, v) for m in ('qwen', 'k2') for p in ('three-tier-bf16', 'three-tier-int4') for t in tasks for v in ('top8', 'threshold')}
    if set(quality) != expected_quality or set(speed) != {(m, p) for m in ('qwen', 'k2') for p in ('three-tier-bf16', 'three-tier-int4')}:
        raise ValueError('missing comparison groups')
    report = dict(stage='complete', serving_phase=serving, probe_phase=probe_phase,
                  independent_cases=24, native_page_tests=66, quality=[], speed=[], attention=[])
    changed = []
    for model, mode in sorted(speed):
        selected_settings = settings[model][mode] if mode in settings[model] else settings[model]
        for task in tasks:
            item = manifest['models'][model][task]
            frozen = Path(item['frozen'])
            if digest(frozen) != item['sha256']:
                raise ValueError(f'changed cohort: {frozen}')
            cohort = {r['doc_id']: r for r in map(json.loads, frozen.read_text().splitlines())}
            expected_tokens = {r['doc_id']: r for r in tokens[model]['tasks'][task]}
            variants = {}
            for variant in ('top8', 'threshold'):
                data = quality[(model, mode, task, variant)]
                expected_options = {} if variant == 'top8' else selected_settings
                expected_checkpoint = {'qwen': 'Qwen/Qwen3.8-27B-FP8', 'k2': 'IFM/K2-Horizon-32B-FP8'}[model]
                if (data['checkpoint'] != expected_checkpoint or data['mode'] != mode or data['batch_size'] != 1 or data['generation_seed'] != 1234
                        or data['selection'] != 'prefix' or data['limit_per_task'] != 32 or data['tasks'] != [task]
                        or data['scheduler_chunk_tokens'] != 16384 or data['decode_threshold_options'] != expected_options
                        or len(data['records']) != 32):
                    raise ValueError(f'bad quality metadata: {model}/{mode}/{task}/{variant}')
                records = {r['doc_id']: r for r in data['records']}
                if set(records) != set(cohort):
                    raise ValueError('inconsistent example IDs')
                for i, r in records.items():
                    ref = cohort[i]
                    if (any(r[k] != ref[k] for k in ('doc_hash', 'prompt_hash', 'target_hash'))
                            or any(r[k] != expected_tokens[i][k] for k in ('input_tokens', 'input_token_sha256'))
                            or r['targets'] != ref['doc']['outputs'] or abs(score(ref, r['prediction'])-r['score']) > 1.e-12):
                        raise ValueError(f'inconsistent request/scoring: {model}/{mode}/{task}/{i}')
                variants[variant] = records
            a, b = variants['top8'], variants['threshold']
            delta = [b[i]['score']-a[i]['score'] for i in sorted(cohort)]
            row = dict(model=model, mode=mode, task=task, dense=100*statistics.mean(r['archived_dense_score'] for r in cohort.values()),
                       top8=100*statistics.mean(r['score'] for r in a.values()), threshold=100*statistics.mean(r['score'] for r in b.values()),
                       difference_points=100*statistics.mean(delta), improved=sum(x>0 for x in delta), declined=sum(x<0 for x in delta))
            report['quality'].append(row)
            changed += [dict(model=model, mode=mode, task=task, doc_id=i, targets=a[i]['targets'], top8=a[i], threshold=b[i])
                        for i in sorted(cohort) if a[i]['prediction'] != b[i]['prediction']]
        measurements = {}
        prompt = None
        for run in speed[(model, mode)]['jobs']:
            data = read(Path(run['output']))
            variant = run['variant']
            if (data['mode'] != mode or data['batch_size'] != 1 or data['decode_tokens'] != 1025
                    or data['measure'] != 'speed' or data['scheduler_chunk_tokens'] != 16384
                    or run['command'][run['command'].index('--repeats')+1] != '2'
                    or data['decode_threshold_options'] != (selected_settings if variant == 'fast' else {})):
                raise ValueError(f'bad speed metadata: {model}/{mode}/{variant}')
            d = data['measurements']['65536']
            hashes = [p['token_sha256'] for p in d['prompts']]
            if prompt is not None and prompt != hashes:
                raise ValueError('different speed prompts')
            prompt = hashes
            if len(d['decode_timings_seconds']) != 2 or len(d['prefill_timings_seconds']) != 2:
                raise ValueError('missing measured speed repetitions')
            measurements[variant] = d
        control = [t*1000/1024 for v in ('top8', 'top8-after') for t in measurements[v]['decode_timings_seconds']]
        fast = measurements['fast']['decode_ms_per_batch_step']
        control_outputs = {tuple(h) for name in ('top8', 'top8-after')
                           for h in measurements[name]['output_token_sha256']}
        report['speed'].append(dict(model=model, mode=mode, control_median_ms=statistics.median(control),
            threshold_ms=fast, change_percent=100*(fast/statistics.median(control)-1),
            before_ms=measurements['top8']['decode_ms_per_batch_step'], after_ms=measurements['top8-after']['decode_ms_per_batch_step'],
            prefill_before_s=measurements['top8']['prefill_seconds'], prefill_threshold_s=measurements['fast']['prefill_seconds'],
            prefill_after_s=measurements['top8-after']['prefill_seconds'], measurements=measurements,
            native_control_outputs_identical=len(control_outputs)==1,
            greedy_repeats_identical={name:d['greedy_output_identical'] for name,d in measurements.items()}))
    probe_jobs = read(root / probe_phase / 'jobs.json')
    source_check(probe_jobs['source'])
    for job in probe_jobs['jobs']:
        finished(job, probe_phase)
        data = read(Path(job['output']))
        selected_settings = settings[job['model']][job['mode']] if job['mode'] in settings[job['model']] else settings[job['model']]
        workers = int(selected_settings['LOD_AMD_THRESHOLD_PAGE_WORKERS'])
        warps = int(selected_settings['LOD_AMD_THRESHOLD_PAGE_WARPS'])
        candidate = [r for r in data['records'] if r['implementation']=='three-tier' and r['page_workers']==workers and r['page_warps']==warps]
        native = [r for r in data['records'] if r['implementation']=='native-top8']
        if data['stage'] != 'complete' or len(candidate) != 3 or len(native) != 3 or any(r['selection_mismatches'] for r in candidate):
            raise ValueError(f'incomplete same-cache kernel comparison: {job["id"]}')
        report['attention'].append(dict(model=job['model'], mode=job['mode'],
            native_us=statistics.mean(r['microseconds'] for r in native), threshold_us=statistics.mean(r['microseconds'] for r in candidate),
            mean_opened_regions=statistics.mean(r['mean_opened_regions'] for r in candidate),
            mean_exact_remote_entry_bound=statistics.mean(r['mean_exact_remote_entry_bound'] for r in candidate)))
    audit = read(root / 'native-audit-v1/jobs.json')
    source_check(audit['source'])
    if digest(Path(audit['old_source'])) != audit['old_source_sha256']:
        raise ValueError('changed pre-extension page source')
    report['native_pre_change_audit'] = []
    for job in audit['jobs']:
        finished(job, 'native-audit-v1')
        data = read(Path(job['output']))
        rows = [r for r in data['records'] if r['implementation']=='native-top8']
        if data['stage'] != 'complete' or len(rows) != 3:
            raise ValueError('incomplete native pre-change audit')
        report['native_pre_change_audit'].append(dict(model=job['model'], mode=job['mode'],
            identical=all(r['pre_change_page_output_identical'] for r in rows),
            max_abs=max(r['max_abs_vs_pre_change_pages'] for r in rows)))
    lines = ['# AMD three-tier threshold decode', '',
        'Qwen3.8-27B-FP8 and K2-Horizon-32B-FP8, batch one, nominal 64K context, BF16 or INT4 KV. '
        'The experiment selects every eligible region with normalized-query proxy mass above 16 times the mean region mass. '
        'Each selected region opens the existing two highest-scoring pages, plus one disjoint residual summary. '
        'Regions over 1,024 entries remain coarse, as in serving. Page ranking and INT4 reconstruction use the native implementation. '
        'Prefill retains top-8 routing and the exact first 16,384-token chunk. The implementation is opt-in; serving defaults are unchanged.', '',
        '## Amortized full-model decode', '',
        'One GPU per model/precision runs native top-8 → threshold → native top-8 serially. Each variant warms up, '
        'then measures two 1,024-step generations, including periodic cache catch-up and quantization maintenance. '
        'Time change compares threshold to the median of all four native-control repetitions. Prompts match; generated continuations can differ.', '',
        '| Model | KV | Native before | Threshold | Native after | Time change vs control median |',
        '|---|---|---:|---:|---:|---:|']
    for r in report['speed']:
        lines.append(f'| {r["model"].upper()} | {r["mode"].removeprefix("three-tier-").upper()} | {r["before_ms"]:.3f} ms | {r["threshold_ms"]:.3f} ms | {r["after_ms"]:.3f} ms | {r["change_percent"]:+.1f}% |')
    drift = [f'{r["model"].upper()}/{r["mode"].removeprefix("three-tier-").upper()}' for r in report['speed'] if not r['native_control_outputs_identical']]
    if drift:
        lines += ['', 'The native before/after controls generated different continuations despite matching prompt hashes for '
                  + ', '.join(drift) + '. The bracket therefore measures both timing and continuation variability; '
                  'small differences should not be read as an isolated kernel-cost effect. Raw repetition hashes and timing ranges are retained.']
    lines += ['', '## Same-query/cache attention', '',
        'Graph-captured complete attention calls on actual queries and retained serving states from first/middle/last LoD layers. '
        'All methods include the same local-length rollback. Model projections, FFNs and cache catch-up are excluded. '
        'Native top-8 opens at most eight regions; threshold uses its actual variable region budget. This is not a work-matched comparison.', '',
        '| Model | KV | Native top-8 | Threshold | Mean regions/head | Exact remote-entry bound/head |',
        '|---|---|---:|---:|---:|---:|']
    for r in report['attention']:
        lines.append(f'| {r["model"].upper()} | {r["mode"].removeprefix("three-tier-").upper()} | {r["native_us"]:.2f} µs | {r["threshold_us"]:.2f} µs | {r["mean_opened_regions"]:.2f} | ≤{r["mean_exact_remote_entry_bound"]:.1f} |')
    lines += ['', 'The entry bound is 32 times the opened-region count, averaged across sampled heads/layers; partial or singleton pages reduce actual work. '
        'It is not a measured exact-entry count. Closed and residual summaries are additional frontier items.', '',
        '## Retrieval screen', '',
        'The first 32 examples of Multikey 2 and Multikey 3 per model, with identical frozen requests, generation settings and seed 1234 across methods and KV precisions. '
        'These are selected retrieval tasks, not a full 13-task RULER score. Dense uses the same archived requests in an older runtime and is descriptive. '
        'Token hashes are checked against the earlier independent CPU token audit.', '',
        '| Model | KV | Task | Archived dense | Fresh top-8 | Threshold | Difference, points | Improved / declined |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in report['quality']:
        lines.append(f'| {r["model"].upper()} | {r["mode"].removeprefix("three-tier-").upper()} | {r["task"]} | {r["dense"]:.2f} | {r["top8"]:.2f} | {r["threshold"]:.2f} | {r["difference_points"]:+.2f} | {r["improved"]} / {r["declined"]} |')
    identical = all(r['identical'] for r in report['native_pre_change_audit'])
    lines += ['', 'The fresh batch-one K2 top-8 controls are substantially below the historical 500-example three-tier report, '
        'which used a different serving revision and schedule. This discrepancy is unresolved, so gains here should not be read '
        'as reproducing an improvement over that historical baseline. A separate [pre-extension native-kernel audit](native-audit-v1/jobs.json) '
        + ('found bitwise-identical full native attention outputs on all 12 sampled real-query/cache cases across both models and KV precisions.'
           if identical else 'found small output differences; inspect the raw audit before attributing the baseline discrepancy.')]
    lines += ['', '## Prefill timing', '', '| Model | KV | Native before | Threshold | Native after |', '|---|---|---:|---:|---:|']
    for r in report['speed']:
        lines.append(f'| {r["model"].upper()} | {r["mode"].removeprefix("three-tier-").upper()} | {r["prefill_before_s"]:.3f} s | {r["prefill_threshold_s"]:.3f} s | {r["prefill_after_s"]:.3f} s |')
    lines += ['', '## Enable the experiment', '',
        'Use the existing `three-tier-bf16` or `three-tier-int4` model mode and add '
        '`LOD_AMD_THRESHOLD_MULTIPLIER=16` to the cluster-run environment. The new path defaults to '
        '64 persistent page workers, normalized-query per-region mass and the precision correction. '
        'The selected settings JSON below records all flags explicitly. The current experiment supports '
        'batch-one ordinary decode on the two tested AMD model geometries; it rejects other routing rules or speculative/batched decode. '
        'The default path without the multiplier remains native top-8.', '',
        '## Correctness and work audit', '',
        'All 24 independent PyTorch frontier cases and all 66 existing page-opening tests passed. Checks cover both model geometries and KV precisions, '
        'partial/singleton pages, empty selections and empty state, region/page ties, overflow hashing, oversized-region exclusion, '
        'multiple regions per worker, KV append, and repeated graph replay.', '',
        'The router computes QK scores once. Selected region IDs use one prefix scan and one scatter, with linear packing work; KV is shared in place. '
        'A fixed worker count processes only valid selected regions on the device, without sorting, CPU count reads or a top-k cap. '
        'Closed-summary value products use the probability-residual precision correction; selected-page attention and residuals retain native FP32 accumulation. '
        'The coarse stage still repeats the tile-normalizer reduction for each routing tile, so that metadata arithmetic is quadratic in tile count. '
        'Prefill and the existing archive catch-up implementation are preserved. These measurements do not establish subquadratic work for complete execution.', '',
        f'[Selected settings]({serving}/settings.json), [serving commands]({serving}/jobs.json), '
        f'[kernel probes]({probe_phase}/jobs.json), [frozen cohorts]({serving}/cohort-manifest.json), '
        '[job status and placement](provenance.json), [raw summary](REPORT.json), [changed predictions](changed-examples.json). '
        'Earlier tuning phases remain separately recorded and do not replace the selected measurements above. '
        'The [superseded eight-worker candidates](serving-v1/superseded.json) include deliberately canceled runs; '
        'their partial quality and speed outputs are excluded. Native quality controls from that phase are retained '
        'with their original commands and frozen source.', '']
    write(root / 'REPORT.json', report)
    write(root / 'changed-examples.json', changed)
    write(root / 'provenance.json', provenance)
    (root / 'REPORT.md').write_text('\n'.join(lines))
    print(root / 'REPORT.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('status', 'collect'))
    parser.add_argument('--root', type=Path, default=Path('results/amd-threshold-three-tier'))
    parser.add_argument('--serving', default='serving-v2')
    parser.add_argument('--probe', default='probe-v3')
    parser.add_argument('--check', default='check-v2')
    args = parser.parse_args()
    if args.action == 'status':
        status(args.root.resolve())
    else:
        collect(args.root.resolve(), args.serving, args.probe, args.check)


if __name__ == '__main__':
    main()
