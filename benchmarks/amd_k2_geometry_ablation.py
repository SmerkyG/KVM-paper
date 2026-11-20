"""Validate and collect the K2 centroid-count and decode-window ablations."""

import argparse
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path

from benchmarks._identity import source_identity
from benchmarks.amd_k2_threshold_ablation import validate_requests, verified_cohort
from benchmarks.amd_threshold_prefill_quality_panel import job_state, write
from benchmarks.amd_threshold_ruler_panel import MODELS, TASKS, digest, paired_interval, read


BASELINE = 'm16-w512'


def candidate_block(settings, baseline):
    (block,) = {options['LOD_BENCH_DECODE_BLOCK']
                for variant, options in settings.items() if variant != baseline}
    return block


def validate_geometry(data, options):
    geometry = data['geometry']
    if not geometry:
        raise ValueError('no pool diagnostics')
    block = int(options.get('LOD_BENCH_DECODE_BLOCK', '256'))
    for pool in geometry:
        if (pool['state_growth_factor'] != int(options['LOD_BENCH_STATE_GROWTH'])
                or pool['configured_window'] != int(options['LOD_BENCH_DECODE_WINDOW'])
                or pool.get('decode_block', 256) != block
                or pool['catch_up_interval'] != block or pool['state_min_len'] != 256
                or pool['prefill_chunk'] != 16384 or pool['prefill_catch_up'] != 16384
                or pool['prefill_local_len'] != 16640 or not pool['exact_first_chunk']):
            raise ValueError(f'wrong runtime geometry: {pool}')


def validate_speed(data, options, sha, job_id):
    if (data['benchmark_identity']['source']['source_sha256'] != sha
            or data['checkpoint'] != MODELS['k2'] or data['mode'] != 'two-tier'
            or data['decode_tokens'] != 1025 or data['batch_size'] != 1
            or set(data['measurements']) != {'65536'}):
        raise ValueError(f'wrong serving check: {job_id}')
    for field, prefix in (('prefill_threshold_options', 'LOD_AMD_PREFILL_'),
                          ('decode_threshold_options', 'LOD_AMD_THRESHOLD_')):
        if data[field] != {k: v for k, v in options.items() if k.startswith(prefix)}:
            raise ValueError(f'wrong {field} in serving check: {job_id}')
    validate_geometry(data, options)


def verify_preflight(root, settings, sha):
    inputs = read(root / 'inputs.json')
    baseline = inputs.get('baseline', BASELINE)
    paired = inputs.get('study') == 'bswa'
    jobs = read(root / 'preflight/jobs.json')['jobs']
    expected = set(settings) - {baseline} if paired else set(settings)
    if len(jobs) != len(expected) or {j['variant'] for j in jobs} != expected:
        raise ValueError('incomplete serving-check submissions')
    probes = {}
    for j in jobs:
        fields, data = job_state(j)
        if not fields:
            continue
        if fields[7:9] != ['finished', '0']:
            raise RuntimeError(f'serving check failed: {j["id"]}')
        validate_speed(data, settings[j['variant']], sha, j['id'])
        if paired:
            control = read(Path(j['control']))
            validate_speed(control, settings[baseline], sha, j['id'])
            control_timing = control['measurements']['65536']
            candidate_timing = data['measurements']['65536']
            if (candidate_timing['prompts'] != control_timing['prompts']
                    or any(len(timing[key]) != 2 for timing in (control_timing, candidate_timing)
                           for key in ('prefill_timings_seconds', 'decode_timings_seconds'))):
                raise ValueError(f'timing pair must match prompts and have two repeats: {j["id"]}')
            data['paired_control'] = control_timing
            probes.setdefault(baseline, control)
        probes[j['variant']] = data
    return probes


def launch(root, ready, settings, sha):
    inputs = read(root / 'inputs.json')
    source = Path(inputs['source'])
    prior = Path(inputs['prior'])
    template = read(Path(inputs['threshold']) / 'jobs.json')['jobs']
    templates = {j['task']: j for j in template if j['variant'] == 'p16-d16'}
    path = root / 'jobs.json'
    manifest = read(path) if path.exists() else {
        'source_sha256': sha, 'jobs': [], 'input_hashes': {
            k: digest(prior / k) for k in ('cohort-manifest.json', 'token-audit.json')
        },
    }
    seen = {(j['variant'], j['task']) for j in manifest['jobs']}
    for task, old in templates.items():
        for variant, options in settings.items():
            if variant not in ready or (variant, task) in seen:
                continue
            # Reuse the exact requests and invocation from the threshold panel.
            command = old['command'].copy()
            start, end = command.index('env') + 1, command.index('uv')
            command[start:end] = [
                f'PYTHONPATH={source}:{source}/integrations/vllm_lod',
                'HF_HUB_OFFLINE=1', 'HF_DATASETS_OFFLINE=1',
                *(f'{k}={v}' for k, v in options.items()),
            ]
            study = (f'bswa{candidate_block(settings, inputs["baseline"])}'
                     if inputs.get('study') == 'bswa' else 'geo')
            command[command.index('--name') + 1] = f'amd-k2-{study}-{variant}-{task}'
            command[command.index('--prefer-nodes') + 1] = inputs['nodes'].split(',')[len(manifest['jobs']) % 4]
            output = root / 'runs' / variant / f'{task}.json'
            command[command.index('--output') + 1] = str(output)
            run = subprocess.run(command, cwd=source, text=True, capture_output=True, check=True)
            match = re.search(r'^Submitted (\S+)', run.stdout, re.MULTILINE)
            if not match:
                raise RuntimeError(run.stdout + run.stderr)
            manifest['jobs'].append({'id': match.group(1), 'variant': variant, 'task': task,
                                     'command': command, 'output': str(output)})
            write(path, manifest)
            print(match.group(1), flush=True)
    return manifest['jobs']


def paired_speed_report(settings, probes, baseline):
    lines = [
        'Each timing pair runs on one GPU in sequence, with one warmup and two measured '
        'requests per setting on the same 64K ProLong prompt. Decode timing covers 1,024 '
        'steps with normal cache catch-up. Greedy continuations can differ. Prefill is '
        'time to first token. Any prefill work deferred to the first decode step is '
        'included in decode.', '',
        '| Candidate | Exact local tokens | Prefill, candidate / control (s) | Decode, candidate / control (ms/token) | Decode delta |',
        '|---|---:|---:|---:|---:|',
    ]
    for variant, options in settings.items():
        if variant == baseline or variant not in probes:
            continue
        probe = probes[variant]
        control = probe['paired_control']
        timing = probe['measurements']['65536']
        window = int(options['LOD_BENCH_DECODE_WINDOW'])
        block = int(options['LOD_BENCH_DECODE_BLOCK'])
        prefill = timing['prefill_seconds']
        decode = timing['decode_ms_per_batch_step']
        reference_decode = control['decode_ms_per_batch_step']
        lines.append(f'| {variant} | {window-block+1}–{window} | {prefill:.2f} / {control["prefill_seconds"]:.2f} | '
                     f'{decode:.2f} / {reference_decode:.2f} | {100*(decode/reference_decode-1):+.1f}% |')
    return lines


def collect(root, jobs, probes, settings, sha):
    inputs = read(root / 'inputs.json')
    baseline = inputs.get('baseline', BASELINE)
    paired = inputs.get('study') == 'bswa'
    prior = Path(inputs['prior'])
    source = Path(inputs['source'])
    tiles = read(root / 'source-manifest.json')['source_files']
    for name, hash_value in tiles.items():
        if digest(source / name) != hash_value:
            raise ValueError(f'changed source: {name}')
    for name, hash_value in read(root / 'jobs.json')['input_hashes'].items():
        if digest(prior / name) != hash_value:
            raise ValueError(f'changed input: {name}')
    cohorts, tokens = verified_cohort(prior)
    expected = {(v, t) for v in settings for t in TASKS}
    if len(jobs) != len(expected) or {(j['variant'], j['task']) for j in jobs} != expected:
        raise ValueError('incomplete or duplicate quality jobs')
    groups = {}
    dest = root / 'provenance'
    dest.mkdir(exist_ok=True)
    spool = Path.home() / '.cluster-run/agent'
    for job in jobs:
        variant, task = job['variant'], job['task']
        fields, data = job_state(job)
        if not fields or fields[7:9] != ['finished', '0']:
            raise ValueError(f'incomplete quality run: {job["id"]}')
        groups[variant, task] = validate_requests(
            data, task, cohorts[task], tokens[task], settings[variant], sha)
        validate_geometry(data, settings[variant])
        shutil.copy2(spool / 'finished' / job['id'], dest / (job['id'] + '.tsv'))
        shutil.copy2(spool / 'logs' / (job['id'] + '.log'), dest / (job['id'] + '.log'))
    task_scores = {
        v: {t: 100 * statistics.mean(row['score'] for row in groups[v, t].values()) for t in TASKS}
        for v in settings
    }
    means = {v: statistics.mean(rows.values()) for v, rows in task_scores.items()}
    intervals = {}
    title = (f'{candidate_block(settings, baseline)}-token BSWA decode blocks'
             if paired else 'centroid count and decode window')
    first_column = 'BSWA block and catch-up' if paired else 'Centroid multiplier'
    lines = [
        f'# K2 Horizon {title} at 64K', '',
        'All 13 RULER tasks use the same 32 requests per setting. K2 Horizon FP8, '
        'two-tier BF16 KV, c=16 in both phases and batch one. Runs keep 16,384-token '
        'prefill chunks and the 16,384-token exact prefix. Prefix caching is disabled.', '',
        f'| {first_column} | Configured decode window | RULER mean | Paired delta vs {baseline}, points | 95% interval |',
        '|---:|---:|---:|---:|---:|',
    ]
    for v, options in settings.items():
        first = options['LOD_BENCH_DECODE_BLOCK' if paired else 'LOD_BENCH_STATE_GROWTH']
        window = options['LOD_BENCH_DECODE_WINDOW']
        mean = means[v]
        if v == baseline:
            lines.append(f'| {first} | {window} | {mean:.2f}% | — | — |')
        else:
            deltas = [[groups[v, t][i]['score'] - groups[baseline, t][i]['score']
                       for i in range(32)] for t in TASKS]
            low, high = paired_interval(deltas)
            intervals[v] = [low, high]
            lines.append(f'| {first} | {window} | {mean:.2f}% | {mean-means[baseline]:+.2f} | [{low:+.2f}, {high:+.2f}] |')
    lines += ['', '| Task | ' + ' | '.join(settings) + ' |', '|---|' + '---:|' * len(settings)]
    for task in TASKS:
        lines.append(f'| {task} | ' + ' | '.join(f'{task_scores[v][task]:.2f}%' for v in settings) + ' |')
    paired_controls = {}
    if paired:
        lines += ['', *paired_speed_report(settings, probes, baseline)]
        paired_controls = {v: p['paired_control'] for v, p in probes.items() if v != baseline}
    else:
        lines += ['', 'The serving checks used 1,024 decode steps on the same ProLong input. '
                  'Their single measured repetitions on separate GPUs validate the configurations '
                  'through multiple catch-up cycles and provide preliminary timings.']
    lines += ['',
              '[Settings](settings.json) · [Serving checks](preflight/jobs.json) · [Commands](jobs.json) · [Scores](REPORT.json)', '']
    write(root / 'REPORT.json', {'stage': 'complete', 'means': means, 'tasks': task_scores,
                                 'paired_bootstrap_95_intervals': intervals,
                                 'serving_checks': {v: p['measurements']['65536'] for v, p in probes.items()},
                                 'paired_speed_controls': paired_controls})
    (root / 'REPORT.md').write_text('\n'.join(lines))
    print(root / 'REPORT.md', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    root = parser.parse_args().output.resolve()
    settings = read(root / 'settings.json')
    inputs = read(root / 'inputs.json')
    sha = source_identity(Path(inputs['source']))['source_sha256']
    reported = set()
    while True:
        probes = verify_preflight(root, settings, sha)
        if inputs.get('study') == 'bswa' and set(probes) != reported:
            (root / 'SPEED.md').write_text('\n'.join([
                f'# K2 Horizon {candidate_block(settings, inputs["baseline"])}-token BSWA timings at 64K', '',
                *paired_speed_report(settings, probes, inputs['baseline']), '',
                '[Timing commands](preflight/jobs.json)', '',
            ]))
            reported = set(probes)
        jobs = launch(root, probes, settings, sha) if probes else []
        complete, examples = 0, 0
        pending = []
        for job in jobs:
            fields, data = job_state(job)
            examples += len(data.get('records', []))
            if fields:
                if fields[7:9] != ['finished', '0'] or data.get('stage') != 'complete':
                    raise RuntimeError(f'quality run failed: {job["id"]}')
                complete += 1
            else:
                pending.append(job['id'])
        progress = {'serving_checks_passed': list(probes), 'quality_jobs': len(jobs),
                    'complete': complete, 'examples': examples}
        write(root / 'status.json', progress)
        print(progress, flush=True)
        if len(probes) == len(settings) and not pending and complete == len(settings) * len(TASKS):
            collect(root, jobs, probes, settings, sha)
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
