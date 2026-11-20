"""Collect two additional K2 threshold settings, one phase at a time."""

import argparse
import json
import math
import re
import statistics
import subprocess
import time
from pathlib import Path

from benchmarks.amd_k2_threshold_ablation import NUM_EXAMPLES, validate_requests, verified_cohort
from benchmarks.amd_threshold_prefill_quality_panel import job_state, write
from benchmarks.amd_threshold_ruler_panel import TASKS, paired_interval, read


def launch_setting(root, jobs, base, settings, source, multiplier, phase, *, shard_size=NUM_EXAMPLES):
    variant = f'p{multiplier}-d16' if phase == 'prefill' else f'p16-d{multiplier}'
    submitted = {(j['task'], j.get('first_doc_id', 0)) for j in jobs if j['variant'] == variant}
    for job in base['jobs']:
        if job['variant'] != 'p16-d16':
            continue
        task = job['task']
        for start in range(0, NUM_EXAMPLES, shard_size):
            if (task, start) in submitted:
                continue
            command = job['command'].copy()
            name = f'amd-k2-c{multiplier}-{phase}-{task}'
            output = root / 'runs' / variant / f'{task}.json'
            row = {}
            if shard_size < NUM_EXAMPLES:
                name += f'-{start}'
                output = output.parent / task / f'{start}.json'
                index = command.index('--samples-root') + 1
                source_cohort = Path(command[index]) / task / 'samples_frozen.jsonl'
                requests = sorted(map(json.loads, source_cohort.read_text().splitlines()),
                                  key=lambda request: request['doc_id'])
                requests = requests[start:start + shard_size]
                subset = root / 'cohort-shards' / str(start)
                dest = subset / task / source_cohort.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(''.join(json.dumps(record) + '\n' for record in requests))
                command[index] = str(subset)
                command[command.index('--limit') + 1] = str(len(requests))
                row = {'first_doc_id': start, 'doc_ids': [r['doc_id'] for r in requests]}
            command[command.index('--name') + 1] = name
            for key, value in settings[variant].items():
                old_value = job['options'][key]
                command[command.index(f'{key}={old_value}')] = f'{key}={value}'
            command[command.index('--output') + 1] = str(output)
            result = subprocess.run(command, cwd=source, text=True, capture_output=True, check=True)
            match = re.search(r'^Submitted (\S+)', result.stdout, re.MULTILINE)
            if not match:
                raise RuntimeError(result.stdout + result.stderr)
            jobs.append({**job, **row, 'id': match.group(1), 'variant': variant, 'output': str(output),
                         'command': command, 'options': settings[variant]})
            write(root / 'jobs.json', {'jobs': jobs})
            print(jobs[-1]['id'], flush=True)


def collect(root, base, jobs, settings, cohorts, tokens, source_sha, multiplier):
    if {(j['variant'], j['task']) for j in jobs} != {
        (variant, task) for variant in settings for task in TASKS
    }:
        raise ValueError('missing or duplicate supplement jobs')
    study = [job for job in base['jobs'] if job['variant'] == 'p16-d16'] + jobs
    data = {}
    for job in study:
        fields, run = job_state(job)
        if not fields or fields[7:9] != ['finished', '0']:
            raise RuntimeError(f'job incomplete: {job["id"]}')
        task, variant = job['task'], job['variant']
        doc_ids = job.get('doc_ids', cohorts[task])
        subset = {i: cohorts[task][i] for i in doc_ids}
        if variant != 'p16-d16' and job['options'] != settings[variant]:
            raise ValueError(f'wrong threshold settings for {job["id"]}')
        rows = validate_requests(run, task, subset, tokens[task], job['options'], source_sha)
        combined = data.setdefault((variant, task), {})
        if combined.keys() & rows.keys():
            raise ValueError(f'duplicate requests for {variant}/{task}')
        combined.update(rows)
    variants = ('p16-d16', *settings)
    for variant in variants:
        for task in TASKS:
            if data[variant, task].keys() != cohorts[task].keys():
                raise ValueError(f'incomplete requests for {variant}/{task}')
    scores = {
        v: {t: 100 * statistics.mean(r['score'] for r in data[v, t].values()) for t in TASKS}
        for v in variants
    }
    means = {v: statistics.mean(s.values()) for v, s in scores.items()}
    comparisons = {}
    prefill_tile = settings[f'p{multiplier}-d16']['LOD_AMD_PREFILL_QUERY_TILE']
    lines = [
        f'# K2 Horizon c={multiplier} supplement at 64K', '',
        'Same frozen 32 prompts in all 13 tasks, c=16 control, '
        'and 16,384-token exact prefix as the [five-setting sweep](../k2-threshold-ablation-v1/REPORT.md).', '',
        f'The c={multiplier} prefill arm uses a {int(prefill_tile):,}-query internal tile. '
        'The c=16 prefill and its decode-only sweep use the original 4,096-query tile.', '',
        '| Prefill c | Decode c | 13-task mean | Paired delta vs (16,16), points | Paired 95% interval |',
        '|---:|---:|---:|---:|---:|',
        f'| 16 | 16 | {means["p16-d16"]:.2f}% | — | — |',
    ]
    for prefill, decode in ((16, multiplier), (multiplier, 16)):
        variant = f'p{prefill}-d{decode}'
        deltas = [[data[variant, task][i]['score'] - data['p16-d16', task][i]['score']
                   for i in range(32)] for task in TASKS]
        low, high = paired_interval(deltas)
        change = means[variant] - means['p16-d16']
        comparisons[variant] = {
            'delta_points': change, 'paired_bootstrap_95_interval_points': [low, high],
            'improved': sum(x > 1e-12 for d in deltas for x in d),
            'declined': sum(x < -1e-12 for d in deltas for x in d),
        }
        lines.append(f'| {prefill} | {decode} | {means[variant]:.2f}% | {change:+.2f} | [{low:+.2f}, {high:+.2f}] |')
    lines += ['', f'| Task | (16, 16) | (16, {multiplier}) | ({multiplier}, 16) |', '|---|---:|---:|---:|']
    for task in TASKS:
        lines.append(f'| {task} | ' + ' | '.join(f'{scores[v][task]:.2f}%' for v in variants) + ' |')
    links = '[Settings](settings.json) · [Jobs](jobs.json) · [Scores](REPORT.json)'
    if (root / 'preflight/jobs.json').exists():
        links += ' · [Memory check](preflight/jobs.json)'
    lines += ['', links, '']
    write(root / 'REPORT.json', {'stage': 'complete', 'means': means, 'tasks': scores, 'comparisons': comparisons})
    (root / 'REPORT.md').write_text('\n'.join(lines))
    print(root / 'REPORT.md', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    root = parser.parse_args().output.resolve()
    inputs = read(root / 'inputs.json')
    base = read(Path(inputs['base']) / 'jobs.json')
    source_sha = inputs['source_sha256']
    cohorts, tokens = verified_cohort(Path(inputs['prior']))
    settings = read(root / 'settings.json')
    multiplier = inputs.get('multiplier', 4)
    shard_size = inputs.get('prefill_shard_size', NUM_EXAMPLES)
    expected_jobs = len(TASKS) * (1 + math.ceil(NUM_EXAMPLES / shard_size))
    jobs_path = root / 'jobs.json'
    jobs = read(jobs_path)['jobs'] if jobs_path.exists() else []
    path = root / 'preflight/jobs.json'
    preflight = read(path)['jobs'] if path.exists() else []
    launch_setting(root, jobs, base, settings, inputs['source'], multiplier, 'decode')
    while True:
        passed, failures = 0, []
        for job in preflight:
            fields, data = job_state(job)
            if not fields:
                continue
            if fields[7:9] != ['finished', '0']:
                failures.append(job['id'])
                continue
            task = job['task']
            rows = {i: cohorts[task][i] for i in job['doc_ids']}
            validate_requests(data, task, rows, tokens[task], settings[f'p{multiplier}-d16'], source_sha)
            passed += 1
        if preflight and passed == len(preflight):
            launch_setting(root, jobs, base, settings, inputs['source'], multiplier, 'prefill', shard_size=shard_size)
        complete = examples = 0
        for job in jobs:
            fields, data = job_state(job)
            examples += len(data.get('records', []))
            if fields:
                if fields[7:9] != ['finished', '0'] or data.get('stage') != 'complete':
                    failures.append(job['id'])
                else:
                    complete += 1
        progress = {'prefill_checks_passed': passed, 'jobs': len(jobs), 'complete': complete,
                    'examples': examples, 'failures': failures}
        write(root / 'status.json', progress)
        print(progress, flush=True)
        if failures:
            raise RuntimeError(f'failed jobs: {failures}')
        if complete == expected_jobs:
            collect(root, base, jobs, settings, cohorts, tokens, source_sha, multiplier)
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
