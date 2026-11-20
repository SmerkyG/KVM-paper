"""Submit and score fixed-K RULER runs after verifying both GPU selectors."""

import argparse
import re
import statistics
import subprocess
import time
from pathlib import Path

from benchmarks._identity import source_identity
from benchmarks.amd_k2_threshold_ablation import validate_requests, verified_cohort
from benchmarks.amd_threshold_prefill_quality_panel import job_state, write
from benchmarks.amd_threshold_ruler_panel import TASKS, digest, paired_interval, read


def verified_probe(root, settings, cohorts, tokens, sha):
    ready = set()
    expected = {'reference', 'prefill-top16', 'decode-top16'}
    jobs = read(root / 'preflight/jobs.json')['jobs']
    if len(jobs) != len(expected) or {j['variant'] for j in jobs} != expected:
        raise ValueError('missing selector checks')
    for job in jobs:
        fields, data = job_state(job)
        if not fields:
            continue
        if fields[7:9] != ['finished', '0'] or data.get('stage') != 'complete':
            raise RuntimeError(f'selector check failed: {job["id"]}')
        variant = job['variant']
        if variant == 'reference':
            if not data['checks'] or any(r['budgets'] != [4, 8, 16] for r in data['checks']):
                raise ValueError('missing fixed-budget cases')
        else:
            task = job['task']
            subset = {i: cohorts[task][i] for i in job['doc_ids']}
            validate_requests(data, task, subset, tokens[task], settings[variant], sha)
        ready.add(variant)
    return ready


def submit(root, inputs, settings):
    source = inputs['source']
    templates = read(Path(inputs['base']) / 'jobs.json')['jobs']
    path = root / 'jobs.json'
    jobs = read(path)['jobs'] if path.exists() else []
    seen = {(j['variant'], j['task']) for j in jobs}
    for old in templates:
        if old['variant'] != 'p16-d16':
            continue
        task = old['task']
        for variant, options in settings.items():
            if (variant, task) in seen:
                continue
            command = old['command'].copy()
            begin = command.index('env') + 1
            command[begin:command.index('uv')] = [
                f'PYTHONPATH={source}:{source}/integrations/vllm_lod',
                'HF_HUB_OFFLINE=1', 'HF_DATASETS_OFFLINE=1',
                *(f'{k}={v}' for k, v in options.items()),
            ]
            command[command.index('--name') + 1] = f'amd-k2-fixed-{variant}-{task}'
            output = root / 'runs' / variant / f'{task}.json'
            command[command.index('--output') + 1] = str(output)
            result = subprocess.run(command, cwd=source, text=True, capture_output=True, check=True)
            match = re.search(r'^Submitted (\S+)', result.stdout, re.MULTILINE)
            if not match:
                raise RuntimeError(result.stdout + result.stderr)
            jobs.append({'id': match.group(1), 'variant': variant, 'task': task,
                         'output': str(output), 'command': command})
            write(path, {'jobs': jobs})
            print(jobs[-1]['id'], flush=True)
    return jobs


def comparison(data, variant):
    deltas = [
        [data[variant, task][i]['score'] - data['threshold', task][i]['score'] for i in range(32)]
        for task in TASKS
    ]
    return {'delta_points': 100 * statistics.mean(statistics.mean(x) for x in deltas),
            'paired_bootstrap_95_interval_points': paired_interval(deltas)}


def collect(root, jobs, settings, inputs, cohorts, tokens, sha):
    expected = {(v, t) for v in settings for t in TASKS}
    if len(jobs) != len(expected) or {(j['variant'], j['task']) for j in jobs} != expected:
        raise ValueError('incomplete or duplicate quality jobs')
    source = Path(inputs['source'])
    for name, hash_value in read(root / 'source-manifest.json')['source_files'].items():
        if digest(source / name) != hash_value:
            raise ValueError(f'changed frozen source: {name}')
    data = {}
    for job in jobs:
        fields, run = job_state(job)
        if not fields or fields[7:9] != ['finished', '0']:
            raise ValueError(f'job incomplete: {job["id"]}')
        variant, task = job['variant'], job['task']
        data[variant, task] = validate_requests(run, task, cohorts[task], tokens[task], settings[variant], sha)
    scores = {v: {t: 100 * statistics.mean(row['score'] for row in data[v, t].values())
                  for t in TASKS} for v in settings}
    means = {v: statistics.mean(r.values()) for v, r in scores.items()}
    comparisons = {v: comparison(data, v) for v in settings if v != 'threshold'}
    lines = [
        '# K2 Horizon normalized fixed region budgets at 64K', '',
        'Two-tier BF16 KV, batch one, same 32 examples in each of 13 RULER tasks. '
        'Top-4, top-8 and top-16 use the same RMS-normalized proxy score and 1,024-entry '
        'eligibility cap as the c=16 threshold. Each condition changes one phase. '
        'The other phase remains at c=16. Ties select the smaller region index first. '
        'Attention and cache updates share the threshold implementation.', '',
        '| Setting | RULER mean | Paired delta vs c=16, points | Paired 95% interval |',
        '|---|---:|---:|---:|',
        f'| Threshold prefill and decode | {means["threshold"]:.2f}% | — | — |',
    ]
    for v, d in comparisons.items():
        lo, hi = d['paired_bootstrap_95_interval_points']
        lines.append(f'| {v} | {means[v]:.2f}% | {d["delta_points"]:+.2f} | [{lo:+.2f}, {hi:+.2f}] |')
    lines += ['', '| Task | ' + ' | '.join(settings) + ' |', '|---|' + '---:|' * len(settings)]
    for task in TASKS:
        lines.append(f'| {task} | ' + ' | '.join(f'{scores[v][task]:.2f}%' for v in settings) + ' |')
    lines += ['', '[GPU checks](preflight/jobs.json) · [Settings](settings.json) · '
              '[Commands](jobs.json) · [Scores](REPORT.json)', '']
    write(root / 'REPORT.json', {'stage': 'complete', 'means': means, 'tasks': scores,
                                 'comparisons': comparisons})
    (root / 'REPORT.md').write_text('\n'.join(lines))
    print(root / 'REPORT.md', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    root = parser.parse_args().output.resolve()
    inputs = read(root / 'inputs.json')
    settings = read(root / 'settings.json')
    sha = source_identity(Path(inputs['source']))['source_sha256']
    cohorts, tokens = verified_cohort(Path(inputs['prior']))
    while True:
        ready = verified_probe(root, settings, cohorts, tokens, sha)
        jobs = submit(root, inputs, settings) if len(ready) == 3 else []
        complete = examples = 0
        failures = []
        for job in jobs:
            fields, data = job_state(job)
            examples += len(data.get('records', []))
            if fields:
                if fields[7:9] == ['finished', '0'] and data.get('stage') == 'complete':
                    complete += 1
                else:
                    failures.append(job['id'])
        progress = {'checks_passed': sorted(ready), 'jobs': len(jobs), 'complete': complete,
                    'examples': examples, 'failures': failures}
        write(root / 'status.json', progress)
        print(progress, flush=True)
        if failures:
            raise RuntimeError(f'failed jobs: {failures}')
        if complete == len(settings) * len(TASKS):
            collect(root, jobs, settings, inputs, cohorts, tokens, sha)
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
