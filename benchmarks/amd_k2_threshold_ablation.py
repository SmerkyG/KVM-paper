"""Run and collect the matched five-setting K2 threshold ablation."""

import argparse
import json
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path

from benchmarks.amd_threshold_prefill_quality_panel import NODES, job_state, write
from benchmarks.amd_threshold_ruler import score
from benchmarks.amd_threshold_ruler_panel import MODELS, PYTHON, TASKS, digest, paired_interval, read


SETTINGS = ((16, 16), (16, 8), (16, 32), (8, 16), (32, 16))
BASELINE = 'p16-d16'
NUM_EXAMPLES = 32
EXPECTED_JOBS = len(TASKS) * len(SETTINGS)


def label(prefill, decode):
    return f'p{prefill}-d{decode}'


def validate_requests(data, task, cohort, tokens, settings, source_sha, *,
                      mode='two-tier', model='k2', prefill_chunk=16384):
    count = len(cohort)
    if (data.get('stage') != 'complete' or data['checkpoint'] != MODELS[model]
            or data['mode'] != mode or data['tasks'] != [task]
            or data['selection'] != 'prefix' or data['limit_per_task'] != count
            or data['generation_seed'] != 1234 or data['kv_cache_dtype'] != 'bfloat16'
            or data['batch_size'] != 1 or data['scheduler_chunk_tokens'] != prefill_chunk
            or data['benchmark_identity']['source']['source_sha256'] != source_sha):
        raise ValueError(f'wrong benchmark metadata for {task}')
    for field, prefix in (('prefill_threshold_options', 'LOD_AMD_PREFILL_'),
                          ('decode_threshold_options', 'LOD_AMD_THRESHOLD_')):
        if data[field] != {k: v for k, v in settings.items() if k.startswith(prefix)}:
            raise ValueError(f'wrong {field} for {task}')
    rows = {r['doc_id']: r for r in data['records']}
    if set(rows) != set(cohort) or len(data['records']) != count:
        raise ValueError(f'missing or duplicate examples for {task}')
    for i, row in rows.items():
        ref = cohort[i]
        if (row['task'] != task or row['targets'] != ref['doc']['outputs']
                or any(row[k] != ref[k] for k in ('prompt_hash', 'doc_hash', 'target_hash'))
                or any(row[k] != tokens[i][k] for k in ('input_tokens', 'input_token_sha256'))
                or abs(row['score'] - score(ref, row['prediction'])) > 1e-12):
            raise ValueError(f'mismatched request or score for {task}/{i}')
    return rows


def verified_cohort(prior, model='k2'):
    manifest = read(prior / 'cohort-manifest.json')
    audit = read(prior / 'token-audit.json')
    if (manifest['count'] != NUM_EXAMPLES or manifest['context'] != 65536
            or audit['stage'] != 'complete' or audit['nominal_context_tokens'] != 65536):
        raise ValueError('expected the audited first-32 64K cohort')
    cohorts, tokens = {}, {}
    for task in TASKS:
        item = manifest['models'][model][task]
        path = Path(item['frozen'])
        if digest(path) != item['sha256']:
            raise ValueError(f'changed cohort {task}')
        rows = {r['doc_id']: r for r in map(json.loads, path.read_text().splitlines())}
        audit_rows = {r['doc_id']: r for r in audit['models'][model]['tasks'][task]}
        if set(rows) != set(range(NUM_EXAMPLES)) or set(rows) != set(audit_rows):
            raise ValueError(f'incomplete cohort {task}')
        cohorts[task], tokens[task] = rows, audit_rows
    return cohorts, tokens


def launch(root):
    preflight = read(root / 'preflight/jobs.json')
    prior = Path(preflight['prior'])
    source = Path(preflight['source'])
    source_sha = read(prior / 'identity.json')['source_sha256']
    cohorts, tokens = verified_cohort(prior)
    for job in preflight['jobs']:
        fields, data = job_state(job)
        if not fields or fields[7:9] != ['finished', '0']:
            raise RuntimeError(f'memory check incomplete or failed: {job["id"]}')
        task = job['task']
        rows = {i: cohorts[task][i] for i in job['doc_ids']}
        validate_requests(data, task, rows, tokens[task], preflight['settings'], source_sha)
    if (preflight['settings']['LOD_AMD_PREFILL_THRESHOLD'] != '8'
            or not any(job['task'] == 'ruler_cwe' and 24 in job['doc_ids']
                       for job in preflight['jobs'])):
        raise ValueError('expected a c=8 prefill check on the prior CWE failure')
    settings = {
        label(prefill, decode): {
            **preflight['settings'],
            'LOD_AMD_PREFILL_THRESHOLD': str(prefill),
            'LOD_AMD_THRESHOLD_MULTIPLIER': str(decode),
        }
        for prefill, decode in SETTINGS
    }
    write(root / 'settings.json', settings)
    path = root / 'jobs.json'
    manifest = read(path) if path.exists() else {
        'source': str(source), 'source_sha256': source_sha, 'prior': str(prior), 'jobs': [],
        'input_hashes': {name: digest(prior / name) for name in ('cohort-manifest.json', 'token-audit.json')},
    }
    seen = {(job['task'], job['variant']) for job in manifest['jobs']}
    manifest['stage'] = 'submitting'
    write(path, manifest)
    # Submit the historically longest jobs early to reduce the final queue tail.
    previous = read(prior / 'provenance.json')
    seconds = {}
    for job in previous:
        if job['model'] == 'k2' and job['variant'] == 'both_threshold':
            rows = read(Path(job['output']))['records']
            seconds[job['task']] = sum(r['elapsed_s'] for r in rows)
    order = sorted(TASKS, key=seconds.__getitem__, reverse=True)
    for task in order:
        for prefill, decode in SETTINGS:
            variant = label(prefill, decode)
            if (task, variant) in seen:
                continue
            options = settings[variant]
            output = root / 'runs' / variant / f'{task}.json'
            name = f'amd-k2-abl-{variant}-{task}'
            preferred = NODES.split(',')[len(manifest['jobs']) % 4]
            command = ['cluster-run', '--detach', '--num-gpus', '1', '--nodes', NODES,
                       '--prefer-nodes', preferred, '--name', name, '--',
                       'env', f'PYTHONPATH={source}:{source}/integrations/vllm_lod',
                       'HF_HUB_OFFLINE=1', 'HF_DATASETS_OFFLINE=1',
                       *(f'{key}={value}' for key, value in options.items()),
                       'uv', 'run', '--offline', '--no-project', '--python', PYTHON, 'python', '-P',
                       '-m', 'benchmarks.amd_threshold_ruler', '--checkpoint', MODELS['k2'],
                       '--samples-root', str(prior / 'cohort/k2'), '--tasks', task,
                       '--selection', 'prefix', '--limit', str(NUM_EXAMPLES), '--output', str(output)]
            run = subprocess.run(command, cwd=source, capture_output=True, text=True, check=True)
            match = re.search(r'^Submitted (\S+)', run.stdout, re.MULTILINE)
            if not match:
                raise RuntimeError(run.stdout + run.stderr)
            job = {'id': match.group(1), 'variant': variant, 'task': task,
                   'output': str(output), 'command': command, 'options': options}
            manifest['jobs'].append(job)
            write(path, manifest)
            print(job['id'], flush=True)
    manifest['stage'] = 'submitted'
    write(path, manifest)


def status(root):
    jobs = read(root / 'jobs.json')['jobs']
    failures, complete, examples = [], 0, 0
    for job in jobs:
        fields, data = job_state(job)
        examples += len(data.get('records', []))
        if fields:
            if fields[7:9] == ['finished', '0'] and data.get('stage') == 'complete':
                complete += 1
            else:
                failures.append(job['id'])
    result = {'jobs': len(jobs), 'complete': complete, 'examples': examples, 'failures': failures}
    write(root / 'status.json', result)
    print(json.dumps(result), flush=True)
    return result


def collect(root):
    manifest = read(root / 'jobs.json')
    prior = Path(manifest['prior'])
    source = Path(manifest['source'])
    variants = [label(*c) for c in SETTINGS]
    expected = {(v, task) for v in variants for task in TASKS}
    if (manifest['stage'] != 'submitted' or len(manifest['jobs']) != EXPECTED_JOBS
            or {(j['variant'], j['task']) for j in manifest['jobs']} != expected):
        raise ValueError('incomplete or duplicate submissions')
    for name, sha in manifest['input_hashes'].items():
        if digest(prior / name) != sha:
            raise ValueError(f'changed input manifest {name}')
    frozen = read(prior / 'source-manifest.json')
    for name, sha in frozen['source_files'].items():
        if digest(source / name) != sha:
            raise ValueError(f'changed frozen source {name}')
    cohorts, tokens = verified_cohort(prior)
    settings = read(root / 'settings.json')
    groups = {}
    provenance = []
    dest = root / 'provenance'
    dest.mkdir(exist_ok=True)
    for job in manifest['jobs']:
        variant, task = job['variant'], job['task']
        fields, data = job_state(job)
        if not fields or fields[7:9] != ['finished', '0'] or job['options'] != settings[variant]:
            raise ValueError(f'job failed or mismatched settings: {job["id"]}')
        groups[variant, task] = validate_requests(
            data, task, cohorts[task], tokens[task], settings[variant], manifest['source_sha256'])
        spool = Path.home() / '.cluster-run/agent'
        shutil.copy2(spool / 'finished' / job['id'], dest / (job['id'] + '.tsv'))
        shutil.copy2(spool / 'logs' / (job['id'] + '.log'), dest / (job['id'] + '.log'))
        provenance.append({**job, 'node': fields[2], 'gpus': fields[3], 'exit_code': fields[8]})
    task_scores = {
        v: {t: 100 * statistics.mean(r['score'] for r in groups[v, t].values()) for t in TASKS}
        for v in variants
    }
    means = {v: statistics.mean(task_scores[v].values()) for v in variants}
    comparisons = {}
    for variant in variants[1:]:
        deltas = [[groups[variant, t][i]['score'] - groups[BASELINE, t][i]['score']
                   for i in range(NUM_EXAMPLES)] for t in TASKS]
        comparisons[variant] = {
            'delta_points': means[variant] - means[BASELINE],
            'improved': sum(x > 1e-12 for d in deltas for x in d),
            'declined': sum(x < -1e-12 for d in deltas for x in d),
            'paired_bootstrap_95_interval_points': paired_interval(deltas),
        }
    summary = {'stage': 'complete', 'context': 65536, 'examples_per_task': NUM_EXAMPLES,
               'task_scores': task_scores, 'means': means, 'comparisons': comparisons}
    write(root / 'REPORT.json', summary)
    write(root / 'provenance.json', provenance)
    tile = int(settings[BASELINE]['LOD_AMD_PREFILL_QUERY_TILE'])
    lines = [
        '# K2 Horizon region threshold ablation at 64K', '',
        'K2-Horizon-32B-FP8 on MI325X, batch one, two-tier BF16 KV. Each configuration evaluates '
        'the same first 32 archived examples of every RULER task with greedy generation. '
        f'All runs use a 16,384-token exact prefix, 16,384-token scheduler chunks and {tile:,}-query '
        'internal tiles. A c=8 prefill smoke test covered the CWE prompt from the previous '
        'scratch allocation failure and three of the longest prompts in the cohort.', '',
        '| Prefill c | Decode c | 13-task mean | Delta vs (16,16), points | Paired 95% interval, points | Improved / declined |',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for prefill, decode in SETTINGS:
        v = label(prefill, decode)
        if v == BASELINE:
            lines.append(f'| {prefill} | {decode} | {means[v]:.2f}% | — | — | — |')
        else:
            stats = comparisons[v]
            low, high = stats['paired_bootstrap_95_interval_points']
            lines.append(f'| {prefill} | {decode} | {means[v]:.2f}% | {stats["delta_points"]:+.2f} | '
                         f'[{low:+.2f}, {high:+.2f}] | {stats["improved"]} / {stats["declined"]} |')
    lines += ['', '| Task | ' + ' | '.join(f'({p}, {d})' for p, d in SETTINGS) + ' |',
              '|---|' + '---:|' * len(SETTINGS)]
    for task in TASKS:
        lines.append(f'| {task} | ' + ' | '.join(f'{task_scores[v][task]:.2f}%' for v in variants) + ' |')
    lines += ['', 'The mean weights the 13 tasks equally. Intervals resample paired scores within '
              'each task and describe uncertainty on this cohort. The cohort was also used for '
              'the preceding threshold quality comparison.', '',
              '[Settings](settings.json) · [Memory check](preflight/jobs.json) · [Commands](jobs.json) · '
              '[Exit statuses](provenance.json) · [Scores](REPORT.json)', '']
    (root / 'REPORT.md').write_text('\n'.join(lines))
    print(root / 'REPORT.md', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'status', 'collect', 'watch'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.action == 'watch':
        while True:
            progress = status(root)
            if progress['failures']:
                raise RuntimeError(f'jobs failed: {progress["failures"]}')
            if progress['complete'] == EXPECTED_JOBS:
                collect(root)
                return
            time.sleep(30)
    else:
        {'launch': launch, 'status': status, 'collect': collect}[args.action](root)


if __name__ == '__main__':
    main()
