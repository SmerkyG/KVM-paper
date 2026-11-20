"""Frozen 64K AMD quality comparison with separate prefill/decode controls.

All GPU work uses uv through cluster-run. No kernel changes or hyperparameter
search are made by this runner. Dense scores are archived, request-matched controls.
"""
import argparse
import json
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from benchmarks._identity import source_identity
from benchmarks.amd_threshold_fast_submit import freeze
from benchmarks.amd_threshold_ruler import score
from benchmarks.amd_threshold_ruler_panel import (
    MODELS, TASKS, PYTHON, digest, paired_interval, read,
)

VARIANTS = ('top8', 'decode_threshold', 'both_threshold')
LABELS = {
    'top8': 'Top-8 prefill / top-8 decode',
    'decode_threshold': 'Top-8 prefill / c=16 decode',
    'both_threshold': 'c=16 prefill / c=16 decode',
}
COMPARISONS = (
    ('decode_threshold', 'top8'), ('both_threshold', 'top8'),
    ('both_threshold', 'decode_threshold'),
)
NODES = ','.join(f'dev-shared-research-{i}.dev-training.svc.cluster.local' for i in (1, 2, 3, 4))


def write(path, data):
    """Use distinct temporary files for simultaneous status and watcher calls."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.',
                                     suffix='.tmp', delete=False) as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        temporary = Path(stream.name)
    temporary.replace(path)


def prefill_options(submission):
    command = read(submission)['command']
    return dict(value.split('=', 1) for value in command
                if value.startswith(('LOD_AMD_PREFILL_', 'LOD_AMD_STATE_')))


def prepare(root, prior):
    if root.exists():
        raise ValueError(f'refusing to overwrite {root}')
    root.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    old = read(prior / 'cohort-manifest.json')
    if old['stage'] != 'complete' or old['count'] != 32 or set(old['tasks']) != set(TASKS):
        raise ValueError('expected completed 32-example/task parent cohort')
    cohort = {'stage': 'complete', 'count': 32, 'context': 65536,
              'selection': 'first doc_ids 0..31; identical to prior validation panel',
              'parent': str(prior), 'tasks': TASKS, 'models': {}}
    for model in MODELS:
        cohort['models'][model] = {}
        for task in TASKS:
            entry = old['models'][model][task]
            src = Path(entry['frozen'])
            if digest(src) != entry['sha256']:
                raise ValueError(f'changed parent cohort: {src}')
            dest = root / 'cohort' / model / task / 'samples_frozen.jsonl'
            dest.parent.mkdir(parents=True)
            shutil.copy2(src, dest)
            cohort['models'][model][task] = {**entry, 'frozen': str(dest)}
    write(root / 'cohort-manifest.json', cohort)
    shutil.copy2(prior / 'token-audit.json', root / 'token-audit.json')
    decode = read(prior / 'settings-ruler32.json')
    base = repo / 'results/amd-threshold-ports'
    prefill = {
        'k2': prefill_options(base / 'k2-prefill-qk-final-v1/amd-k2-final-fp32-64k-submission.json'),
        'qwen': prefill_options(base / 'k2-prefill-grouped-v1/amd-qwen-prefill-speed-reference-submission.json'),
    }
    settings = {}
    for model in MODELS:
        # The packed-max state update is shared by all three fresh arms.
        common = {'LOD_AMD_STATE_PACKED_MAX': '1',
                  'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
        settings[model] = {
            'top8': common,
            'decode_threshold': {**common, **decode[model]},
            'both_threshold': {**common, **decode[model], **prefill[model]},
        }
        if settings[model]['both_threshold']['LOD_AMD_PREFILL_THRESHOLD'] != '16':
            raise ValueError('wrong prefill multiplier')
    write(root / 'settings.json', settings)
    source = freeze(repo, root)
    write(root / 'identity.json', source_identity(source))
    write(root / 'jobs.json', {'stage': 'prepared', 'source': str(source), 'jobs': []})
    write(root / 'frozen-inputs.json', {name: digest(root / name) for name in (
        'cohort-manifest.json', 'token-audit.json', 'settings.json', 'identity.json')})
    (root / 'REPORT.md').write_text(
        '# AMD threshold prefill and decode quality at 64K\n\n'
        'Panel prepared. Scores are pending. All 13 RULER tasks, 32 frozen examples per task '
        'and model, Qwen3.8-27B-FP8 and K2-Horizon-32B-FP8, BF16 two-tier KV, batch one. '
        'Three fresh arms compare top-eight throughout, c=16 decode with top-eight prefill, '
        'and c=16 in both stages. The exact initial prefill stays at 16,384 tokens. '
        'Qwen uses the probability-residual decode correction. Dense scores are matched archived results.\n\n'
        '[Settings](settings.json) · [Source freeze](source-manifest.json) · '
        '[Requests](cohort-manifest.json) · [Jobs](jobs.json)\n')
    print(f'Prepared {root}', flush=True)


def submit(root):
    manifest = read(root / 'jobs.json')
    settings = read(root / 'settings.json')
    source = Path(manifest['source'])
    seen = {(j['model'], j['task'], j['variant']) for j in manifest['jobs']}
    priority = ('niah_multikey_2', 'niah_multikey_3', 'niah_multikey_1',
                'ruler_qa_hotpot', 'ruler_qa_squad', 'ruler_cwe', 'ruler_fwe')
    order = (*priority, *(task for task in TASKS if task not in priority))
    manifest['stage'] = 'submitting'
    write(root / 'jobs.json', manifest)
    for task in order:
        for model in MODELS:
            for variant in ('both_threshold', 'decode_threshold', 'top8'):
                if (model, task, variant) in seen:
                    continue
                options = settings[model][variant]
                output = root / 'runs' / model / task / f'{variant}.json'
                name = f'amd-final32-{model}-{task}-{variant}'
                command = ['cluster-run', '--detach', '--num-gpus', '1', '--nodes', NODES,
                           '--name', name, '--', 'env',
                           f'PYTHONPATH={source}:{source}/integrations/vllm_lod',
                           'HF_HUB_OFFLINE=1', 'HF_DATASETS_OFFLINE=1',
                           *(f'{key}={value}' for key, value in options.items()),
                           'uv', 'run', '--offline', '--no-project', '--python', PYTHON,
                           'python', '-P', '-m', 'benchmarks.amd_threshold_ruler',
                           '--checkpoint', MODELS[model], '--samples-root', str(root / 'cohort' / model),
                           '--tasks', task, '--selection', 'prefix', '--limit', '32', '--output', str(output)]
                run = subprocess.run(command, cwd=source, text=True, capture_output=True, check=True)
                match = re.search(r'^Submitted (\S+)', run.stdout, re.MULTILINE)
                if not match:
                    raise RuntimeError(run.stdout + run.stderr)
                job = {'id': match.group(1), 'model': model, 'task': task, 'variant': variant,
                       'output': str(output), 'options': options, 'command': command}
                manifest['jobs'].append(job)
                write(root / 'jobs.json', manifest)
                print(f'Submitted {job["id"]}', flush=True)
    manifest['stage'] = 'submitted'
    write(root / 'jobs.json', manifest)


def job_state(job):
    final = Path.home() / '.cluster-run/agent/finished' / job['id']
    fields = final.read_text().strip().split('\t') if final.exists() else None
    path = Path(job['output'])
    data = read(path) if path.exists() else {}
    return fields, data


def status(root):
    jobs = read(root / 'jobs.json')['jobs']
    states = [job_state(j) for j in jobs]
    failures = [j['id'] for j, (fields, data) in zip(jobs, states)
                if fields and (fields[7:9] != ['finished', '0'] or data.get('stage') != 'complete')]
    result = {'jobs': len(jobs), 'complete': sum(bool(f and f[7:9] == ['finished', '0']
               and d.get('stage') == 'complete') for f, d in states),
              'examples': sum(len(d.get('records', [])) for _, d in states), 'failures': failures}
    write(root / 'status.json', result)
    print(json.dumps(result), flush=True)
    return result


def collect(root):
    manifest = read(root / 'jobs.json')
    for name, sha in read(root / 'frozen-inputs.json').items():
        if digest(root / name) != sha:
            raise ValueError(f'changed frozen metadata: {name}')
    source_manifest = read(root / 'source-manifest.json')
    for name, sha in source_manifest['source_files'].items():
        if digest(Path(manifest['source']) / name) != sha:
            raise ValueError(f'changed source: {name}')
    expected = {(m, t, v) for m in MODELS for t in TASKS for v in VARIANTS}
    if (manifest['stage'] != 'submitted' or len(manifest['jobs']) != len(expected)
            or {(j['model'], j['task'], j['variant']) for j in manifest['jobs']} != expected):
        raise ValueError('incomplete or duplicate submissions')
    cohort_manifest = read(root / 'cohort-manifest.json')
    audit = read(root / 'token-audit.json')
    if audit['stage'] != 'complete' or audit['nominal_context_tokens'] != 65536:
        raise ValueError('invalid token audit')
    settings = read(root / 'settings.json')
    identity = read(root / 'identity.json')['source_sha256']
    cohorts, tokens, groups, provenance = {}, {}, {}, []
    retry_notes = []
    for model in MODELS:
        for task in TASKS:
            item = cohort_manifest['models'][model][task]
            path = Path(item['frozen'])
            if digest(path) != item['sha256']:
                raise ValueError(f'changed requests: {path}')
            cohorts[model, task] = {r['doc_id']: r for r in map(json.loads, path.read_text().splitlines())}
            tokens[model, task] = {r['doc_id']: r for r in audit['models'][model]['tasks'][task]}
            if set(cohorts[model, task]) != set(range(32)):
                raise ValueError('incomplete cohort')
    for job in manifest['jobs']:
        fields, data = job_state(job)
        model, task, variant = (job[k] for k in ('model', 'task', 'variant'))
        options = settings[model][variant]
        retry_tile = job.get('oom_retry_query_tile')
        if retry_tile is not None:
            if (variant != 'both_threshold'
                    or not 2 <= int(retry_tile) < int(options['LOD_AMD_PREFILL_QUERY_TILE'])):
                raise ValueError(f'invalid OOM retry tile: {job["id"]}')
            options = {**options, 'LOD_AMD_PREFILL_QUERY_TILE': str(retry_tile)}
            retry_notes.append(
                f'{model.upper()} {task} was rerun in full after a scratch allocation failed. '
                f'The successful run used an internal {int(retry_tile):,}-query tile with the '
                'same threshold and 16,384-token exact prefix.'
            )
        if options != job['options']:
            raise ValueError(f'job options mismatch: {job["id"]}')
        if not fields or fields[7:9] != ['finished', '0'] or data.get('stage') != 'complete':
            raise ValueError(f'job incomplete or failed: {job["id"]}')
        if (data['checkpoint'] != MODELS[model] or data['mode'] != 'two-tier'
                or data['limit_per_task'] != 32 or data['tasks'] != [task]
                or data['batch_size'] != 1 or data['kv_cache_dtype'] != 'bfloat16'
                or data['generation_seed'] != 1234 or data['selection'] != 'prefix'
                or data['scheduler_chunk_tokens'] != 16384
                or data['benchmark_identity']['source']['source_sha256'] != identity
                or data['decode_threshold_options'] != {k: v for k, v in options.items() if k.startswith('LOD_AMD_THRESHOLD_')}
                or data['prefill_threshold_options'] != {k: v for k, v in options.items() if k.startswith('LOD_AMD_PREFILL_')}):
            raise ValueError(f'run metadata mismatch: {job["id"]}')
        records = {r['doc_id']: r for r in data['records']}
        if len(data['records']) != 32 or set(records) != set(range(32)):
            raise ValueError('missing or duplicate examples')
        for i, row in records.items():
            ref, token = cohorts[model, task][i], tokens[model, task][i]
            if (row['task'] != task or any(row[k] != ref[k] for k in ('prompt_hash', 'doc_hash', 'target_hash'))
                    or row['targets'] != ref['doc']['outputs']
                    or any(row[k] != token[k] for k in ('input_tokens', 'input_token_sha256'))
                    or abs(row['score'] - score(ref, row['prediction'])) > 1e-12):
                raise ValueError(f'request/score mismatch: {model}/{task}/{variant}/{i}')
        groups[model, task, variant] = records
        prov = root / 'provenance'
        prov.mkdir(exist_ok=True)
        shutil.copy2(Path.home() / '.cluster-run/agent/finished' / job['id'], prov / (job['id'] + '.tsv'))
        shutil.copy2(Path.home() / '.cluster-run/agent/logs' / (job['id'] + '.log'), prov / (job['id'] + '.log'))
        provenance.append({**job, 'node': fields[2], 'gpus': fields[3], 'exit_code': fields[8]})
    summary, changes = {'stage': 'complete', 'examples_per_task': 32, 'models': {}}, []
    lines = ['# AMD threshold prefill and decode quality at 64K', '',
             'All 13 RULER tasks, 32 matched examples per task, two-tier BF16 KV, batch one. '
             'Qwen3.8-27B-FP8 and K2-Horizon-32B-FP8 use c=16 for threshold routing. '
             'The initial exact prefix and scheduler chunks remain 16,384 tokens. '
             'Qwen threshold decode includes its probability-residual correction. '
             'The three fresh arms share one source freeze and the packed-max cache update.', '',
             'Dense is an archived control with identical requests and targets, from a different runtime. '
             'The frozen first-32 cohort has been used in earlier validation; these results are not an independent holdout. '
             'All-task means weight tasks equally and retain fractional recall. '
             'Quality-run elapsed times include setup effects and are not speed-benchmark results.', '',
             '| Model | Archived dense | Top-8 throughout | Threshold decode only | Threshold prefill + decode |',
             '|---|---:|---:|---:|---:|']
    details = []
    for model in MODELS:
        rows = []
        for task in TASKS:
            row = {'task': task, 'archived_dense': 100 * statistics.mean(
                r['archived_dense_score'] for r in cohorts[model, task].values())}
            row.update({v: 100 * statistics.mean(r['score'] for r in groups[model, task, v].values()) for v in VARIANTS})
            rows.append(row)
            for i in range(32):
                recs = {v: groups[model, task, v][i] for v in VARIANTS}
                if len({r['prediction'] for r in recs.values()}) > 1:
                    changes.append({'model': model, 'task': task, 'doc_id': i,
                                    'targets': recs['top8']['targets'], 'variants': recs})
        means = {v: statistics.mean(r[v] for r in rows) for v in ('archived_dense', *VARIANTS)}
        comparisons = {}
        for candidate, control in COMPARISONS:
            deltas = [[groups[model, task, candidate][i]['score'] - groups[model, task, control][i]['score']
                       for i in range(32)] for task in TASKS]
            comparisons[f'{candidate}_vs_{control}'] = {
                'delta_points': means[candidate] - means[control],
                'improved': sum(x > 1e-12 for d in deltas for x in d),
                'declined': sum(x < -1e-12 for d in deltas for x in d),
                'paired_bootstrap_95_interval_points': paired_interval(deltas),
            }
        summary['models'][model] = {'mean': means, 'comparisons': comparisons, 'tasks': rows}
        lines.append(f'| {model.upper()} | ' + ' | '.join(f'{means[v]:.2f}' for v in ('archived_dense', *VARIANTS)) + ' |')
        details += ['', f'## {model.upper()} per-task results', '',
                    '| Task | Archived dense | Top-8 throughout | Threshold decode only | Threshold prefill + decode |',
                    '|---|---:|---:|---:|---:|']
        for r in rows:
            details.append(f'| {r["task"]} | ' + ' | '.join(f'{r[v]:.2f}' for v in ('archived_dense', *VARIANTS)) + ' |')
        for (candidate, control), comparison in zip(COMPARISONS, comparisons.values()):
            lo, hi = comparison['paired_bootstrap_95_interval_points']
            details += ['', f'{LABELS[candidate]} minus {LABELS[control]}: '
                        f'{comparison["delta_points"]:+.2f} points; '
                        f'{comparison["improved"]} improved / {comparison["declined"]} declined examples. '
                        f'Paired bootstrap 95% interval [{lo:+.2f}, {hi:+.2f}] points.']
    lines += details + ['', 'Thirty-two examples per task can expose large differences but cannot establish '
                        'small improvements reliably. Bootstrap intervals resample within each task with 2,000 draws and seed zero.', '',
                        *[note + '\n' for note in retry_notes],
                        '[Settings](settings.json) · [Source](source-manifest.json) · [Requests](cohort-manifest.json) · '
                        '[Token audit](token-audit.json) · [Commands](jobs.json) · [Exit statuses](provenance.json) · '
                        '[Changed predictions](changed-examples.json) · [Scores](REPORT.json)', '']
    write(root / 'REPORT.json', summary)
    write(root / 'changed-examples.json', changes)
    write(root / 'provenance.json', provenance)
    (root / 'REPORT.md').write_text('\n'.join(lines))
    print(root / 'REPORT.md', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'submit', 'status', 'collect', 'watch'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior', type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.action == 'prepare':
        prepare(root, args.prior.resolve())
    elif args.action == 'submit':
        submit(root)
    elif args.action == 'status':
        status(root)
    elif args.action == 'collect':
        collect(root)
    else:
        while True:
            progress = status(root)
            if progress['failures']:
                raise RuntimeError(f'failed jobs: {progress["failures"]}')
            if progress['complete'] == 78:
                collect(root)
                break
            time.sleep(30)


if __name__ == '__main__':
    main()
