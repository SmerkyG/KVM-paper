"""Freeze, submit and collect a paired 32-example/task AMD RULER panel at 64K.

Submit uses uv and cluster-run for every GPU job. Collection rejects missing
tasks, failed jobs, altered source/cohort files, mismatched inputs and settings.
Historical dense scores are paired by request, but are not fresh runtime controls.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess

from benchmarks.amd_threshold_fast_submit import freeze
from benchmarks.amd_threshold_ruler import score


MODELS = {'qwen': 'Qwen/Qwen3.8-27B-FP8', 'k2': 'IFM/K2-Horizon-32B-FP8'}
TASKS = (
    'niah_single_1', 'niah_single_2', 'niah_single_3',
    'niah_multikey_1', 'niah_multikey_2', 'niah_multikey_3',
    'niah_multiquery', 'niah_multivalue', 'ruler_cwe', 'ruler_fwe',
    'ruler_qa_hotpot', 'ruler_qa_squad', 'ruler_vt',
)
DEFAULT_ARCHIVE = Path('/home/dan/subusers/agent/nogit-kvm-paper-dg/branches/lod-paper-release/code/results/centroid-close-1024/official/ruler-v0413-chat')
DEFAULT_OUTPUT = Path('results/amd-threshold-ruler32-64k')
PYTHON = '/home/dan/subusers/agent/.venvs/vllm-rocm-0.27.1/bin/python'


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def archived_prefix(archive, model, mode, task, count):
    paths = list((archive / model / mode / task).rglob('samples_*.jsonl'))
    if len(paths) != 1:
        raise ValueError(f'expected one archive for {model}/{mode}/{task}: {paths}')
    path = paths[0]
    rows = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['doc_id'] >= count:
                continue
            if row['doc_id'] in rows:
                raise ValueError(f'duplicate archived example: {path}/{row["doc_id"]}')
            if abs(score(row, row['filtered_resps'][0]) - row['65536']) > 1e-12:
                raise ValueError(f'archived scorer mismatch: {path}/{row["doc_id"]}')
            rows[row['doc_id']] = row
            if len(rows) == count:
                break
    if set(rows) != set(range(count)):
        raise ValueError(f'incomplete archived prefix: {path}')
    return rows, {'path': str(path), 'sha256': digest(path)}


def submit(args):
    root = args.output.resolve()
    if root.exists():
        raise ValueError(f'refusing to overwrite experiment directory: {root}')
    root.mkdir(parents=True)
    settings = read(args.settings)
    manifest = {'stage': 'preparing', 'count': args.count, 'context': 65536,
                'selection': 'first doc_ids 0..count-1; no score-based selection',
                'tasks': TASKS, 'models': {}}
    for model in MODELS:
        manifest['models'][model] = {}
        for task in TASKS:
            rows, source = archived_prefix(args.archive, model, 'two-tier', task, args.count)
            dense, dense_source = archived_prefix(args.archive, model, 'dense', task, args.count)
            compact = []
            for i in range(args.count):
                row, full = rows[i], dense[i]
                if (any(row[k] != full[k] for k in ('doc_hash', 'prompt_hash', 'target_hash', 'arguments'))
                        or row['doc']['outputs'] != full['doc']['outputs']):
                    raise ValueError(f'dense request mismatch: {model}/{task}/{i}')
                item = {k: row[k] for k in ('doc_id', 'arguments', 'filtered_resps', 'doc_hash',
                                           'prompt_hash', 'target_hash', '65536')}
                item['doc'] = {'outputs': row['doc']['outputs']}
                item['archived_dense_score'] = full['65536']
                compact.append(item)
            dest = root / 'cohort' / model / task / 'samples_frozen.jsonl'
            dest.parent.mkdir(parents=True)
            dest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in compact))
            manifest['models'][model][task] = {'source': source, 'dense_source': dense_source,
                'frozen': str(dest), 'sha256': digest(dest), 'doc_ids': list(range(args.count))}
            print(f'Frozen {model}/{task}: {args.count} paired requests', flush=True)
    manifest['stage'] = 'complete'
    write(root / 'cohort-manifest.json', manifest)
    write(root / 'settings.json', settings)
    repo = Path(__file__).resolve().parents[1]
    source = freeze(repo, root)
    prior = read(repo / 'results/amd-threshold-fast-v2/speed-v4/source-manifest.json')
    now = read(root / 'source-manifest.json')
    for name in ('benchmarks/amd_threshold_materialized.py', 'benchmarks/amd_threshold_shared.py',
                 'benchmarks/amd_threshold_merge.py'):
        if prior['source_files'][name] != now['source_files'][name]:
            raise ValueError(f'optimized kernels changed since speed measurement: {name}')
    jobs = {'stage': 'submitting', 'source': str(source), 'jobs': []}
    for model in MODELS:
        # Interleave task controls and candidates so both make progress together.
        for task in TASKS:
            for variant in ('legacy', 'optimized'):
                options = {'LOD_AMD_THRESHOLD_MULTIPLIER': '16'}
                if variant == 'optimized':
                    options.update(settings[model])
                output = root / 'runs' / model / task / (variant + '.json')
                name = f'amd-ruler32-{model}-{task}-{variant}'
                command = ['cluster-run', '--detach', '--num-gpus', '1', '--name', name, '--',
                    'env', f'PYTHONPATH={source}:{source}/integrations/vllm_lod',
                    'HF_HUB_OFFLINE=1', 'HF_DATASETS_OFFLINE=1',
                    *(f'{k}={v}' for k, v in options.items()),
                    'uv', 'run', '--offline', '--no-project', '--python', PYTHON,
                    'python', '-m', 'benchmarks.amd_threshold_ruler', '--checkpoint', MODELS[model],
                    '--samples-root', str(root / 'cohort' / model), '--tasks', task,
                    '--selection', 'prefix', '--limit', str(args.count), '--output', str(output)]
                run = subprocess.run(command, cwd=source, text=True, capture_output=True, check=True)
                match = re.search(r'^Submitted (\S+)', run.stdout, re.MULTILINE)
                if not match:
                    raise RuntimeError(run.stdout + run.stderr)
                jobs['jobs'].append({'id': match.group(1), 'model': model, 'task': task,
                    'variant': variant, 'output': str(output), 'options': options, 'command': command})
                write(root / 'jobs.json', jobs)
                print(f'Submitted {match.group(1)}', flush=True)
    jobs['stage'] = 'submitted'
    write(root / 'jobs.json', jobs)


def status(root):
    jobs = read(root / 'jobs.json')['jobs']
    done = failed = examples = 0
    for job in jobs:
        path = Path(job['output'])
        data = read(path) if path.exists() else {}
        examples += len(data.get('records', []))
        done += data.get('stage') == 'complete'
        final = Path.home() / '.cluster-run/agent/finished' / job['id']
        if final.exists() and final.read_text().strip().split('\t')[8] != '0':
            failed += 1
    print(json.dumps({'completed_jobs': done, 'jobs': len(jobs), 'failed_jobs': failed,
                      'completed_examples': examples}, indent=2))


def paired_interval(task_deltas):
    rng = random.Random(0)
    draws = sorted(100 * statistics.mean(statistics.mean(rng.choices(x, k=len(x)))
                    for x in task_deltas) for _ in range(2000))
    return [draws[49], draws[1949]]


def collect(root):
    manifest = read(root / 'cohort-manifest.json')
    job_manifest = read(root / 'jobs.json')
    token_audit = read(root / 'token-audit.json')
    if (token_audit['stage'] != 'complete' or token_audit['nominal_context_tokens'] != manifest['context']
            or set(token_audit['models']) != set(MODELS)):
        raise ValueError('invalid independent prompt-token audit')
    if digest(Path(__file__).with_name('amd_threshold_ruler_token_audit.py')) != token_audit['audit_source_sha256']:
        raise ValueError('changed token audit source')
    if manifest['stage'] != 'complete' or job_manifest['stage'] != 'submitted':
        raise ValueError('experiment submission is incomplete')
    source_manifest = read(root / 'source-manifest.json')
    for name, sha in source_manifest['source_files'].items():
        if digest(Path(source_manifest['source']) / name) != sha:
            raise ValueError(f'changed frozen source: {name}')
    expected = {(m, t, v) for m in MODELS for t in TASKS for v in ('legacy', 'optimized')}
    groups = {}
    provenance = []
    for job in job_manifest['jobs']:
        key = (job['model'], job['task'], job['variant'])
        if key not in expected or key in groups:
            raise ValueError(f'duplicate or unexpected job: {key}')
        final = Path.home() / '.cluster-run/agent/finished' / job['id']
        if not final.exists():
            raise ValueError(f'job still running: {job["id"]}')
        fields = final.read_text().strip().split('\t')
        if fields[7:9] != ['finished', '0']:
            raise ValueError(f'job failed: {job["id"]}: {fields[7:9]}')
        data = read(Path(job['output']))
        if (data['stage'] != 'complete' or data['limit_per_task'] != manifest['count']
                or data['selection'] != 'prefix' or data['tasks'] != [job['task']]
                or data['checkpoint'] != MODELS[job['model']] or data['batch_size'] != 1
                or data['threshold_multiplier'] != 16 or data['kv_cache_dtype'] != 'bfloat16'
                or data['generation_seed'] != 1234 or data['scheduler_chunk_tokens'] != 16384
                or data['decode_threshold_options'] != job['options']):
            raise ValueError(f'invalid run metadata: {key}')
        groups[key] = data
        dest = root / 'provenance'
        dest.mkdir(exist_ok=True)
        shutil.copy2(final, dest / (job['id'] + '.tsv'))
        log = Path.home() / '.cluster-run/agent/logs' / (job['id'] + '.log')
        shutil.copy2(log, dest / log.name)
        provenance.append({**job, 'node': fields[2], 'gpus': fields[3], 'started': fields[5],
                           'ended': fields[6], 'status': fields[7], 'exit_code': fields[8]})
    if set(groups) != expected:
        raise ValueError('missing model/task/variant results')
    summary = {'stage': 'complete', 'examples_per_task': manifest['count'], 'models': {}}
    changed = []
    overview = []
    detail = []
    for model in MODELS:
        audit = token_audit['models'][model]
        if audit['checkpoint'] != MODELS[model] or set(audit['tasks']) != set(TASKS):
            raise ValueError(f'incorrect token audit model or tasks: {model}')
        rows = []
        deltas = []
        for task in TASKS:
            item = manifest['models'][model][task]
            frozen = Path(item['frozen'])
            if digest(frozen) != item['sha256']:
                raise ValueError(f'changed cohort: {frozen}')
            cohort = {r['doc_id']: r for r in map(json.loads, frozen.read_text().splitlines())}
            token_rows = {r['doc_id']: r for r in audit['tasks'][task]}
            if len(audit['tasks'][task]) != len(cohort) or set(token_rows) != set(cohort):
                raise ValueError(f'incomplete token audit: {model}/{task}')
            variants = {}
            for variant in ('legacy', 'optimized'):
                data = groups[(model, task, variant)]
                records = {r['doc_id']: r for r in data['records']}
                if len(data['records']) != len(cohort) or set(records) != set(cohort):
                    raise ValueError(f'cohort mismatch: {model}/{task}/{variant}')
                for i, row in records.items():
                    ref = cohort[i]
                    if (row['task'] != task or any(row[k] != ref[k] for k in ('doc_hash', 'prompt_hash', 'target_hash'))
                            or row['targets'] != ref['doc']['outputs']
                            or abs(score(ref, row['prediction']) - row['score']) > 1e-12
                            or not 0 < row['input_tokens'] <= 66560
                            or any(row[k] != token_rows[i][k] for k in ('input_tokens', 'input_token_sha256'))):
                        raise ValueError(f'bad request or score: {model}/{task}/{variant}/{i}')
                variants[variant] = records
            a, b = variants['legacy'], variants['optimized']
            for i in cohort:
                if any(a[i][k] != b[i][k] for k in ('input_token_sha256', 'input_tokens', 'targets')):
                    raise ValueError(f'input token mismatch: {model}/{task}/{i}')
                if a[i]['prediction'] != b[i]['prediction'] or a[i]['score'] != b[i]['score']:
                    changed.append({'model': model, 'task': task, 'doc_id': i,
                                    'targets': a[i]['targets'], 'legacy': a[i]['prediction'],
                                    'optimized': b[i]['prediction'], 'old_score': a[i]['score'], 'new_score': b[i]['score']})
            delta = [b[i]['score'] - a[i]['score'] for i in sorted(cohort)]
            deltas.append(delta)
            row = {'task': task, 'archived_dense': 100 * statistics.mean(r['archived_dense_score'] for r in cohort.values()),
                   'archived_top8': 100 * statistics.mean(r['65536'] for r in cohort.values()),
                   'legacy': 100 * statistics.mean(r['score'] for r in a.values()),
                   'optimized': 100 * statistics.mean(r['score'] for r in b.values()),
                   'delta_points': 100 * statistics.mean(delta),
                   'improved': sum(x > 1e-12 for x in delta), 'declined': sum(x < -1e-12 for x in delta)}
            rows.append(row)
        mean = {v: statistics.mean(r[v] for r in rows) for v in ('archived_dense', 'legacy', 'optimized', 'delta_points')}
        interval = paired_interval(deltas)
        wins, losses = (sum(r[k] for r in rows) for k in ('improved', 'declined'))
        summary['models'][model] = {'mean': mean, 'improved': wins, 'declined': losses,
                                    'paired_bootstrap_95_interval_points': interval, 'tasks': rows,
                                    'input_token_range': [min(r['input_tokens'] for t in audit['tasks'].values() for r in t),
                                                          max(r['input_tokens'] for t in audit['tasks'].values() for r in t)]}
        overview.append(f'| {model.upper()} | {mean["archived_dense"]:.2f} | {mean["legacy"]:.2f} | {mean["optimized"]:.2f} | {mean["delta_points"]:+.2f} | {wins} / {losses} |')
        detail += [f'## {model.upper()} task results', '',
                   '| Task | Archived dense | Old c=16 | Optimized c=16 | Difference, points | Improved / declined |',
                   '|---|---:|---:|---:|---:|---:|']
        for r in rows:
            detail.append(f'| {r["task"]} | {r["archived_dense"]:.2f} | {r["legacy"]:.2f} | {r["optimized"]:.2f} | {r["delta_points"]:+.2f} | {r["improved"]} / {r["declined"]} |')
        detail += ['', f'Paired bootstrap 95% interval for the mean difference: [{interval[0]:+.2f}, {interval[1]:+.2f}] points. '
                   'This describes sampling uncertainty within the observed cohort; it does not establish noninferiority on the full benchmark.', '']
    token_lengths = '; '.join(f'{model.upper()}: {data["input_token_range"][0]:,}–{data["input_token_range"][1]:,}'
                              for model, data in summary['models'].items())
    lines = ['# Optimized AMD decode: 32-example RULER panel at 64K', '',
        'All 13 tasks, 32 examples per task and model, 416 paired requests per model. '
        'Qwen3.8-27B-FP8 and K2-Horizon-32B-FP8 use batch-one two-tier LoD with BF16 KV and equal-per-region c=16 routing. '
        'The new path is the materialized-index/GQA-shared implementation from the [kernel optimization report](../amd-threshold-fast-v2/REPORT.md).', '',
        'The old and optimized kernels use the same frozen first-32 requests, answers, generation budgets and seed 1234. '
        'There is no failure-based selection. Prefill retains top-8 routing and its exact first 16,384-token chunk. '
        'Only decode implementation settings differ; optimized settings match the preceding speed measurements. '
        'All GPU jobs use uv through cluster-run with frozen source and request hashes.', '',
        f'The archived dataset has a nominal 65,536-token context. Actual model-tokenized prompt ranges are {token_lengths} tokens. '
        'Every generated request is checked against a separate CPU retokenization of its frozen archived chat; prompts are not padded or shortened.', '',
        '## Mean over all 13 tasks', '',
        '| Model | Archived dense | Old c=16 | Optimized c=16 | Difference, points | Improved / declined requests |',
        '|---|---:|---:|---:|---:|---:|', *overview, '',
        'Scores are percentages; means weight all 13 tasks equally. The original RULER substring-recall metric permits fractional per-example scores. '
        'Archived dense scores use the identical requests and answers, checked by hashes and request arguments, but come from a different runtime and scheduling configuration. '
        'They provide historical context. The paired implementation comparison is between the two fresh c=16 runs. '
        'Thirty-two examples per task can reveal regressions but cannot certify unchanged quality on the 500-example benchmark.', '',
        'The earlier [500-example threshold versus fresh top-8 comparison](../amd-threshold/full500/REPORT.md) '
        'covers four Qwen tasks and five K2 tasks. It evaluates the old c=16 implementation, '
        'not the optimized implementation tested here; it is not a complete 13-task threshold panel.', '',
        'A separate [precision diagnostic](precision-diagnostic/REPORT.md) investigates changed Qwen examples. '
        'The [complete Qwen precision follow-up](precision-panel/REPORT.md) evaluates all 13 tasks with '
        '`LOD_AMD_THRESHOLD_PRECISE=1`; its results do not replace the original candidate below.', '',
        *detail, '## Reproduction and audit', '',
        '[All commands and jobs](jobs.json), [frozen requests and historical sources](cohort-manifest.json), '
        '[independent prompt-token audit](token-audit.json), '
        '[optimized settings](settings.json), [source hashes](source-manifest.json), '
        '[placement, status and exit codes](provenance.json), [changed predictions](changed-examples.json), [raw summary](REPORT.json).', '',
        'Bootstrap intervals resample paired differences separately within each task, with 2,000 replicates and seed zero. '
        'The report rejects missing tasks, altered snapshots, failed jobs, inconsistent scoring, mismatched tokenized inputs or incorrect decode settings. '
        'The implementation remains opt-in; this report does not change serving defaults.', '']
    write(root / 'REPORT.json', summary)
    write(root / 'changed-examples.json', changed)
    write(root / 'provenance.json', provenance)
    (root / 'REPORT.md').write_text('\n'.join(lines))
    print(root / 'REPORT.md')


def collect_precision(root):
    """Collect the Qwen precision follow-up without replacing the first candidate."""
    main_report = read(root / 'REPORT.json')
    if main_report['stage'] != 'complete':
        raise ValueError('collect the main panel first')
    phase = root / 'precision-panel'
    jobs = read(phase / 'jobs.json')
    if jobs['stage'] != 'submitted' or {j['task'] for j in jobs['jobs']} != set(TASKS) or len(jobs['jobs']) != len(TASKS):
        raise ValueError('incomplete precision panel submission')
    settings = {'LOD_AMD_THRESHOLD_MULTIPLIER': '16', **read(root / 'settings.json')['qwen'],
                'LOD_AMD_THRESHOLD_PRECISE': '1'}
    if jobs['settings'] != settings:
        raise ValueError('incorrect precision settings')
    source_manifest = read(root / 'source-manifest.json')
    for name, sha in source_manifest['source_files'].items():
        if digest(Path(source_manifest['source']) / name) != sha:
            raise ValueError(f'changed frozen source: {name}')
    provenance = []

    def verify_job(job):
        final = Path.home() / '.cluster-run/agent/finished' / job['id']
        if not final.exists():
            raise ValueError(f'job still running: {job["id"]}')
        fields = final.read_text().strip().split('\t')
        if fields[7:9] != ['finished', '0']:
            raise ValueError(f'job failed: {job["id"]}: {fields[7:9]}')
        dest = phase / 'provenance'
        dest.mkdir(exist_ok=True)
        shutil.copy2(final, dest / (job['id'] + '.tsv'))
        log = Path.home() / '.cluster-run/agent/logs' / (job['id'] + '.log')
        shutil.copy2(log, dest / log.name)
        provenance.append({**job, 'node': fields[2], 'gpus': fields[3], 'started': fields[5],
                           'ended': fields[6], 'status': fields[7], 'exit_code': fields[8]})

    manifest = read(root / 'cohort-manifest.json')
    token_audit = read(root / 'token-audit.json')['models']['qwen']['tasks']
    rows = []
    paired = []
    changed = []
    for job in jobs['jobs']:
        verify_job(job)
        task = job['task']
        old = read(root / 'runs/qwen' / task / 'legacy.json')
        fast = read(root / 'runs/qwen' / task / 'optimized.json')
        precise = read(Path(job['output']))
        shared = ('checkpoint', 'mode', 'batch_size', 'kv_cache_dtype', 'generation_seed',
                  'scheduler_chunk_tokens', 'tasks', 'limit_per_task', 'selection',
                  'threshold_multiplier', 'threshold_rule')
        if (precise['stage'] != 'complete' or len(precise['records']) != 32 or job['options'] != settings
                or precise['decode_threshold_options'] != settings
                or precise['threshold_implementation'] != 'materialized'
                or any(precise[k] != old[k] for k in shared)):
            raise ValueError(f'incorrect precision run metadata: {task}')
        cohort_path = Path(manifest['models']['qwen'][task]['frozen'])
        if digest(cohort_path) != manifest['models']['qwen'][task]['sha256']:
            raise ValueError(f'changed requests: {task}')
        cohort = {r['doc_id']: r for r in map(json.loads, cohort_path.read_text().splitlines())}
        old, fast, precise = ({r['doc_id']: r for r in d['records']} for d in (old, fast, precise))
        expected = {r['doc_id']: r for r in token_audit[task]}
        if len(precise) != 32 or set(precise) != set(cohort):
            raise ValueError(f'incomplete precision cohort: {task}')
        for i, row in precise.items():
            ref = cohort[i]
            if (row['task'] != task or row['targets'] != ref['doc']['outputs']
                    or any(row[k] != ref[k] for k in ('doc_hash', 'prompt_hash', 'target_hash'))
                    or any(row[k] != expected[i][k] for k in ('input_token_sha256', 'input_tokens'))
                    or abs(score(ref, row['prediction']) - row['score']) > 1e-12):
                raise ValueError(f'bad precision request or score: {task}/{i}')
            if row['prediction'] != old[i]['prediction'] or row['prediction'] != fast[i]['prediction']:
                changed.append({'task': task, 'doc_id': i, 'targets': row['targets'],
                                'old': old[i], 'optimized': fast[i], 'precise': row})
        delta = [precise[i]['score'] - old[i]['score'] for i in sorted(cohort)]
        paired.append(delta)
        rows.append({'task': task, 'old': 100 * statistics.mean(r['score'] for r in old.values()),
                     'optimized': 100 * statistics.mean(r['score'] for r in fast.values()),
                     'precise': 100 * statistics.mean(r['score'] for r in precise.values()),
                     'difference_points': 100 * statistics.mean(delta),
                     'improved': sum(x > 1e-12 for x in delta), 'declined': sum(x < -1e-12 for x in delta)})
    verify_job(jobs['probe'])
    probe = read(Path(jobs['probe']['output']))
    if probe['stage'] != 'complete' or probe['model'] != 'qwen' or probe['context'] != 65536:
        raise ValueError('incomplete precision kernel probe')
    kernel = {}
    for impl in ('legacy', 'materialized', 'native-top8'):
        measurements = [r['microseconds'] for r in probe['records'] if r['implementation'] == impl]
        if len(measurements) != 3:
            raise ValueError(f'missing precision probe layers: {impl}')
        kernel[impl] = statistics.mean(measurements)
    means = {k: statistics.mean(r[k] for r in rows) for k in ('old', 'optimized', 'precise', 'difference_points')}
    interval = paired_interval(paired)
    summary = {'stage': 'complete', 'model': 'qwen', 'examples_per_task': 32, 'tasks': rows, 'mean': means,
               'improved': sum(r['improved'] for r in rows), 'declined': sum(r['declined'] for r in rows),
               'paired_bootstrap_95_interval_points': interval, 'kernel_microseconds': kernel}
    lines = ['# Qwen precision follow-up: all 13 RULER tasks', '',
        'Qwen3.8-27B-FP8, batch-one two-tier LoD, BF16 KV, equal-per-region c=16 decode routing and nominal 64K archived RULER requests. '
        'The first [32-example/task panel](../REPORT.md) found some changed answers with the optimized BF16-probability path. '
        'This follow-up changes only `LOD_AMD_THRESHOLD_PRECISE=1`: the exact-entry value product adds a second BF16 dot product '
        'for the probability-rounding residual. KV storage, routing, local/sink attention, prefill, prompts and generation settings are unchanged. '
        'It uses the same frozen first 32 examples of every task, not just examples with changed scores.', '',
        '| Task | Old c=16 | Optimized | Precision correction | Correction vs old, points | Improved / declined vs old |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["task"]} | {r["old"]:.2f} | {r["optimized"]:.2f} | {r["precise"]:.2f} | {r["difference_points"]:+.2f} | {r["improved"]} / {r["declined"]} |')
    lines += [f'| **13-task mean** | **{means["old"]:.2f}** | **{means["optimized"]:.2f}** | **{means["precise"]:.2f}** | **{means["difference_points"]:+.2f}** | **{summary["improved"]} / {summary["declined"]}** |', '',
        f'Paired bootstrap 95% interval for correction minus old: [{interval[0]:+.2f}, {interval[1]:+.2f}] points. '
        'Thirty-two examples per task cannot certify noninferiority on the full benchmark. Fractional RULER recall scores are retained.', '',
        '## Attention timing', '',
        '| Old c=16 | Optimized with precision correction | Native top-8 |', '|---:|---:|---:|',
        f'| {kernel["legacy"]:.2f} µs | {kernel["materialized"]:.2f} µs | {kernel["native-top8"]:.2f} µs |', '',
        'These are graph-captured attention calls on identical actual query/cache states from the first, middle and last layers. '
        'The scope includes routing, selected entries, coarse/local/sink attention and a common local-length rollback; it excludes the rest '
        'of the model and cache catch-up. It is not a full-model speed measurement. The earlier uncorrected timing used different '
        'generated queries and must not be subtracted from this result to estimate correction overhead. '
        'Native top-8 retains its own routing rule and refinement budget, so this is not a work-matched comparison.', '',
        'A separate [paired precision-cost check](../precision-paired-timing/REPORT.md) runs the correction off/on '
        'on identical saved queries and KV states; use that check to quantify precision overhead.', '',
        '[Commands and settings](jobs.json), [all job statuses and placement](provenance.json), [kernel probe](kernel-probe.json), '
        '[raw summary](REPORT.json), [changed predictions](changed-examples.json), [parent token audit](../token-audit.json). '
        'This experiment leaves serving defaults unchanged.', '']
    write(phase / 'REPORT.json', summary)
    write(phase / 'changed-examples.json', changed)
    write(phase / 'provenance.json', provenance)
    (phase / 'REPORT.md').write_text('\n'.join(lines))
    print(phase / 'REPORT.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('submit', 'status', 'collect', 'collect-precision'))
    parser.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--settings', type=Path, default=Path('results/amd-threshold-fast-v2/settings-v4.json'))
    parser.add_argument('--count', type=int, default=32)
    args = parser.parse_args()
    if args.count < 1 or args.count > 500:
        parser.error('--count must be between 1 and 500')
    if args.action == 'submit':
        submit(args)
    elif args.action == 'status':
        status(args.output.resolve())
    elif args.action == 'collect':
        collect(args.output.resolve())
    else:
        collect_precision(args.output.resolve())


if __name__ == '__main__':
    main()
