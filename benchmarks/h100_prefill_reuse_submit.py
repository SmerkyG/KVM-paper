"""Submit score-reuse diagnostics on the H100 without losing queued RULER work.

Queued matrix jobs are resubmitted from their immutable source snapshots after
the short diagnostic jobs. A running job is never interrupted.
"""
import json
import pathlib
import subprocess

root = pathlib.Path('/home/ubuntu/lod-h100/code')
state = pathlib.Path.home() / '.local/state/lod-cluster-run'
manifest = pathlib.Path('/home/ubuntu/comparison/results/ruler32k-prefill-matrix-runs.json')
output = pathlib.Path('/home/ubuntu/comparison/results/prefill-score-reuse-runs.json')
uv = str(pathlib.Path.home() / '.local/bin/uv')
cluster = [uv, 'run', '--offline', '--no-project', '--python', '/usr/bin/python3',
           'python', '-m', 'benchmarks.h100_cluster_run']


def metadata(run_id):
    return json.loads((state / run_id / 'run.json').read_text())


def submit(name, command, cwd=root):
    text = subprocess.check_output(cluster + ['--name', name, '--detach', '--num-gpus', '1', '--'] + command,
                                   cwd=cwd, text=True)
    return text.splitlines()[0].split()[-1]


def replace_output(command, filename):
    command = command.copy()
    command[command.index('--output') + 1] = '/home/ubuntu/comparison/results/' + filename
    return command


def add_env(command, *values):
    return [command[0], *values, *command[1:]]


matrix = json.loads(manifest.read_text())
deferred = []
for job in matrix['jobs']:
    meta = metadata(job['id'])
    if meta['state'] == 'queued':
        subprocess.run(cluster + ['cancel', job['id']], cwd=root, check=True)
        after = metadata(job['id'])
        if after.get('exit_code') == -15:
            deferred.append((job, meta))

try:
    # Same source and inputs for both variants; diagnostics are separate from timing.
    base = metadata('1794188763480512353-prefill-attention-entry16-32k-gpu-events')['command']
    base = [x for x in base if x not in ('LOD_H100_PREFILL_PROFILE=1', 'LOD_H100_PREFILL_GPU_EVENTS=1')]
    jobs = []
    python_index = base.index(uv)
    env = base[:python_index]
    runner = base[python_index:base.index('benchmarks.h100_checked_run') - 1]
    smoke = add_env(env, 'LOD_H100_PREFILL_REUSE_SCORES=1') + runner + [
        '-m', 'benchmarks.h100_threshold_prefill_check', '--queries', '8', '--tile-queries', '8',
        '--slots', '48', '--entry-only', '--output', '/home/ubuntu/comparison/results/prefill-score-reuse-smoke.json']
    jobs.append({'name': 'prefill-score-reuse-smoke', 'id': submit('prefill-score-reuse-smoke', smoke)})
    for length in (32768, 65536):
        for variant in ('baseline', 'reuse'):
            name = f'prefill-score-{variant}-{length//1024}k'
            cmd = replace_output(base, name + '.json')
            cmd[cmd.index('--length') + 1] = str(length)
            cmd[cmd.index('--repeats') + 1] = '3'
            if variant == 'reuse':
                cmd = add_env(cmd, 'LOD_H100_PREFILL_REUSE_SCORES=1')
            jobs.append({'name': name, 'id': submit(name, cmd)})
    audit = replace_output(base, 'prefill-score-reuse-audit.json')
    audit[audit.index('--repeats') + 1] = '1'
    audit[audit.index('--warmups') + 1] = '0'
    audit = add_env(audit, 'LOD_H100_PREFILL_REUSE_SCORES=1', 'LOD_H100_PREFILL_REUSE_AUDIT=1')
    jobs.append({'name': 'prefill-score-reuse-audit', 'id': submit('prefill-score-reuse-audit', audit)})
    qa = next(j for j in matrix['jobs'] if j['decode'] == 'entry' and j['prefill'] == 'entry' and j['group'] == 'qa1')
    qa_cmd = metadata(qa['id'])['command']
    for task in ('qa_1', 'niah_multikey_2'):
        name = 'prefill-score-reuse-' + task.replace('_', '-')
        cmd = replace_output(qa_cmd, name + '.json')
        cmd[cmd.index('--tasks') + 1] = task
        cmd = add_env(cmd, 'LOD_H100_PREFILL_REUSE_SCORES=1')
        jobs.append({'name': name, 'id': submit(name, cmd), 'task': task})
    output.write_text(json.dumps({'jobs': jobs}, indent=2) + '\n')
    print(json.dumps(jobs), flush=True)
finally:
    for job, meta in deferred:
        old_id = job['id']
        name = f"ruler32k-prefill-matrix-{job['decode']}-{job['prefill']}-{job['group']}-reuse-requeue"
        job['id'] = submit(name, meta['command'], pathlib.Path(meta['source']))
        job['previous_id'] = old_id
        temp = manifest.with_suffix('.tmp')
        temp.write_text(json.dumps(matrix, indent=2) + '\n')
        temp.replace(manifest)
