"""Queue same-source baseline RULER tasks for the one-pass experiment."""
import json
import pathlib
import subprocess

root = pathlib.Path('/home/ubuntu/lod-h100/code')
state = pathlib.Path.home() / '.local/state/lod-cluster-run'
matrix_path = pathlib.Path('/home/ubuntu/comparison/results/ruler32k-prefill-matrix-runs.json')
output_path = pathlib.Path('/home/ubuntu/comparison/results/prefill-onepass-quality-control-runs.json')
uv = str(pathlib.Path.home() / '.local/bin/uv')
cluster = [uv, 'run', '--offline', '--no-project', '--python', '/usr/bin/python3',
           'python', '-m', 'benchmarks.h100_cluster_run']


def read(rid):
    return json.loads((state / rid / 'run.json').read_text())


def submit(name, command, cwd=root):
    result = subprocess.check_output(cluster + ['--name', name, '--detach', '--num-gpus', '1', '--'] + command,
                                     cwd=cwd, text=True)
    return result.splitlines()[0].split()[-1]


matrix = json.loads(matrix_path.read_text())
deferred = []
for job in matrix['jobs']:
    meta = read(job['id'])
    if meta['state'] == 'queued':
        subprocess.run(cluster + ['cancel', job['id']], cwd=root, check=True)
        if read(job['id']).get('exit_code') == -15:
            deferred.append((job, meta))
try:
    reference = next(j for j in matrix['jobs'] if j['decode'] == 'entry' and j['prefill'] == 'entry' and j['group'] == 'qa1')
    base = read(reference['id'])['command']
    jobs = []
    for task in ('qa_1', 'niah_multikey_2'):
        name = 'prefill-onepass-control-' + task.replace('_', '-')
        cmd = base.copy()
        cmd[cmd.index('--tasks') + 1] = task
        cmd[cmd.index('--output') + 1] = '/home/ubuntu/comparison/results/' + name + '.json'
        jobs.append({'name': name, 'id': submit(name, cmd), 'task': task})
    output_path.write_text(json.dumps({'jobs': jobs}, indent=2) + '\n')
    print(json.dumps(jobs), flush=True)
finally:
    for job, meta in deferred:
        old = job['id']
        name = f"ruler32k-prefill-matrix-{job['decode']}-{job['prefill']}-{job['group']}-onepass-control-requeue"
        job['id'] = submit(name, meta['command'], pathlib.Path(meta['source']))
        job['previous_id'] = old
        tmp = matrix_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(matrix, indent=2) + '\n')
        tmp.replace(matrix_path)
