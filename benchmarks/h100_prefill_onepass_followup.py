"""Submit same-source controls, stage profiles, and paired RULER checks."""
import json
import pathlib
import subprocess

root = pathlib.Path('/home/ubuntu/lod-h100/code')
state = pathlib.Path.home() / '.local/state/lod-cluster-run'
matrix_path = pathlib.Path('/home/ubuntu/comparison/results/ruler32k-prefill-matrix-runs.json')
output_path = pathlib.Path('/home/ubuntu/comparison/results/prefill-onepass-followup-runs.json')
uv = str(pathlib.Path.home() / '.local/bin/uv')
cluster = [uv, 'run', '--offline', '--no-project', '--python', '/usr/bin/python3',
           'python', '-m', 'benchmarks.h100_cluster_run']


def read(run_id):
    return json.loads((state / run_id / 'run.json').read_text())


def submit(name, command, cwd=root):
    result = subprocess.check_output(cluster + ['--name', name, '--detach', '--num-gpus', '1', '--'] + command,
                                     cwd=cwd, text=True)
    return result.splitlines()[0].split()[-1]


def new_output(command, name):
    command = command.copy()
    command[command.index('--output') + 1] = '/home/ubuntu/comparison/results/' + name + '.json'
    return command


matrix = json.loads(matrix_path.read_text())
deferred = []
for job in matrix['jobs']:
    meta = read(job['id'])
    if meta['state'] == 'queued':
        subprocess.run(cluster + ['cancel', job['id']], cwd=root, check=True)
        if read(job['id']).get('exit_code') == -15:
            deferred.append((job, meta))
try:
    original = read('1794188763480512353-prefill-attention-entry16-32k-gpu-events')['command']
    speed = [x for x in original if x not in ('LOD_H100_PREFILL_PROFILE=1', 'LOD_H100_PREFILL_GPU_EVENTS=1')]
    jobs = []
    for length in (32768, 65536):
        for variant in ('baseline', 'producer'):
            name = f'prefill-onepass-followup-{variant}-{length//1024}k'
            cmd = new_output(speed, name)
            cmd[cmd.index('--length') + 1] = str(length)
            cmd[cmd.index('--repeats') + 1] = '3'
            if variant == 'producer':
                cmd = [cmd[0], 'LOD_H100_PREFILL_PRODUCER_LSE=1', *cmd[1:]]
            jobs.append({'name': name, 'id': submit(name, cmd)})
    for variant in ('baseline', 'producer'):
        name = f'prefill-onepass-followup-{variant}-profile'
        cmd = new_output(original, name)
        if variant == 'producer':
            cmd = [cmd[0], 'LOD_H100_PREFILL_PRODUCER_LSE=1', *cmd[1:]]
        jobs.append({'name': name, 'id': submit(name, cmd)})
    reference = next(j for j in matrix['jobs'] if j['decode'] == 'entry' and j['prefill'] == 'entry' and j['group'] == 'qa1')
    base_qa = read(reference['id'])['command']
    for task in ('qa_1', 'niah_multikey_2'):
        name = 'prefill-onepass-followup-' + task.replace('_', '-')
        cmd = new_output(base_qa, name)
        cmd[cmd.index('--tasks') + 1] = task
        cmd = [cmd[0], 'LOD_H100_PREFILL_PRODUCER_LSE=1', *cmd[1:]]
        jobs.append({'name': name, 'id': submit(name, cmd), 'task': task})
    output_path.write_text(json.dumps({'jobs': jobs}, indent=2) + '\n')
    print(json.dumps(jobs), flush=True)
finally:
    for job, meta in deferred:
        old = job['id']
        name = f"ruler32k-prefill-matrix-{job['decode']}-{job['prefill']}-{job['group']}-onepass-followup-requeue"
        job['id'] = submit(name, meta['command'], pathlib.Path(meta['source']))
        job['previous_id'] = old
        tmp = matrix_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(matrix, indent=2) + '\n')
        tmp.replace(matrix_path)
