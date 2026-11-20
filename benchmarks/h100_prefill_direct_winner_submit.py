"""Validate and benchmark the promising direct-packing selection geometry."""
import json
import pathlib
import subprocess

root = pathlib.Path('/home/ubuntu/lod-h100/code')
state = pathlib.Path.home() / '.local/state/lod-cluster-run'
results = pathlib.Path('/home/ubuntu/comparison/results')
uv = str(pathlib.Path.home() / '.local/bin/uv')
cluster = [uv, 'run', '--offline', '--no-project', '--python', '/usr/bin/python3',
           'python', '-m', 'benchmarks.h100_cluster_run']


def read(rid):
    return json.loads((state / rid / 'run.json').read_text())


def submit(name, command, cwd=root):
    out = subprocess.check_output(cluster + ['--name', name, '--detach', '--num-gpus', '1', '--'] + command,
                                  cwd=cwd, text=True)
    return out.splitlines()[0].split()[-1]


paths = [results / n for n in ('prefill-onepass-followup-runs.json',
                               'prefill-onepass-quality-control-runs.json',
                               'ruler32k-prefill-matrix-runs.json')]
manifests = [(p, json.loads(p.read_text())) for p in paths]
deferred = []
for path, data in manifests:
    for job in data['jobs']:
        meta = read(job['id'])
        if meta['state'] == 'queued':
            subprocess.run(cluster + ['cancel', job['id']], cwd=root, check=True)
            if read(job['id']).get('exit_code') == -15:
                deferred.append((path, data, job, meta))
try:
    direct = json.loads((results / 'prefill-direct-pack-runs.json').read_text())['jobs']
    speed_job = next(j for j in direct if j['name'] == 'prefill-onepass-producer-direct-32k')
    speed_meta = read(speed_job['id'])
    source = pathlib.Path(speed_meta['source'])
    def tuned(command, name):
        cmd = [x for x in command if not x.startswith('LOD_H100_PREFILL_SELECT_N=')]
        cmd[1:1] = ['LOD_H100_PREFILL_SELECT_N=64']
        cmd[cmd.index('--output') + 1] = str(results / (name + '.json'))
        return cmd
    jobs = []
    for length in (32768, 65536):
        name = f'prefill-direct-winner-{length//1024}k'
        cmd = tuned(speed_meta['command'], name)
        cmd[cmd.index('--length') + 1] = str(length)
        cmd[cmd.index('--repeats') + 1] = '3'
        jobs.append({'name': name, 'id': submit(name, cmd, source)})
    verify_job = next(j for j in direct if j['name'] == 'prefill-onepass-producer-direct-verify')
    name = 'prefill-direct-winner-verify'
    verify = submit(name, tuned(read(verify_job['id'])['command'], name), source)
    jobs.append({'name': name, 'id': verify})
    followup = next(data for path, data in manifests if path.name == 'prefill-onepass-followup-runs.json')
    for task in ('qa_1', 'niah_multikey_2'):
        name = 'prefill-direct-winner-' + task.replace('_', '-')
        job = next(j for j in followup['jobs'] if j.get('task') == task)
        # The original source snapshot is identical for inference to the speed snapshot.
        cmd = tuned(read(job['id'])['command'], name)
        cmd[1:1] = ['LOD_H100_PREFILL_PACK_DIRECT=1']
        i = cmd.index('benchmarks.h100_checked_run') + 1
        cmd[i:i] = ['--require-run', str(state / verify / 'run.json')]
        jobs.append({'name': name, 'id': submit(name, cmd, source), 'task': task})
    (results / 'prefill-direct-winner-runs.json').write_text(json.dumps({'jobs': jobs}, indent=2) + '\n')
    print(json.dumps(jobs), flush=True)
finally:
    for path, data, job, meta in deferred:
        old = job['id']
        job['id'] = submit(old.split('-', 1)[-1] + '-winner-requeue', meta['command'], pathlib.Path(meta['source']))
        job['previous_id'] = old
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2) + '\n')
        tmp.replace(path)
