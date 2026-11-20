"""Test direct-packing tile geometries without changing the timed computation."""
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


def submit(name, cmd, cwd=root):
    text = subprocess.check_output(cluster + ['--name', name, '--detach', '--num-gpus', '1', '--'] + cmd,
                                   cwd=cwd, text=True)
    return text.splitlines()[0].split()[-1]


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
    runs = json.loads((results / 'prefill-direct-pack-runs.json').read_text())['jobs']
    target = next(j for j in runs if j['name'] == 'prefill-onepass-producer-direct-32k')
    original = read(target['id'])
    jobs = []
    for m, n in ((16, 64), (16, 128), (16, 256), (32, 64), (32, 256)):
        name = f'prefill-direct-tune-m{m}-n{n}'
        cmd = [x for x in original['command'] if not x.startswith('LOD_H100_PREFILL_SELECT_N=')]
        cmd[1:1] = [f'LOD_H100_PREFILL_SELECT_M={m}', f'LOD_H100_PREFILL_SELECT_N={n}']
        cmd[cmd.index('--output') + 1] = str(results / (name + '.json'))
        cmd[cmd.index('--repeats') + 1] = '2'
        jobs.append({'name': name, 'id': submit(name, cmd, pathlib.Path(original['source'])), 'm': m, 'n': n})
    (results / 'prefill-direct-tune-runs.json').write_text(json.dumps({'jobs': jobs}, indent=2) + '\n')
    print(json.dumps(jobs), flush=True)
finally:
    for path, data, job, meta in deferred:
        old = job['id']
        job['id'] = submit(old.split('-', 1)[-1] + '-tune-requeue', meta['command'], pathlib.Path(meta['source']))
        job['previous_id'] = old
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2) + '\n')
        tmp.replace(path)
