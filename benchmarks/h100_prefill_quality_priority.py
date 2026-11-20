"""Let completed mask-free quality runs reach their matched controls sooner."""
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


def submit(name, cmd, cwd):
    text = subprocess.check_output(cluster + ['--name', name, '--detach', '--num-gpus', '1', '--'] + cmd,
                                   cwd=cwd, text=True)
    return text.splitlines()[0].split()[-1]


paths = [results / 'prefill-onepass-followup-runs.json', results / 'ruler32k-prefill-matrix-runs.json']
deferred = []
for path in paths:
    data = json.loads(path.read_text())
    for job in data['jobs']:
        meta = read(job['id'])
        if meta['state'] == 'queued':
            subprocess.run(cluster + ['cancel', job['id']], cwd=root, check=True)
            if read(job['id']).get('exit_code') == -15:
                deferred.append((path, data, job, meta))
for path, data, job, meta in deferred:
    old = job['id']
    job['id'] = submit(old.split('-', 1)[-1] + '-controlfirst', meta['command'], pathlib.Path(meta['source']))
    job['previous_id'] = old
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)
print('Reordered', len(deferred), 'queued jobs; running jobs were left alone.', flush=True)
