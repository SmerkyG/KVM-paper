"""Run the direct-packing diagnostic before queued experiments, without changing their snapshots."""
import json
import pathlib
import shutil
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
    result = subprocess.check_output(cluster + ['--name', name, '--detach', '--num-gpus', '1', '--'] + command,
                                     cwd=cwd, text=True)
    return result.splitlines()[0].split()[-1]


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
    old = results / 'prefill-direct-pack-runs.json'
    shutil.copyfile(old, results / 'prefill-direct-pack-second-failed-runs.json')
    subprocess.run(cluster[:-2] + ['-m', 'benchmarks.h100_prefill_onepass_submit', '--direct'],
                   cwd=root, check=True)
finally:
    for path, data, job, meta in deferred:
        old = job['id']
        job['id'] = submit(old.split('-', 1)[-1] + '-debug-requeue', meta['command'], pathlib.Path(meta['source']))
        job['previous_id'] = old
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2) + '\n')
        tmp.replace(path)
