"""Prioritize a paired synthetic check while preserving queued source snapshots."""
import json
import pathlib
import subprocess

root = pathlib.Path('/home/ubuntu/lod-h100/code')
state = pathlib.Path.home() / '.local/state/lod-cluster-run'
paths = [pathlib.Path('/home/ubuntu/comparison/results/prefill-score-reuse-runs.json'),
         pathlib.Path('/home/ubuntu/comparison/results/ruler32k-prefill-matrix-runs.json')]
uv = str(pathlib.Path.home() / '.local/bin/uv')
cluster = [uv, 'run', '--offline', '--no-project', '--python', '/usr/bin/python3',
           'python', '-m', 'benchmarks.h100_cluster_run']


def read(run_id):
    return json.loads((state / run_id / 'run.json').read_text())


def submit(name, command, cwd):
    result = subprocess.check_output(cluster + ['--name', name, '--detach', '--num-gpus', '1', '--'] + command,
                                     cwd=cwd, text=True)
    return result.splitlines()[0].split()[-1]


manifests = [(p, json.loads(p.read_text())) for p in paths]
deferred = []
for path, manifest in manifests:
    for job in manifest['jobs']:
        meta = read(job['id'])
        if meta['state'] == 'queued':
            subprocess.run(cluster + ['cancel', job['id']], cwd=root, check=True)
            if read(job['id']).get('exit_code') == -15:
                deferred.append((path, manifest, job, meta))
try:
    smoke = read('1794196167488653745-prefill-score-reuse-smoke')['command']
    command = [x for x in smoke if x != 'LOD_H100_PREFILL_REUSE_SCORES=1']
    command[command.index('--output') + 1] = '/home/ubuntu/comparison/results/prefill-score-reuse-paired.json'
    command.append('--compare-reuse')
    run_id = submit('prefill-score-reuse-paired', command, root)
    path, manifest = manifests[0]
    manifest['jobs'].insert(0, {'name': 'prefill-score-reuse-paired', 'id': run_id})
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(run_id, flush=True)
finally:
    for path, manifest, job, meta in deferred:
        old = job['id']
        name = old.split('-', 1)[-1] + '-paired-requeue'
        job['id'] = submit(name, meta['command'], pathlib.Path(meta['source']))
        job['previous_id'] = old
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(path)
