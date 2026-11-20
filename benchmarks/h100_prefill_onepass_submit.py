"""Submit producer-normalization and bitmask prefill experiments on H100.

Deferred matrix jobs are requeued from their original source snapshots.
"""
import json
import pathlib
import subprocess
import sys

root = pathlib.Path('/home/ubuntu/lod-h100/code')
state = pathlib.Path.home() / '.local/state/lod-cluster-run'
manifest_path = pathlib.Path('/home/ubuntu/comparison/results/ruler32k-prefill-matrix-runs.json')
direct = '--direct' in sys.argv[1:]
experiment_path = pathlib.Path('/home/ubuntu/comparison/results/' +
                               ('prefill-direct-pack-runs.json' if direct else 'prefill-onepass-runs.json'))
uv = str(pathlib.Path.home() / '.local/bin/uv')
cluster = [uv, 'run', '--offline', '--no-project', '--python', '/usr/bin/python3',
           'python', '-m', 'benchmarks.h100_cluster_run']


def meta(run_id):
    return json.loads((state / run_id / 'run.json').read_text())


def submit(name, cmd, cwd=root):
    result = subprocess.check_output(cluster + ['--name', name, '--detach', '--num-gpus', '1', '--'] + cmd,
                                     cwd=cwd, text=True)
    return result.splitlines()[0].split()[-1]


def add_env(cmd, *values):
    return [cmd[0], *values, *cmd[1:]]


def output(cmd, name):
    cmd = cmd.copy()
    cmd[cmd.index('--output') + 1] = '/home/ubuntu/comparison/results/' + name + '.json'
    return cmd


matrix = json.loads(manifest_path.read_text())
deferred = []
for job in matrix['jobs']:
    old = meta(job['id'])
    if old['state'] == 'queued':
        subprocess.run(cluster + ['cancel', job['id']], cwd=root, check=True)
        if meta(job['id']).get('exit_code') == -15:
            deferred.append((job, old))
try:
    base = meta('1794188763480512353-prefill-attention-entry16-32k-gpu-events')['command']
    base = [x for x in base if x not in ('LOD_H100_PREFILL_PROFILE=1', 'LOD_H100_PREFILL_GPU_EVENTS=1')]
    python_index = base.index(uv)
    runner = base[python_index:base.index('benchmarks.h100_checked_run') - 1]
    env = base[:python_index]
    variants = {'producer': ['LOD_H100_PREFILL_PRODUCER_LSE=1'],
                'bitmask': ['LOD_H100_PREFILL_BITMASK=1'],
                'combined': ['LOD_H100_PREFILL_PRODUCER_LSE=1', 'LOD_H100_PREFILL_BITMASK=1']}
    if direct:
        variants = {'direct': ['LOD_H100_PREFILL_PACK_DIRECT=1'],
                    'producer-direct': ['LOD_H100_PREFILL_PRODUCER_LSE=1', 'LOD_H100_PREFILL_PACK_DIRECT=1']}
    jobs = []
    for variant, flags in variants.items():
        name = 'prefill-onepass-' + variant + '-smoke'
        cmd = add_env(env, *flags) + runner + ['-m', 'benchmarks.h100_threshold_prefill_check',
              '--queries', '8', '--tile-queries', '8', '--slots', '49' if direct else '48', '--entry-only',
              '--output', '/home/ubuntu/comparison/results/' + name + '.json']
        rid = submit(name, cmd)
        jobs.append({'name': name, 'id': rid, 'variant': variant})
    # No performance or quality result is valid unless its synthetic correctness check succeeds.
    for length in (32768, 65536):
        for variant, flags in variants.items():
            name = f'prefill-onepass-{variant}-{length//1024}k'
            cmd = add_env(output(base, name), *flags)
            cmd[cmd.index('--length') + 1] = str(length)
            cmd[cmd.index('--repeats') + 1] = '3'
            smoke = next(j['id'] for j in jobs if j['name'] == 'prefill-onepass-' + variant + '-smoke')
            i = cmd.index('benchmarks.h100_checked_run') + 1
            cmd[i:i] = ['--require-run', str(state / smoke / 'run.json')]
            jobs.append({'name': name, 'id': submit(name, cmd), 'variant': variant})
    original_verify = meta('1794185269455954088-prefill-next-packed128-ruler32k-verify')['command']
    for variant, flags in variants.items():
        name = f'prefill-onepass-{variant}-verify'
        cmd = add_env(output(original_verify, name), *flags)
        smoke = next(j['id'] for j in jobs if j['name'] == 'prefill-onepass-' + variant + '-smoke')
        i = cmd.index('benchmarks.h100_checked_run') + 1
        cmd[i:i] = ['--require-run', str(state / smoke / 'run.json')]
        jobs.append({'name': name, 'id': submit(name, cmd), 'variant': variant})
    experiment_path.write_text(json.dumps({'jobs': jobs}, indent=2) + '\n')
    print(json.dumps(jobs), flush=True)
finally:
    for job, old in deferred:
        old_id = job['id']
        name = f"ruler32k-prefill-matrix-{job['decode']}-{job['prefill']}-{job['group']}-onepass-requeue"
        job['id'] = submit(name, old['command'], pathlib.Path(old['source']))
        job['previous_id'] = old_id
        temp = manifest_path.with_suffix('.tmp')
        temp.write_text(json.dumps(matrix, indent=2) + '\n')
        temp.replace(manifest_path)
