"""Compare capped threshold routes in the indexed and original grouped leaf kernels.

Every run snapshots the same repository state through the H100 cluster-run adapter.
"""
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


def submit(name, command):
    out = subprocess.check_output(cluster + ['--name', name, '--detach', '--num-gpus', '1', '--'] + command,
                                  cwd=root, text=True)
    return out.splitlines()[0].split()[-1]


def variant(command, name, method):
    command = [x for x in command if x not in ('LOD_H100_PREFILL_PACK_DIRECT=1',
                'LOD_H100_PREFILL_PROFILE=1', 'LOD_H100_PREFILL_GPU_EVENTS=1')]
    command[1:1] = ['LOD_H100_PREFILL_MAX_REGIONS=8', 'LOD_H100_PREFILL_PRODUCER_LSE=1',
                    'LOD_H100_PREFILL_GROUPED8=1' if method == 'grouped' else 'LOD_H100_PREFILL_PACK_DIRECT=1']
    command[command.index('--output') + 1] = str(results / (name + '.json'))
    return command


jobs = []
smokes = {}
smoke = read('1794198801713400820-prefill-onepass-direct-smoke')['command']
for method in ('direct', 'grouped'):
    name = 'grouped8-' + method + '-smoke'
    cmd = variant(smoke, name, method)
    cmd[cmd.index('--slots') + 1] = '128'
    smokes[method] = submit(name, cmd)
    jobs.append({'name': name, 'id': smokes[method]})

verify_base = read('1794185269455954088-prefill-next-packed128-ruler32k-verify')['command']
verifies = {}
for method in ('direct', 'grouped'):
    name = 'grouped8-' + method + '-verify'
    cmd = variant(verify_base, name, method)
    cmd = ['LOD_H100_PREFILL_TILE_QUERIES=512' if x == 'LOD_H100_PREFILL_TILE_QUERIES=1024' else x for x in cmd]
    i = cmd.index('benchmarks.h100_checked_run') + 1
    cmd[i:i] = ['--require-run', str(state / smokes[method] / 'run.json')]
    verifies[method] = submit(name, cmd)
    jobs.append({'name': name, 'id': verifies[method]})

base = read('1794188763480512353-prefill-attention-entry16-32k-gpu-events')['command']
for length in (32768, 65536):
    for method in ('direct', 'grouped'):
        name = f'grouped8-{method}-{length//1024}k'
        cmd = variant(base, name, method)
        cmd[cmd.index('--length') + 1] = str(length)
        cmd[cmd.index('--repeats') + 1] = '3'
        i = cmd.index('benchmarks.h100_checked_run') + 1
        cmd[i:i] = ['--require-run', str(state / verifies[method] / 'run.json')]
        jobs.append({'name': name, 'id': submit(name, cmd)})

for method in ('direct', 'grouped'):
    name = f'grouped8-{method}-audit'
    cmd = variant(base, name, method)
    cmd[1:1] = ['LOD_H100_PREFILL_CAP_AUDIT=1']
    cmd[cmd.index('--warmups') + 1] = '0'
    cmd[cmd.index('--repeats') + 1] = '1'
    i = cmd.index('benchmarks.h100_checked_run') + 1
    cmd[i:i] = ['--require-run', str(state / verifies[method] / 'run.json')]
    jobs.append({'name': name, 'id': submit(name, cmd)})

top8 = read('1794184867838551085-prefill-next-top8-32768-timing')['command']
for length in (32768, 65536):
    name = f'grouped8-original-top8-{length//1024}k'
    cmd = top8.copy()
    cmd[cmd.index('--length') + 1] = str(length)
    cmd[cmd.index('--output') + 1] = str(results / (name + '.json'))
    cmd[cmd.index('--repeats') + 1] = '3'
    jobs.append({'name': name, 'id': submit(name, cmd)})

(results / 'grouped8-runs.json').write_text(json.dumps({'jobs': jobs}, indent=2) + '\n')
print(json.dumps(jobs), flush=True)
