"""Compare grouped attention with top-8-style fused route bookkeeping."""
import json
import pathlib
import subprocess

root = pathlib.Path('/home/ubuntu/lod-h100/code')
state = pathlib.Path.home() / '.local/state/lod-cluster-run'
results = pathlib.Path('/home/ubuntu/comparison/results')
uv = str(pathlib.Path.home() / '.local/bin/uv')
cluster = [uv, 'run', '--offline', '--no-project', '--python', '/usr/bin/python3',
           'python', '-m', 'benchmarks.h100_cluster_run']
prior = json.loads((results / 'grouped8-runs.json').read_text())['jobs']


def meta(name):
    rid = next(j['id'] for j in prior if j['name'] == name)
    return json.loads((state / rid / 'run.json').read_text())


def submit(name, command):
    out = subprocess.check_output(cluster + ['--name', name, '--detach', '--num-gpus', '1', '--'] + command,
                                  cwd=root, text=True)
    return out.splitlines()[0].split()[-1]


def configure(command, name, method):
    cmd = [x for x in command if not (method != 'direct' and x == 'LOD_H100_PREFILL_PACKED=1')]
    if method == 'fused':
        cmd[1:1] = ['LOD_H100_PREFILL_GROUPED8_FUSED_COUNTS=1']
    cmd[cmd.index('--output') + 1] = str(results / (name + '.json'))
    return cmd


jobs = []
name = 'grouped8-fused-counts-smoke'
smoke = submit(name, configure(meta('grouped8-grouped-smoke')['command'], name, 'fused'))
jobs.append({'name': name, 'id': smoke})
name = 'grouped8-fused-counts-verify'
cmd = configure(meta('grouped8-grouped-verify')['command'], name, 'fused')
i = cmd.index('benchmarks.h100_checked_run') + 1
cmd[i:i] = ['--require-run', str(state / smoke / 'run.json')]
verify = submit(name, cmd)
jobs.append({'name': name, 'id': verify})
for length in (32, 64):
    for method in ('direct', 'grouped', 'fused'):
        name = f'grouped8-fusedtest-{method}-{length}k'
        base_name = f'grouped8-{"direct" if method == "direct" else "grouped"}-{length}k'
        cmd = configure(meta(base_name)['command'], name, method)
        if method == 'fused':
            i = cmd.index('benchmarks.h100_checked_run') + 1
            cmd[i:i] = ['--require-run', str(state / verify / 'run.json')]
        jobs.append({'name': name, 'id': submit(name, cmd)})
name = 'grouped8-fused-counts-crosscheck'
cmd = configure(meta('grouped8-grouped-audit')['command'], name, 'fused')
cmd[1:1] = ['LOD_H100_PREFILL_GROUPED8_CROSSCHECK=1']
i = cmd.index('benchmarks.h100_checked_run') + 1
cmd[i:i] = ['--require-run', str(state / verify / 'run.json')]
jobs.append({'name': name, 'id': submit(name, cmd)})
for length in (32, 64):
    name = f'grouped8-fused-counts-{length}k-profile'
    cmd = configure(meta(f'grouped8-grouped-{length}k')['command'], name, 'fused')
    cmd[1:1] = ['LOD_H100_PREFILL_PROFILE=1', 'LOD_H100_PREFILL_GPU_EVENTS=1']
    cmd[cmd.index('--repeats') + 1] = '2'
    i = cmd.index('benchmarks.h100_checked_run') + 1
    cmd[i:i] = ['--require-run', str(state / verify / 'run.json')]
    jobs.append({'name': name, 'id': submit(name, cmd)})
(results / 'grouped8-fused-runs.json').write_text(json.dumps({'jobs': jobs}, indent=2) + '\n')
print(json.dumps(jobs), flush=True)
