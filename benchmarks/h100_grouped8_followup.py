"""Profile the capped routes and record the grouped backend's effective settings."""
import json
import pathlib
import subprocess

root = pathlib.Path('/home/ubuntu/lod-h100/code')
state = pathlib.Path.home() / '.local/state/lod-cluster-run'
results = pathlib.Path('/home/ubuntu/comparison/results')
uv = str(pathlib.Path.home() / '.local/bin/uv')
cluster = [uv, 'run', '--offline', '--no-project', '--python', '/usr/bin/python3',
           'python', '-m', 'benchmarks.h100_cluster_run']
jobs_base = json.loads((results / 'grouped8-runs.json').read_text())['jobs']


def meta(name):
    rid = next(j['id'] for j in jobs_base if j['name'] == name)
    return json.loads((state / rid / 'run.json').read_text())


def submit(name, command, source):
    out = subprocess.check_output(cluster + ['--name', name, '--detach', '--num-gpus', '1', '--'] + command,
                                  cwd=source, text=True)
    return out.splitlines()[0].split()[-1]


def output(command, name, *, group):
    cmd = [x for x in command if not (group and x == 'LOD_H100_PREFILL_PACKED=1')]
    cmd[cmd.index('--output') + 1] = str(results / (name + '.json'))
    return cmd


jobs = []
old = meta('grouped8-grouped-verify')
name = 'grouped8-grouped-effective-verify'
verification = submit(name, output(old['command'], name, group=True), pathlib.Path(old['source']))
jobs.append({'name': name, 'id': verification})
for length in (32, 64):
    old = meta(f'grouped8-grouped-{length}k')
    name = f'grouped8-grouped-effective-{length}k'
    cmd = output(old['command'], name, group=True)
    at = cmd.index('benchmarks.h100_checked_run') + 1
    cmd[at:at] = ['--require-run', str(state / verification / 'run.json')]
    jobs.append({'name': name, 'id': submit(name, cmd, pathlib.Path(old['source']))})
old = meta('grouped8-grouped-audit')
name = 'grouped8-grouped-effective-audit'
cmd = output(old['command'], name, group=True)
jobs.append({'name': name, 'id': submit(name, cmd, pathlib.Path(old['source']))})
for length in (32, 64):
    for method in ('direct', 'grouped'):
        old = meta(f'grouped8-{method}-{length}k')
        name = f'grouped8-{method}-{length}k-profile'
        cmd = output(old['command'], name, group=method == 'grouped')
        cmd[1:1] = ['LOD_H100_PREFILL_PROFILE=1', 'LOD_H100_PREFILL_GPU_EVENTS=1']
        cmd[cmd.index('--repeats') + 1] = '2'
        jobs.append({'name': name, 'id': submit(name, cmd, pathlib.Path(old['source']))})
(results / 'grouped8-followup-runs.json').write_text(json.dumps({'jobs': jobs}, indent=2) + '\n')
print(json.dumps(jobs), flush=True)
