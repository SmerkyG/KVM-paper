"""Compare grouped and indexed leaf kernels on identical model states and routes."""
import json
import pathlib
import subprocess

root = pathlib.Path('/home/ubuntu/lod-h100/code')
state = pathlib.Path.home() / '.local/state/lod-cluster-run'
results = pathlib.Path('/home/ubuntu/comparison/results')
old = json.loads((state / '1794240997520392472-grouped8-grouped-audit' / 'run.json').read_text())
cmd = [x for x in old['command'] if x != 'LOD_H100_PREFILL_PACKED=1']
cmd[1:1] = ['LOD_H100_PREFILL_GROUPED8_CROSSCHECK=1']
cmd[cmd.index('--output') + 1] = str(results / 'grouped8-identical-state-crosscheck.json')
uv = str(pathlib.Path.home() / '.local/bin/uv')
cluster = [uv, 'run', '--offline', '--no-project', '--python', '/usr/bin/python3',
           'python', '-m', 'benchmarks.h100_cluster_run']
name = 'grouped8-identical-state-crosscheck'
out = subprocess.check_output(cluster + ['--name', name, '--detach', '--num-gpus', '1', '--'] + cmd,
                              cwd=root, text=True)
rid = out.splitlines()[0].split()[-1]
(results / 'grouped8-crosscheck-runs.json').write_text(json.dumps({'jobs': [{'name': name, 'id': rid}]}, indent=2) + '\n')
print(rid, flush=True)
