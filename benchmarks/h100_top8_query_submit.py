"""Run the original top-eight selector with grouped and per-query leaf backends."""
import json
import pathlib
import subprocess

root = pathlib.Path('/home/ubuntu/lod-h100/code')
state = pathlib.Path.home() / '.local/state/lod-cluster-run'
results = pathlib.Path('/home/ubuntu/comparison/results')
cluster = [str(pathlib.Path.home()/'.local/bin/uv'), 'run', '--offline', '--no-project',
           '--python', '/usr/bin/python3', 'python', '-m', 'benchmarks.h100_cluster_run']
base = json.loads((state/'1794240997648891343-grouped8-original-top8-32k'/'run.json').read_text())['command']


def submit(name, cmd):
    out = subprocess.check_output(cluster + ['--name', name, '--detach', '--num-gpus', '1', '--'] + cmd,
                                  cwd=root, text=True)
    return out.splitlines()[0].split()[-1]


def command(name, length, direct=False, audit=False):
    cmd = base.copy()
    if direct:
        cmd[1:1] = ['LOD_H100_TOP8_QUERY_BACKEND=1']
    if audit:
        cmd[1:1] = ['LOD_H100_TOP8_QUERY_AUDIT=1']
        cmd[cmd.index('--warmups')+1] = '0'
        cmd[cmd.index('--repeats')+1] = '1'
    cmd[cmd.index('--length')+1] = str(length)
    cmd[cmd.index('--output')+1] = str(results/(name+'.json'))
    return cmd

jobs=[]
name='top8-query-32k-audit'
audit=submit(name, command(name,32768,direct=True,audit=True))
jobs.append({'name':name,'id':audit})
for length in (32768,65536):
    for backend in ('grouped','query'):
        name=f'top8-query-{backend}-{length//1024}k'
        cmd=command(name,length,direct=backend=='query')
        if backend=='query':
            i=cmd.index('benchmarks.h100_checked_run')+1
            cmd[i:i]=['--require-run',str(state/audit/'run.json')]
        jobs.append({'name':name,'id':submit(name,cmd)})
(results/'top8-query-runs.json').write_text(json.dumps({'jobs':jobs},indent=2)+'\n')
print(json.dumps(jobs),flush=True)
