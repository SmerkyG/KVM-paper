"""Audit region sharing for top-8 and capped threshold on matched prompts."""
import json
import pathlib
import subprocess
root=pathlib.Path('/home/ubuntu/lod-h100/code')
state=pathlib.Path.home()/'.local/state/lod-cluster-run'
results=pathlib.Path('/home/ubuntu/comparison/results')
cluster=[str(pathlib.Path.home()/'.local/bin/uv'),'run','--offline','--no-project',
         '--python','/usr/bin/python3','python','-m','benchmarks.h100_cluster_run']

def prior(manifest, name):
    j=next(x for x in json.loads((results/manifest).read_text())['jobs'] if x['name']==name)
    return json.loads((state/j['id']/'run.json').read_text())['command']

def submit(name,cmd):
    o=subprocess.check_output(cluster+['--name',name,'--detach','--num-gpus','1','--']+cmd,cwd=root,text=True)
    return o.splitlines()[0].split()[-1]

jobs=[]
for length in (32,64):
    for method in ('top8','threshold'):
        name=f'prefill-overlap-{method}-{length}k'
        if method=='top8':
            cmd=prior('top8-query-runs.json',f'top8-query-grouped-{length}k')
        else:
            cmd=prior('grouped8-fused-runs.json',f'grouped8-fusedtest-grouped-{length}k')
        cmd[1:1]=['LOD_H100_OVERLAP_AUDIT=1']
        cmd[cmd.index('--warmups')+1]='0'
        cmd[cmd.index('--repeats')+1]='1'
        cmd[cmd.index('--output')+1]=str(results/(name+'.json'))
        jobs.append({'name':name,'id':submit(name,cmd)})
(results/'prefill-overlap-runs.json').write_text(json.dumps({'jobs':jobs},indent=2)+'\n')
print(json.dumps(jobs),flush=True)
