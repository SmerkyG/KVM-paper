"""Record GPU event stages without per-stage synchronization for fast prefill."""
import argparse,json,pathlib,subprocess
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--label',default='v2');label=parser.parse_args().label
p=pathlib.Path('/home/ubuntu/comparison/results');s=pathlib.Path.home()/'.local/state/lod-cluster-run';root=pathlib.Path('/home/ubuntu/lod-h100/code')
cluster=[str(pathlib.Path.home()/'.local/bin/uv'),'run','--offline','--no-project','--python','/usr/bin/python3','python','-m','benchmarks.h100_cluster_run']
prior=json.loads((p/f'threshold-fast-{label}-runs.json').read_text())['jobs'];jobs=[]
for mode,length in [('cap8',32),('grouped',32),('direct',32),('direct',64)]:
 j=next(x for x in prior if x['name']==f'threshold-fast-{label}-{mode}-{length}k')
 cmd=json.loads((s/j['id']/'run.json').read_text())['command'];cmd[1:1]=['LOD_H100_PREFILL_PROFILE=1'];cmd[cmd.index('--repeats')+1]='2'
 name=f'threshold-fast-{label}-profile-{mode}-{length}k';cmd[cmd.index('--output')+1]=str(p/(name+'.json'))
 out=subprocess.check_output(cluster+['--name',name,'--detach','--num-gpus','1','--']+cmd,cwd=root,text=True);rid=out.splitlines()[0].split()[-1];jobs.append({'name':name,'id':rid})
(p/f'threshold-fast-{label}-profile-runs.json').write_text(json.dumps({'jobs':jobs},indent=2)+'\n');print(json.dumps(jobs),flush=True)
