"""Validate device-only fast threshold prefill, then measure matched latencies."""
import json,pathlib,subprocess
root=pathlib.Path('/home/ubuntu/lod-h100/code');s=pathlib.Path.home()/'.local/state/lod-cluster-run';p=pathlib.Path('/home/ubuntu/comparison/results')
cluster=[str(pathlib.Path.home()/'.local/bin/uv'),'run','--offline','--no-project','--python','/usr/bin/python3','python','-m','benchmarks.h100_cluster_run']
base=json.loads((s/'1794242313619807234-top8-query-grouped-32k'/'run.json').read_text())['command']
jobs=[]
def submit(name,cmd):
 out=subprocess.check_output(cluster+['--name',name,'--detach','--num-gpus','1','--']+cmd,cwd=root,text=True);rid=out.splitlines()[0].split()[-1];jobs.append({'name':name,'id':rid});return rid

def configure(name,length,cap,tile,verify=False,prerequisites=()):
 cmd=base.copy()
 cmd[1:1]=['LOD_H100_PREFILL_RULE=entry','LOD_H100_PREFILL_IMPLEMENTATION=fast',f'LOD_H100_PREFILL_MAX_REGIONS={cap}',f'LOD_H100_FAST_TILE={tile}']
 if verify:
  cmd[1:1]=['LOD_H100_FAST_VERIFY=1'];cmd[cmd.index('--warmups')+1]='0';cmd[cmd.index('--repeats')+1]='1'
 cmd[cmd.index('--length')+1]=str(length);cmd[cmd.index('--output')+1]=str(p/(name+'.json'))
 i=cmd.index('benchmarks.h100_checked_run')+1
 cmd[i:i]=[x for rid in prerequisites for x in ('--require-run',str(s/rid/'run.json'))]
 return cmd
smoke_base=base[:base.index('benchmarks.h100_checked_run')]
name='threshold-fast-grouped-smoke';cmd=smoke_base+['benchmarks.h100_threshold_fast_check','--output',str(p/(name+'.json'))];smoke=submit(name,cmd)
name='threshold-fast-grouped-overflow-smoke';cmd=smoke_base+['benchmarks.h100_threshold_fast_check','--output',str(p/(name+'.json'))];cmd[1:1]=['LOD_H100_FAST_ROUTE_CAPACITY=1'];overflow=submit(name,cmd)
checks={}
for length in (32768,65536):
 for cap in (8,0):
  name=f'threshold-fast-grouped-cap{cap}-{length//1024}k-audit'
  checks[(length,cap)]=submit(name,configure(name,length,cap,16384,verify=True,prerequisites=(smoke,overflow)))
for length in (32768,65536):
 for cap in (8,0):
  name=f'threshold-fast-grouped-cap{cap}-{length//1024}k'
  submit(name,configure(name,length,cap,16384,prerequisites=(checks[(length,cap)],)))
 name=f'threshold-fast-top8-{length//1024}k';cmd=base.copy();cmd[cmd.index('--length')+1]=str(length);cmd[cmd.index('--output')+1]=str(p/(name+'.json'));submit(name,cmd)
(p/'threshold-fast-grouped-runs.json').write_text(json.dumps({'jobs':jobs},indent=2)+'\n');print(json.dumps(jobs),flush=True)
