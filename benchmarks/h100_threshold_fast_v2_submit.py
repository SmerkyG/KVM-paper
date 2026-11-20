"""Compare final-merge tiling and both exact consumers of fast threshold prefill."""
import argparse,json,pathlib,subprocess
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--label',default='v2')
parser.add_argument('--saved-correction',action='store_true');parser.add_argument('--select-m',type=int,default=1)
parser.add_argument('--route-capacity',type=int,default=128);parser.add_argument('--tile',type=int,default=16384);options=parser.parse_args();label=options.label
root=pathlib.Path('/home/ubuntu/lod-h100/code');s=pathlib.Path.home()/'.local/state/lod-cluster-run';p=pathlib.Path('/home/ubuntu/comparison/results')
cluster=[str(pathlib.Path.home()/'.local/bin/uv'),'run','--offline','--no-project','--python','/usr/bin/python3','python','-m','benchmarks.h100_cluster_run']
base=json.loads((s/'1794242313619807234-top8-query-grouped-32k'/'run.json').read_text())['command']
jobs=[]
def submit(name,cmd):
 name=name.replace('threshold-fast-v2',f'threshold-fast-{label}')
 cmd=[x.replace('threshold-fast-v2',f'threshold-fast-{label}') for x in cmd]
 out=subprocess.check_output(cluster+['--name',name,'--detach','--num-gpus','1','--']+cmd,cwd=root,text=True);rid=out.splitlines()[0].split()[-1];jobs.append({'name':name,'id':rid});return rid

def configure(name,length,mode,verify=False,prerequisites=()):
 cmd=base.copy();cap=8 if mode=='cap8' else 0;backend='direct' if mode=='direct' else 'grouped'
 cmd[1:1]=['LOD_H100_PREFILL_RULE=entry','LOD_H100_PREFILL_IMPLEMENTATION=fast',f'LOD_H100_PREFILL_MAX_REGIONS={cap}',f'LOD_H100_FAST_TILE={options.tile}',f'LOD_H100_FAST_LEAF={backend}']
 cmd[1:1]=[f'LOD_H100_FAST_SAVED_CORRECTION={int(options.saved_correction)}',f'LOD_H100_FAST_SELECT_M={options.select_m}',f'LOD_H100_FAST_ROUTE_CAPACITY={options.route_capacity}']
 if verify:
  cmd[1:1]=['LOD_H100_FAST_VERIFY=1'];cmd[cmd.index('--warmups')+1]='0';cmd[cmd.index('--repeats')+1]='1'
 cmd[cmd.index('--length')+1]=str(length);cmd[cmd.index('--output')+1]=str(p/(name+'.json'))
 i=cmd.index('benchmarks.h100_checked_run')+1;cmd[i:i]=[x for rid in prerequisites for x in ('--require-run',str(s/rid/'run.json'))]
 return cmd
sb=base[:base.index('benchmarks.h100_checked_run')]
sb[1:1]=[f'LOD_H100_FAST_SAVED_CORRECTION={int(options.saved_correction)}',f'LOD_H100_FAST_SELECT_M={options.select_m}']
name='threshold-fast-v2-smoke';smoke=submit(name,sb+['benchmarks.h100_threshold_fast_check','--output',str(p/(name+'.json'))])
name='threshold-fast-v2-overflow-smoke';cmd=sb+['benchmarks.h100_threshold_fast_check','--output',str(p/(name+'.json'))];cmd[1:1]=['LOD_H100_FAST_ROUTE_CAPACITY=1'];overflow=submit(name,cmd)
checks={}
for mode in ('cap8','grouped','direct'):
 name=f'threshold-fast-v2-{mode}-64k-audit';checks[mode]=submit(name,configure(name,65536,mode,verify=True,prerequisites=(smoke,overflow)))
for length in (32768,65536):
 for mode in ('cap8','grouped','direct'):
  name=f'threshold-fast-v2-{mode}-{length//1024}k';submit(name,configure(name,length,mode,prerequisites=(checks[mode],)))
 name=f'threshold-fast-v2-top8-{length//1024}k';cmd=base.copy();cmd[cmd.index('--length')+1]=str(length);cmd[cmd.index('--output')+1]=str(p/(name+'.json'));submit(name,cmd)
 # Old residual-coarse pipeline, retaining all entry-16 routes.
 prior=json.loads((p/'grouped8-fused-runs.json').read_text())['jobs'];j=next(x for x in prior if x['name']==f'grouped8-fusedtest-direct-{length//1024}k')
 cmd=json.loads((s/j['id']/'run.json').read_text())['command'];cmd=[x for x in cmd if not x.startswith('LOD_H100_PREFILL_MAX_REGIONS=')]
 name=f'threshold-fast-v2-legacy-{length//1024}k';cmd[cmd.index('--output')+1]=str(p/(name+'.json'));submit(name,cmd)
(p/f'threshold-fast-{label}-runs.json').write_text(json.dumps({'jobs':jobs},indent=2)+'\n');print(json.dumps(jobs),flush=True)
