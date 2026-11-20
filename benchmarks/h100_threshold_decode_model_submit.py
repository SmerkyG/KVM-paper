"""Validate and time uncapped threshold decode with complete amortized model steps.

Checks compare the optimized graph to legacy eager decode on the same cache,
including update boundaries. Separate timings retain 512 warmup + 1024 measured
decode steps, whole-model graphs, top-eight prefill, and all cache catch-up.
"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--label',default='v3');a=p.parse_args()
    root=Path('/home/ubuntu/lod-h100/code');state=Path.home()/'.local/state/lod-cluster-run'
    results=Path('/home/ubuntu/comparison/results');label=f'threshold-decode-{a.label}'
    prior=json.loads((state/'1794148564636551304-threshold-region-32768-guard-amortized'/'run.json').read_text())['command']
    tune=json.loads((results/'threshold-decode-v2-runs.json').read_text())['jobs']
    tune_checks=[j['id'] for j in tune if '-tune-' in j['name']]
    cluster=[str(Path.home()/'.local/bin/uv'),'run','--offline','--no-project','--python','/usr/bin/python3','python','-m','benchmarks.h100_cluster_run']
    jobs=[]
    def submit(name,cmd):
        output=subprocess.check_output(cluster+['--name',name,'--detach','--num-gpus','1','--']+cmd,cwd=root,text=True)
        rid=output.splitlines()[0].split()[-1];jobs.append(dict(name=name,id=rid));return rid
    def configure(rule,length,mode,*,check=False,required=()):
        name=f'{label}-{rule}-{mode}-{length//1024}k'+('-check' if check else '')
        impl,splits=('indexed',64) if rule=='region' else (('combined',16) if length==32768 else ('indexed',32))
        if mode=='legacy':impl,splits='legacy',64
        cmd=[x for x in prior if not x.startswith(('LOD_H100_THRESHOLD_RULE=','LOD_H100_THRESHOLD_SPLITS='))]
        cmd[1:1]=[f'LOD_H100_THRESHOLD_RULE={rule}',f'LOD_H100_THRESHOLD_IMPLEMENTATION={impl}',f'LOD_H100_THRESHOLD_SPLITS={splits}']
        cmd[cmd.index('--length')+1]=str(length);cmd[cmd.index('--output')+1]=str(results/(name+'.json'))
        if check:
            cmd[1:1]=['LOD_H100_DECODE_CHECK_REFERENCE=1']
            for key,value in [('--tokens','514'),('--warmups','0'),('--repeats','1'),('--decode-warmup-steps','0')]:cmd[cmd.index(key)+1]=value
            cmd.append('--check-graph-same-cache')
        index=cmd.index('benchmarks.h100_checked_run')+1
        cmd[index:index]=[x for rid in required for x in ('--require-run',str(state/rid/'run.json'))]
        return submit(name,cmd)
    checks={}
    for length in (32768,65536):
        for rule in ('region','entry'):
            checks[rule,length]=configure(rule,length,'fast',check=True,required=tune_checks)
    for length in (32768,65536):
        for rule in ('region','entry'):
            for mode in ('legacy','fast'):
                configure(rule,length,mode,required=(checks[rule,length],))
    (results/(label+'-runs.json')).write_text(json.dumps(dict(jobs=jobs),indent=2)+'\n')
    print(json.dumps(jobs,indent=2),flush=True)


if __name__=='__main__':main()
