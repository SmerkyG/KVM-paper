"""Submit logged uv/cluster-run threshold-decode validation and fixed-cache probes."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--label',default='v1');a=p.parse_args()
    root=Path('/home/ubuntu/lod-h100/code');state=Path.home()/'.local/state/lod-cluster-run'
    results=Path('/home/ubuntu/comparison/results')
    prior=json.loads((state/'1794148564636551304-threshold-region-32768-guard-amortized'/'run.json').read_text())['command']
    prefix=prior[:prior.index('benchmarks.h100_checked_run')]
    cluster=[str(Path.home()/'.local/bin/uv'),'run','--offline','--no-project','--python','/usr/bin/python3','python','-m','benchmarks.h100_cluster_run']
    jobs=[]
    def submit(name,command):
        output=subprocess.check_output(cluster+['--name',name,'--detach','--num-gpus','1','--']+command,cwd=root,text=True)
        rid=output.splitlines()[0].split()[-1];jobs.append(dict(name=name,id=rid));return rid
    checks=[]
    for impl in ('packed','indexed','combined'):
        name=f'threshold-decode-{a.label}-{impl}-check'
        checks.append(submit(name,prefix+['benchmarks.h100_threshold_decode_check','--implementation',impl,'--output',str(results/(name+'.json'))]))
    for length in (32768,65536):
        name=f'threshold-decode-{a.label}-tune-{length//1024}k'
        cmd=prefix+['benchmarks.h100_checked_run']+[x for rid in checks for x in ('--require-run',str(state/rid/'run.json'))]
        cmd+=['--','python','-m','benchmarks.h100_threshold_decode_tune','--manifest','/home/ubuntu/lod-h100/results/prompts-b1.json',
              '--length',str(length),'--output',str(results/(name+'.json'))]
        submit(name,cmd)
    (results/f'threshold-decode-{a.label}-runs.json').write_text(json.dumps(dict(jobs=jobs),indent=2)+'\n')
    print(json.dumps(jobs,indent=2),flush=True)


if __name__=='__main__':main()
