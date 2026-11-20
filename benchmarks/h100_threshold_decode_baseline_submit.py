"""Measure top-eight and dense controls for the optimized threshold comparison.

Same H100, BF16 checkpoint, batch-one replayed prompts, whole-model CUDA graphs,
one warmup repetition plus two measurements, 512 untimed and 1024 timed decode
steps. LoD updates are included. Every experiment uses uv and cluster-run.
"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--label',default='baselines');a=p.parse_args()
    root=Path('/home/ubuntu/lod-h100/code');state=Path.home()/'.local/state/lod-cluster-run'
    results=Path('/home/ubuntu/comparison/results')
    manifest=json.loads((results/'threshold-decode-v4-runs.json').read_text())['jobs']
    cluster=[str(Path.home()/'.local/bin/uv'),'run','--offline','--no-project','--python','/usr/bin/python3','python','-m','benchmarks.h100_cluster_run']
    jobs=[]
    for length in (32,64):
        prior=next(j for j in manifest if j['name']==f'threshold-decode-v4-region-legacy-{length}k')
        base=json.loads((state/prior['id']/'run.json').read_text())['command']
        for mode in ('top8','dense'):
            name=f'threshold-decode-{a.label}-{mode}-{length}k'
            command=[x for x in base if not x.startswith('LOD_H100_THRESHOLD_')]
            if mode=='dense':
                command=[x for x in command if not x.startswith(('LOD_RETRO_TWO_TIER=','LOD_TRACK_UPDATES='))]
                command[command.index('--method')+1]='retro-full'
                command[command.index('--lod-graph')]='--native-graph'
            command[command.index('--output')+1]=str(results/(name+'.json'))
            index=command.index('benchmarks.h100_checked_run')+1
            command[index:index]=['--require-run',str(state/prior['id']/'run.json')]
            if mode=='dense':
                check='1794001137902607466-retro-full-graph-same-cache-32k' if length==32 else '1794001213542288980-retro-full-graph-same-cache-64k'
                command[index:index]=['--require-run',str(state/check/'run.json')]
            output=subprocess.check_output(cluster+['--name',name,'--detach','--num-gpus','1','--']+command,cwd=root,text=True)
            run_id=output.splitlines()[0].split()[-1];jobs.append(dict(name=name,id=run_id))
    (results/f'threshold-decode-{a.label}-runs.json').write_text(json.dumps(dict(jobs=jobs),indent=2)+'\n')
    print(json.dumps(jobs,indent=2),flush=True)


if __name__=='__main__':main()
