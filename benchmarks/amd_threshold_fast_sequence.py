"""Run AMD speed variants sequentially on one cluster-run allocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=('qwen','k2'),required=True)
    p.add_argument('--mode',choices=('two-tier','three-tier-bf16','three-tier-int4'),default='two-tier')
    p.add_argument('--variants',nargs='+',choices=('top8','legacy','fast','top8-after'),required=True)
    p.add_argument('--settings-json',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();settings=json.loads(a.settings_json)
    if a.mode != 'two-tier' and 'legacy' in a.variants:
        p.error('three-tier speed comparisons use native top8 and the new threshold path')
    checkpoint={'qwen':'Qwen/Qwen3.8-27B-FP8','k2':'IFM/K2-Horizon-32B-FP8'}[a.model]
    result=dict(stage='running',model=a.model,mode=a.mode,serial=True,jobs=[])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():a.output.write_text(json.dumps(result,indent=2)+'\n')
    for variant in a.variants:
        output=a.output.with_name(a.output.stem+'-'+variant+'.json')
        env={k:v for k,v in os.environ.items() if not k.startswith('LOD_AMD_THRESHOLD_')}
        if variant not in ('top8','top8-after'):env['LOD_AMD_THRESHOLD_MULTIPLIER']='16'
        if variant=='fast':env.update(settings)
        command=['uv','run','--offline','--no-project','--python',sys.executable,'python','-m','benchmarks.prolong',
                 '--checkpoint',checkpoint,'--mode',a.mode,'--measure','speed','--lengths','65536',
                 '--batch-size','1','--decode-tokens','1025','--repeats','2','--output',str(output)]
        row=dict(variant=variant,command=command,output=str(output),
                 settings={k:v for k,v in env.items() if k.startswith('LOD_AMD_THRESHOLD_')})
        result['jobs'].append(row);save();print(json.dumps(row),flush=True)
        process=subprocess.run(command,env=env)
        row['exit_code']=process.returncode;save()
        if process.returncode:
            result['stage']='failed';save();raise SystemExit(process.returncode)
    result['stage']='complete';save()


if __name__=='__main__':main()
