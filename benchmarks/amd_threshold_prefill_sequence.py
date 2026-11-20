"""Measure top-8 / threshold / top-8 prefill serially on one AMD allocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lengths', default='32768,65536')
    p.add_argument('--multiplier', type=float, default=16.)
    p.add_argument('--backend', choices=('blas','triton'), default='blas')
    p.add_argument('--variants', nargs='+', choices=('top8','threshold','top8-after','dense'), default=['top8','threshold','top8-after','dense'])
    a = p.parse_args()
    result = dict(stage='running', serial=True, scope='full model prefill including cache construction; decode unchanged', jobs=[])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for variant in a.variants:
        env = {k:v for k,v in os.environ.items() if not k.startswith(('LOD_AMD_PREFILL_', 'LOD_AMD_THRESHOLD_'))}
        if variant == 'threshold':
            env['LOD_AMD_PREFILL_THRESHOLD'] = str(a.multiplier)
            env['LOD_AMD_PREFILL_MATMUL'] = a.backend
        output = a.output.with_name(a.output.stem+'-'+variant+'.json')
        command = ['uv','run','--offline','--no-project','--python',sys.executable,'python','-m','benchmarks.prolong',
            '--checkpoint','Qwen/Qwen3.8-27B-FP8','--mode','full' if variant=='dense' else 'two-tier','--measure','speed','--lengths',a.lengths,
            '--batch-size','1','--decode-tokens','2','--repeats','3','--output',str(output)]
        row = dict(variant=variant, command=command, output=str(output), prefill_multiplier=a.multiplier if variant=='threshold' else None)
        result['jobs'].append(row)
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        proc = subprocess.run(command,env=env)
        row['exit_code'] = proc.returncode
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        if proc.returncode:
            raise SystemExit(proc.returncode)
    result['stage']='complete'
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
