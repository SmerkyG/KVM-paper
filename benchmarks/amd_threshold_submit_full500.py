"""Submit frozen, disjoint AMD RULER shards through cluster-run."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


MODELS={'qwen':'Qwen/Qwen3.8-27B-FP8','k2':'IFM/K2-Horizon-32B-FP8'}
VARIANTS={'top8':{},'c16':{'LOD_AMD_THRESHOLD_MULTIPLIER':'16'},
          'c32':{'LOD_AMD_THRESHOLD_MULTIPLIER':'32'},
          'c32-union':{'LOD_AMD_THRESHOLD_MULTIPLIER':'32','LOD_AMD_THRESHOLD_UNION':'1'}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=tuple(MODELS),required=True)
    p.add_argument('--tasks',nargs='+',required=True)
    p.add_argument('--variants',nargs='+',choices=tuple(VARIANTS),required=True)
    p.add_argument('--submission-log',type=Path,required=True)
    args=p.parse_args();frozen=args.frozen.resolve();source=args.source.resolve();output=args.output.resolve()
    manifest=json.loads((frozen/'manifest.json').read_text())
    if manifest['stage']!='complete' or manifest['samples_per_task']!=500:
        raise ValueError('frozen cohort is incomplete')
    if set(args.tasks)-set(manifest['models'][args.model]):raise ValueError('task missing from frozen cohort')
    if args.submission_log.exists():raise ValueError('refusing to overwrite a submission log')
    log={'model':args.model,'tasks':args.tasks,'variants':args.variants,'frozen_manifest':str(frozen/'manifest.json'),
         'source':str(source),'stage':'submitting','jobs':[]}
    args.submission_log.parent.mkdir(parents=True,exist_ok=True)
    def save():
        temp=args.submission_log.with_suffix(args.submission_log.suffix+'.tmp')
        temp.write_text(json.dumps(log,indent=2)+'\n');temp.replace(args.submission_log)
    save()
    for shard in range(500//manifest['shard_size']):
        for task in args.tasks:
            info=manifest['models'][args.model][task]['shards'][shard]
            for variant in args.variants:
                dest=output/args.model/task/variant/f'shard-{shard:02d}.json'
                if dest.exists():raise ValueError(f'refusing to overwrite existing output: {dest}')
                name=f'full500-{args.model}-{task}-{variant}-{shard:02d}'
                env=[f'PYTHONPATH={source}:{source}/integrations/vllm_lod','HF_HUB_OFFLINE=1','HF_DATASETS_OFFLINE=1']
                env.extend(f'{key}={value}' for key,value in VARIANTS[variant].items())
                cmd=['cluster-run','--detach','--num-gpus','1','--name',name,'--','env',*env,
                     'uv','run','--offline','--no-project','--python',
                     '/home/dan/subusers/agent/.venvs/vllm-rocm-0.27.1/bin/python',
                     'python','-m','benchmarks.amd_threshold_ruler','--checkpoint',MODELS[args.model],
                     '--samples-root',str(frozen/args.model/f'shard-{shard:02d}'),
                     '--tasks',task,'--limit',str(info['count']),'--output',str(dest)]
                run=subprocess.run(cmd,cwd=source,text=True,capture_output=True,check=True)
                match=re.search(r'^Submitted (\S+)',run.stdout,re.MULTILINE)
                if not match:raise RuntimeError(f'cannot parse cluster-run submission: {run.stdout}')
                log['jobs'].append({'id':match.group(1),'model':args.model,'task':task,'variant':variant,
                                    'shard':shard,'output':str(dest),'expected_ids':info['doc_ids']})
                save();print(json.dumps(log['jobs'][-1]),flush=True)
    log['stage']='submitted';save()


if __name__=='__main__':main()
