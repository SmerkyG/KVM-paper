"""Monitor the detached AMD RULER shards, retry failures, and build the report."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

from benchmarks.amd_threshold_submit_full500 import MODELS, VARIANTS


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def submit_again(job,log):
    source=Path(log['source']);frozen=Path(log['frozen_manifest']).parent
    env=[f'PYTHONPATH={source}:{source}/integrations/vllm_lod','HF_HUB_OFFLINE=1','HF_DATASETS_OFFLINE=1']
    env.extend(f'{k}={v}' for k,v in VARIANTS[job['variant']].items())
    name=f'full500-{job["model"]}-{job["task"]}-{job["variant"]}-{job["shard"]:02d}-retry'
    cmd=['cluster-run','--detach','--num-gpus','1','--name',name,'--','env',*env,
         'uv','run','--offline','--no-project','--python',
         '/home/dan/subusers/agent/.venvs/vllm-rocm-0.27.1/bin/python',
         'python','-m','benchmarks.amd_threshold_ruler','--checkpoint',MODELS[job['model']],
         '--samples-root',str(frozen/job['model']/f'shard-{job["shard"]:02d}'),
         '--tasks',job['task'],'--limit',str(len(job['expected_ids'])),'--output',job['output']]
    result=subprocess.run(cmd,cwd=source,text=True,capture_output=True,check=True)
    match=re.search(r'^Submitted (\S+)',result.stdout,re.MULTILINE)
    if not match:raise RuntimeError('cannot parse cluster-run retry')
    return match.group(1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--poll-seconds',type=int,default=30)
    p.add_argument('--max-attempts',type=int,default=3)
    args=p.parse_args();args.results=args.results.resolve();finished=Path('/home/dan/subusers/agent/.cluster-run/agent/finished')
    submissions=sorted(args.results.glob('*-submissions.json'))
    if len(submissions)!=5:raise ValueError('expected all five submission groups before monitoring')
    if args.poll_seconds<1 or args.max_attempts<1:raise ValueError('invalid monitor settings')
    while True:
        total=completed=records=0;blocked=[];retried=[]
        for path in submissions:
            log=json.loads(path.read_text());changed=False
            if log['stage']!='submitted':raise ValueError(f'submission incomplete: {path}')
            for job in log['jobs']:
                total+=1;marker=finished/job['id'];output=Path(job['output']);data=None
                if output.exists():
                    try:data=json.loads(output.read_text());records+=len(data['records'])
                    except (OSError,json.JSONDecodeError,KeyError):data=None
                if not marker.exists():continue
                fields=marker.read_text().split('\t')
                succeeded=len(fields)>=9 and fields[7]=='finished' and fields[8]=='0' and data is not None and data.get('stage')=='complete'
                if succeeded:
                    completed+=1;continue
                attempts=job.setdefault('attempts',[])
                if len(attempts)+1>=args.max_attempts:
                    blocked.append({'id':job['id'],'output':job['output'],'attempts':len(attempts)+1});continue
                old=job['id']
                if output.exists():
                    saved=output.with_name(output.stem+f'.failed-{old}'+output.suffix)
                    if saved.exists():raise ValueError(f'retry preservation path exists: {saved}')
                    os.replace(output,saved)
                new=submit_again(job,log);attempts.append({'id':old,'finished_record':str(marker)})
                job['id']=new;changed=True;retried.append({'old':old,'new':new})
            if changed:save(path,log)
        state={'stage':'running','submitted_shards':total,'completed_shards':completed,
               'sample_outputs_observed':records,'retries':retried,'blocked':blocked,
               'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        if blocked:
            state['stage']='blocked';save(args.results/'progress.json',state);print(json.dumps(state),flush=True);return
        if total==180 and completed==total:
            report=args.results/'REPORT.md'
            subprocess.run(['uv','run','--offline','--no-project','--python',
                            '/home/dan/subusers/agent/.venvs/vllm-rocm-0.27.1/bin/python',
                            'python','-m','benchmarks.amd_threshold_collect_full500',
                            '--frozen',str(args.frozen.resolve()),'--results',str(args.results),'--output',str(report)],check=True)
            state['stage']='complete';state['report']=str(report);save(args.results/'progress.json',state);print(json.dumps(state),flush=True);return
        save(args.results/'progress.json',state);print(json.dumps(state),flush=True)
        time.sleep(args.poll_seconds)


if __name__=='__main__':main()
