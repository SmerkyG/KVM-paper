"""Collect threshold follow-ups and finish gated Qwen author quality comparisons.

CPU bookkeeping only; author GPU work is queued exclusively through the
self-documenting cluster-run submitter after all independent checks pass.
Failed or incomplete experiments never produce quality averages.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

REMOTE=r'''
import glob,json,pathlib
r=pathlib.Path('/home/ubuntu/comparison/results');s=pathlib.Path.home()/'.local/state/lod-cluster-run'
rows=[]
for p in s.glob('*-threshold-next-*/run.json'):
 d=json.loads(p.read_text());rows.append(dict(id=p.parent.name,name=p.parent.name.split('-',1)[1],state=d['state'],exit_code=d.get('exit_code')))
print(json.dumps(dict(jobs=rows,full_submitted=(r/'threshold-next-author-full-v1-author-quality-runs.json').exists())))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--host',required=True);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--watch',action='store_true');p.add_argument('--max-hours',type=float,default=6);a=p.parse_args()
    root=a.directory.resolve();root.mkdir(parents=True,exist_ok=True);repo=Path(__file__).resolve().parents[1]
    statefile=root/'finalization-status.json';started=time.monotonic();errors=0;last_collect=0
    def save(**state):statefile.write_text(json.dumps(state,indent=2)+'\n')
    def remote(code):
        return subprocess.check_output(['ssh','-o','StrictHostKeyChecking=no',a.host,
            '/home/ubuntu/.local/bin/uv','run','--offline','--no-project','--python','/usr/bin/python3',
            'python','-c',shlex.quote(code)],timeout=45)
    try:
        while time.monotonic()-started<a.max_hours*3600:
            try:
                current=json.loads(remote(REMOTE));errors=0
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as e:
                errors+=1;save(stage='waiting',connection_error=str(e),retry=errors)
                if not a.watch or errors>=4:raise
                time.sleep(30);continue
            jobs=current['jobs'];by_name={j['name']:j for j in jobs}
            gates=[by_name.get(f'threshold-next-author-v2-qwen-{mode}-{budget}')
                   for mode in ('multipole','squeezed-multipole') for budget in (512,2048)]
            gates.append(by_name.get('threshold-next-author-v2-author-frontier-check'))
            if not current['full_submitted'] and all(j is not None and j['state']=='finished' and j['exit_code']==0 for j in gates):
                # Labels are frozen and unique; the remote manifest is also the
                # restart guard, preventing duplicate full comparisons.
                command=['cd','/home/ubuntu/lod-h100/code','&&','/home/ubuntu/.local/bin/uv','run',
                    '--offline','--no-project','--python','/usr/bin/python3','python','-m',
                    'benchmarks.h100_threshold_next_submit','--stage','author-quality',
                    '--label','author-full-v1','--checks-label','author-v2','--require-run',gates[-1]['id']]
                response=subprocess.check_output(['ssh','-o','StrictHostKeyChecking=no',a.host,
                    ' '.join(x if x=='&&' else shlex.quote(x) for x in command)],timeout=45,text=True)
                print(response,flush=True);continue
            pending=[j for j in jobs if j['state']!='finished']
            if not pending or time.monotonic()-last_collect>90:
                subprocess.run([sys.executable,'-m','benchmarks.h100_three_tier_repair_collect','--host',a.host,
                    '--prefix','threshold-next','--directory',str(root)],check=True,timeout=65)
                subprocess.run([sys.executable,'-m','benchmarks.h100_threshold_next_report',
                    '--directory',str(root)],check=True,timeout=45)
                last_collect=time.monotonic()
                ledger=dict(report_command=[sys.executable,'-m','benchmarks.h100_threshold_next_report','--directory',str(root)],
                    watcher_command=sys.argv,score_policy='Validated tokenizer decode; first EOS/EOT; substring metrics; matched input token hashes',
                    source_sha256={str(path.relative_to(repo)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                        (repo/'benchmarks/h100_threshold_next_report.py',Path(__file__).resolve())},
                    environment=dict(python=sys.version))
                (root/'analysis-commands.json').write_text(json.dumps(ledger,indent=2)+'\n')
            save(stage='running' if pending else 'complete',pending=[j['name'] for j in pending],
                excluded_failures=[j['name'] for j in jobs if j.get('exit_code') not in (None,0)],
                author_full_submitted=current['full_submitted'])
            if not pending or not a.watch:return
            time.sleep(30)
        save(stage='timeout',message='Stopped bounded collection; queued cluster jobs retain their own state')
    except Exception as e:
        save(stage='failed',error=str(e));raise


if __name__=='__main__':main()
