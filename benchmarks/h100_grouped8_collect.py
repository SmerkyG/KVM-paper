"""Collect logged capped-threshold H100 runs and refresh the report."""
import argparse
import io
import json
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REMOTE = r'''
import io,json,pathlib,sys,tarfile
results=pathlib.Path('/home/ubuntu/comparison/results')
state=pathlib.Path.home()/'.local/state/lod-cluster-run'
names=('grouped8-runs.json','grouped8-followup-runs.json','grouped8-crosscheck-runs.json','grouped8-fused-runs.json')
status=[]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for name in names:
  manifest=results/name
  if not manifest.exists():continue
  tar.add(manifest,arcname=name)
  for job in json.loads(manifest.read_text())['jobs']:
   directory=state/job['id']; meta=json.loads((directory/'run.json').read_text())
   status.append({'name':job['name'],'id':job['id'],'state':meta['state'],'exit_code':meta.get('exit_code')})
   for part in ('run.json','output.log'):
    p=directory/part
    if p.exists():tar.add(p,arcname='provenance/'+job['id']+'/'+part)
   output=results/(job['name']+'.json')
   if meta['state']=='finished' and meta.get('exit_code')==0 and output.exists():tar.add(output,arcname=output.name)
 data=json.dumps(status).encode();info=tarfile.TarInfo('_remote_status.json');info.size=len(data)
 tar.addfile(info,io.BytesIO(data))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--poll-seconds', type=int, default=45)
    args = p.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            raw = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=no',
                                           '-o', 'ConnectTimeout=8', args.host, 'python3', '-c', shlex.quote(REMOTE)],
                                          timeout=60)
            with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
                for member in archive:
                    path = Path(member.name)
                    if not member.isfile() or path.is_absolute() or '..' in path.parts:
                        raise ValueError(f'unsafe archive member {member.name}')
                    dest = args.directory / path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    temp = dest.with_name(dest.name + '.tmp')
                    temp.write_bytes(archive.extractfile(member).read())
                    temp.replace(dest)
            rows = json.loads((args.directory / '_remote_status.json').read_text())
            failures = [r for r in rows if r['state'] == 'finished' and r['exit_code'] != 0]
            pending = [r for r in rows if r['state'] != 'finished']
            status = {'stage': 'running' if pending else ('failed' if failures else 'complete'),
                      'pending_count': len(pending), 'failures': failures, 'jobs': rows}
            (args.directory / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
            subprocess.run([sys.executable, '-m', 'benchmarks.h100_grouped8_report',
                            '--directory', str(args.directory), '--tokenizer', str(args.tokenizer)], check=True)
            print(json.dumps({'stage': status['stage'], 'pending_count': len(pending), 'failures': failures}), flush=True)
            if not pending:
                return
        except Exception as exc:
            print(json.dumps({'stage': 'collector_error', 'error': repr(exc)}), flush=True)
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
