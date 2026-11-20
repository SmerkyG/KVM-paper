"""Collect immutable H100 closed-prefill results, including excluded trials."""
import argparse
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

REMOTE = r'''
import io,json,pathlib,sys,tarfile
p=pathlib.Path('/home/ubuntu/comparison/results')
s=pathlib.Path.home()/'.local/state/lod-cluster-run'
rows=[]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for f in sorted(p.glob('prefill-closed-*-runs.json')):t.add(f,arcname=f.name)
 for d in sorted(s.glob('*-prefill-closed-*')):
  meta=json.loads((d/'run.json').read_text())
  name=d.name.partition('-')[2]
  rows.append(dict(name=name,id=d.name,state=meta['state'],exit_code=meta.get('exit_code')))
  for part in ('run.json','output.log'):
   if (d/part).exists():t.add(d/part,arcname='provenance/'+d.name+'/'+part)
  for part in ('benchmarks/h100_comparison.py','benchmarks/h100_threshold_prefill_fast.py','benchmarks/h100_threshold_fast_check.py'):
   if (d/'source'/part).exists():t.add(d/'source'/part,arcname='provenance/'+d.name+'/source/'+part)
  result=p/(name+'.json')
  if result.exists():t.add(result,arcname=result.name)
 if sys.argv[1]:
  source=s/sys.argv[1]/'source'
  for name in ('benchmarks','lod_attention','integrations','tests','pyproject.toml','uv.lock','README.md'):
   t.add(source/name,arcname='source/'+name)
 raw=json.dumps(rows,indent=2).encode();i=tarfile.TarInfo('status.json');i.size=len(raw);t.addfile(i,io.BytesIO(raw))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--source-run', default='')
    a = p.parse_args()
    raw = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=no', a.host,
        '/home/ubuntu/.local/bin/uv', 'run', '--offline', '--no-project', '--python', '/usr/bin/python3',
        'python', '-c', shlex.quote(REMOTE), shlex.quote(a.source_run)], timeout=55)
    a.directory.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        for member in archive:
            path = Path(member.name)
            if member.isdir():
                continue
            if not member.isfile() or path.is_absolute() or '..' in path.parts:
                raise ValueError('invalid archive member')
            target = a.directory / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
    rows = json.loads((a.directory / 'status.json').read_text())
    print(json.dumps(dict(pending=[r['name'] for r in rows if r['state'] != 'finished'],
        excluded=[r['name'] for r in rows if r['state'] == 'finished' and r['exit_code'] != 0])))


if __name__ == '__main__':
    main()
