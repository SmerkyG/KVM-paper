"""Copy H100 AMD-transfer experiments, logs and immutable source provenance."""
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
manifests=sorted({f for pattern in ('threshold-shared-*-runs.json','threshold-materialized-*-runs.json','threshold-transfer-*-runs.json') for f in p.glob(pattern)})
rows=[]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for f in manifests:
  t.add(f,arcname=f.name)
  for j in json.loads(f.read_text())['jobs']:
   d=s/j['id'];meta=json.loads((d/'run.json').read_text())
   rows.append(dict(name=j['name'],id=j['id'],state=meta['state'],exit_code=meta.get('exit_code')))
   for part in ('run.json','output.log'):t.add(d/part,arcname='provenance/'+j['id']+'/'+part)
   result=p/(j['name']+'.json')
   if result.exists():t.add(result,arcname=result.name)
 if manifests:
  models=[r for r in rows if 'transfer-' in r['name']]
  if models:
   source=s/models[0]['id']/'source'
   for name in ('benchmarks','lod_attention','integrations','tests','pyproject.toml','uv.lock','README.md'):
    t.add(source/name,arcname='source/'+name)
 raw=json.dumps(rows,indent=2).encode();i=tarfile.TarInfo('status.json');i.size=len(raw);t.addfile(i,io.BytesIO(raw))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True)
    p.add_argument('--directory', type=Path, required=True)
    a = p.parse_args()
    raw = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=no', a.host,
                                   '/home/ubuntu/.local/bin/uv', 'run', '--offline', '--no-project', '--python',
                                   '/usr/bin/python3', 'python', '-c', shlex.quote(REMOTE)], timeout=55)
    a.directory.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        for member in archive:
            if member.isdir():
                continue
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts:
                raise ValueError('invalid archive member')
            target = a.directory / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
    rows = json.loads((a.directory / 'status.json').read_text())
    print(json.dumps(dict(pending=[r['name'] for r in rows if r['state'] != 'finished'],
                          failed=[r['name'] for r in rows if r['state'] == 'finished' and r['exit_code'] != 0])))


if __name__ == '__main__':
    main()
