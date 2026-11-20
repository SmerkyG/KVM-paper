"""Collect direct-summary experiments and content-verified immutable sources."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

REMOTE=r'''
import hashlib,io,json,pathlib,sys,tarfile
r=pathlib.Path('/home/ubuntu/comparison/results');s=pathlib.Path.home()/'.local/state/lod-cluster-run'
rows={};objects=set(KNOWN_OBJECTS)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for f in sorted(r.glob(MANIFEST_GLOB)):
  t.add(f,arcname=f.name)
  for j in json.loads(f.read_text())['jobs']:
   d=s/j['id'];m=json.loads((d/'run.json').read_text())
   rows[j['id']]={k:j[k]for k in ('id','name','variant','length')}
   rows[j['id']].update(state=m['state'],exit_code=m.get('exit_code'))
   for part in ('run.json','output.log'):
    if (d/part).exists():t.add(d/part,arcname='provenance/'+j['id']+'/'+part)
   out=pathlib.Path(j['output'])
   if out.exists():t.add(out,arcname=out.name)
   for name,digest in m['source_files'].items():
    if digest in objects:continue
    p=d/'source'/name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,p
    t.add(p,arcname='source-objects/'+digest);objects.add(digest)
 raw=json.dumps(list(rows.values()),indent=2).encode();i=tarfile.TarInfo('status.json');i.size=len(raw);t.addfile(i,io.BytesIO(raw))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--host',required=True)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--manifest-glob',default='threshold-directopt-*-runs.json')
    p.add_argument('--timeout',type=int,default=180);a=p.parse_args()
    # Keep follow-up tuning panels separate from the much larger historical
    # quality archive, while retaining every selected immutable source object.
    known=[]
    for obj in (a.directory/'source-objects').glob('*'):
        assert hashlib.sha256(obj.read_bytes()).hexdigest()==obj.name,obj
        known.append(obj.name)
    program='MANIFEST_GLOB='+repr(a.manifest_glob)+'\nKNOWN_OBJECTS='+repr(known)+'\n'+REMOTE
    raw=subprocess.check_output(['ssh','-o','StrictHostKeyChecking=no',a.host,
        '/home/ubuntu/.local/bin/uv','run','--offline','--no-project','--python','/usr/bin/python3',
        'python','-c',shlex.quote(program)],timeout=a.timeout)
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
        for member in archive:
            if member.isdir():continue
            path=Path(member.name)
            if not member.isfile() or path.is_absolute() or '..'in path.parts:raise ValueError(member.name)
            content=archive.extractfile(member).read()
            if path.parts[0]=='source-objects':assert hashlib.sha256(content).hexdigest()==path.name
            dest=a.directory/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
    rows=json.loads((a.directory/'status.json').read_text())
    print(json.dumps(dict(pending=[r['name']for r in rows if r['state']!='finished'],
        failed=[r['name']for r in rows if r['state']=='finished' and r['exit_code']!=0])))


if __name__=='__main__':main()
