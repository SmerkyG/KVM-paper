"""Collect threshold-decode results and immutable cluster-run provenance."""
import argparse
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

REMOTE=r'''
import io,json,pathlib,sys,tarfile
p=pathlib.Path('/home/ubuntu/comparison/results');s=pathlib.Path.home()/'.local/state/lod-cluster-run'
rows=[]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for path in sorted(p.glob('threshold-decode-*-runs.json')):
  archive.add(path,arcname=path.name)
  for job in json.loads(path.read_text())['jobs']:
   directory=s/job['id'];meta=json.loads((directory/'run.json').read_text())
   rows.append(dict(name=job['name'],id=job['id'],state=meta['state'],exit_code=meta.get('exit_code')))
   for part in ('run.json','output.log'):archive.add(directory/part,arcname='provenance/'+job['id']+'/'+part)
   output=p/(job['name']+'.json')
   if meta['state']=='finished' and meta.get('exit_code')==0 and output.exists():archive.add(output,arcname=output.name)
 data=json.dumps(rows).encode();info=tarfile.TarInfo('status.json');info.size=len(data);archive.addfile(info,io.BytesIO(data))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--host',required=True)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--once',action='store_true')
    a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
    while True:
        raw=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=no',a.host,'python3','-c',shlex.quote(REMOTE)],timeout=55)
        with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
            for member in archive:
                path=Path(member.name)
                if not member.isfile() or path.is_absolute() or '..' in path.parts:raise ValueError('invalid archive member')
                dest=a.directory/path;dest.parent.mkdir(parents=True,exist_ok=True)
                temp=dest.with_name(dest.name+'.tmp');temp.write_bytes(archive.extractfile(member).read());temp.replace(dest)
        rows=json.loads((a.directory/'status.json').read_text())
        pending=[r for r in rows if r['state']!='finished']
        print(json.dumps(dict(pending=len(pending),failed=[r['name'] for r in rows if r['state']=='finished' and r['exit_code']!=0])),flush=True)
        if not pending or a.once:return
        time.sleep(45)


if __name__=='__main__':main()
