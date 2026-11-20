"""Collect completed fast-threshold cluster runs and their immutable provenance."""
import argparse,io,json,shlex,subprocess,tarfile,time
from pathlib import Path
REMOTE=r'''
import json,pathlib,sys,tarfile,io
p=pathlib.Path('/home/ubuntu/comparison/results');s=pathlib.Path.home()/'.local/state/lod-cluster-run'
names=sorted(x.name for x in p.glob('threshold-fast-*-runs.json'))
rows=[]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for name in names:
  f=p/name
  if not f.exists():continue
  t.add(f,arcname=name)
  for j in json.loads(f.read_text())['jobs']:
   d=json.loads((s/j['id']/'run.json').read_text());rows.append({'name':j['name'],'id':j['id'],'state':d['state'],'exit_code':d.get('exit_code')})
   for part in ('run.json','output.log'):t.add(s/j['id']/part,arcname='provenance/'+j['id']+'/'+part)
   out=p/(j['name']+'.json')
   if d['state']=='finished' and d.get('exit_code')==0 and out.exists():t.add(out,arcname=out.name)
 raw=json.dumps(rows).encode();i=tarfile.TarInfo('status.json');i.size=len(raw);t.addfile(i,io.BytesIO(raw))
'''
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--host',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--once',action='store_true');a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
 while True:
  try:
   data=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=no',a.host,'python3','-c',shlex.quote(REMOTE)],timeout=55)
   with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as t:
    for m in t:
     path=Path(m.name)
     if not m.isfile() or path.is_absolute() or '..' in path.parts:raise ValueError('invalid archive member')
     dest=a.directory/path;dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name(dest.name+'.tmp');tmp.write_bytes(t.extractfile(m).read());tmp.replace(dest)
   rows=json.loads((a.directory/'status.json').read_text());pending=[r for r in rows if r['state']!='finished'];failures=[r for r in rows if r['state']=='finished' and r['exit_code']!=0]
   print(json.dumps({'pending':len(pending),'failed':[r['name'] for r in failures]}),flush=True)
   if a.once or not pending:return
  except Exception as e:print(json.dumps({'error':repr(e)}),flush=True)
  time.sleep(45)
if __name__=='__main__':main()
