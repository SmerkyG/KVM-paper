"""Collect repair jobs with raw outputs, logs and their immutable source snapshots."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

REMOTE=r'''
import gzip,hashlib,io,json,pathlib,sys,tarfile
r=pathlib.Path('/home/ubuntu/comparison/results');s=pathlib.Path.home()/'.local/state/lod-cluster-run';rows=[];objects=set();known=set(json.load(sys.stdin))
with gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1) as compressed, tarfile.open(fileobj=compressed,mode='w|') as t:
 def add(path,name):
  if name not in known:t.add(path,arcname=name)
 for d in sorted(s.glob('*-three-tier-repair-*')):
  m=json.loads((d/'run.json').read_text());name=d.name.split('-',1)[1]
  rows.append(dict(id=d.name,name=name,state=m['state'],exit_code=m.get('exit_code')))
  for part in ('run.json','output.log'):add(d/part,'provenance/'+d.name+'/'+part)
  for source_name,digest in m['source_files'].items():
   if digest in objects:continue
   objects.add(digest)
   if 'source-objects/'+digest in known:continue
   f=d/'source'/source_name
   if hashlib.sha256(f.read_bytes()).hexdigest()!=digest:raise ValueError('snapshot source changed: '+str(f))
   add(f,'source-objects/'+digest)
  f=r/(name+'.json')
  if f.exists() and m['state']=='finished':
   add(f,f.name)
   for child in r.glob(name+'.json.*.json'):add(child,child.name)
  if m['state']=='finished' and m.get('exit_code')==0:
   for part in ('benchmarks/amd_threshold_three_tier.py','benchmarks/h100_threshold_three_tier.py','lod_attention/kernels/paged_prefill.py','benchmarks/h100_three_tier_diagnose.py','benchmarks/amd_threshold_three_tier_check.py','benchmarks/retro_ruler_run.py','benchmarks/h100_threshold_pages.py','benchmarks/h100_page_threshold_probe.py','benchmarks/h100_page_threshold_quality.py','benchmarks/h100_comparison.py'):
    f=d/'source'/part
    if f.exists():add(f,'executed-source/'+d.name+'/'+part)
 for f in sorted(r.glob('three-tier-repair-*-runs.json')):t.add(f,arcname=f.name)
 f=r/'three-tier-repair-first-response-audit.json'
 if f.exists():t.add(f,arcname=f.name)
 for pattern in ('three-tier-repair-*.md','three-tier-repair-analysis-commands.json','three-tier-repair-cpu-compile-check.json'):
  for f in r.glob(pattern):t.add(f,arcname=f.name)
 raw=json.dumps(rows,indent=2).encode();i=tarfile.TarInfo('status.json');i.size=len(raw);t.addfile(i,io.BytesIO(raw))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--host',required=True);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--prefix',default='three-tier-repair',help='Collect this experiment family with identical provenance checks')
    a=p.parse_args()
    if not a.prefix or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in a.prefix):p.error('invalid run prefix')
    known=set()
    def cached(path):
        if path.exists():known.add(path.relative_to(a.directory).as_posix())
    for path in (a.directory/'source-objects').glob('*'):
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==path.name:cached(path)
    for path in (a.directory/'provenance').glob('*/run.json'):
        metadata=json.loads(path.read_text())
        if metadata['state']!='finished':continue
        cached(path);cached(path.with_name('output.log'))
        result=a.directory/(path.parent.name.split('-',1)[1]+'.json')
        if result.exists():
            data=json.loads(result.read_text())
            if isinstance(data,dict) and (data.get('stage')=='complete' or data.get('median_decode_ms_per_token') is not None):cached(result)
        sources=a.directory/'executed-source'/path.parent.name
        for name,digest in metadata['source_files'].items():
            source=sources/name
            if source.exists() and hashlib.sha256(source.read_bytes()).hexdigest()==digest:cached(source)
    remote=REMOTE.replace('three-tier-repair',a.prefix)
    raw=subprocess.check_output(['ssh','-o','StrictHostKeyChecking=no',a.host,'/home/ubuntu/.local/bin/uv','run','--offline','--no-project','--python','/usr/bin/python3','python','-c',shlex.quote(remote)],input=json.dumps(sorted(known)).encode(),timeout=55)
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
        for m in archive:
            if m.isdir():continue
            path=Path(m.name)
            if not m.isfile() or path.is_absolute() or '..' in path.parts:raise ValueError('unsafe archive entry')
            target=a.directory/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.extractfile(m).read())
    rows=json.loads((a.directory/'status.json').read_text());print(json.dumps(dict(pending=[r['name'] for r in rows if r['state']!='finished'],failed=[r['name'] for r in rows if r['state']=='finished' and r['exit_code']!=0])))


if __name__=='__main__':main()
