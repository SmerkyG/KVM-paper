"""Monitor and report matched H100 RULER prefill/decode configurations.

GPU jobs are submitted separately with cluster-run. This collector performs
no inference and marks incomplete jobs and tasks clearly.
"""
import argparse
import hashlib
import inspect
import json
import shlex
import subprocess
import time
from pathlib import Path

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase
from benchmarks.retro_ruler_prompt_compare import score

TASKS=('niah_single_1','niah_single_2','niah_single_3','niah_multikey_1',
       'niah_multikey_2','niah_multikey_3','niah_multivalue','niah_multiquery',
       'vt','fwe','cwe','qa_1','qa_2')
PREFILLS=('dense','top8','region','entry')
REMOTE=r'''
import hashlib,json,pathlib
done=set(DONE_IDS)
base=pathlib.Path('/home/ubuntu/.local/state/lod-cluster-run')
path=pathlib.Path('/home/ubuntu/comparison/results/ruler32k-prefill-matrix-runs.json')
manifest=json.loads(path.read_text());rows=[]
for job in manifest['jobs']:
 d=base/job['id'];m=json.loads((d/'run.json').read_text());files=m.pop('source_files')
 digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
 item={'job':job,'metadata':m,'source_digest':digest}
 if not rows:item['source_files']=files
 if job['id'] not in done:
  log=d/'output.log'
  if log.exists():item['log']=log.read_text(errors='replace')
  out=pathlib.Path(job['output'])
  if out.exists():
   try:item['output']=json.loads(out.read_text())
   except (ValueError,OSError):pass
 rows.append(item)
print(json.dumps({'manifest_stage':manifest['stage'],'rows':rows}))
'''


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--host',required=True);p.add_argument('--directory',type=Path,required=True)
 p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--poll-seconds',type=int,default=45)
 args=p.parse_args();args.directory.mkdir(parents=True,exist_ok=True)
 tok=Tokenizer.from_file(str(args.tokenizer));stop={tok.token_to_id('<|end_of_text|>'),tok.token_to_id('<|eot_id|>')}
 if None in stop:raise ValueError('missing stop token')
 cleanup=PreTrainedTokenizerBase.clean_up_tokenization
 def first(row):
  ids=row['output_token_ids'];end=next((i for i,t in enumerate(ids) if t in stop),len(ids))
  text=tok.decode(ids[:end],skip_special_tokens=True)
  text=cleanup(None,text) if len(inspect.signature(cleanup).parameters)==2 else cleanup(text)
  return score(row['task'],text,row['answers'])
 done=set()
 while True:
  try:
   remote=REMOTE.replace('DONE_IDS',repr(sorted(done)))
   raw=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=no',
                                '-o','ConnectTimeout=8',args.host,'python3','-c',shlex.quote(remote)],
                               text=True,timeout=60)
   data=json.loads(raw);rows=data['rows']
   if data['manifest_stage']!='submitted' or len(rows)!=40:raise ValueError('matrix submission incomplete')
   digests={item['source_digest'] for item in rows}
   if len(digests)!=1:raise ValueError('matrix source snapshots do not match')
   (args.directory/'source-files.json').write_text(json.dumps(rows[0]['source_files'],indent=2)+'\n')
   results={};pending=[];failures=[];meta_ref=None
   for item in rows:
    job=item['job'];m=item['metadata'];rid=job['id'];state=m['state'];code=m.get('exit_code')
    dest=args.directory/'provenance'/rid;dest.mkdir(parents=True,exist_ok=True)
    (dest/'run.json').write_text(json.dumps(m|{'source_digest':item['source_digest']},indent=2)+'\n')
    if 'log' in item:(dest/'output.log').write_text(item['log'])
    if state!='finished':pending.append(rid)
    elif code!=0:failures.append({'id':rid,'exit_code':code})
    out=args.directory/Path(job['output']).name
    if 'output' in item:out.write_text(json.dumps(item['output'],indent=2)+'\n')
    if not out.exists() or (state=='finished' and code!=0):continue
    result=json.loads(out.read_text())
    if state=='finished' and result.get('stage')!='complete':raise ValueError(f'incomplete {rid}')
    if result.get('mode')!='two-tier' or result.get('model_precision')!='bf16' or result.get('kv_precision')!='bf16':
     raise ValueError(f'wrong model mode {rid}')
    route=result.get('threshold_routing') or {}
    if route.get('mass_multiplier')!=16 or route.get('rule','region')!=job['decode']:
     raise ValueError(f'wrong decode threshold {rid}')
    pref=result.get('prefill_threshold')
    if job['prefill']=='dense':
     if result.get('prefill_attention')!='dense with LoD cache construction' or pref is not None:
      raise ValueError(f'wrong dense prefill {rid}')
    elif job['prefill']=='top8':
     if result.get('prefill_attention')!='LoD sparse' or pref is not None:raise ValueError(f'wrong top8 {rid}')
    elif not pref or pref.get('rule')!=job['prefill']:
     raise ValueError(f'wrong prefill threshold {rid}')
    if pref and pref.get('prefill_block_size')!=16384:raise ValueError(f'wrong prefill size {rid}')
    for key in ('model_name','model_revision','cohort_sha256','generation','prompt_format'):
     if meta_ref is None:continue
     if result.get(key)!=meta_ref.get(key):raise ValueError(f'mismatch {key} {rid}')
    if meta_ref is None:meta_ref=result
    grouped={task:[] for task in job['tasks']}
    for record in result['records']:
     if record['task'] not in grouped:raise ValueError(f'wrong task {rid}')
     grouped[record['task']].append(record)
    for task,records in grouped.items():
     if len(records)>32 or len({r['source_row'] for r in records})!=len(records):raise ValueError(f'bad records {rid}')
     if len(records)==32:
      key=(job['decode'],job['prefill'],task)
      if key in results:raise ValueError(f'duplicate task {key}')
      results[key]={'rows':{r['source_row']:r for r in records},'provisional':state!='finished'}
    if state=='finished' and code==0:done.add(rid)
   lines=['# H100 Llama 32K: matched prefill comparison','',
          'BF16 model and KV, batch one, 32 frozen examples per task. All four '
          'prefill modes use the same source snapshot and the same two-tier decode '
          'rule within each table. Dense prefill retains LoD cache construction; '
          'its first token is checked against the original dense model. The 16K '
          'exact prefill size is unchanged. The threshold methods are experimental.','',
          'First-response scores stop at the first end-of-turn or end-of-text token. '
          'An asterisk marks a task completed inside a job still running, pending '
          'the job’s successful completion. A dash means not yet complete.','']
   summary={}
   for decode in ('region','entry'):
    lines += [f'## {decode.capitalize()}-threshold decode','',
              '| Task | Full prefill | Top-8 prefill | Region c=16 prefill | Entry-16 prefill |',
              '|---|---:|---:|---:|---:|']
    for task in TASKS:
     anchor=results.get((decode,'dense',task));cells=[]
     for pref in PREFILLS:
      value=results.get((decode,pref,task))
      if value is None or anchor is None:cells.append('—');continue
      if set(value['rows'])!=set(anchor['rows']):raise ValueError(f'cohort mismatch {decode}/{task}/{pref}')
      for k,row in value['rows'].items():
       if any(row[x]!=anchor['rows'][k][x] for x in ('prompt_sha256','answers','input_token_sha256')):
        raise ValueError(f'input mismatch {decode}/{task}/{pref}/{k}')
      value['mean']=100*sum(first(r) for r in value['rows'].values())/32
      cells.append(f"{value['mean']:.2f}%"+('*' if value['provisional'] else ''))
      summary[f'{decode}/{pref}/{task}']={'score':value['mean'],'provisional':value['provisional']}
     lines.append('| '+task+' | '+' | '.join(cells)+' |')
    lines.append('')
    for pref in PREFILLS:
     values=[results.get((decode,pref,t)) for t in TASKS]
     if all(v is not None and not v['provisional'] and 'mean' in v for v in values):
      lines.append(f"{pref} unweighted 13-task average: {sum(v['mean'] for v in values)/13:.2f}%.")
    lines.append('')
   lines += ['All comparisons use matched inputs and first-response scoring. '
             'Complete-source digests and cluster-run commands are in the provenance '
             'directory. The dense-prefill diagnostic also constructs the LoD cache; '
             'it is not a prefill speed test.','']
   (args.directory/'REPORT.md').write_text('\n'.join(lines))
   state={'stage':'running' if pending else 'failed' if failures else 'complete',
          'pending_count':len(pending),'failures':failures,
          'completed_jobs':len(done),'completed_task_cells':sum(not v['provisional'] for v in summary.values())}
   (args.directory/'status.json').write_text(json.dumps(state,indent=2)+'\n')
   (args.directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
   print(json.dumps(state),flush=True)
   if not pending:return
  except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:print(f'Collection error: {exc}',flush=True)
  time.sleep(args.poll_seconds)


if __name__=='__main__':main()
