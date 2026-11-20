"""Collect and validate the H100 threshold dense-prefill RULER experiments.

This is a read-only monitor. GPU experiments are separately submitted via
cluster-run; the monitor copies their results and builds a paired report.
"""
import argparse
import inspect
import json
import shlex
import subprocess
import time
from pathlib import Path

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase

from benchmarks.retro_ruler_prompt_compare import score

TASKS = ('niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1',
         'niah_multikey_2', 'niah_multikey_3', 'niah_multivalue', 'niah_multiquery',
         'vt', 'fwe', 'cwe', 'qa_1', 'qa_2')
REMOTE = r'''
import json, pathlib
root=pathlib.Path('/home/ubuntu/.local/state/lod-cluster-run')
p=pathlib.Path('/home/ubuntu/comparison/results/dense-prefill-threshold32-runs.json')
manifest=json.loads(p.read_text()); rows=[]
for j in manifest['jobs']:
    directory=root/j['id']; meta=json.loads((directory/'run.json').read_text())
    old=json.loads((root/j['source_run']/'run.json').read_text())
    result={'job':j, 'metadata':meta, 'same_source':meta['source_files']==old['source_files']}
    log=directory/'output.log'
    if log.exists(): result['log']=log.read_text(errors='replace')
    out=pathlib.Path(j['output'])
    if out.exists():
        try: result['output']=json.loads(out.read_text())
        except (ValueError, OSError): pass
    rows.append(result)
print(json.dumps({'manifest':manifest,'rows':rows}))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--baselines',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--poll-seconds',type=int,default=45)
    args=p.parse_args()
    args.directory.mkdir(parents=True,exist_ok=True)
    tokenizer=Tokenizer.from_file(str(args.tokenizer))
    stop={tokenizer.token_to_id('<|end_of_text|>'),tokenizer.token_to_id('<|eot_id|>')}
    if None in stop:raise ValueError('missing stop token')
    baselines={}
    for name,file in [('region','ruler-32k-llama-two-tier-threshold-c16-fused-full13.json'),
                      ('entry','threshold-entry-guard-ruler-32k-full13.json')]:
        d=json.loads((args.baselines/file).read_text())
        if d['stage']!='complete' or d['prefill_attention']!='LoD sparse':raise ValueError('bad baseline')
        baselines[name]=d
    dense_rows={}
    for file in ('ruler-32k-dense-full.json','ruler-32k-dense-missing8.json'):
        d=json.loads((args.baselines/file).read_text())
        if d['stage']!='complete' or d['mode']!='dense':raise ValueError('bad dense reference')
        for row in d['records']:
            key=(row['task'],row['source_row'])
            if key in dense_rows:raise ValueError('duplicate dense reference')
            dense_rows[key]=row
    if len(dense_rows)!=416:raise ValueError('incomplete dense reference')
    def first(row):
        ids=row['output_token_ids']
        end=next((i for i,t in enumerate(ids) if t in stop),len(ids))
        text=tokenizer.decode(ids[:end],skip_special_tokens=True)
        cleanup=PreTrainedTokenizerBase.clean_up_tokenization
        text=cleanup(None,text) if len(inspect.signature(cleanup).parameters)==2 else cleanup(text)
        return score(row['task'],text,row['answers'])
    while True:
        try:
            raw=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=no',
                                         '-o','ConnectTimeout=8',args.host,'python3','-c',shlex.quote(REMOTE)],
                                        text=True,timeout=60)
            collection=json.loads(raw)
            if collection['manifest']['stage']!='submitted' or len(collection['rows'])!=8:
                raise ValueError('submission manifest incomplete')
            summaries={};pending=[];failures=[]
            for item in collection['rows']:
                job=item['job'];meta=item['metadata'];variant=job['variant'];runid=job['id']
                if job.get('superseded'):
                    continue
                dest=args.directory/'provenance'/runid;dest.mkdir(parents=True,exist_ok=True)
                (dest/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
                (dest/'output.log').write_text(item.get('log',''))
                if not item['same_source']:raise ValueError(f'source differs from baseline: {runid}')
                status=meta['state'];code=meta.get('exit_code')
                if status!='finished':pending.append(runid)
                elif code!=0:failures.append({'id':runid,'exit_code':code})
                result=item.get('output')
                if result is None:continue
                (args.directory/Path(job['output']).name).write_text(json.dumps(result,indent=2)+'\n')
                baseline=baselines[variant]
                for key in ('mode','model_name','model_revision','model_precision','kv_precision',
                            'cohort_sha256','generation','prompt_format','threshold_routing'):
                    if result.get(key)!=baseline.get(key):raise ValueError(f'{runid}: mismatched {key}')
                if result.get('prefill_attention')!='dense with LoD cache construction':
                    raise ValueError(f'{runid}: wrong prefill mode')
                if status=='finished' and (code!=0 or result.get('stage')!='complete'):continue
                ref={(r['task'],r['source_row']):r for r in baseline['records']}
                groups={task:[] for task in job['tasks']}
                for row in result['records']:
                    key=(row['task'],row['source_row'])
                    if row['task'] not in groups or key not in ref or key not in dense_rows:
                        raise ValueError(f'{runid}: wrong example {key}')
                    before=ref[key];dense=dense_rows[key]
                    if any(row[k]!=before[k] for k in ('prompt_sha256','answers','input_token_sha256')):
                        raise ValueError(f'{runid}: unmatched prompt {key}')
                    if row['output_token_ids'][0]!=dense['output_token_ids'][0]:
                        raise ValueError(f'{runid}: dense prefill first-token mismatch {key}')
                    groups[row['task']].append((before,row))
                for task,pairs in groups.items():
                    if len(pairs)>32 or len({x[0]['source_row'] for x in pairs})!=len(pairs):
                        raise ValueError(f'{runid}: invalid cohort {task}')
                    if len(pairs)!=32:continue
                    key=(variant,task)
                    if key in summaries:raise ValueError(f'duplicate result {key}')
                    summaries[key]={'provisional':status!='finished',
                                    'first_before':100*sum(first(a) for a,b in pairs)/32,
                                    'first_after':100*sum(first(b) for a,b in pairs)/32,
                                    'full_before':100*sum(a['score'] for a,b in pairs)/32,
                                    'full_after':100*sum(b['score'] for a,b in pairs)/32,
                                    'improved':sum(first(b)>first(a) for a,b in pairs),
                                    'declined':sum(first(b)<first(a) for a,b in pairs)}
            lines=['# H100 Llama 32K: dense vs. sparse prefill with threshold decode','',
                   'BF16 weights/KV; batch one; 32 matched frozen examples per task. '
                   'Both are two-tier LoD decode. Region uses c=16 times mean region mass; '
                   'entry uses the strict 16√n per-head bound. Dense-prefill runs retain '
                   'LoD cache construction and match the complete source snapshot of '
                   'their respective sparse-prefill baseline. Each first token is '
                   'checked against the dense model. This diagnostic is not a speed test.','',
                   'An asterisk marks a complete 32-example task from a run still in '
                   'progress; it remains provisional until that run succeeds. '
                   'Remaining queued historical-snapshot controls were superseded by '
                   'the complete same-revision prefill matrix.','',
                   '| Task | Region sparse | Region dense | Change | Entry sparse | Entry dense | Change |',
                   '|---|---:|---:|---:|---:|---:|---:|']
            for task in TASKS:
                cells=[]
                for variant in ('region','entry'):
                    v=summaries.get((variant,task))
                    if v is None:cells.extend(('—','—','—'));continue
                    mark='*' if v['provisional'] else ''
                    cells.extend((f"{v['first_before']:.2f}%",f"{v['first_after']:.2f}%{mark}",
                                  f"{v['first_after']-v['first_before']:+.2f}{mark}"))
                lines.append('| '+task+' | '+' | '.join(cells)+' |')
            for variant in ('region','entry'):
                values=[summaries.get((variant,t)) for t in TASKS]
                if all(v is not None and not v['provisional'] for v in values):
                    before=sum(v['first_before'] for v in values)/13
                    after=sum(v['first_after'] for v in values)/13
                    lines += ['',f'{variant.capitalize()} unweighted 13-task first-response mean: '
                              f'{before:.2f}% sparse, {after:.2f}% dense ({after-before:+.2f} points).']
            lines += ['', 'First-response scoring stops at the first end-of-turn or '
                      'end-of-text token. A 32-example task has limited precision. '
                      'Original run commands, source hashes and logs are in the '
                      'provenance directory.','']
            (args.directory/'REPORT.md').write_text('\n'.join(lines))
            state={'stage':'failed' if failures else 'running' if pending else 'complete',
                   'pending':pending,'failures':failures,
                   'completed_tasks':sum(not x['provisional'] for x in summaries.values()),
                   'provisional_tasks':sum(x['provisional'] for x in summaries.values())}
            (args.directory/'status.json').write_text(json.dumps(state,indent=2)+'\n')
            (args.directory/'summary.json').write_text(json.dumps(
                {f'{a}/{b}':v for (a,b),v in summaries.items()},indent=2)+'\n')
            print(json.dumps(state),flush=True)
            if not pending:return
        except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
            print(f'Collection error: {exc}',flush=True)
        time.sleep(args.poll_seconds)


if __name__=='__main__':main()
