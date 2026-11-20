"""Compare matched 64K retrieval examples for both threshold rules and dense."""
import argparse
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase
from benchmarks.retro_ruler_compare import paired_interval
from benchmarks.retro_ruler_prompt_compare import score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cohort','dense','entry','region','tokenizer','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--entry24',type=Path)
    p.add_argument('--entry32',type=Path)
    p.add_argument('--cap24',type=Path)
    p.add_argument('--cap32',type=Path)
    a=p.parse_args();raw=a.cohort.read_bytes();cohort=json.loads(raw);sha=hashlib.sha256(raw).hexdigest()
    frozen={(r['task'],r['source_row']):r for r in cohort['samples']}
    tasks=tuple(cohort['source_files']);n=cohort['samples_per_task']
    if cohort['context_label']!='64k' or n!=16 or len(tasks)!=4 or len(frozen)!=64:raise ValueError('wrong 64K retrieval cohort')
    groups=[]
    specs=[(a.dense,'dense',None,None,None),(a.region,'two-tier','region',None,None),(a.entry,'two-tier','entry',16,16)]
    if a.entry24:specs.append((a.entry24,'two-tier','entry',24,24))
    if a.entry32:specs.append((a.entry32,'two-tier','entry',32,32))
    if a.cap24:specs.append((a.cap24,'two-tier','entry',24,16))
    if a.cap32:specs.append((a.cap32,'two-tier','entry',32,16))
    for path,mode,rule,bound,work_bound in specs:
        data=json.loads(path.read_text())
        if data['stage']!='complete' or data['mode']!=mode or data['cohort_sha256']!=sha:raise ValueError(f'bad metadata: {path}')
        if rule and data['threshold_routing']['rule']!=rule:raise ValueError('wrong threshold rule')
        if bound and data['threshold_routing'].get('entry_bound',16)!=bound:raise ValueError('wrong entry multiplier')
        if work_bound and data['threshold_routing'].get('work_bound',bound)!=work_bound:raise ValueError('wrong work bound')
        rows={(r['task'],r['source_row']):r for r in data['records']}
        if set(rows)!=set(frozen) or len(rows)!=len(data['records']):raise ValueError(f'bad sample set: {path}')
        groups.append(rows)
    dense,region,entry=groups[:3]
    for key,src in frozen.items():
        for group in groups:
            r=group[key]
            if r['prompt_sha256']!=src['prompt_sha256'] or r['answers']!=src['answers'] or r['input_token_sha256']!=dense[key]['input_token_sha256']:raise ValueError(f'input mismatch: {key}')
    tok=Tokenizer.from_file(str(a.tokenizer));stop={tok.token_to_id('<|end_of_text|>'),tok.token_to_id('<|eot_id|>')}
    if None in stop:raise ValueError('missing stop token')
    def first(r):
        ids=r['output_token_ids'];end=next((i for i,t in enumerate(ids) if t in stop),len(ids))
        text=tok.decode(ids[:end],skip_special_tokens=True)
        try:text=PreTrainedTokenizerBase.clean_up_tokenization(text)
        except TypeError:text=PreTrainedTokenizerBase.clean_up_tokenization(None,text)
        return score(r['task'],text,r['answers'])
    labels=['Dense','Per-region','Entry 16']+(['Entry 24'] if a.entry24 else [])+(['Entry 32'] if a.entry32 else [])+(['Entry 24 cap 16'] if a.cap24 else [])+(['Entry 32 cap 16'] if a.cap32 else [])
    lines=['# Llama 3.1 64K RULER retrieval: per-entry versus per-region','',
           'Four selected retrieval tasks, 16 matched examples per task; BF16 model and KV, greedy decoding. LoD uses the same top-eight sparse prefill for both rules; dense uses dense prefill and decode. '
           'These selected tasks and small sample do not represent a full RULER average. Region c=16 retains the existing 1,024-entry cap; entry multiple B uses the B sqrt(n) remote-entry bound and has no fixed per-region cap.', '',
           '| Task | '+' | '.join(labels)+' | Entry 16 minus region, pp (paired 95% interval) |', '|---|'+'---:|'*(len(labels)+1)]
    for i,task in enumerate(tasks):
        keys=sorted(k for k in frozen if k[0]==task);vals=[[first(g[k]) for k in keys] for g in groups]
        avg=[sum(v)*100/n for v in vals];lo,hi=paired_interval([x-y for x,y in zip(vals[2],vals[1])],4920+i)
        lines.append(f'| {task} | '+' | '.join(f'{x:.2f}' for x in avg)+f' | {avg[2]-avg[1]:+.2f} [{lo:+.2f}, {hi:+.2f}] |')
    lines += ['', 'Scores stop at the first end-of-turn or end-of-text token. Intervals are paired bootstrap intervals; each answer can move a single-answer task score by 6.25 points.', '',
              '## Last-step remote work and proxy coverage', '',
              'The measurements sample the final decode step for every example, layer, and query head. Proxy mass is normalized over remote regions and is not exact attention mass. Exact local and sink entries are excluded.', '',
              '| Task | Rule | Mean entries/head | Mean regions/head | Mean proxy mass opened | Max bound fraction | Empty heads |', '|---|---|---:|---:|---:|---:|---:|']
    for task in tasks:
        keys=sorted(k for k in frozen if k[0]==task)
        for name,group in zip(labels[1:],groups[1:]):
            ls=[l for k in keys for l in group[k]['threshold_route_audit']['layers']];heads=sum(l['heads'] for l in ls)
            entries=sum(l['entries_sum'] for l in ls)/heads;regions=sum(l['regions_sum'] for l in ls)/heads;mass=sum(l['proxy_mass_sum'] for l in ls)/heads
            ratio=max(l['max_bound_fraction'] for l in ls);empty=sum(l['empty_heads'] for l in ls)/heads
            lines.append(f'| {task} | {name} | {entries:.1f} | {regions:.2f} | {mass:.2%} | {ratio:.3f} | {empty:.2%} |')
    bound_note = 'The maximum bound fraction uses 16 sqrt(n) for the region control and each entry rule’s work bound.'
    if a.cap24 or a.cap32:
        bound_note += ' The capped variants use a looser candidate cutoff but retain a 16 sqrt(n) work bound by keeping a deterministic prefix in region order, which can reject later candidates.'
    lines += ['', bound_note, '',
              f'Frozen cohort SHA-256: `{sha}`.',
              'Results: '+' · '.join(f'[{label}]({path.name})' for label,(path,_,_,_,_) in zip(labels,specs))+'.','']
    a.output.write_text('\n'.join(lines));print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
