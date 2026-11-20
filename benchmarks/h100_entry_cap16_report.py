"""Compare looser entry thresholds with and without the original work cap."""
import argparse
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase
from benchmarks.retro_ruler_prompt_compare import score

TASKS=('niah_single_3','niah_multikey_2','niah_multikey_3')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cohort','region','region-audit','entry16','entry24','entry32','cap24','cap32','tokenizer','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--timing',type=Path,nargs=12,required=True)
    a=p.parse_args();raw=a.cohort.read_bytes();cohort=json.loads(raw);digest=hashlib.sha256(raw).hexdigest()
    frozen={(r['task'],r['source_row']):r for r in cohort['samples'] if r['task'] in TASKS}
    if len(frozen)!=96:raise ValueError('wrong cohort')
    specs=[('Region c=16',a.region,'region',None,None),('Entry 16',a.entry16,'entry',16,16),
           ('Entry 24',a.entry24,'entry',24,24),('Entry 32',a.entry32,'entry',32,32),
           ('Entry 24, cap 16',a.cap24,'entry',24,16),('Entry 32, cap 16',a.cap32,'entry',32,16)]
    groups=[]
    for label,path,rule,bound,cap in specs:
        data=json.loads(path.read_text());cfg=data['threshold_routing']
        if data['stage']!='complete' or data['mode']!='two-tier' or data['cohort_sha256']!=digest or cfg.get('rule','region')!=rule:raise ValueError(f'bad result: {path}')
        if bound and (cfg.get('entry_bound',16)!=bound or cfg.get('work_bound',cfg.get('entry_bound',16))!=cap):raise ValueError(f'wrong threshold or cap: {path}')
        rows={(r['task'],r['source_row']):r for r in data['records'] if r['task'] in TASKS}
        if set(rows)!=set(frozen):raise ValueError(f'missing inputs: {path}')
        groups.append(rows)
    audit_data=json.loads(a.region_audit.read_text());audit={(r['task'],r['source_row']):r for r in audit_data['records']}
    if audit_data['stage']!='complete' or audit_data['cohort_sha256']!=digest:raise ValueError('bad region audit')
    for key,src in frozen.items():
        for group in groups:
            r=group[key]
            if r['prompt_sha256']!=src['prompt_sha256'] or r['answers']!=src['answers'] or r['input_token_sha256']!=groups[0][key]['input_token_sha256']:raise ValueError(f'input mismatch: {key}')
        if key in audit and audit[key]['input_token_sha256']!=groups[0][key]['input_token_sha256']:raise ValueError('audit mismatch')
    tok=Tokenizer.from_file(str(a.tokenizer));stop={tok.token_to_id('<|end_of_text|>'),tok.token_to_id('<|eot_id|>')}
    if None in stop:raise ValueError('missing stop token')
    def first(row):
        ids=row['output_token_ids'];end=next((i for i,t in enumerate(ids) if t in stop),len(ids));text=tok.decode(ids[:end],skip_special_tokens=True)
        try:text=PreTrainedTokenizerBase.clean_up_tokenization(text)
        except TypeError:text=PreTrainedTokenizerBase.clean_up_tokenization(None,text)
        return score(row['task'],text,row['answers'])
    labels=[s[0] for s in specs]
    lines=['# Llama 3.1: looser entry cutoffs with the original 16 sqrt(n) cap','',
           'Three selected 32K retrieval tasks, 32 matched examples each, BF16 model and KV, batch one, and identical sparse prefill. '
           'Entry B considers a region when p_r/n_r > 1/(B sqrt(n)). Uncapped variants use the strict B sqrt(n) guard; cap-16 variants use a strict 16 sqrt(n) guard. '
           'When candidates exceed the cap, the guard opens a deterministic prefix in region order and leaves all later candidates as summaries. This can discard a later candidate even if it would fit. '
           'All rules are experimental and these selected tasks are not a full RULER average.', '',
           '| Task | '+' | '.join(labels)+' |','|---|'+'---:|'*len(labels)]
    for task in TASKS:
        keys=sorted(k for k in frozen if k[0]==task);means=[100*sum(first(g[k]) for k in keys)/32 for g in groups]
        lines.append('| '+task+' | '+' | '.join(f'{x:.2f}' for x in means)+' |')
    lines += ['', 'Scores use the first response. A single-answer miss changes a task score by 3.125 points.', '',
              '## Remote work on the final decode step', '',
              'Means cover all layers and query heads at each example’s final decode step. Proxy mass is not exact attention mass. The region work audit did not include single-needle 3. The bound fraction uses each entry variant’s work cap; the region fraction uses 16 sqrt(n) for comparison.', '',
              '| Task | Rule | Mean entries/head | Mean opened proxy mass | Max bound fraction |','|---|---|---:|---:|---:|']
    for task in TASKS:
        keys=sorted(k for k in frozen if k[0]==task)
        for label,group in [(labels[0],audit)]+list(zip(labels[1:],groups[1:])):
            if not all(k in group for k in keys):continue
            layers=[l for key in keys for l in group[key]['threshold_route_audit']['layers']];heads=sum(l['heads'] for l in layers)
            lines.append(f'| {task} | {label} | {sum(l["entries_sum"] for l in layers)/heads:.1f} | {sum(l["proxy_mass_sum"] for l in layers)/heads:.2%} | {max(l["max_bound_fraction"] for l in layers):.3f} |')
    lines += ['', '## Whole-model timing','',
              'H100, batch-one BF16 Llama, graph-captured decode, matched teacher-forced prompts, one warmup and two measured repetitions. Each repetition has 512 untimed and 1,024 timed steps, including cache catch-up. Timings include page padding and all model layers.', '',
              '| Context | Rule | Decode ms/token | Timed layer update events |', '|---|---|---:|---:|']
    for path in a.timing:
        d=json.loads(path.read_text());records=[r for r in d['records'] if not r['warmup']]
        if len(records)!=2 or d['timed_decode_steps']!=1024:raise ValueError(f'incomplete timing: {path}')
        name=path.name
        if 'region' in name:label=labels[0]
        elif 'entry-24-cap16' in name:label=labels[4]
        elif 'entry-32-cap16' in name:label=labels[5]
        elif 'entry-24-' in name:label=labels[2]
        elif 'entry-32-' in name:label=labels[3]
        else:label=labels[1]
        lines.append(f'| {d["prompt_tokens"]:,} | {label} | {d["median_decode_ms_per_batch_step"]:.3f} | '+', '.join(str(r['lod_layer_catch_up_events_timed']) for r in records)+' |')
    lines += ['', 'The hard cap covers exact remote KV entries per query head, not local/sink work, padded page lanes, summary scoring, prefill, or cache maintenance. The full execution has not been proven subquadratic.', '',
              f'Frozen cohort SHA-256: `{digest}`.', 'Outputs: '+' · '.join(f'[{label}]({path.name})' for label,path,_,_,_ in specs)+'.', '']
    a.output.write_text('\n'.join(lines));print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
