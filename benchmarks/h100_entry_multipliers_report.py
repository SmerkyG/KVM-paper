"""Compare entry-bound multipliers on matched retrieval examples."""
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
    for name in ('cohort','region','region-audit','entry16','entry24','entry32','tokenizer','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--timing',type=Path,nargs=8,required=True)
    a=p.parse_args();raw=a.cohort.read_bytes();cohort=json.loads(raw);digest=hashlib.sha256(raw).hexdigest()
    frozen={(x['task'],x['source_row']):x for x in cohort['samples'] if x['task'] in TASKS}
    if len(frozen)!=96:raise ValueError('expected 32 examples for each selected task')
    groups=[]
    for path,rule,bound in ((a.region,'region',None),(a.entry16,'entry',16),(a.entry24,'entry',24),(a.entry32,'entry',32)):
        data=json.loads(path.read_text());cfg=data['threshold_routing']
        if data['stage']!='complete' or data['mode']!='two-tier' or data['cohort_sha256']!=digest or cfg.get('rule','region')!=rule:raise ValueError(f'bad run: {path}')
        if bound and cfg.get('entry_bound',16)!=bound:raise ValueError(f'bad entry bound: {path}')
        rows={(r['task'],r['source_row']):r for r in data['records'] if r['task'] in TASKS}
        if set(rows)!=set(frozen):raise ValueError(f'missing examples: {path}')
        groups.append(rows)
    region,e16,e24,e32=groups
    audit_data=json.loads(a.region_audit.read_text())
    audit={(r['task'],r['source_row']):r for r in audit_data['records']}
    if audit_data['stage']!='complete' or audit_data['cohort_sha256']!=digest:raise ValueError('bad region audit')
    for key,src in frozen.items():
        for group in groups:
            row=group[key]
            if row['prompt_sha256']!=src['prompt_sha256'] or row['answers']!=src['answers'] or row['input_token_sha256']!=region[key]['input_token_sha256']:raise ValueError(f'mismatched input: {key}')
        if key in audit and (audit[key]['prompt_sha256']!=src['prompt_sha256'] or audit[key]['answers']!=src['answers'] or audit[key]['input_token_sha256']!=region[key]['input_token_sha256']):
            raise ValueError(f'mismatched region audit input: {key}')
    tok=Tokenizer.from_file(str(a.tokenizer));stop={tok.token_to_id('<|end_of_text|>'),tok.token_to_id('<|eot_id|>')}
    if None in stop:raise ValueError('missing stop token')
    def first(row):
        ids=row['output_token_ids'];end=next((i for i,t in enumerate(ids) if t in stop),len(ids))
        text=tok.decode(ids[:end],skip_special_tokens=True)
        try:text=PreTrainedTokenizerBase.clean_up_tokenization(text)
        except TypeError:text=PreTrainedTokenizerBase.clean_up_tokenization(None,text)
        return score(row['task'],text,row['answers'])
    lines=['# Llama 3.1 32K retrieval: entry-bound multiples 16, 24, and 32','',
           'Three selected tasks, 32 matched examples per task, BF16 model and KV, batch one, same sparse prefill. '
           'An entry multiple B opens a region when proxy mass per entry exceeds 1/(B sqrt(n)), with a strict B sqrt(n) remote-entry guard. '
           'Higher B opens more candidates and allows more exact work. The per-region control opens regions above 16 times mean region proxy mass, capped at 1,024 entries per region. '
           'These tasks were selected because entry multiple 16 had misses; this is not a full RULER panel.', '',
           '| Task | Per-region | Entry 16 | Entry 24 | Entry 32 |','|---|---:|---:|---:|---:|']
    for task in TASKS:
        keys=sorted(k for k in frozen if k[0]==task)
        means=[100*sum(first(g[k]) for k in keys)/32 for g in groups]
        lines.append('| '+task+' | '+' | '.join(f'{x:.2f}' for x in means)+' |')
    lines += ['', 'First-response scoring stops at the first end-of-turn or end-of-text token. One example changes a single-answer score by 3.125 points.', '',
              '## Last-step remote work and proxy mass','',
              'Means cover every layer and query head on the final decode step only. Proxy mass is normalized over remote regions and is not exact attention mass. The region control was not rerun with work instrumentation for single-needle 3.', '',
              '| Task | Rule | Mean entries/head | Mean regions/head | Mean opened proxy mass | Max fraction of that rule’s bound |', '|---|---|---:|---:|---:|---:|']
    for task in TASKS:
        keys=sorted(k for k in frozen if k[0]==task)
        for label,group in [('Per-region',audit),('Entry 16',e16),('Entry 24',e24),('Entry 32',e32)]:
            if not all(k in group for k in keys):continue
            layers=[l for k in keys for l in group[k]['threshold_route_audit']['layers']];heads=sum(l['heads'] for l in layers)
            lines.append(f'| {task} | {label} | {sum(l["entries_sum"] for l in layers)/heads:.1f} | {sum(l["regions_sum"] for l in layers)/heads:.2f} | {sum(l["proxy_mass_sum"] for l in layers)/heads:.2%} | {max(l["max_bound_fraction"] for l in layers):.3f} |')
    lines += ['', 'The per-region fraction is measured against 16 sqrt(n) for reference; the entry fractions use each rule’s own bound.', '',
              '## Whole-model speed','',
              'H100 graph-captured BF16 Llama, batch one, identical teacher-forced prompt tokens. One warmup and two measured repetitions; 512 untimed plus 1,024 timed decode steps. Timed steps include cache catch-up, with layer-event counts shown per repetition.', '',
              '| Context | Rule | Decode ms/token | Layer update events |', '|---|---|---:|---:|']
    for path in a.timing:
        d=json.loads(path.read_text());recs=[x for x in d['records'] if not x['warmup']]
        if len(recs)!=2 or d['timed_decode_steps']!=1024:raise ValueError(f'incomplete timing: {path}')
        name=path.name
        if 'region' in name:label='Per-region'
        elif 'entry-24-' in name:label='Entry 24'
        elif 'entry-32-' in name:label='Entry 32'
        else:label='Entry 16'
        events=', '.join(str(r['lod_layer_catch_up_events_timed']) for r in recs)
        lines.append(f'| {d["prompt_tokens"]:,} | {label} | {d["median_decode_ms_per_batch_step"]:.3f} | {events} |')
    lines += ['', 'Each entry rule’s bound covers remote exact KV entries per query head; local/sink attention, page padding, prefill, and cache maintenance are separate. The bound is not a guarantee of retrieval quality.', '',
              f'Frozen cohort SHA-256: `{digest}`.',f'[Entry 16]({a.entry16.name}) · [Entry 24]({a.entry24.name}) · [Entry 32]({a.entry32.name}) · [Per-region control]({a.region.name}).','']
    a.output.write_text('\n'.join(lines));print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
