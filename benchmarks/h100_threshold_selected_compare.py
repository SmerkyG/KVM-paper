"""Compare H100 threshold against the prior two-tier fixed-eight task subset."""
import argparse
import json
from pathlib import Path
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase
from benchmarks.retro_ruler_compare import paired_interval
from benchmarks.retro_ruler_prompt_compare import score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixed',type=Path,required=True)
    p.add_argument('--threshold',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    fixed=json.loads(a.fixed.read_text());new=json.loads(a.threshold.read_text())
    if fixed['stage']!='complete' or new['stage']!='complete' or fixed['mode']!='two-tier' or new['mode']!='two-tier':
        raise ValueError('expected complete two-tier runs')
    if new['threshold_routing']['mass_multiplier']!=16 or fixed['model_name']!=new['model_name']:
        raise ValueError('routing or model mismatch')
    prior={(r['task'],r['source_row']):r for r in fixed['records']}
    current={(r['task'],r['source_row']):r for r in new['records']}
    if len(prior)!=len(fixed['records']) or not set(prior)<=set(current):raise ValueError('missing or duplicate paired examples')
    for key,row in prior.items():
        if any(row[x]!=current[key][x] for x in ('prompt_sha256','input_token_sha256','answers')):
            raise ValueError(f'input mismatch: {key}')
    tok=Tokenizer.from_file(str(a.tokenizer));stops={tok.token_to_id('<|end_of_text|>'),tok.token_to_id('<|eot_id|>')}
    if None in stops:raise ValueError('missing Llama stop tokens')
    def first(row):
        ids=row['output_token_ids'];end=next((i for i,x in enumerate(ids) if x in stops),len(ids))
        text=PreTrainedTokenizerBase.clean_up_tokenization(tok.decode(ids[:end],skip_special_tokens=True))
        return score(row['task'],text,row['answers'])
    tasks=('niah_multikey_1','niah_multikey_2','niah_multikey_3','vt','fwe','cwe','qa_1','qa_2')
    lines=['# Llama 3.1 32K: fixed-eight versus threshold on paired tasks','',
           'The prior fixed-eight run covered these eight selected tasks, not the full panel. '
           'Each task uses 32 identical tokenized inputs and first-response scoring. '
           'Both runs use two-tier BF16 LoD with sparse prefill; only decode routing changes. '
           'Confidence intervals are paired bootstrap intervals over examples.','',
           '| Task | Fixed eight | c=16 threshold | Difference, points (paired 95% interval) |',
           '|---|---:|---:|---:|']
    for index,task in enumerate(tasks):
        keys=sorted(k for k in prior if k[0]==task)
        if len(keys)!=32:raise ValueError(f'expected 32 examples for {task}')
        old=[first(prior[k]) for k in keys];now=[first(current[k]) for k in keys]
        x=sum(old)*100/32;y=sum(now)*100/32;lo,hi=paired_interval([n-o for n,o in zip(now,old)],6240+index)
        lines.append(f'| {task} | {x:.2f}% | {y:.2f}% | {y-x:+.2f} [{lo:+.2f}, {hi:+.2f}] |')
    lines += ['', 'The tasks were selected for analysis; no full-panel fixed-eight average can be inferred.','']
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(lines));print('\n'.join(lines))


if __name__=='__main__':main()
