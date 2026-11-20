"""Paired comparison of 8- and 36-region LoD against dense and RetroInfer."""
import argparse,json
from pathlib import Path
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase
from benchmarks.retro_ruler_compare import paired_interval
from benchmarks.retro_ruler_prompt_compare import score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dense','prior','retroinfer','control','candidate','tokenizer','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); data=[json.loads(path.read_text()) for path in (a.dense,a.prior,a.retroinfer,a.control,a.candidate)]
    if any(d['stage']!='complete' for d in data):raise ValueError('incomplete runs')
    if data[3]['routes']!=8 or data[4]['routes']!=36:raise ValueError('wrong route count')
    task='niah_multikey_3'
    def make_group(j):return {(r['task'],r.get('row',r.get('source_row'))):r for r in j['records'] if r['task']==task}
    groups=[make_group(d) for d in data];keys=sorted(groups[-1])
    if len(keys)!=32 or any(set(g)!=set(keys) for g in groups):raise ValueError('unpaired or missing rows')
    for k in keys:
        baseline=groups[1][k]
        if any(g[k]['input_token_sha256']!=baseline['input_token_sha256'] or g[k]['answers']!=baseline['answers'] for g in groups):
            raise ValueError(f'input mismatch {k}')
        if any(g[k]['output_token_ids'][0]!=baseline['output_token_ids'][0] for g in groups[3:]):
            raise ValueError(f'prefill token mismatch {k}')
    for x in (data[3],data[4]):
        checks=x['records'][0]['reference_checks']
        if len(checks)!=32 or any(row['max_abs']>0.2 or row['mean_abs']>0.01 for row in checks):
            raise ValueError('reference checks missing or failed')
    tok=Tokenizer.from_file(str(a.tokenizer));stop={tok.token_to_id('<|end_of_text|>'),tok.token_to_id('<|eot_id|>')}
    if None in stop:raise ValueError('missing stop tokens')
    def first(row):
        ids=row['output_token_ids'];end=next((i for i,t in enumerate(ids) if t in stop),len(ids))
        text=PreTrainedTokenizerBase.clean_up_tokenization(tok.decode(ids[:end],skip_special_tokens=True))
        return score(task,text,row['answers'])
    values=[[first(g[k]) for k in keys] for g in groups]
    labels=['Dense','Optimized two-tier, 8 regions','RetroInfer, 36 shared clusters',
            'PyTorch two-tier control, 8 regions','PyTorch two-tier, 36 regions']
    lines=['# Two-tier LoD with 36 open regions: Llama multi-key 3', '',
        'All rows use the same 32 frozen 32K prompts, Llama-3.1-8B-Instruct, BF16 and greedy first-response scoring. '
        'The PyTorch runs use the original optimized two-tier sparse prefill and cache updates, '
        'but evaluate the two-tier decode frontier in eager FP32/PyTorch. '
        'Each of its query heads opens up to the specified number of whole regions; '
        'RetroInfer selects 36 clusters shared by its four-query-head GQA group, so equal counts are not equal work. '
        'This is a quality experiment, not a latency benchmark.', '',
        '| Method | First-response score | Change from PyTorch eight-region control, pp (paired 95% interval) |',
        '|---|---:|---:|']
    for i,(label,vals) in enumerate(zip(labels,values)):
        diff=[v-b for v,b in zip(vals,values[3])];lo,hi=paired_interval(diff,10612+i)
        lines.append(f'| {label} | {sum(vals)*100/32:.2f}% | {sum(diff)*100/32:+.2f} [{lo:+.2f}, {hi:+.2f}] |')
    for label,idx in [('36 versus PyTorch 8',4),('Optimized 8 versus PyTorch 8',1)]:
        base=values[3];v=values[idx]
        improved=[keys[i][1] for i in range(32) if v[i]>base[i]]
        worsened=[keys[i][1] for i in range(32) if v[i]<base[i]]
        lines += ['',f'{label}: {len(improved)} improved, {len(worsened)} worsened. '
                  f'Improved source rows: {improved}; worsened source rows: {worsened}.']
    for j in (3,4):
        checks=data[j]['records'][0]['reference_checks']
        lines+=['',f'{data[j]["routes"]}-region PyTorch first-step check against the independent '
                 f'readable reference: maximum absolute component error {max(x["max_abs"] for x in checks):.6f}; '
                 f'average layer-wise mean absolute error {sum(x["mean_abs"] for x in checks)/32:.6f}.']
    lines+=['','Checks cover the first decode step on one example; later states were not checked against the reference. '
             'The PyTorch versions use FP32 accumulation and can choose different tokens from the BF16 optimized kernel, '
             'so the 8-region control is included. Dense and RetroInfer use dense prefill. '
             'Paired bootstrap intervals are over these 32 examples.', '']
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(lines));print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
