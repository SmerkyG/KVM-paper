"""Summarize adaptive two-tier quality and counts by layer and query head."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase
from benchmarks.retro_ruler_compare import paired_interval
from benchmarks.retro_ruler_prompt_compare import score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('adaptive','control','fixed36','dense','retroinfer','tokenizer','output','head-csv'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    data=[json.loads(path.read_text()) for path in (a.adaptive,a.control,a.fixed36,a.dense,a.retroinfer)]
    if any(d['stage']!='complete' for d in data):raise ValueError('incomplete inputs')
    if data[0]['top_p']!=0.70 or data[1]['routes']!=8 or data[2]['routes']!=36:raise ValueError('wrong settings')
    def group(d):return {(r['task'],r.get('row',r.get('source_row'))):r for r in d['records'] if r['task']=='niah_multikey_3'}
    groups=[group(d) for d in data];keys=sorted(groups[0])
    if len(keys)!=32 or any(set(g)!=set(keys) for g in groups):raise ValueError('unpaired cohort')
    for k in keys:
        if any(g[k]['input_token_sha256']!=groups[0][k]['input_token_sha256'] or g[k]['answers']!=groups[0][k]['answers'] for g in groups):
            raise ValueError(f'input mismatch {k}')
        if any(g[k]['output_token_ids'][0]!=groups[0][k]['output_token_ids'][0] for g in groups[1:3]):
            raise ValueError(f'sparse prefill mismatch {k}')
    checks=data[0]['records'][0]['reference_checks']
    if len(checks)!=9 or any(c['max_abs']>0.001 for c in checks):raise ValueError('independent checks missing or failed')
    tokenizer=Tokenizer.from_file(str(a.tokenizer));stop={tokenizer.token_to_id('<|end_of_text|>'),tokenizer.token_to_id('<|eot_id|>')}
    if None in stop:raise ValueError('missing stop token')
    def first(r):
        ids=r['output_token_ids'];end=next((i for i,t in enumerate(ids) if t in stop),len(ids))
        text=PreTrainedTokenizerBase.clean_up_tokenization(tokenizer.decode(ids[:end],skip_special_tokens=True))
        return score('niah_multikey_3',text,r['answers'])
    values=[[first(g[k]) for k in keys] for g in groups]
    labels=['Adaptive 70% centroid mass','Fixed 8 regions','Fixed 36 regions','Dense','RetroInfer']
    lines=['# Llama 32K multi-key 3: adaptive 70% region mass','',
        'All rows use the same 32 frozen prompts and BF16 model/KV. The adaptive run uses original two-tier LoD sparse prefill '
        'and cache updates, and the same FP32 PyTorch decode approach as the fixed-8 and fixed-36 controls. '
        'For every query head it ranks the remote centroids by their **native-query, count-corrected** scores, '
        'opens the smallest prefix covering 70% of their estimated mass, and retains all other regions as summaries. '
        'Local and sink entries are excluded from the 70% denominator and receive exact attention. '
        'Regions over 1,024 entries stay summarized; any resulting coverage shortfall is reported below. '
        'Fixed-8/36 LoD ranks regions with a normalized query, so both the count rule and routing score differ. '
        'This is a quality diagnostic, not a speed measurement.','',
        '| Method | First-response score | Change from fixed 8, pp (paired 95% interval) |',
        '|---|---:|---:|']
    for i,(label,v) in enumerate(zip(labels,values)):
        diff=[x-b for x,b in zip(v,values[1])];lo,hi=paired_interval(diff,19110+i)
        lines.append(f'| {label} | {sum(v)*100/32:.2f}% | {sum(diff)*100/32:+.2f} [{lo:+.2f}, {hi:+.2f}] |')
    improved=[keys[i][1] for i in range(32) if values[0][i]>values[1][i]]
    worsened=[keys[i][1] for i in range(32) if values[0][i]<values[1][i]]
    samples=[groups[0][k] for k in keys];steps=sum(r['decode_steps'] for r in samples)
    sums=np.sum([np.asarray(r['routing_sums'],dtype=float) for r in samples],axis=0)
    mins=np.min([np.asarray(r['routing_mins'],dtype=float) for r in samples],axis=0)
    maxs=np.max([np.asarray(r['routing_maxs'],dtype=float) for r in samples],axis=0)
    if sums.shape!=(32,32,4) or mins.shape!=(32,32) or maxs.shape!=(32,32):raise ValueError('wrong head geometry')
    means=sums/steps;avg=means[:,:,0];head_mean=avg.mean(axis=0);layer_mean=avg.mean(axis=1)
    a.head_csv.parent.mkdir(parents=True,exist_ok=True)
    with a.head_csv.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['layer','query_head','mean_regions_opened','mean_exact_remote_entries',
            'mean_estimated_remote_mass_opened','mean_oversized_regions_skipped','minimum_regions','maximum_regions','decode_steps'])
        for layer in range(32):
            for head in range(32):
                w.writerow([layer,head,*[f'{v:.8f}' for v in means[layer,head]],int(mins[layer,head]),int(maxs[layer,head]),steps])
    lines+=['',f'Adaptive versus fixed 8: {len(improved)} improved and {len(worsened)} worsened; '
        f'improved source rows {improved}; worsened rows {worsened}.','',
        f'There were {steps} decode steps across the 32 answers. Every query head and layer is sampled at every step. '
        f'The mean is **{avg.mean():.2f} open regions per query head**, with '
        f'{means[:,:,1].mean():.2f} exact remote KV entries. Across individual head/layer/step observations, '
        f'the minimum was {int(mins.min())} regions and the maximum was {int(maxs.max())}. '
        f'Mean selected estimated remote mass was {100*means[:,:,2].mean():.3f}%; '
        f'mean skipped oversized regions per head-step was {means[:,:,3].mean():.6f}.','',
        '## Per query head','',
        'These are query-head indices 0–31 averaged across all layers and decode steps; '
        'head 0 in different layers refers to different learned heads. The '
        '[layer-by-head CSV]('+a.head_csv.name+') gives all 1,024 separate heads, including minimum, maximum and exact-entry counts.','',
        '| Head | Mean open regions | Head | Mean open regions | Head | Mean open regions | Head | Mean open regions |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for i in range(8):
        inds=[i,i+8,i+16,i+24]
        lines.append('| '+' | '.join(f'{j} | {head_mean[j]:.2f}' for j in inds)+' |')
    lines+=['',f'The lowest layer-averaged head was {int(head_mean.argmin())} ({head_mean.min():.2f}); '
        f'the highest was {int(head_mean.argmax())} ({head_mean.max():.2f}). '
        f'Layer means range from {layer_mean.min():.2f} (layer {int(layer_mean.argmin())}) '
        f'to {layer_mean.max():.2f} (layer {int(layer_mean.argmax())}).','',
        f'Nine independent first-step selection/frontier checks (three heads at layers 0, 15 and 31) '
        f'had maximum absolute output error {max(c["max_abs"] for c in checks):.9f}. '
        'These checks use the same optimized cache and do not validate the entire autoregressive run. '
        'The other two fixed-budget PyTorch controls were previously compared with the readable reference.','']
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(lines));print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
