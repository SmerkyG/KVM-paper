"""Build the paired Llama threshold quality, work, and timing report."""
import argparse
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase
from benchmarks.retro_ruler_compare import paired_interval
from benchmarks.retro_ruler_prompt_compare import score
from benchmarks.retro_ruler_threshold_full_compare import TASKS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cohort','entry','region','region-audit','tokenizer','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--timing',type=Path,nargs=4,required=True)
    a=p.parse_args()
    cohort_raw=a.cohort.read_bytes();cohort=json.loads(cohort_raw)
    frozen={(r['task'],r['source_row']):r for r in cohort['samples']}
    if len(frozen)!=13*32 or set(t for t,_ in frozen)!=set(TASKS):raise ValueError('wrong full cohort')
    digest=hashlib.sha256(cohort_raw).hexdigest()
    def read(path,rule,full):
        data=json.loads(path.read_text())
        if data['stage']!='complete' or data['mode']!='two-tier' or data['cohort_sha256']!=digest:raise ValueError(f'incomplete or mismatched {path}')
        routing=data['threshold_routing']
        if routing.get('rule','region')!=rule or routing['mass_multiplier']!=16:raise ValueError('wrong routing')
        rows={(r['task'],r['source_row']):r for r in data['records']}
        if len(rows)!=len(data['records']) or (full and set(rows)!=set(frozen)):raise ValueError('missing or duplicate rows')
        return rows,data
    entry,em=read(a.entry,'entry',True);region,rm=read(a.region,'region',True);audit,am=read(a.region_audit,'region',False)
    for key in entry:
        for group in (entry,region):
            row=group[key];src=frozen[key]
            if row['prompt_sha256']!=src['prompt_sha256'] or row['answers']!=src['answers'] or row['input_token_sha256']!=region[key]['input_token_sha256']:raise ValueError(f'input mismatch {key}')
    for key in audit:
        if key not in entry or audit[key]['input_token_sha256']!=entry[key]['input_token_sha256'] or audit[key]['answers']!=entry[key]['answers']:raise ValueError('audit cohort mismatch')
    tok=Tokenizer.from_file(str(a.tokenizer));stop={tok.token_to_id('<|end_of_text|>'),tok.token_to_id('<|eot_id|>')}
    if None in stop:raise ValueError('missing stop token')
    def first(row):
        ids=row['output_token_ids'];end=next((i for i,t in enumerate(ids) if t in stop),len(ids))
        text=tok.decode(ids[:end],skip_special_tokens=True)
        try:text=PreTrainedTokenizerBase.clean_up_tokenization(text)
        except TypeError:text=PreTrainedTokenizerBase.clean_up_tokenization(None,text)
        return score(row['task'],text,row['answers'])
    lines=['# Llama 3.1 two-tier threshold: per-entry versus per-region', '',
           '32K RULER, BF16 model and KV, batch one, 32 matched examples per task. Both use the same sparse top-eight prefill and normalized-query proxy scores in decode. '
           'The existing region rule opens eligible regions above 16 times mean proxy mass and leaves regions over 1,024 entries coarse. '
           'The entry rule compares proxy mass per remote entry to 1/(16 sqrt(n)), where n is the current remote-entry count; it can open larger regions. '
           'The kernel uses a one-part-per-million conservative log threshold and an exact integer work guard, so a numerical boundary can close a region that the ideal rule would open.', '',
           '## First-response quality', '', '| Task | Per-region c=16 | Per-entry | Entry minus region, pp (paired 95% interval) |','|---|---:|---:|---:|']
    means=[]
    for i,task in enumerate(TASKS):
        keys=sorted(k for k in frozen if k[0]==task);r=[first(region[k]) for k in keys];e=[first(entry[k]) for k in keys]
        rmv=100*sum(r)/32;emv=100*sum(e)/32;lo,hi=paired_interval([x-y for x,y in zip(e,r)],7120+i)
        means.append((rmv,emv));lines.append(f'| {task} | {rmv:.2f} | {emv:.2f} | {emv-rmv:+.2f} [{lo:+.2f}, {hi:+.2f}] |')
    x=sum(x for x,_ in means)/13;y=sum(y for _,y in means)/13
    lines += [f'| **RULER average** | **{x:.2f}** | **{y:.2f}** | {y-x:+.2f} |','',
              'Scores stop at the first end-of-turn or end-of-text token; task means are weighted equally. The sample is small and task intervals are paired bootstrap intervals.', '',
              '## Opened entries and proxy coverage', '',
              'The following is sampled only on the **last decode step** of each example, across all layers and query heads. Proxy coverage is the sum of normalized proxy masses of opened regions; it is not exact attention mass. Counts exclude exact local and sink entries. The maximum ratio divides opened remote entries by 16 sqrt(n).', '',
              '| Task | Rule | Mean opened entries/head | Mean opened regions/head | Mean proxy mass opened | Max bound fraction | Empty heads |',
              '|---|---|---:|---:|---:|---:|---:|']
    for task in TASKS:
        keys=sorted(k for k in audit if k[0]==task)
        if not keys:continue
        for name,group in [('Per-region',audit),('Per-entry',entry)]:
            layers=[l for key in keys for l in group[key]['threshold_route_audit']['layers']]
            heads=sum(l['heads'] for l in layers)
            entries=sum(l['entries_sum'] for l in layers)/heads
            regions=sum(l['regions_sum'] for l in layers)/heads
            mass=sum(l['proxy_mass_sum'] for l in layers)/heads
            maxratio=max(l['max_bound_fraction'] for l in layers);empty=100*sum(l['empty_heads'] for l in layers)/heads
            lines.append(f'| {task} | {name} | {entries:.1f} | {regions:.2f} | {mass:.2%} | {maxratio:.3f} | {empty:.2f}% |')
    changed=sum(audit[k]['output_token_ids']!=region[k]['output_token_ids'] for k in audit)
    lines += ['',f'The fresh per-region audit differs from the earlier per-region run in {changed} of {len(audit)} generated token sequences; the quality table uses the earlier full-panel run.', '',
              '## Whole-model speed', '',
              'H100, BF16 Llama, batch one, graph-captured decode and identical teacher-forced prompt tokens. Each result uses one warmup plus two measured repeats, 512 untimed decode steps and 1,024 timed steps. Prefill is reported separately. Timed decode includes host-side cache catch-up; the update count is summed over model layers.', '',
              '| Context | Rule | Decode ms/token | Prefill seconds, median | Layer update events per timed window |', '|---|---|---:|---:|---:|']
    for path in a.timing:
        data=json.loads(path.read_text());recs=[r for r in data['records'] if not r['warmup']]
        if len(recs)!=2 or data['timed_decode_steps']!=1024:raise ValueError(f'incomplete timing {path}')
        name='Per-entry' if 'entry' in path.name else 'Per-region'
        pf=sorted(r['prefill_s'] for r in recs); updates=[r['lod_layer_catch_up_events_timed'] for r in recs]
        lines.append(f'| {data["prompt_tokens"]:,} | {name} | {data["median_decode_ms_per_batch_step"]:.3f} | {(pf[0]+pf[1])/2:.3f} | {updates[0]}, {updates[1]} |')
    lines += ['', '## Scope of the work bound', '',
              'The sum of opened remote region sizes is strictly below 16 sqrt(n) per query head: each opened region contributes more than its size divided by that bound to a proxy distribution summing to one. The integer guard checks the same bound against actual page-list lengths. It can conservatively discard a suffix of candidates at a numerical boundary. Padded page lanes and per-page bookkeeping mean this is not an identical bound on hardware operations. Local and sink work, region scoring, prefill, and cache updates are outside this bound.', '',
              'The packing scan processes fixed 32-element tiles and carries totals forward once, giving linear work in the number of region tiles. The existing prefill still uses top-eight routing and a 1,024-entry per-region cap, rather than this entry rule. A normal catch-up routes each fixed-size overflow block against current centroids and updates the page index; the timed decode includes those updates. The usual unsplit 16 sqrt(T) state schedule suggests subquadratic work for idealized routing and fixed-size updates, but this is not a proof for complete execution: allocation, page-index maintenance, fallback paths, prefill kernels, model layers, and state-capacity behavior must all be included. No end-to-end asymptotic claim is made here.', '',
              f'Frozen cohort SHA-256: `{digest}`.',
              f'[Per-entry RULER output]({a.entry.name}) · [Prior per-region RULER output]({a.region.name}) · [Fresh per-region audit]({a.region_audit.name}).', '']
    a.output.write_text('\n'.join(lines));print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
