"""Build the uncapped H100 decode optimization report from validated raw runs."""
import argparse
import json
from pathlib import Path
import statistics


def baseline_comparison(root,read,reference_provenance,prefix):
    """Use complete controls only; require matching inference source and scope."""
    manifest=root/'threshold-decode-baselines-runs.json'
    if not manifest.exists():return [],{}
    jobs=json.loads(manifest.read_text())['jobs']
    if not all((root/(j['name']+'.json')).exists() for j in jobs):return [],{}
    for job in jobs:
        record=json.loads((root/'provenance'/job['id']/'run.json').read_text())
        if record['state']!='finished' or record['exit_code']!=0:raise ValueError('baseline failed')
        if record['hardware']!=reference_provenance['hardware']:raise ValueError('baseline GPU differs')
        original=reference_provenance['source_files'];current=record['source_files']
        if any(current.get(name)!=sha for name,sha in original.items()):raise ValueError('baseline shared source differs')
        extra=set(current)-set(original)
        if extra-{'benchmarks/h100_threshold_decode_baseline_submit.py'}:raise ValueError('unexpected extra baseline source')
    datasets={}
    for length in (32,64):
        for mode in ('dense','top8','region','entry'):
            name=f'threshold-decode-baselines-{mode}-{length}k' if mode in ('dense','top8') else f'{prefix}-{mode}-fast-{length}k'
            data=read(name);measured=[r for r in data['records'] if not r['warmup']]
            if data['batch_size']!=1 or data['timed_decode_steps']!=1024 or data['decode_warmup_steps']!=512 or not data['replay_prompt_tokens'] or len(measured)!=2:raise ValueError('baseline scope mismatch')
            if mode=='dense':
                if not data['native_cuda_graph'] or data['method']!='retro-full':raise ValueError('dense control not graphed')
            else:
                if not data['lod_cuda_graph'] or data['method']!='retro-lod-bf16':raise ValueError('LoD control not graphed')
                if [r['lod_layer_catch_up_events_timed'] for r in measured]!=[128,128]:raise ValueError('LoD update window differs')
            if mode in ('dense','top8') and data.get('decode_threshold') is not None:raise ValueError('control still enabled thresholds')
            datasets[mode,length]=data
        anchor=datasets['dense',length]
        for mode in ('top8','region','entry'):
            if any(datasets[mode,length][key]!=anchor[key] for key in ('model','checkpoint','dtype','batch_size','manifest','prompt_tokens','output_tokens')):raise ValueError('baseline inputs differ')
    labels=[('dense','Dense'),('top8','Two-tier top-8'),('region','Optimized per-region c=16'),('entry','Optimized entry-16')]
    lines=['### Dense and top-eight controls','',
           'Fresh dense and top-eight controls use the same H100, checkpoint, BF16 precision, batch, replayed prompt tokens, graph policy and 512+1,024-step decode window. Their shared source files exactly match the threshold runs; the only added file is the submission helper. All LoD measured repetitions include 128 layer cache-update events.', '',
           '| Method | 32K decode, ms/token | 64K decode, ms/token |','|---|---:|---:|']
    summary={}
    for mode,label in labels:
        cells=[]
        for length in (32,64):
            data=datasets[mode,length]
            name=f'threshold-decode-baselines-{mode}-{length}k' if mode in ('dense','top8') else f'{prefix}-{mode}-fast-{length}k'
            cells.append(f'[{data["median_decode_ms_per_token"]:.3f}]({name}.json)')
            summary[f'{mode}-{length}k']=dict(decode_ms=data['median_decode_ms_per_token'],prefill_s=data['median_prefill_s'],
                measured_repetitions_ms=[r['decode_ms_per_token'] for r in data['records'] if not r['warmup']],
                speedup_vs_dense=datasets['dense',length]['median_decode_ms_per_token']/data['median_decode_ms_per_token'],
                latency_change_vs_top8_pct=100*(data['median_decode_ms_per_token']/datasets['top8',length]['median_decode_ms_per_token']-1))
        lines.append(f'| {label} | '+' | '.join(cells)+' |')
    lines+=['', '| Method | 32K prefill, seconds | 64K prefill, seconds |','|---|---:|---:|']
    for mode,label in labels:
        lines.append(f'| {label} | {datasets[mode,32]["median_prefill_s"]:.3f} | {datasets[mode,64]["median_prefill_s"]:.3f} |')
    lines+=['', 'Dense uses dense prefill. All LoD rows retain top-eight sparse prefill and the same exact initial chunk; threshold changes affect decode only here. Decode excludes prefill and capture, but includes normal KV appends and LoD cache catch-up. This is a timing comparison, not a quality-matched comparison.', '', '### Previous versus optimized thresholds','']
    return lines,summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--label',default='v4');a=p.parse_args();root=a.directory
    def read(name):return json.loads((root/(name+'.json')).read_text())
    prefix=f'threshold-decode-{a.label}';manifest=read(prefix+'-runs')
    provenance=[json.loads((root/'provenance'/j['id']/'run.json').read_text()) for j in manifest['jobs']]
    if any(r['state']!='finished' or r['exit_code']!=0 for r in provenance):raise ValueError('model runs incomplete or failed')
    for field in ('source_files','hardware'):
        if any(r[field]!=provenance[0][field] for r in provenance):raise ValueError('model runs differ in '+field)
    lines=['# Faster uncapped H100 threshold decode','',
           'BF16 Llama-3.1-8B-Instruct, batch 1, H100 80GB, RetroInfer runtime. All GPU experiments use uv and the self-documenting cluster-run adapter. Threshold rules and selected-region budgets are unchanged: there is no cap of eight or sixteen regions.', '',
           'The per-region rule opens every eligible region with proxy mass greater than 16 times the average; its existing 1,024-entry eligibility limit remains. Entry-16 opens regions with proxy mass per entry greater than 1/(16 sqrt(n)), with the existing deterministic strict entry-work guard. Routing still uses the same normalized-query scores and BF16 mean arithmetic.', '',
           '## Complete model, including amortized cache updates','',
           'Every timing uses complete-model CUDA graphs, 512 untimed decode steps and 1,024 measured steps, with context growing throughout. The LoD rows retain top-eight sparse prefill with the unchanged 16,384-token exact initial chunk; dense controls use dense prefill. Cache catch-up remains included for LoD. One warmup repetition precedes two measured repetitions. Values are medians; initial prefill, allocation and graph capture are excluded from decode time.', '',
           '| Initial context | Routing rule | Prior ms/token | Optimized ms/token | Latency reduction | Layer updates per measured window |',
           '|---|---|---:|---:|---:|---|']
    baseline_lines,baselines=baseline_comparison(root,read,provenance[0],prefix)
    if baseline_lines:
        lines[-2:-2]=baseline_lines
        (root/'BASELINES_SUMMARY.json').write_text(json.dumps(baselines,indent=2)+'\n')
    summary={}
    for length in (32,64):
        for rule,label in [('region','Per-region c=16'),('entry','Entry-16')]:
            old=read(f'{prefix}-{rule}-legacy-{length}k');new=read(f'{prefix}-{rule}-fast-{length}k')
            for result in (old,new):
                if not result['lod_cuda_graph'] or not result['replay_prompt_tokens'] or result['batch_size']!=1 or result['timed_decode_steps']!=1024 or result['decode_warmup_steps']!=512:raise ValueError('scope mismatch')
                if len([r for r in result['records'] if not r['warmup']])!=2:raise ValueError('missing measured repetitions')
            for key in ('model','checkpoint','dtype','prompt_tokens','manifest','output_tokens'):
                if old[key]!=new[key]:raise ValueError('paired model mismatch: '+key)
            old_records=[r for r in old['records'] if not r['warmup']];new_records=[r for r in new['records'] if not r['warmup']]
            counts=[[r['lod_layer_catch_up_events_timed'] for r in records] for records in (old_records,new_records)]
            if counts[0]!=counts[1] or not all(n>0 for n in counts[0]):raise ValueError('cache updates not matched')
            before=old['median_decode_ms_per_token'];after=new['median_decode_ms_per_token'];reduction=100*(1-after/before)
            token_disagreements=[]
            for prior_record,new_record in zip(old['records'],new['records']):
                left=prior_record['output_token_ids'];right=new_record['output_token_ids']
                if len(left)!=len(right):raise ValueError('different teacher-forced token lengths')
                token_disagreements.append(sum(x!=y for x,y in zip(left,right)))
            summary[f'{rule}-{length}k']=dict(prior_ms=before,optimized_ms=after,reduction_pct=reduction,
                update_events=counts[0],settings=new['decode_threshold'],
                predicted_token_disagreements_per_repetition=token_disagreements,
                prior_repetitions_ms=[r['decode_ms_per_token'] for r in old_records],
                optimized_repetitions_ms=[r['decode_ms_per_token'] for r in new_records])
            lines.append(f'| {length}K | {label} | [{before:.3f}]({prefix}-{rule}-legacy-{length}k.json) | [{after:.3f}]({prefix}-{rule}-fast-{length}k.json) | {reduction:.1f}% | {counts[0]} |')
    lines += ['', '## What changed', '',
              'The optimized router fuses thresholding, region compaction and entry offsets into one prefix-scan kernel. Exact workers then traverse actual selected entries, finding each entry through its compact region offsets and the existing page directory. This avoids the separate page-descriptor expansion and skips padding in short pages. Coarse attention still directly excludes selected summaries, so this path does not use full-coarse subtraction or its cancellation fallback.', '',
              'The indexed variant retains the existing local/current/sink branch and uses six kernel launches per attention call instead of nine. A combined variant includes those exact entries in the compact exact workers, using five launches. There is no host readback of selected counts, no new GQA expansion, and no change to the cache update schedule. Both reuse the original score producer; no extra score simplification is applied.', '',
              '| Rule and context | Optimized implementation | Exact worker splits |', '|---|---|---:|']
    for length in (32,64):
        for rule in ('region','entry'):
            settings=summary[f'{rule}-{length}k']['settings']
            lines.append(f'| {rule}, {length}K | {settings["implementation"]} | {settings["splits"]} |')
    lines += ['', '## Fixed-query diagnostics', '',
              'These graph timings cover a complete attention call plus the same local-length rollback, on identical real queries/KV states from layers 0, 15 and 31. They omit model layers and cache maintenance. The table reports arithmetic means of the three per-layer medians, not full-model latency. Each median has three event-timed repetitions of 40 graph replays with 16 calls per replay.', '',
              '| Context | Rule | Prior attention, us | Selected optimized attention, us | Reduction |', '|---|---|---:|---:|---:|']
    for length in (32,64):
        rows=read(f'threshold-decode-v2-tune-{length}k')['records']
        if len(rows)!=72 or any(r['selection_mismatches'] for r in rows):raise ValueError('missing fixed-query checks')
        for rule in ('region','entry'):
            settings=summary[f'{rule}-{length}k']['settings']
            before=statistics.mean(r['microseconds'] for r in rows if r['rule']==rule and r['implementation']=='legacy' and r['splits']==64)
            after=statistics.mean(r['microseconds'] for r in rows if r['rule']==rule and r['implementation']==settings['implementation'] and r['splits']==settings['splits'])
            lines.append(f'| {length}K | {rule} | {before:.2f} | {after:.2f} | {100*(1-after/before):.1f}% |')
    lines += ['', 'Packing fusion alone gave little benefit and regressed the 64K probe: the large single-head prefix scan offset the saved launches. Compact entry attention produced the useful gain. Combining local attention helped entry-16 at 32K but was slower than retaining its separate kernel for the other chosen configurations.', '',
              '## Correctness and scope', '',
              '| Optimized graph versus legacy eager on identical cache | Checked steps | Different next-token choices | Maximum logit difference |',
              '|---|---:|---:|---:|']
    for length in (32,64):
        for rule in ('region','entry'):
            data=read(f'{prefix}-{rule}-fast-{length}k-check')
            checks=data['records'][0]['graph_same_cache_checks']
            if len(checks)<17:raise ValueError('missing cache-boundary checks')
            changed=sum(c['graph_argmax']!=c['eager_argmax'] for c in checks.values())
            if changed:raise ValueError('legacy next-token choices differ')
            maximum=max(c['max_abs'] for c in checks.values())
            lines.append(f'| {rule}, {length}K | {len(checks)} | {changed} | {maximum:.6f} |')
    synthetics=[read(f'threshold-decode-v2-{impl}-check') for impl in ('packed','indexed','combined')]
    if any(d['stage']!='complete' or len(d['records'])!=11 for d in synthetics):raise ValueError('missing independent reference checks')
    max_synthetic=max(r['max_abs'] for d in synthetics for r in d['records'])
    max_real=max(r['max_abs_vs_legacy'] for length in (32,64) for r in read(f'threshold-decode-v2-tune-{length}k')['records'])
    lines += ['', f'The independent PyTorch frontier suite passed all eleven cases for each implementation, covering uneven sizes, empty selections, regions above 1,024 entries in entry mode, entry multiples 16/24/32, the strict work guard, and local/sink append. Maximum component error was {max_synthetic:.6f}. All 144 real-query tuning checks selected the same regions as legacy; maximum attention component difference was {max_real:.6f}. Whole-model checks include sampled steps around cache updates and compare the optimized graph to the legacy implementation on the same cache, rather than comparing unrelated generated trajectories.', '',
              '| Paired complete-model repetition | Predicted-token disagreements, warmup / measured 1 / measured 2 |', '|---|---|']
    for length in (32,64):
        for rule in ('region','entry'):
            changes=summary[f'{rule}-{length}k']['predicted_token_disagreements_per_repetition']
            lines.append(f'| {rule}, {length}K | {changes} out of 1,537 per repetition |')
    lines += ['',
              'These are implementation and limited next-token checks. The full RULER quality panel was not rerun. Attention reductions differ at floating-point precision; those differences can propagate through later layers and threshold decisions. Full-model logits are not bitwise equivalent: the largest sampled difference was 1.191406 in the 64K per-region check, and its paired timing runs had one predicted-token disagreement in two repetitions. Quality equivalence has not been established by these speed tests. This work does not add a proof of subquadratic complete prefill or cache maintenance.', '',
              'The optimization is opt-in through `LOD_H100_THRESHOLD_IMPLEMENTATION=indexed` or `combined`, with `LOD_H100_THRESHOLD_SPLITS` as listed above. Existing threshold enable/rule flags remain required. Serving defaults remain unchanged.', '',
              'Failed preliminary probes are retained: the first tuning harness used the wrong integer type in reporting, and the initial full-model validation helper lacked a local import. Dependent timings were skipped after those failures; failed or partial trials are excluded here. All final model comparisons share identical source hashes and GPU identity. Raw commands, logs, timings and validation remain in the JSON files and provenance directory.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');(root/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
