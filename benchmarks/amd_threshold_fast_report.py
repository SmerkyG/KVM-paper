"""Validate and summarize the opt-in AMD two-tier compact-decode experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('results/amd-threshold-fast'))
    a=p.parse_args();root=a.root.resolve();summary=dict(stage='complete',models={},correctness={})
    def read(path):return json.loads(path.read_text())
    def complete(path):
        d=read(path)
        if d.get('stage')!='complete':raise ValueError(f'incomplete experiment: {path}')
        return d
    provenance=[]
    core_hashes={}
    for manifest in sorted(root.glob('*/jobs.json')):
        phase=read(manifest);frozen=read(manifest.parent/'source-manifest.json')
        if manifest.parent.name in ('check-v2','probe-v2','quality-v1','speed-controls-v1','speed-final-v1'):
            for name in ('benchmarks/amd_threshold_decode.py','benchmarks/amd_threshold_decode_fast.py','benchmarks/h100_threshold_decode_fast.py'):
                digest=frozen['source_files'][name]
                if name in core_hashes and core_hashes[name]!=digest:raise ValueError(f'different tested kernels: {name}')
                core_hashes[name]=digest
        for name,digest in frozen['source_files'].items():
            if hashlib.sha256((Path(frozen['source'])/name).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'frozen source changed: {manifest.parent.name}/{name}')
        for job in phase['jobs']:
            final=Path.home()/'.cluster-run/agent/finished'/job['id']
            log=Path.home()/'.cluster-run/agent/logs'/(job['id']+'.log')
            if not final.exists():raise ValueError(f'job has not finished: {job["id"]}')
            fields=final.read_text().strip().split('\t')
            row=dict(id=job['id'],phase=manifest.parent.name,node=fields[2],gpus=fields[3],
                     submitted=int(fields[4]),started=int(fields[5]),ended=int(fields[6]),
                     status=fields[7],exit_code=fields[8],command=job['command'])
            used=manifest.parent.name in ('check-v1','check-v2','probe-v2','quality-v1','speed-final-v1') or (manifest.parent.name=='speed-controls-v1' and job['name'].endswith('-top8'))
            if used and (row['status']!='finished' or row['exit_code']!='0'):
                raise ValueError(f'valid experiment did not exit successfully: {job["id"]}')
            dest=root/'provenance';dest.mkdir(exist_ok=True)
            shutil.copy2(final,dest/(job['id']+'.tsv'))
            if log.exists():shutil.copy2(log,dest/log.name)
            provenance.append(row)
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    for phase in ('check-v1','check-v2'):
        cases=[]
        for impl in ('packed','indexed','combined'):
            cases+=complete(root/phase/f'amd-fast-{phase}-{impl}.json')['records']
        if len(cases)!=54 or any(r['selection_mismatches'] for r in cases):raise ValueError('reference checks failed')
        summary['correctness'][phase]=dict(cases=len(cases),max_abs_vs_reference=max(r['max_abs'] for r in cases))
    lines=['# AMD two-tier compact threshold decode','',
           'Batch one, 65,536-token prompts, BF16 KV and the FP8-weight Qwen3.8-27B / K2-Horizon-32B models on MI325X. '
           'The primary threshold opens regions above 16 times mean normalized-query proxy mass, retaining the existing 1,024-entry eligibility limit. '
           'This is an opt-in decode change. Exact first-chunk prefill remains 16,384 tokens; later prefill uses the existing top-8 path.','',
           'The port fuses threshold selection and linear prefix packing, then traverses selected entries directly. '
           'It removes page-descriptor expansion and keeps AMD FP32 count division, coarse summaries, local/sink attention and KV append. '
           'The selected setting for both models is `indexed`, 64 splits, 32 entries per block, two warps and routing tile 64. '
           'It uses six attention kernels instead of nine. The five-kernel combined-local variant was also checked and timed.','',
           '## Full-model autoregressive decode','',
           'One warmup generation followed by two measured generations, each with 1,024 timed decode steps. '
           'Prefill is excluded from these milliseconds per token; serving overhead and normal amortized cache updates are included. '
           'Controls and variants use the same physical GPU within each model and run serially. '
           'Methods can generate different continuations, so this is not a quality-matched or identical-query speed comparison. '
           'Native AMD top-8 uses native-query ranking; thresholds use normalized-query ranking and a variable exact-work budget.','',
           '| Model | Top-8 ms/token | Old c=16 | Compact c=16 | Less time than old threshold |',
           '|---|---:|---:|---:|---:|']
    quality_lines=[];fixed_lines=[];repeat_lines=[]
    for model in ('qwen','k2'):
        probes=complete(root/'probe-v2'/f'amd-fast-probe-v2-{model}.json')['records']
        if len(probes)!=108 or any(r['selection_mismatches'] for r in probes):raise ValueError('real-query validation failed')
        serial=complete(root/'speed-final-v1'/f'amd-fast-speed-final-v1-{model}.json')
        speeds={'top8':read(root/'speed-controls-v1'/f'amd-fast-speed-controls-v1-{model}-top8.json')}
        speeds.update({r['variant']:read(Path(r['output'])) for r in serial['jobs']})
        required={'top8','legacy','fast'}
        if set(speeds)!=required:raise ValueError('missing speed variants')
        ms={k:v['measurements']['65536'] for k,v in speeds.items()}
        if len({r['prompts'][0]['token_sha256'] for r in ms.values()})!=1:raise ValueError('different speed prompts')
        for variant,d in speeds.items():
            if d['batch_size']!=1 or d['decode_tokens']!=1025 or d['mode']!='two-tier' or d['scheduler_chunk_tokens']!=16384:
                raise ValueError('speed scope mismatch')
            if len(ms[variant]['decode_timings_seconds'])!=2:raise ValueError('missing repeats')
        timing={k:v['decode_ms_per_batch_step'] for k,v in ms.items()}
        saving=100*(1-timing['fast']/timing['legacy'])
        paths={'top8':f'speed-controls-v1/amd-fast-speed-controls-v1-{model}-top8.json',
               'legacy':f'speed-final-v1/amd-fast-speed-final-v1-{model}-legacy.json',
               'fast':f'speed-final-v1/amd-fast-speed-final-v1-{model}-fast.json'}
        cells={key:f'[{timing[key]:.3f}]({paths[key]})' for key in timing}
        lines.append(f'| {model.upper()} | {cells["top8"]} | {cells["legacy"]} | {cells["fast"]} | {saving:.1f}% |')
        model_result=dict(timing=timing,measurements=ms,fixed_query={})
        for rule in ('region','entry'):
            selected=[r for r in probes if r['rule']==rule and r['implementation']=='indexed' and r['splits']==64 and r['entry_block']==32 and r['entry_warps']==2]
            old=[r for r in probes if r['rule']==rule and r['implementation']=='legacy']
            if len(old)!=3 or len(selected)!=3:raise ValueError('missing fixed-query layers')
            previous=statistics.mean(r['microseconds'] for r in old);now=statistics.mean(r['microseconds'] for r in selected)
            regions=statistics.mean(r['mean_opened_regions'] for r in selected)
            entries=statistics.mean(r['mean_opened_entries'] for r in selected)
            fixed_lines.append(f'| {model.upper()} | {rule} | {previous:.2f} | {now:.2f} | {100*(1-now/previous):.1f}% | {regions:.2f} | {entries:.1f} |')
            model_result['fixed_query'][rule]=dict(old_us=previous,compact_us=now,max_abs=max(r['max_abs_vs_legacy'] for r in selected),layers=selected)
        quality={v:complete(root/'quality-v1'/f'amd-fast-quality-v1-{model}-{v}.json') for v in ('legacy','fast')}
        pairs=list(zip(quality['legacy']['records'],quality['fast']['records'],strict=True))
        for old,new in pairs:
            if any(old[k]!=new[k] for k in ('task','doc_id','input_token_sha256','target_hash')):raise ValueError('unpaired quality examples')
        wins=sum(new['score']>old['score'] for old,new in pairs);losses=sum(new['score']<old['score'] for old,new in pairs)
        predictions=sum(old['prediction']!=new['prediction'] for old,new in pairs)
        model_result['quality']=dict(old_scores=quality['legacy']['task_scores'],compact_scores=quality['fast']['task_scores'],
                                    improved=wins,declined=losses,changed_predictions=predictions)
        for task in quality['legacy']['tasks']:
            quality_lines.append(f'| {model.upper()} | {task} | {quality["legacy"]["task_scores"][task]:.2f}% | {quality["fast"]["task_scores"][task]:.2f}% |')
        repeat_lines.append(f'- {model.upper()}: old/compact retrieval examples improved {wins}, declined {losses}; changed prediction texts {predictions}/16.')
        repeat_lines.append(f'- {model.upper()} speed repeat checks against warmup: '+', '.join(f'{k}: '+('identical' if v['greedy_output_identical'] else f'differed at {v["first_mismatch_positions"]}') for k,v in ms.items())+'.')
        summary['models'][model]=model_result
    lines += ['',
              'The compact worker improves the existing threshold implementation, but remains slower than native AMD top-8 in these full-model runs. '
              'The Qwen difference is small and comes from one prompt and two timed repetitions. '
              'Native top-8 already shares KV work across GQA heads, whereas the compact exact workers remain per query head. '
              'The Qwen model has 16 LoD attention layers and K2 has 64, so the per-attention-call improvements contribute differently to total latency.','',
              '## Fixed real-query attention diagnostics','',
              'Identical query, KV state and selected regions, sampled from first/middle/last attention layers after a three-token generation. '
              'Times include the complete attention pipeline and the same local-length rollback. Each timing is a median of three measurements of a graph with eight calls, replayed 32 times. '
              'These are averages over three sampled layers, excluding projections, FFNs, model scheduling and cache catch-up. '
              'The worker retains the diagnostic request cache after completion; normal speed and quality runs do not install that hook.','',
              '| Model | Rule | Old threshold µs | Compact µs | Less time | Mean open regions/head | Mean exact remote entries/head |','|---|---|---:|---:|---:|---:|---:|',*fixed_lines,'',
              'Opened-work counts average over sampled layers and query heads at this single decode step, not across generated tokens.','',
              'Both per-region c=16 and per-entry bound-16 routing were checked. The per-entry rule has kernel-only timings here; no full-model or RULER claim is made for that rule. '
              'All 216 real-query configurations matched the old ordered region selections exactly. Floating reduction orders differ: maximum attention-component difference across all tuned configurations was 0.001953125 for Qwen and 0.00390625 for K2.','',
              '## Retrieval implementation smoke check','',
              'Eight archived 64K examples per task: four historical failures plus four successes, separately selected for each model. '
              'This small, deliberately enriched subset checks implementation regression; it is not a full RULER panel or an accuracy estimate. '
              'Old and compact c=16 use identical tokenized requests and references.','',
              '| Model | Task | Old c=16 | Compact c=16 |','|---|---|---:|---:|',*quality_lines,'',*repeat_lines,'',
              '## Validation and provenance','',
              'The independent PyTorch reference suite passed two sets of 54 cases across the packed, indexed and combined variants, including the chosen two-warp exact worker, covering both model geometries, empty selections, oversized regions, region imbalance, and a strict per-entry work guard. '
              'The largest reference attention-component error was 0.000553534, with zero selection mismatches. '
              'Passing these checks does not establish bitwise equivalence or full-panel quality.','',
              'The initial `probe-v1` jobs failed because completed-request cleanup had cleared the cache; they contribute no timings. '
              'The initial legacy speed submissions in `speed-controls-v1` and `speed-legacy-v1` were cancelled before measurement because explicit GPU filters allow overlapping one’s own jobs. '
              'Those cancelled runs contribute no timings. Final legacy/compact measurements execute sequentially inside one allocation per model. '
              'Source snapshots are immutable and hash-validated; every GPU run uses `uv` through `cluster-run`.','',
              '[Raw summary](REPORT.json), [chosen settings](settings.json), [jobs and placement](provenance.json), '
              '[Qwen fixed-query inputs/timings](probe-v2/amd-fast-probe-v2-qwen.json), '
              '[K2 fixed-query inputs/timings](probe-v2/amd-fast-probe-v2-k2.json), '
              '[retrieval cohort manifest](quality-v1/cohort-manifest.json).','',
              'Serving defaults are unchanged. Enable the experimental worker with `LOD_AMD_THRESHOLD_MULTIPLIER=16`, '
              '`LOD_AMD_THRESHOLD_IMPLEMENTATION=indexed`, `LOD_AMD_THRESHOLD_SPLITS=64`, '
              '`LOD_AMD_THRESHOLD_ENTRY_BLOCK=32`, and `LOD_AMD_THRESHOLD_ENTRY_WARPS=2`. '
              'Prefill and complete-execution asymptotic work were not changed or re-audited by this decode port.','']
    (root/'REPORT.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'REPORT.md').write_text('\n'.join(lines))
    print(root/'REPORT.md')


if __name__=='__main__':main()
