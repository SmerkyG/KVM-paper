"""Collect validated AMD balanced-threshold results and all experiment provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('results/amd-threshold-fast-v2'));a=p.parse_args();root=a.root.resolve()
    def read(path):return json.loads(path.read_text())
    def complete(path):
        data=read(path)
        if data.get('stage')!='complete':raise ValueError(f'incomplete result: {path}')
        return data
    provenance=[];core={};required={'check-final-v2','verify-model-v2','quality-v2','speed-v2'}
    for path in sorted(root.glob('*/jobs.json')):
        manifest=read(path);phase=path.parent.name;snapshot=read(path.parent/'source-manifest.json')
        for name,digest in snapshot['source_files'].items():
            if hashlib.sha256((Path(snapshot['source'])/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'changed frozen source: {phase}/{name}')
        if phase in required:
            for name in ('benchmarks/amd_threshold_balanced.py','benchmarks/amd_threshold_frontier.py','benchmarks/amd_threshold_merge.py',
                         'integrations/vllm_lod/vllm_lod_plugin/pool.py'):
                digest=snapshot['source_files'][name]
                if name in core and core[name]!=digest:raise ValueError(f'different final kernels: {name}')
                core[name]=digest
        for job in manifest['jobs']:
            final=Path.home()/'.cluster-run/agent/finished'/job['id']
            if not final.exists():raise ValueError(f'job still running: {job["id"]}')
            f=final.read_text().strip().split('\t');row=dict(id=job['id'],phase=phase,node=f[2],gpu=f[3],
                submitted=int(f[4]),started=int(f[5]),ended=int(f[6]),status=f[7],exit_code=f[8],command=job['command'],final_validation=phase in required)
            if phase in required and (row['status']!='finished' or row['exit_code']!='0'):raise ValueError(f'final run failed: {row}')
            dest=root/'provenance';dest.mkdir(exist_ok=True);shutil.copy2(final,dest/(job['id']+'.tsv'))
            log=Path.home()/'.cluster-run/agent/logs'/(job['id']+'.log')
            if log.exists():shutil.copy2(log,dest/log.name)
            provenance.append(row)
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    settings=read(root/'settings-v2.json');summary=dict(stage='complete',settings=settings,models={},core_sources=core)
    rows=[];quality_rows=[];diagnostic_rows=[];details=[]
    for model in ('qwen','k2'):
        controller=complete(root/'speed-v2'/f'amd-fast-speed-v2-{model}.json')
        speed={job['variant']:read(Path(job['output'])) for job in controller['jobs']}
        if set(speed)!={'top8','fast','top8-after'}:raise ValueError('missing before/after controls')
        measures={v:d['measurements']['65536'] for v,d in speed.items()}
        if len({x['prompts'][0]['token_sha256'] for x in measures.values()})!=1:raise ValueError('different speed prompts')
        for variant,data in speed.items():
            if data['decode_tokens']!=1025 or data['batch_size']!=1 or data['scheduler_chunk_tokens']!=16384 or data['mode']!='two-tier':raise ValueError('scope mismatch')
            if len(measures[variant]['decode_timings_seconds'])!=2:raise ValueError('missing timed repetitions')
        for key,value in settings[model].items():
            if speed['fast']['decode_threshold_options'].get(key)!=value:raise ValueError(f'wrong speed setting: {key}')
        ms={v:d['decode_ms_per_batch_step'] for v,d in measures.items()}
        native_times=[1000*t/1024 for v in ('top8','top8-after') for t in measures[v]['decode_timings_seconds']]
        control=statistics.median(native_times);improvement=100*(1-ms['fast']/control)
        prefix=f'speed-v2/amd-fast-speed-v2-{model}'
        rows.append(f'| {model.upper()} | [{ms["top8"]:.3f}]({prefix}-top8.json) | [{ms["fast"]:.3f}]({prefix}-fast.json) | [{ms["top8-after"]:.3f}]({prefix}-top8-after.json) | {improvement:+.1f}% |')
        probe=complete(root/'verify-model-v2'/f'amd-fast-verify-model-v2-{model}.json')['records']
        if len(probe)!=6 or any(x['selection_mismatches'] for x in probe):raise ValueError('invalid serving-cache validation')
        old=[x for x in probe if x['implementation']=='legacy'];new=[x for x in probe if x['implementation']=='balanced']
        old_us=statistics.mean(x['microseconds'] for x in old);new_us=statistics.mean(x['microseconds'] for x in new)
        max_error=max(x['max_abs_vs_legacy'] for x in new)
        diagnostic_rows.append(f'| {model.upper()} | {old_us:.2f} | {new_us:.2f} | {100*(1-new_us/old_us):.1f}% | {max_error:g} |')
        quality={v:complete(root/'quality-v2'/f'amd-fast-quality-v2-{model}-{v}.json') for v in ('legacy','fast')}
        pairs=list(zip(quality['legacy']['records'],quality['fast']['records'],strict=True))
        if len(pairs)!=16:raise ValueError('incomplete retrieval cohort')
        for old,new in pairs:
            if any(old[key]!=new[key] for key in ('task','doc_id','input_token_sha256','target_hash')):raise ValueError('quality inputs differ')
        wins=sum(y['score']>x['score'] for x,y in pairs);losses=sum(y['score']<x['score'] for x,y in pairs)
        for task in quality['legacy']['tasks']:
            quality_rows.append(f'| {model.upper()} | {task} | {quality["legacy"]["task_scores"][task]:.2f}% | {quality["fast"]["task_scores"][task]:.2f}% |')
        check=complete(root/'check-final-v2'/f'amd-fast-check-final-v2-{model}.json')
        if len(check['records'])!=12 or any(x['selection_mismatches'] for x in check['records']):raise ValueError('reference checks failed')
        details.append(f'- {model.upper()} retrieval: {wins} improved, {losses} declined; text changed on {sum(x["prediction"]!=y["prediction"] for x,y in pairs)}/16 requests.')
        details.append(f'- {model.upper()} greedy repeat checks against each run’s warmup: '+', '.join(f'{v}: '+('identical' if d['greedy_output_identical'] else f'first mismatch {d["first_mismatch_positions"]}') for v,d in measures.items())+'.')
        summary['models'][model]=dict(ms_per_token=ms,native_median_ms=control,less_time_percent=improvement,
            faster_than_both_controls=ms['fast']<min(ms['top8'],ms['top8-after']),measurements=measures,
            kernel_diagnostic=dict(old_us=old_us,new_us=new_us,max_abs=max_error,records=probe),
            quality=dict(old=quality['legacy']['task_scores'],new=quality['fast']['task_scores'],improved=wins,declined=losses),
            max_reference_error=max(x['max_abs'] for x in check['records']))
    lines=['# AMD two-tier threshold decode: balanced persistent-index workers','',
        'Final setting: equal-per-region **c=16**, batch one, 65,536-token ProLong input, BF16 KV and FP8 model weights on MI325X. '
        'The router retains normalized-query ranking and the existing 1,024-entry eligibility limit. No top-k cap was added. '
        'The exact first prefill chunk remains 16,384 tokens, and later prefill retains top-8 routing. The per-entry rule is outside this result.','',
        '## Full-model decode, including cache updates','',
        'Each model uses one GPU and serial **top-8 → optimized c=16 → top-8** runs. Each run warms up with one full generation, then measures two generations of 1,024 decode steps. '
        'The final top-8 control checks baseline drift. “Less time” compares optimized latency to the median of all four native-control repetitions. '
        'These are measurements on one prompt; autoregressive continuations can differ, and the methods are not quality- or work-budget matched.','',
        '| Model | Top-8 before, ms/token | Optimized c=16 | Top-8 after | Less time vs control median |',
        '|---|---:|---:|---:|---:|',*rows,'',
        '## Implementation','',
        'The five-stage and grouped-GQA trials led to a four-stage path:','',
        '1. Share summary-key loads across GQA heads, produce native and normalized proxy logits plus tile normalizers, and place the current KV entry in the local arena without advancing its length.',
        '2. Compute the global threshold from those normalizers; evaluate closed summaries and pack selected regions and entry prefixes **within each routing tile**.',
        '3. Balance exact workers by actual entry count. Each worker scans only the short tile-total array in registers, finds the owning tile and region, and reads the persistent physical index. Local, sink and current entries use the same arena loads. There is no global selected-list packing kernel or page-directory traversal in this path.',
        '4. Merge partial attention results and advance the local length.','',
        'Both models use routing tile 64, 256 exact workers per query head, 16 entries per block, one warp and workers ordered together by query head. '
        'Counts remain FP32 when forming means. GPU routing does not read selected counts back to the CPU. '
        'The persistent index is already maintained by Qwen’s native two-tier path. The opt-in K2 path now maintains the same index at normal cache catch-up boundaries. '
        '**That extra K2 update cost is included in the full-model timings.** This index rebuild visits the remote archive; these measurements make no complete-execution subquadratic-work claim.','',
        '## Same-query kernel checks','',
        'The first, middle and last attention layers use actual model queries and serving caches after a three-token generation. '
        'Old and optimized calls see identical state, with the same local-length rollback. The table averages three sampled layers. '
        'This scope excludes projections, FFNs, scheduling and cache catch-up and must not be read as a full-model speedup.','',
        '| Model | Old c=16, µs/call | Optimized c=16 | Less time | Max attention-component difference |',
        '|---|---:|---:|---:|---:|',*diagnostic_rows,'',
        'All sampled selected-region masks matched exactly. The chosen settings passed 12 independent PyTorch frontier cases over both head geometries, repeated under both model configurations. '
        'These cover empty selections, a 1,024-entry region, exclusion of a 1,025-entry region, and repeated fixed-cache calls.','',
        '## Retrieval smoke check','',
        'Eight examples per task and model: four historical misses plus four successes. Requests and reference answers are paired exactly. '
        'This deliberately enriched 32-example total checks implementation regression and is **not a full RULER panel or an accuracy estimate**.','',
        '| Model | Task | Old c=16 | Optimized c=16 |','|---|---|---:|---:|',*quality_rows,'',*details,'',
        '## Reproduction and exclusions','',
        'Every GPU run uses `uv` through `cluster-run`. Sources are frozen per phase and hash-validated. '
        'The report requires successful completion of the final speed, serving-cache, reference and quality jobs. '
        'Earlier alternatives and failed attempts remain in the provenance but contribute no rows to the final tables: grouped GQA, small GQA subgroups, whole-frontier consumers, region-owned workers, resource-limited configurations, and initial numerical-validation failures. '
        'The first K2 serving attempt failed at initialization because its native path did not allocate the persistent index; this was fixed and rerun with its maintenance cost included.','',
        '[Raw summary](REPORT.json), [chosen settings](settings-v2.json), [all jobs, placement and exit status](provenance.json), '
        '[retrieval cohort](quality-v2/cohort-manifest.json), '
        '[Qwen serving-cache checks](verify-model-v2/amd-fast-verify-model-v2-qwen.json), '
        '[K2 serving-cache checks](verify-model-v2/amd-fast-verify-model-v2-k2.json).','',
        'Enable this experiment with `LOD_AMD_THRESHOLD_MULTIPLIER=16` and the environment settings in `settings-v2.json`. '
        'The serving default remains the existing native router.','']
    (root/'REPORT.json').write_text(json.dumps(summary,indent=2)+'\n');(root/'REPORT.md').write_text('\n'.join(lines));print(root/'REPORT.md')


if __name__=='__main__':main()
