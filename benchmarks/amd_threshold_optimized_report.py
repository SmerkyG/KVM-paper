"""Collect AMD threshold optimization results with paired native controls and provenance."""
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
    provenance=[];core={};required={'check-final-v4','verify-model-v4','quality-v4','speed-v4'}
    for path in sorted(root.glob('*/jobs.json')):
        manifest=read(path);phase=path.parent.name;snapshot=read(path.parent/'source-manifest.json')
        for name,digest in snapshot['source_files'].items():
            if hashlib.sha256((Path(snapshot['source'])/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'changed frozen source: {phase}/{name}')
        if phase in required:
            for name in ('benchmarks/amd_threshold_materialized.py','benchmarks/amd_threshold_shared.py','benchmarks/amd_threshold_decode.py','benchmarks/amd_threshold_frontier.py','benchmarks/amd_threshold_merge.py',
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
    settings=read(root/'settings-v4.json');summary=dict(stage='complete',settings=settings,models={},core_sources=core)
    rows=[];quality_rows=[];diagnostic_rows=[];details=[];conclusions=[]
    for model in ('qwen','k2'):
        controller=complete(root/'speed-v4'/f'amd-fast-speed-v4-{model}.json')
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
        prefix=f'speed-v4/amd-fast-speed-v4-{model}'
        rows.append(f'| {model.upper()} | [{ms["top8"]:.3f}]({prefix}-top8.json) | [{ms["fast"]:.3f}]({prefix}-fast.json) | [{ms["top8-after"]:.3f}]({prefix}-top8-after.json) | {improvement:+.1f}% |')
        same_native_output=measures['top8']['output_token_sha256']==measures['top8-after']['output_token_sha256']
        if not same_native_output:details.append(f'- {model.upper()} native before/after controls generated different token sequences despite matching prompt hashes. The bracketing spread cannot be attributed to clock drift alone.')
        fast_times=[1000*t/1024 for t in measures['fast']['decode_timings_seconds']]
        separated=max(fast_times)<min(native_times)
        conclusions.append(f'{model.upper()}: '+('both optimized repetitions are below all four top-8 control repetitions in this run.' if separated else 'timing ranges overlap or favor top-8; a consistent speedup is not established.'))
        probe=complete(root/'verify-model-v4'/f'amd-fast-verify-model-v4-{model}.json')['records']
        if len(probe)!=9 or any(x.get('selection_mismatches',0) for x in probe):raise ValueError('invalid serving-cache validation')
        old=[x for x in probe if x['implementation']=='legacy'];new=[x for x in probe if x['implementation']=='materialized']
        native_us=statistics.mean(x['microseconds'] for x in probe if x['implementation']=='native-top8')
        old_us=statistics.mean(x['microseconds'] for x in old);new_us=statistics.mean(x['microseconds'] for x in new)
        max_error=max(x['max_abs_vs_legacy'] for x in new)
        diagnostic_rows.append(f'| {model.upper()} | {old_us:.2f} | {new_us:.2f} | {native_us:.2f} | {100*(1-new_us/old_us):.1f}% | {max_error:g} |')
        quality={v:complete(root/'quality-v4'/f'amd-fast-quality-v4-{model}-{v}.json') for v in ('legacy','fast')}
        pairs=list(zip(quality['legacy']['records'],quality['fast']['records'],strict=True))
        if len(pairs)!=16:raise ValueError('incomplete retrieval cohort')
        for old,new in pairs:
            if any(old[key]!=new[key] for key in ('task','doc_id','input_token_sha256','target_hash')):raise ValueError('quality inputs differ')
        wins=sum(y['score']>x['score'] for x,y in pairs);losses=sum(y['score']<x['score'] for x,y in pairs)
        for task in quality['legacy']['tasks']:
            quality_rows.append(f'| {model.upper()} | {task} | {quality["legacy"]["task_scores"][task]:.2f}% | {quality["fast"]["task_scores"][task]:.2f}% |')
        check=complete(root/'check-final-v4'/f'amd-fast-check-final-v4-{model}.json')
        if len(check['records'])!=12 or any(x['selection_mismatches'] for x in check['records']):raise ValueError('reference checks failed')
        details.append(f'- {model.upper()} retrieval: {wins} improved, {losses} declined; text changed on {sum(x["prediction"]!=y["prediction"] for x,y in pairs)}/16 requests.')
        details.append(f'- {model.upper()} greedy repeat checks against each run’s warmup: '+', '.join(f'{v}: '+('identical' if d['greedy_output_identical'] else f'first mismatch {d["first_mismatch_positions"]}') for v,d in measures.items())+'.')
        summary['models'][model]=dict(ms_per_token=ms,native_median_ms=control,less_time_percent=improvement,
            faster_than_both_controls=ms['fast']<min(ms['top8'],ms['top8-after']),faster_than_all_control_repetitions=separated,
            native_repetitions_ms=native_times,optimized_repetitions_ms=fast_times,native_controls_output_identical=same_native_output,measurements=measures,
            kernel_diagnostic=dict(old_us=old_us,new_us=new_us,native_top8_us=native_us,max_abs=max_error,records=probe),
            quality=dict(old=quality['legacy']['task_scores'],new=quality['fast']['task_scores'],improved=wins,declined=losses),
            max_reference_error=max(x['max_abs'] for x in check['records']))
    lines=['# AMD two-tier threshold decode optimization','',
        'Final setting: equal-per-region **c=16**, batch one, 65,536-token ProLong input, BF16 KV and FP8 model weights on MI325X. '
        'The router retains normalized-query ranking and the existing 1,024-entry eligibility limit. No top-k cap was added. '
        'The exact first prefill chunk remains 16,384 tokens, and later prefill retains top-8 routing. The per-entry rule is outside this result.','',
        '## Full-model decode, including cache updates','',
        'Each model uses one GPU and serial **top-8 → optimized c=16 → top-8** runs. Each run warms up with one full generation, then measures two generations of 1,024 decode steps. '
        'The final top-8 control checks baseline drift; overlapping repetition ranges are treated as near parity. “Less time” compares optimized latency to the median of all four native-control repetitions. '
        'These are measurements on one prompt; autoregressive continuations can differ, and the methods are not quality- or work-budget matched.','',
        '| Model | Top-8 before, ms/token | Optimized c=16 | Top-8 after | Less time vs control median |',
        '|---|---:|---:|---:|---:|',*rows,'',*conclusions,'',
        '## Implementation','',
        'The chosen experimental path has five stages:', '',
        '1. Share summary-key reads across GQA heads, produce native and normalized proxy logits plus tile normalizers, and append the current KV entry without advancing its length.',
        '2. Compute the c=16 threshold; evaluate closed summaries and pack selected regions, lengths and per-head membership bits inside each routing tile.',
        '3. Use parallel prefix-scan shards to compact physical KV indices and membership bits into a shared table. No KV data is copied. The number of shards is fixed, so packing work is linear in routing-tile count plus the selected union’s entry count.',
        '4. Use AITER-shaped segmented attention over the compact table, sharing KV reads across GQA heads while retaining each head’s own frontier. Local, sink and current entries are included.',
        '5. Merge partial attention results and advance the local length.', '',
        'The selected settings use 64-region routing tiles, 128 exact segments, 64 entries per attention block, eight index-preparation shards, one warp per preparation worker, four warps per exact worker and a 32-dimension merge block. '
        'The tensor-core value product uses native-style BF16 attention probabilities; stored KV remains BF16. Counts remain FP32 when forming means. '
        'There is no top-k cap, sorting, selected-count CPU readback, change to c=16, or removal of local/sink attention.', '',
        'The persistent index already exists in Qwen’s native path. The opt-in K2 path maintains it at normal cache catch-up boundaries. '
        '**That extra K2 update cost is included in the full-model timings.**', '',
        '## Work audit', '',
        'Cached-summary QK scores are computed once per call and reused for routing and coarse attention. The chosen index-packing stage uses prefix scans and a fixed number of workers; it does not compare every region with every other region. '
        'Its flat tables have linear remote-token capacity, but exact attention visits only the selected union plus local/sink/current entries. '
        'The per-tile coarse stage still repeats the reduction of all tile normalizers: with T routing tiles, that metadata arithmetic is O(T²), while summary matrix products are linear in region count at fixed head dimension. '
        'Persistent-index catch-up visits the archive. Prefill still uses its existing top-8 implementation after the exact first chunk and has not been audited as subquadratic here. '
        '**No complete-execution subquadratic claim follows from these measurements.**', '',
        '## Same-query kernel checks','',
        'The first, middle and last attention layers use actual model queries and serving caches after a three-token generation. '
        'Old threshold, optimized threshold and native top-8 calls see identical query/cache state, with the same local-length rollback. Native top-8 uses its own routing rule and refinement budget. The table averages three sampled layers. '
        'This scope excludes projections, FFNs, scheduling and cache catch-up and must not be read as a full-model speedup.','',
        '| Model | Old c=16, µs/call | Optimized c=16 | Native top-8 | Less time vs old c=16 | Max attention-component difference |',
        '|---|---:|---:|---:|---:|---:|',*diagnostic_rows,'',
        'All sampled old/optimized threshold selected-region masks matched exactly. Tensor results are checked with BF16-aware tolerances (atol=0.004, rtol=0.008) plus independent PyTorch frontier tests; absolute differences are reported above. The chosen settings passed 12 independent PyTorch frontier cases over both head geometries, repeated under both model configurations. '
        'These cover empty selections, a 1,024-entry region, exclusion of a 1,025-entry region, and repeated fixed-cache calls.','',
        '## Retrieval smoke check','',
        'Eight examples per task and model: four historical misses plus four successes. Requests and reference answers are paired exactly. '
        'This deliberately enriched 32-example total checks implementation regression and is **not a full RULER panel or an accuracy estimate**.','',
        '| Model | Task | Old c=16 | Optimized c=16 |','|---|---|---:|---:|',*quality_rows,'',*details,'',
        '## Reproduction and exclusions','',
        'Every GPU run uses `uv` through `cluster-run`. Sources are frozen per phase and hash-validated. '
        'The report requires successful completion of the final speed, serving-cache, reference and quality jobs. '
        'Earlier alternatives and failed attempts remain in the provenance but contribute no rows to the final tables: grouped GQA, small GQA subgroups, whole-frontier consumers, region-owned workers, balanced scalar workers, active-block workers, fused tile-local tables, resource-limited configurations, a compiler-rejected pointer-select reduction, and initial numerical-validation failures. The absolute-only 0.04 validation gate rejected a one-BF16-ULP difference (0.0625 at a large output component); the final gate uses the explicit dtype-aware tolerance above and reports maximum absolute errors. '
        'The first K2 serving attempt failed at initialization because its native path did not allocate the persistent index; this was fixed and rerun with its maintenance cost included.','',
        '[Raw summary](REPORT.json), [chosen settings](settings-v4.json), [all jobs, placement and exit status](provenance.json), '
        '[retrieval cohort](quality-v4/cohort-manifest.json), '
        '[Qwen serving-cache checks](verify-model-v4/amd-fast-verify-model-v4-qwen.json), '
        '[K2 serving-cache checks](verify-model-v4/amd-fast-verify-model-v4-k2.json).','',
        'Enable this experiment with `LOD_AMD_THRESHOLD_MULTIPLIER=16` and the environment settings in `settings-v4.json`. '
        'The serving default remains the existing native router. After freezing these measurements, CPU dispatch guards were added for the tested two-tier BF16 head geometries; the independent frontier checks were rerun, with no change to the measured GPU kernels.','']
    (root/'REPORT.json').write_text(json.dumps(summary,indent=2)+'\n');(root/'REPORT.md').write_text('\n'.join(lines));print(root/'REPORT.md')


if __name__=='__main__':main()
