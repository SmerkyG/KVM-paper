"""Audit frozen AMD prefill runs and generate their consolidated report."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from benchmarks._identity import source_identity


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('results/amd-threshold-prefill'))
    p.add_argument('--phase',default='final-v15')
    a=p.parse_args();root=a.root.resolve();phase=root/a.phase
    jobs=read(phase/'jobs.json');source=Path(jobs['source'])
    for name,expected in read(phase/'source-manifest.json')['source_files'].items():
        if digest(source/name)!=expected:raise ValueError('frozen source changed: '+name)
    source_sha=source_identity(source)['source_sha256']
    for item in read(phase/'cohort-manifest.json').values():
        if digest(Path(item['path']))!=item['sha256']:raise ValueError('quality cohort changed')
    finished=Path.home()/'.cluster-run/agent/finished'
    for job in jobs['jobs']:
        fields=(finished/job['id']).read_text().strip().split('\t')
        if fields[7:9]!=['finished','0']:raise ValueError('unsuccessful required job: '+job['id'])
        if 'uv' not in job['command'] or job['command'][0]!='cluster-run':raise ValueError('unlogged GPU experiment')
    sequence=read(phase/'speed.json')
    if sequence['stage']!='complete' or not sequence['serial']:raise ValueError('incomplete speed sequence')
    speed={item['variant']:read(Path(item['output'])) for item in sequence['jobs']}
    for variant,result in speed.items():
        if result['benchmark_identity']['source']['source_sha256']!=source_sha:raise ValueError('speed source differs')
        if result['batch_size']!=1 or result['decode_tokens']!=2 or result['scheduler_chunk_tokens']!=16384:
            raise ValueError('speed execution scope changed')
        if result['decode_threshold_multiplier'] is not None:raise ValueError('speed decode must remain native')
        for value in result['measurements'].values():
            if len(value['prefill_timings_seconds'])!=3 or len(value['decode_timings_seconds'])!=3:
                raise ValueError('expected three warmed measurements')
    timing=[]
    for length in ('32768','65536'):
        baseline=speed['top8']['measurements'][length]
        for result in speed.values():
            if result['measurements'][length]['prompts']!=baseline['prompts']:raise ValueError('speed prompts differ')
        control=[x for v in ('top8','top8-after') for x in speed[v]['measurements'][length]['prefill_timings_seconds']]
        native=statistics.median(control)
        row=dict(context=int(length),top8_before=baseline['prefill_seconds'],
            threshold=speed['threshold']['measurements'][length]['prefill_seconds'],
            top8_after=speed['top8-after']['measurements'][length]['prefill_seconds'],
            dense=speed['dense']['measurements'][length]['prefill_seconds'],control_median=native)
        row['threshold_change_percent']=100*(row['threshold']/native-1)
        row['dense_speedup']=row['dense']/row['threshold']
        row['completion_seconds']={}
        for variant,result in speed.items():
            value=result['measurements'][length]
            row['completion_seconds'][variant]=statistics.median(
                x+y for x,y in zip(value['prefill_timings_seconds'],value['decode_timings_seconds'],strict=True))
        timing.append(row)
    quality=[]
    for task in ('niah_multikey_2','niah_multikey_3'):
        pair={v:read(phase/f'ruler-{task}-{v}.json') for v in ('top8','threshold')}
        for result in pair.values():
            if result['benchmark_identity']['source']['source_sha256']!=source_sha:raise ValueError('quality source differs')
            if result['stage']!='complete' or len(result['records'])!=32 or result['scheduler_chunk_tokens']!=16384:
                raise ValueError('incomplete quality panel')
        if pair['top8']['decode_threshold_options']!=pair['threshold']['decode_threshold_options']:
            raise ValueError('quality decode settings differ')
        if pair['top8']['threshold_multiplier']!=16 or pair['top8']['generation_seed']!=pair['threshold']['generation_seed']:
            raise ValueError('quality generation differs')
        before={r['doc_id']:r for r in pair['top8']['records']};after={r['doc_id']:r for r in pair['threshold']['records']}
        if before.keys()!=after.keys():raise ValueError('quality cohorts differ')
        identity=('input_token_sha256','input_tokens','targets','doc_hash','prompt_hash','target_hash')
        if any(before[i][k]!=after[i][k] for i in before for k in identity):raise ValueError('quality requests differ')
        quality.append(dict(task=task,examples=32,top8=sum(r['score'] for r in before.values())/32*100,
            threshold=sum(r['score'] for r in after.values())/32*100,
            improved=sum(after[i]['score']>before[i]['score'] for i in before),
            declined=sum(after[i]['score']<before[i]['score'] for i in before),
            min_tokens=min(r['input_tokens'] for r in before.values()),max_tokens=max(r['input_tokens'] for r in before.values())))
    checks=read(phase/'check.json')
    if checks['stage']!='complete' or len(checks['checks'])!=14:raise ValueError('missing frontier checks')
    replay=read(phase/'replay.json')
    if replay['benchmark_identity']['source']['source_sha256']!=source_sha:raise ValueError('replay source differs')
    if replay['stage']!='complete' or len(replay['records'])!=6:raise ValueError('incomplete real-state diagnostics')
    audit=[]
    for manifest in sorted(root.glob('*/jobs.json')):
        for job in read(manifest)['jobs']:
            status=finished/job['id']
            fields=status.read_text().strip().split('\t') if status.exists() else []
            audit.append(dict(phase=manifest.parent.name,id=job['id'],selected=manifest.parent==phase,
                status=fields[7] if fields else 'not-finished',exit_code=fields[8] if fields else None))
    output=dict(stage='complete',phase=a.phase,timing=timing,quality=quality,
                frontier_checks=checks['checks'],real_state=replay['records'],all_runs=audit)
    (root/'REPORT.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=['# AMD Qwen threshold prefill','',
        'Qwen3.8-27B-FP8 weights, BF16 two-tier KV, batch one, MI325X. The opt-in router opens every eligible region '
        'whose normalized-query proxy mass exceeds **16/R**, where R counts nonempty regions. Regions larger than '
        '1,024 entries remain coarse. This report measures the equal-per-region rule; it does not measure entry-16 routing.','',
        '## Full-model prefill','',
        'Each GPU allocation runs top-8 → threshold → top-8 → dense serially. Each variant has one full warmup '
        'and three measured runs on identical nonrepeating ProLong prompts. The exact first chunk, scheduler chunks and '
        'prefill cache-update schedule remain 16,384 tokens. The existing local window and sink attention are retained. '
        'Later remote attention uses 4,096-query scratch tiles. '
        'Decode remains native top-8 in every speed run. Timings are first-token latency from scheduling; initialization '
        'and compilation are outside the measurements.','',
        '| Context | Top-8 before | Threshold | Top-8 after | Change vs native median | Dense | Dense / threshold |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in timing:
        lines.append(f"| {r['context']//1024}K | {r['top8_before']:.3f} s | {r['threshold']:.3f} s | {r['top8_after']:.3f} s | {r['threshold_change_percent']:+.1f}% | {r['dense']:.3f} s | {r['dense_speedup']:.2f}× |")
    lines+=['','The following completion times include the first decode step. This charges any deferred final prefill '
        'cache update to the comparison, since decode waits for its completion. These are two-token request latencies, '
        'not steady-state decode measurements.','',
        '| Context | Top-8 before | Threshold | Top-8 after | Dense |','|---|---:|---:|---:|---:|']
    for r in timing:
        c=r['completion_seconds'];lines.append(f"| {r['context']//1024}K | {c['top8']:.3f} s | {c['threshold']:.3f} s | {c['top8-after']:.3f} s | {c['dense']:.3f} s |")
    lines+=['','## Retrieval check','',
        'First 32 frozen examples per task at nominal 64K; identical prompts, token hashes, targets and seed. Both arms '
        'use the previously tested precise c=16 decoder. The variable is prefill routing. This is a 64-example retrieval '
        'pilot, not a full RULER panel.','',
        '| Task | Top-8 prefill | Threshold prefill | Improved | Declined | Actual input tokens |',
        '|---|---:|---:|---:|---:|---|']
    for r in quality:
        lines.append(f"| {r['task']} | {r['top8']:.2f}% | {r['threshold']:.2f}% | {r['improved']} | {r['declined']} | {r['min_tokens']:,}–{r['max_tokens']:,} |")
    lines+=['','## Identical-query remote attention','',
        'These graph-captured diagnostics replay three actual Qwen layers from the last 16K chunk of the same 64K '
        'native-prefill prompt. They include routing, coarse attention, grouping, exact leaves, sink and final merge. '
        'They exclude local-attention computation, the rest of the model and cache updates; the full-model table includes '
        'those costs. The methods open different amounts of work.','',
        '| Layer | Method | Route/coarse | Leaves, packing and reduction | Complete remote call | Regions/head | Entries/head |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in replay['records']:
        lines.append(f"| {r['layer']} | {r['variant']} | {r['route_coarse_ms']:.2f} ms | {r['leaves_ms']:.2f} ms | {r['remote_attention_ms']:.2f} ms | {r['mean_open_regions']:.2f} | {r['mean_open_entries']:.1f} |")
    lines+=['','## Implementation and validation','',
        'A grouped BLAS product computes query–mean-key scores once with FP32 output, reading shared GQA KV storage '
        'directly. One row pass normalizes routing, computes closed-summary probabilities and packs selected IDs. '
        'The value product uses BF16 probabilities. Bitmasks and integer ranks feed deterministic query-tile counts '
        'and prefix scans; packing has linear work and no global counter contention. Fixed persistent leaf workers '
        'read the actual GPU work count. The final reduction uses 32/64-route fast paths and a general uncapped fallback. '
        'Selection never truncates to eight or sixteen regions. No selected-count CPU readback occurs.','',
        'All 14 independent PyTorch frontier checks passed, including empty selections, uneven sizes, eligibility limits, '
        'random page ownership, query-tile boundaries, more than 64 opened regions, both D/GQA geometries and graph replay. '
        'The native page/hash regressions also passed. Real-state validation independently checks selection and closed '
        'coarse attention on sampled actual queries; exact frontier/reference tolerances and raw validation results are '
        'retained in the JSON artifacts.','',
        'The first exact chunk and local attention are unchanged. Cache initialization and amortized semantic/page '
        'updates use the existing code. The complete execution has not been given a new asymptotic bound by this work. '
        'The attention score matrix and grouping scans scale with queries × regions; exact work depends on selected region '
        'sizes and the existing eligibility limit.','',
        'Enable the tested prefill path with `LOD_AMD_PREFILL_THRESHOLD=16`. This option does not enable or change '
        'threshold decode. The complete experimental settings are in [settings.json](settings.json). '
        'Serving defaults remain unchanged. Model validation here covers BF16 two-tier Qwen at batch one.','',
        '## Provenance and exclusions','',
        f'Final commands, frozen source hashes and outputs: [{a.phase}/jobs.json]({a.phase}/jobs.json). '
        'All GPU runs use uv through cluster-run. [REPORT.json](REPORT.json) records every submitted experiment, '
        'its completion status, and which runs were selected. Failed or cancelled tuning runs are excluded from the tables. '
        'These include the initial whole-chunk allocation failures, an incorrect atomic grouping prototype, its debugging '
        'checks and superseded routing probes. A fused full-coarse experiment was slower than BLAS and was removed from '
        'the final implementation. Wider leaf tile candidates that failed the comparison tolerance were rejected.','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    print(root/'REPORT.md')


if __name__=='__main__':main()
