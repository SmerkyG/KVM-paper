"""Publish matched full-prefill timings and independently score frozen checks.

Warmup, instrumented profiles, failed jobs and cancelled jobs are never used
as final timings. The complete 13-task RULER panel is not rerun by this panel.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from benchmarks.h100_threshold_next_report import score_files


def publish(root,repo):
    status=json.loads((root/'status.json').read_text())
    summary=dict(models={},profiles={},quality={},quality_trials={},verification={},checks={},independent_checks={},inputs_sha256={},hardware=[])
    hardware=set();verified_objects=set();candidate=('threshold-directopt-accepted-v3-select-',
        'threshold-directopt-accepted128-v3-select-')
    for job in status:
        if job['state']!='finished' or job['exit_code']!=0:continue
        path=root/(job['name']+'.json');d=json.loads(path.read_text())
        meta_path=root/'provenance'/job['id']/'run.json';meta=json.loads(meta_path.read_text())
        hardware.add(meta['hardware'])
        for p in (path,meta_path):summary['inputs_sha256'][str(p.relative_to(repo))]=hashlib.sha256(p.read_bytes()).hexdigest()
        for name,digest in meta['source_files'].items():
            if digest not in verified_objects:
                assert hashlib.sha256((root/'source-objects'/digest).read_bytes()).hexdigest()==digest,name
                verified_objects.add(digest)
        if job['name'].endswith('-quality'):continue
        if job['name'].endswith(('-statecheck','-rmscheck')):
            if job['name'] in ('threshold-directopt-rms-final-v3-select-0-rmscheck',
                'threshold-directopt-packedmax-fixed-v3-select-0-statecheck'):
                assert d['stage']=='complete'
                if job['name'].endswith('-rmscheck'):
                    cases=[r for r in d['records'] if r['style']==3]
                    assert all(r[k]==0 for r in cases for k in
                        ('different_bits','mean_differences','root_differences','inverse_differences'))
                    summary['independent_checks']['rms']=dict(file=path.name,elements=sum(r['elements']for r in cases),different_bits=0)
                else:
                    assert all(r['different_indices']==r['different_finite_value_bits']==0 and r['matching_nan_positions']for r in d['cases'])
                    summary['independent_checks']['packed_max']=dict(file=path.name,rows=sum(r['rows']for r in d['cases']),different_indices=0,different_finite_value_bits=0)
            continue
        if job['name'].endswith(('-numerics','-probe','-rounding')):continue
        if job['name'].endswith('-checks'):
            if job['name'].startswith(candidate):
                assert d['stage']=='complete'
                assert d['cases'] and all(case['cap']==0 and case['checks'] and all(
                    c['selection_mismatches']==c['selection_outside_tolerance']==0 for c in case['checks'])for case in d['cases'])
                summary['checks'][path.name]=d
            continue
        rows=[r for r in d['records']if not r['warmup']]
        if job['name'].endswith('-verify'):
            if not job['name'].startswith(candidate):continue
            assert len(rows)==1
            checks=[c for layer in rows[0]['prefill_fast_audit']for c in layer]
            graph=rows[0]['graph_same_cache_checks']
            assert checks and all(c['selection_outside_tolerance']==0 for c in checks)
            assert graph and all(c['max_abs']==0 for c in graph.values())
            summary['verification'][str(job['length'])]=dict(file=path.name,frontiers=len(checks),
                selection_outside_tolerance=0,max_reference_error=max(c['max_abs_output_difference']for c in checks),
                graph_same_cache_max_abs=0)
            continue
        profiles=job['name'].endswith('-profiles')
        assert len(rows)==(1 if profiles else 3),(path,len(rows))
        assert d['dtype']=='bfloat16' and d['batch_size']==1 and d['prefill_mlp_chunk']==8192
        if job['variant']!='dense':assert d['prefill_includes_deferred_cache_completion']
        identity=[d[k]for k in ('manifest_sha256','prompt_token_sha256','checkpoint','dtype','prefill_mlp_chunk')]
        item=dict(file=path.name,length=job['length'],variant=job['variant'],identity=identity,
            median_s=statistics.median(r['prefill_s']for r in rows),samples_s=[r['prefill_s']for r in rows],
            peak_allocated_gib=max(r['memory_bytes']['peak_allocated']for r in rows)/2**30,
            configuration=d['prefill_threshold'])
        if job['name'].startswith(candidate):
            cfg=item['configuration']
            assert cfg['mass_multiplier']==16 and cfg['max_regions']is None and cfg['prefill_block_size']==16384
            assert cfg['closed_probability_precision']=='scaled-fp16'and cfg['closed_arithmetic']=='direct'
            assert cfg['closed_reduce_backend']=='tc-wide'and cfg['tc_reduce_rows']==4 and cfg['tc_reduce_parts']==3
            assert cfg['fast_tile_queries']==(4096 if job['length']==128000 else 'auto') and cfg['fast_score_reuse']
            assert cfg['fast_producer_tile']==[64,128,4,'auto'] and cfg['state_fused_rms'] and cfg['state_packed_max']
        if profiles:
            totals=defaultdict(float);aux=defaultdict(float)
            for layer in rows[0]['prefill_leaf_profile']:
                for k,v in layer.items():totals[k]+=v
            audit=rows[0]['prefill_scaling_audit'];work=[r for l in audit['layers']for r in l['rows']]
            for layer in audit['layers']:
                for k,v in layer['gpu_seconds'].items():aux[k]+=v
            qh=sum(r['query_heads']for r in work)
            item.update(gpu_seconds=dict(totals),auxiliary_gpu_seconds=dict(aux),
                regions_per_query_head=sum(r['opened_regions']for r in work)/qh,
                entries_per_query_head=sum(r['opened_entries']for r in work)/qh)
        summary['profiles'if profiles else'models'][path.name]=item
    assert len(hardware)==1;summary['hardware']=sorted(hardware)
    final={}
    for item in summary['models'].values():
        name=item['file']
        if name.startswith(('threshold-directopt-accepted-v3-','threshold-directopt-accepted128-v3-')):
            arm='optimized'if item['variant']=='select'else item['variant']
            final.setdefault(str(item['length']),{})[arm]=item
        elif name.startswith('threshold-directopt-retime-v3-'):
            final.setdefault(str(item['length']),{}).setdefault(item['variant'],item)
    old=root.parent/'direct-opt';offset=root.parent/'closed-offset'
    if '128000'in final:
        prior_path=old/'SUMMARY.json';prior=json.loads(prior_path.read_text())
        assert prior['validated_final']and prior['stage']=='complete'and prior['hardware']==summary['hardware']
        item=prior['final']['128000']['select'].copy();path=old/item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==prior['inputs_sha256'][item['file']]
        d=json.loads(path.read_text());rows=[r for r in d['records']if not r['warmup']]
        assert len(rows)==3 and statistics.median(r['prefill_s']for r in rows)==item['median_s']
        assert item['identity']==[d[k]for k in ('manifest_sha256','prompt_token_sha256','checkpoint','dtype','prefill_mlp_chunk')]
        item['peak_allocated_gib']=item.pop('peak_allocated_bytes')/2**30
        item['file']='../direct-opt/'+item['file']
        final['128000']['select']=item
        for p in (path,prior_path):summary['inputs_sha256'][str(p.relative_to(repo))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for length,arms in final.items():
        assert len({tuple(v['identity'])for v in arms.values()})==1,length
    references=[old/'threshold-directopt-baseline-cwe-v2-baseline-32768-quality.json',
        offset/'threshold-offset-scaled-fp16-panel-v1-prefix-closed-tuned-ruler.json']
    tokenizer=root.parent/'ruler-32k-panel/llama-tokenizer.json'
    reference,expected=score_files(references,tokenizer)
    summary['quality_reference']=reference
    for p in (*references,tokenizer):summary['inputs_sha256'][str(p.relative_to(repo))]=hashlib.sha256(p.read_bytes()).hexdigest()
    quality_jobs=[j for j in status if j['name'].endswith('-quality')and j['state']=='finished'and j['exit_code']==0]
    for job in quality_jobs:
        path=root/(job['name']+'.json');scored,samples=score_files([path],tokenizer)
        assert all(a is None or a==b for a,b in zip(scored['identity'],reference['identity']))
        for key,row in samples.items():
            assert key in expected and all(row[k]==expected[key][k]for k in
                ('prompt_sha256','effective_prompt_sha256','input_token_sha256','answers'))
        scored.update(n=len(samples),same_scores=sum(v['score']==expected[k]['score']for k,v in samples.items()),
            improved_scores=sum(v['score']>expected[k]['score']for k,v in samples.items()),
            lower_scores=sum(v['score']<expected[k]['score']for k,v in samples.items()))
        summary['quality_trials'][path.name]=scored
        if job['name'].startswith(('threshold-directopt-acceptedcwe-v3-','threshold-directopt-acceptedpanel-v3-')):
            assert json.loads(path.read_text())['source_rows_filter']is None
            summary['quality'][path.name]=scored
    tasks={k:v for scored in summary['quality'].values()for k,v in scored['tasks'].items()}
    summary.update(final=final,pending=[j for j in status if j['state']!='finished'],
        failed=[j for j in status if j['state']=='finished'and j['exit_code']not in(0,-15,143)],
        cancelled=[j for j in status if j['exit_code']in(-15,143)])
    required_lengths=(32768,65536,128000)if'128000'in final else(32768,65536)
    summary['validated_final']=all({'dense','top8','select','optimized'}<=final.get(str(n),{}).keys()
        for n in required_lengths)and set(summary['verification'])=={str(n)for n in required_lengths}and bool(summary['checks'])and set(summary['independent_checks'])=={'rms','packed_max'}and all(
        tasks.get(k,{}).get('n')==n and tasks[k]['first']>=reference['tasks'][k]['first']
        for k,n in (('cwe',32),('niah_multikey_2',2),('niah_multikey_3',2),('qa_1',2)))
    summary['stage']='running'if summary['pending']else'complete'
    lines=['# H100 threshold prefill: optimized retime','',
        'Llama-3.1-8B-Instruct, BF16 weights/KV, batch one, same H100 80GB. '
        'All GPU runs use uv and immutable cluster-run snapshots.','',
        '## Full-model prefill','',
        'One warmup and three measured repetitions per arm; medians below exclude warmup. '
        'Timing includes local/sink attention, the complete initial exact field, state/cache updates and deferred cache completion. '
        'Initial allocation, loading and decode graph capture are excluded consistently. '
        'Prefill is eager for every arm; these are not graph-replay or attention-only timings.','',
        '| Context | Dense | Top-eight | Previous optimized threshold | New threshold | New versus dense |',
        '|---:|---:|---:|---:|---:|---:|']
    for n,arms in sorted(final.items(),key=lambda x:int(x[0])):
        if not {'dense','top8','select','optimized'}<=arms.keys():continue
        new=arms['optimized']['median_s'];dense=arms['dense']['median_s'];delta=100*(new/dense-1)
        vals=[f'{arms[k]["median_s"]:.3f} s'for k in ('dense','top8','select','optimized')]
        lines.append('| '+' | '.join([f'{int(n):,}',*vals,f'{abs(delta):.1f}% '+('more time'if delta>0 else'less time')])+' |')
    lines+=['','Uncapped equal-per-region c=16, the 1,024-entry region eligibility limit and the '
        '16,384-token initial exact field are unchanged. Internal scratch tiles now use 16,384 queries '
        'through 2,048 active regions, 8,192 through 4,096 regions and 4,096 above that '
        '(previously 4,096 throughout). The 128,000-token run explicitly retains 4,096-query scratch '
        'tiles throughout for memory headroom; this does not change the exact initial field or routing. '
        'MLP chunks remain 8,192. This is not the per-entry bounded rule.','',
        '## Implementation and limits','',
        '- Native QK is saved in FP32. Final threshold selection and closed-weight preparation share the same saved-score pass, retaining the previously validated globally shifted FP16 coarse projection.',
        '- Selection bits are compacted and packed without sorting, count readbacks or a region cap. Query scaling retains its explicit rounded multiplication before log-count addition.',
        '- Common exact frontiers use a block diagonal tensor-core reduction with a three-part BF16 expansion of FP32 weights. Frontiers larger than 32 regions use the uncapped original reduction.',
        '- Producer pipelining uses two stages through 2,048 regions and one stage for larger states. The QK tile remains 64 queries by 128 regions with four warps.',
        '- Cache-key RMS normalization fuses the existing PyTorch 2.5 operations while reproducing its reduction order and rounding. The guarded BF16/D128 path falls back to PyTorch for unsupported shapes or versions.',
        '- State assignment keeps the original cuBLAS score matrix. A packed unsigned reduction returns its BF16 maximum and first winning index without separate value/index reductions. NaN positions and indices match; NaN payload bits are outside the comparison contract.',
        '- All cache work remains inside full-prefill timing. The fresh top-eight control uses the same new cache normalization and max reductions.','',
        'The increased scratch tile trades memory for speed. Peak PyTorch allocation below includes weights '
        'and excludes allocations made outside PyTorch. Dense and top-eight controls are freshly rerun at every length; '
        'the previous optimized-threshold column uses earlier completed matched retimes. '
        'The longest prompt is 128,000 tokens, not 131,072. No 128K quality panel has been rerun. '
        'No complete-execution subquadratic bound is claimed. '
        'The remaining 32K gap is reported explicitly; this result does not establish a tie with dense.','',
        '| Context | Previous threshold peak | New threshold peak | New measured samples |',
        '|---:|---:|---:|---|']
    for n,arms in sorted(final.items(),key=lambda x:int(x[0])):
        if not {'select','optimized'}<=arms.keys():continue
        a,b=arms['select'],arms['optimized']
        lines.append(f'| {int(n):,} | {a["peak_allocated_gib"]:.2f} GiB | {b["peak_allocated_gib"]:.2f} GiB | '+
            ', '.join(f'{x:.4f} s'for x in b['samples_s'])+' |')
    validation_note = 'The table below covers the frozen quality checks used during tuning.'
    if (root/'RULER_32K.md').is_file():
        validation_note = 'The [full 32K RULER panel](RULER_32K.md) uses the accepted timing configuration. ' + validation_note
    lines+=['','## Validation','',f'Candidate validation complete: **{summary["validated_final"]}**. '
        + validation_note,'',
        '| Task | Examples | Fresh previous-selector control | Candidate |','|---|---:|---:|---:|']
    for k in ('cwe','niah_multikey_2','niah_multikey_3','qa_1'):
        if k in tasks:lines.append(f'| {k} | {tasks[k]["n"]} | {reference["tasks"][k]["first"]:.2f}% | {tasks[k]["first"]:.2f}% |')
    if summary['quality']:
        counts={k:sum(v[k]for v in summary['quality'].values())for k in ('n','same_scores','improved_scores','lower_scores')}
        lines += [f'\nAcross these {counts["n"]} examples: {counts["same_scores"]} scores are unchanged, '
            f'{counts["improved_scores"]} improve and {counts["lower_scores"]} are lower than the matched previous-selector control. '
            'These differences are numerical generation outcomes; they are not evidence that this implementation improves the routing rule.']
    for n,v in summary['verification'].items():
        lines.append(f'\nAt {int(n):,} tokens: {v["frontiers"]} checked frontiers, zero selection differences outside tolerance, '
            f'maximum frontier-reference error {v["max_reference_error"]:.7f}, and exact same-cache eager/graph decode logits.')
    if set(summary['independent_checks'])=={'rms','packed_max'}:
        c=summary['independent_checks']
        lines += [f'\nIndependent PyTorch checks find zero BF16 output-bit differences across {c["rms"]["elements"]:,} normalized values, '
            'with identical FP32 means, square roots and reciprocals. An ordinary Triton reduction is retained as a negative control and does differ. '
            f'The packed maximum check covers {c["packed_max"]["rows"]:,} rows, including ties, signed zeros, infinities and NaNs, '
            'with identical indices, non-NaN value bits and NaN positions.']
    if summary['checks']:
        cases=[case for run in summary['checks'].values()for case in run['cases']]
        largest=max(c['full_tile_max_regions']for case in cases for c in case['checks'])
        lines += [f'\nThe {len(cases)} uncapped synthetic stress cases have zero selection mismatches and reach {largest} opened regions, '
            'exercising both the tensor-core reducer and its larger-frontier fallback.']
    lines+=['','The faster online-coarse and tiled-cache candidates were rejected after quality checks. '
        'With both changes CWE scored 29.38%; restoring the original cache update alone scored 30.00%. '
        'Keeping the original coarse projection with tensor-core reduction scored 36.56% in the intermediate full CWE control. '
        'The table above uses the final full 32-example rerun, with no diagnostic source-row filter.','',
        'Apply these overrides to the existing experimental scaled-FP16 direct configuration:','',
        '```text','LOD_H100_CLOSED_REDUCE=tc-wide','LOD_H100_TC_REDUCE_ROWS=4',
        'LOD_H100_TC_REDUCE_PARTS=3','LOD_H100_FAST_TILE=auto',
        'LOD_H100_FAST_PRODUCE_STAGES=auto','LOD_H100_STATE_FUSED_RMS=1','LOD_H100_STATE_PACKED_MAX=1','```','',
        'Use `LOD_H100_FAST_TILE=4096` at 128,000 tokens. The scratch dispatch is separate from the unchanged 16K initial exact field. '
        'Serving defaults and other models are unchanged.','']
    lines+=['','## Diagnostic trials','',
        'The following are one-repetition instrumented profiles, with warmup excluded. '
        'Nested stage timings overlap and must not be added to their parent timings. '
        'They are separate from the three-repetition full-model table.','',
        '<details>','<summary>Profiles, including regressing variants</summary>','',
        '| Raw profile | Context | Wall seconds | Routing pass | Closed pass | Exact kernel | Exact reduction |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,v in sorted(summary['profiles'].items()):
        g=v['gpu_seconds'];vals=[f'{g.get(k,0):.4f}'for k in ('fast_produce','fast_closed_project','kernel','reduce')]
        lines.append('| '+' | '.join([f'[{name}]({name})',str(v['length']),f'{v["median_s"]:.4f}',*vals])+' |')
    lines+=['','</details>','',f'{len(summary["failed"])} failed and {len(summary["cancelled"])} cancelled trials are excluded '
        'from all accepted timing and quality results. Their logs and statuses remain in provenance and SUMMARY.json. '
        'The compact-output prototypes included compilation failures; quality jobs for the slower matched-subtraction '
        'prototype were cancelled after timing showed a regression.','',
        'Exact commands, immutable source hashes, source objects, raw outputs, and individual samples are retained '
        'in this directory and [SUMMARY.json](SUMMARY.json).','']
    (root/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'REPORT.md').write_text('\n'.join(lines))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];root=a.directory.resolve();s=publish(root,repo)
    print(json.dumps({k:s[k]for k in ('stage','validated_final')}))


if __name__=='__main__':main()
