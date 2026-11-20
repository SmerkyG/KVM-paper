"""Report verified 64K threshold optimizations without substituting kernel pilots."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

from tokenizers import Tokenizer
from benchmarks.h100_threshold_next_report import cleanup,metric


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--final-label',default='final-v1');a=p.parse_args();root=a.directory
    status=json.loads((root/'status.json').read_text());jobs={v['name']:v for v in status}
    summary=dict(models={},profiles={},checks={},verify={},quality={},probes={},excluded={},input_sha256={})
    hardware=set()
    for path in sorted(root.glob('*.json')):
        job=jobs.get(path.stem)
        if not job or job['state']!='finished' or job.get('exit_code')!=0:continue
        d=json.loads(path.read_text());meta=json.loads((root/'provenance'/job['id']/'run.json').read_text())
        hardware.add(meta['hardware']);assert len(hardware)==1
        summary['input_sha256'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stem=='threshold-select64-v1-fused-65536-profiles':
            summary['excluded'][path.name]='Invalid: later full-model reference check failed; 32-bit fallback indexing overflowed.';continue
        if '-live-'in path.name or '-producer-'in path.name:
            assert d['stage']=='complete';summary['probes'][path.name]=d;continue
        if 'cases'in d:
            assert d['stage']=='complete';summary['checks'][path.name]=d;continue
        if path.stem.endswith('-ruler'):
            assert d['stage']=='complete';summary['quality'][path.name]=d;continue
        if path.stem.endswith('-verify'):
            rows=[r for r in d['records']if not r['warmup']];assert len(rows)==1
            layers=rows[0]['prefill_fast_audit'];assert len(layers)==32 and all(layers)
            checks=[z for l in layers for z in l]
            assert all(z['selection_outside_tolerance']==0 for z in checks)
            assert all(v['max_abs']==0 for v in rows[0]['graph_same_cache_checks'].values())
            summary['verify'][path.name]=dict(frontiers=len(checks),
                max_abs_error=max(z['max_abs_output_difference']for z in checks),
                threshold_boundary_differences=sum(z['selection_mismatches']for z in checks),
                outside_tolerance=sum(z['selection_outside_tolerance']for z in checks))
            if all('selection_backend'in z for z in checks):
                observed=sorted({(z['selection_backend'],z['active_regions'])for z in checks})
                summary['verify'][path.name]['observed_dispatch']=observed
                if '-auto-65536-'in path.name:assert {b for b,s in observed}=={'prefix'}
            continue
        match=re.fullmatch(r'threshold-select64-(.+?)-(control|scan|sparse|fused(?:-tuned)?|combined|dense|top8|auto|bitmap(?:-fused)?|prefix(?:-(?:full|closed))?(?:-fused|-bucket)?(?:-tuned)?|recompute(?:-tuned)?|bucket|bitmap-bucket)-(\d+)-(models|profiles)',path.stem)
        if not match:continue
        series,variant,n,kind=match.groups();rows=[r for r in d['records']if not r['warmup']]
        assert len(rows)==(3 if kind=='models'else 1)
        assert d['batch_size']==1 and d['dtype']=='bfloat16'
        if variant not in ('dense','top8'):
            assert d['prefill_threshold']['mass_multiplier']==16 and d['prefill_threshold']['max_regions']is None
            assert d['prefill_threshold']['prefill_block_size']==16384
        if variant!='dense':assert d['prefill_includes_deferred_cache_completion']
        entry=dict(seconds=statistics.median(r['prefill_s']for r in rows),file=path.name,
            identity=[d[k]for k in ('manifest_sha256','prompt_token_sha256','checkpoint','dtype','prefill_mlp_chunk')],
            tokens=[r['output_token_ids']for r in rows],peak_allocated_bytes=max(r['memory_bytes']['peak_allocated']for r in rows))
        if kind=='profiles':
            keys={k for layer in rows[0]['prefill_leaf_profile']for k in layer}
            entry['gpu_seconds']={k:sum(layer.get(k,0)for layer in rows[0]['prefill_leaf_profile'])for k in keys}
        summary[kind].setdefault(series,{}).setdefault(n,{})[variant]=entry
    lines=['# H100 threshold prefill at 64K','',
        'Llama-3.1-8B-Instruct, BF16, batch one, H100 80GB. The c=16 per-region rule '
        'opens every eligible region, with the existing 1,024-entry limit. The '
        'exact initial 16,384-token field, local/sink attention, model/MLP chunks '
        'and cache updates remain fixed. Experiments use uv and source-snapshotting '
        'cluster-run, serially on one GPU.','']
    quality_headline_pos=None
    final=summary['models'].get(a.final_label,{}).get('65536',{})
    if set(final)=={'control','prefix-closed-tuned','dense','top8'}:
        assert len({tuple(v['identity'])for v in final.values()})==1
        c=final['control']['seconds'];s=final['prefix-closed-tuned']['seconds'];dense=final['dense']['seconds']
        safe=summary['models'].get('prefix-safe-v1',{}).get('65536',{}).get('prefix')
        if safe:assert safe['identity']==final['control']['identity']
        default=safe['seconds']if safe else None
        summary['selected_result']=dict(before_seconds=c,selection_only_seconds=default,experimental_seconds=s,dense_seconds=dense,
            experimental_reduction_pct=100*(1-s/c),experimental_change_vs_dense_pct=100*(s/dense-1))
        lines+=['## Selected result','',
            'Fresh matched runs: one warmup and three measured repetitions per method. '
            'This is an opt-in threshold-prefill implementation; serving defaults are unchanged.','',
            '| Method | 65,536-token prefill |','|---|---:|']
        for key,label in (('control','Previous threshold'),('prefix-closed-tuned','Direct-closed candidate (experimental)'),('dense','Dense'),('top8','Top-eight')):
            lines.append(f"| {label} | {final[key]['seconds']:.3f} s |")
        if safe:lines.insert(len(lines)-3,f"| Selection-only optimization | {default:.3f} s |")
        relation='slower' if s>dense else'faster'
        lines+=['',f'The direct-closed candidate is **{100*(1-s/c):.1f}% faster** than the control and **{abs(100*(s/dense-1)):.1f}% {relation} than dense**. '
            'The 16K exact initial field and uncapped c=16 selections are retained.']
        if safe:
            lines+=['',f'The selection-only implementation is **{100*(1-default/c):.1f}% faster** than the control and **{100*(default/dense-1):.1f}% slower than dense**. '
                'It keeps the previous coarse-attention and correction arithmetic. '
                'Within the opt-in fast threshold path, automatic dispatch now uses '
                'this prefix filter at 2,048–4,095 active region slots; larger '
                'problems retain the optimized prefix-scan packer. Set '
                '`LOD_H100_FAST_PREFIX_SELECT=0` to disable the filter.']
        quality_headline_pos=len(lines)
        lines+=['',
            'The experimental direct-closed path uses conservative prefix filtering, direct closed-summary '
            'normalization with shared FP32 log-counts, and cuBLAS BF16-to-FP32 value '
            'projection without KV duplication. Producer tiles are 64 queries × 128 '
            'regions with four warps; normalization handles one query at a time. '
            'Exact leaves retain the existing wide reducer. Full enable flags and '
            'commands are in `threshold-select64-final-v1-models-runs.json`.','']
    combination_path=root/'COMBINATION.json'
    if combination_path.exists():
        followup=json.loads(combination_path.read_text())
        if 'cwe32'in followup and 'online_cwe32'in followup:
            cwe=followup['cwe32'];online=followup['online_cwe32']
            pair=followup['models']['combine-online-v1-prefix-closed-tuned-65536-models']['median_s']
            single=followup['models']['combine-online-single-v1-prefix-closed-tuned-65536-models']['median_s']
            lines+=['','The [coarse-arithmetic ablation and combination follow-up](COMBINATION.md) '
                f'extends CWE to all 32 frozen examples: conservative {cwe["conservative_pct"]:.2f}%, '
                f'direct-closed {cwe["aggressive_pct"]:.2f}%, online-compatible combination '
                f'{online["percent"]:.2f}%. The combination recovers the eight checked '
                f'scores but takes {pair:.3f} s at 64K; a single-projection version takes '
                f'{single:.3f} s and retains the original CWE example\'s regression. Neither '
                'is promoted. These task-specific scores are not a full RULER average.','']
    normalization_path=root.parent/'closed-offset'/'SUMMARY.json'
    if normalization_path.exists():
        latest=json.loads(normalization_path.read_text())
        if latest.get('stage')=='complete' and 'scaled-fp16'in latest['models'] and 'scaled-fp16'in latest['quality']:
            seconds=latest['models']['scaled-fp16']['median_s'];score=latest['quality']['scaled-fp16']['percent']
            lines+=['','**Subsequent result:** the [normalization and offset follow-up](../closed-offset/REPORT.md) '
                f'keeps direct closed-summary attention using one FP16 coarse projection '
                f'and FP32 normalization afterward: **{seconds:.3f} s at 64K**, '
                f'**{score:.2f}% CWE recall on all 32 frozen examples at 32K**. '
                'Model/KV storage stay BF16. This supersedes the arithmetic '
                'combination\'s speed/quality tradeoff; it is not a full RULER average '
                'or a serving-default change.','']
    automatic=summary['models'].get('auto-v1',{})
    if automatic:
        lines+=['## Additional automatic-dispatch checks','',
            'One warmup and three measured repetitions, using the same frozen '
            'prompt manifest and full cache-completion scope. These are additional '
            'context checks; dense/top-eight controls were rerun only at 64K in '
            'this follow-up.','',
            '| Context tokens | Automatic threshold prefill |','|---:|---:|']
        for n,arms in sorted(automatic.items(),key=lambda v:int(v[0])):
            if 'auto'in arms:lines.append(f"| {int(n):,} | {arms['auto']['seconds']:.3f} s |")
        lines.append('')
    lines+=['<details>','<summary>Full tuning history and instrumented stage pilots</summary>','',
        '## Full-model timings','',
        'One warmup and three measured repetitions; median wall-clock prefill '
        'includes deferred LoD cache completion and excludes model loading, initial '
        'cache allocation and decode graph capture. MLP chunks are 8,192 tokens '
        'for every method; internal attention scratch remains 4,096 queries. '
        'All arms use expandable allocation segments.','',
        '| Series | Context tokens | Method | Prefill | Peak PyTorch allocation |',
        '|---|---:|---|---:|---:|']
    for series,lengths in summary['models'].items():
        for n,arms in lengths.items():
            for variant,d in arms.items():
                lines.append(f"| {series} | {int(n):,} | {variant} | {d['seconds']:.3f} s | {d['peak_allocated_bytes']/2**30:.2f} GiB |")
    lines+=['','Full-model results are kept separate from instrumented pilots and '
        'isolated kernel probes. Earlier invalid fusion timings are excluded even '
        'when the timing job exited successfully.','',
        '## Instrumented stage pilots','',
        'One measured repetition after warmup, summed CUDA-event time over all '
        'layers. These diagnose costs and are not the final timing medians.','',
        '| Series | Method | Prefill | Producer | Selection/packing | Closed projection | Exact including dispatch/reduction | Correction/merge |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for series,lengths in summary['profiles'].items():
        for n,arms in lengths.items():
            for variant,d in arms.items():
                g=d['gpu_seconds'];lines.append(f"| {series} | {variant} | {d['seconds']:.3f} s | {g.get('fast_produce',0):.3f} s | {g.get('fast_select',0):.3f} s | {g.get('fast_closed_project',0):.3f} s | {g.get('fast_exact',0):.3f} s | {g.get('fast_merge',0):.3f} s |")
    lines+=['','</details>','','## Implementations','',
        '- Bitmap ranks replace boolean-tile scans with population counts and word prefixes. '
        'They preserve complete selection and ascending region order.','- Sparse tile staging '
        'packs only selected pairs after tile counts, but reserves extra scratch capacity; '
        'it did not outperform the scan on the initial live fixture.','- Fused route merging '
        'combines exact-route reduction and coarse-summary replacement. Its rare numerical '
        'fallback uses 64-bit route indexing and checks groups of 128 flags before doing '
        'repair work. Bucketed merging uses 32/64/128-route fast paths and an uncapped fallback.',
        '- Prefix filtering reuses the producer\'s running routing partition. Since '
        'prefix mass is no greater than final mass, a region that fails the prefix '
        'threshold cannot pass the final threshold. A conservative FP32 boundary '
        'guard retains extra candidates. The final comparison uses the original '
        'saved FP32 score and final normalizer. Packing compacts nonempty bitmap '
        'words and reads only their candidate scores. No selection cap or CPU '
        'selected-count readback is introduced.','- Direct closed-summary projection '
        'omits full-coarse value attention in the producer, then normalizes only '
        'unopened summaries and projects their values with shared-GQA Triton or '
        'cuBLAS BF16-to-FP32 multiplication. It avoids subtraction and its numerical '
        'fallback. The final local/sink merge is unchanged.','- The recomputed-QK '
        'prototype omits the saved score matrix and fuses selection/packing with '
        'a second QK pass and closed-summary attention. Its timing is retained as '
        'a separate pilot, not substituted for the selected implementation.',
        '- Separate query-count reducers compact query IDs into 32/64/128-route '
        'buckets and retain a general uncapped reducer. Counts stay on the GPU. '
        'The standalone oracle records BF16 output rounding separately from '
        'differences against the BF16-rounded PyTorch reference.','',
        'These are implementation changes within the experimental threshold path. '
        'They do not change the exact initial field or establish subquadratic '
        'complete prefill/cache maintenance. The producer still visits every '
        'query–region pair; prefix filtering has no improved worst-case candidate bound.','',
        '## Correctness','',
        f"{len(summary['checks'])} successful independent synthetic/merge fixtures are retained. "
        'The dedicated cancellation fixture tests 1,560 repaired queries for each '
        'merge tile, and separately addresses a route value beyond 2^31 elements.','']
    for name,d in summary['verify'].items():
        lines.append(f"- `{name}`: {d['frontiers']} sampled frontiers across 32 layers, maximum absolute attention-output difference {d['max_abs_error']:.6f}, "
            f"{d['threshold_boundary_differences']} threshold-boundary differences and {d['outside_tolerance']} outside tolerance. Same-cache eager/graph logits match exactly.")
    if summary['quality']:
        tokpath=root.parent/'ruler-32k-panel'/'llama-tokenizer.json';tok=Tokenizer.from_file(str(tokpath))
        summary['input_sha256'][str(tokpath)]=hashlib.sha256(tokpath.read_bytes()).hexdigest()
        stops={tok.token_to_id(v)for v in ('<|end_of_text|>','<|eot_id|>')};assert None not in stops
        def first(ids):return ids[:next((i for i,v in enumerate(ids)if v in stops),len(ids))]
        qrows={}
        for name,d in summary['quality'].items():
            rows={}
            for r in d['records']:
                ids=first(r['output_token_ids']);text=cleanup(tok.decode(ids,skip_special_tokens=True))
                rows[(r['task'],r['source_row'])]=dict(input_sha256=r['input_token_sha256'],tokens=ids,
                    score=metric(r['task'],text,r['answers']))
            qrows[name]=rows
        summary['quality_comparisons']={}
        for name,rows in qrows.items():
            if '-control-ruler.json'not in name:continue
            prefix=name.removesuffix('-control-ruler.json')
            for other,vals in qrows.items():
                if other==name or not other.startswith(prefix):continue
                assert rows.keys()==vals.keys() and all(rows[k]['input_sha256']==vals[k]['input_sha256']for k in rows)
                equal=sum(rows[k]['score']==vals[k]['score']for k in rows);same=sum(rows[k]['tokens']==vals[k]['tokens']for k in rows)
                changes=[dict(task=k[0],source_row=k[1],before=rows[k]['score'],after=vals[k]['score'])for k in rows if rows[k]['score']!=vals[k]['score']]
                summary['quality_comparisons'][other]=dict(n=len(rows),same_score=equal,same_first_response_tokens=same,changed_scores=changes)
                lines+=['',f'Paired 32K RULER correctness panel `{other}`: {equal}/{len(rows)} first-response scores unchanged; '
                    f'{same}/{len(rows)} first-response token sequences identical. Two examples/task is a correctness panel, '
                    'not a new full RULER estimate.']
        if quality_headline_pos is not None:
            remarks=['','**Quality limitation:** these are eight-example 32K RULER checks, not full-panel estimates.']
            for name,d in summary['quality_comparisons'].items():
                label='Selection-only'if 'prefix-safe' in name else'Direct-closed candidate'
                remarks.append(f"{label}: {d['same_score']}/{d['n']} first-response scores unchanged; {d['same_first_response_tokens']}/{d['n']} first-response token sequences identical.")
                for c in d['changed_scores']:
                    remarks.append(f"{c['task']} source row {c['source_row']}: **{100*c['before']:.0f}% → {100*c['after']:.0f}%**. The direct-closed candidate remains experimental.")
            lines[quality_headline_pos:quality_headline_pos]=remarks
    lines+=['','## Live-input kernel probes','',
        'These intercept real layer-zero inputs in GPU memory, then intentionally '
        'stop execution after validation/tuning. No Q/K/V snapshot is saved. They '
        'exclude the rest of the model, exact attention and cache updates. Each '
        'selection candidate checks complete masks/counts, ascending selected IDs '
        'and unique gap-free expert offsets against the original pass.','']
    for name,d in summary['probes'].items():
        lines.append(f'- `{name}`: see raw fixture/records for every tested geometry and its validation.')
    pending=[j for j in status if j['state']!='finished'];failed=[j for j in status if j['state']=='finished'and j.get('exit_code')!=0]
    summary.update(stage='running'if pending else'complete',pending=pending,failed=failed,hardware=sorted(hardware))
    lines+=['','## Provenance and exclusions','',
        'Raw results and immutable executed-source hashes are retained alongside '
        'provenance logs and content-verified source-objects/. SUMMARY.json hashes '
        'every analyzed raw output. Failed, partial and known-invalid runs are '
        'excluded from timing tables.','',f'{len(pending)} pending; {len(failed)} failed; {len(summary["excluded"])} additional invalid completed timings.','']
    for j in failed:lines.append(f"- `{j['name']}`: exit {j['exit_code']}; see its provenance log.")
    for name,reason in summary['excluded'].items():lines.append(f'- `{name}`: {reason}')
    lines+=['','The initial fusion exposed 32-bit fallback indexing overflow; its '
        'timing is invalid. The first bitmap implementation hit a compiler gather '
        'layout assertion; broadcast/reshape fixes it. The initial dense retry '
        'incorrectly requested LoD-specific cache completion and is excluded; '
        'the corrected dense run uses the native cache.','']
    (root/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(stage=summary['stage'],models=list(summary['models']),verified=len(summary['verify']))))


if __name__=='__main__':main()
