"""Report matched direct-summary profiles, optimization trials and validation."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from benchmarks.h100_threshold_next_report import score_files


def publish(root,repo,final_label='final-v2'):
    status=json.loads((root/'status.json').read_text());hardware=set()
    summary=dict(profiles={},models={},checks={},verification={},quality={},numerics={},inputs_sha256={})
    dense_root=root.parent/'ruler-32k-panel';tokenizer=dense_root/'llama-tokenizer.json'
    old=root.parent/'closed-offset'
    cwe_reference=old/'threshold-offset-scaled-fp16-cwe32-v1-prefix-closed-tuned-ruler.json'
    fresh_name='threshold-directopt-baseline-cwe-v2-baseline-32768-quality'
    if any(j['name']==fresh_name and j['state']=='finished' and j['exit_code']==0 for j in status):
        cwe_reference=root/(fresh_name+'.json')
    reference,expected=score_files([cwe_reference,
        old/'threshold-offset-scaled-fp16-panel-v1-prefix-closed-tuned-ruler.json'],tokenizer)
    summary['quality_reference']=reference
    summary['inputs_sha256'][str(tokenizer.relative_to(repo))]=hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    for name in reference['files']:
        path=Path(name)
        summary['inputs_sha256'][str(path.relative_to(repo))]=reference['sha256'][path.name]
    for job in status:
        if job['state']!='finished' or job['exit_code']!=0:continue
        path=root/(job['name']+'.json');raw=path.read_bytes();d=json.loads(raw)
        summary['inputs_sha256'][path.name]=hashlib.sha256(raw).hexdigest()
        meta_path=root/'provenance'/job['id']/'run.json';meta=json.loads(meta_path.read_text())
        summary['inputs_sha256'][str(meta_path.relative_to(root))]=hashlib.sha256(meta_path.read_bytes()).hexdigest()
        hardware.add(meta['hardware']);assert len(hardware)==1
        if job['name'].endswith('-rounding'):
            assert d['stage']=='complete' and d['legacy_max_abs']==0 and d['fma_false_positive_decisions']>0
            summary['checks'][path.name]=d;continue
        if job['name'].endswith('-checks'):
            assert d['stage']=='complete'
            checks=[c for case in d['cases']for c in case['checks']]
            assert all(c['selection_outside_tolerance']==0 for c in checks)
            summary['checks'][path.name]=dict(cases=len(d['cases']),frontiers=len(checks));continue
        if job['name'].endswith('-probe'):continue
        if job['name'].endswith('-numerics'):
            def differs(v):
                if isinstance(v,dict):return v.get('different',0)>0 or any(differs(x)for x in v.values())
                if isinstance(v,list):return any(differs(x)for x in v)
                return False
            records=d['records'];summary['numerics'][path.name]=dict(calls=len(records)if isinstance(records,list)else 1,
                different=differs(records));continue
        if job['name'].endswith('-quality'):
            result,samples=score_files([path],tokenizer)
            for key,row in samples.items():
                assert key in expected and all(row[k]==expected[key][k]for k in
                    ('prompt_sha256','effective_prompt_sha256','input_token_sha256','answers'))
            assert all(prior is None or current==prior for current,prior in zip(result['identity'],reference['identity']))
            result.update(same_scores=sum(v['score']==expected[k]['score']for k,v in samples.items()),n=len(samples),
                variant=job['variant'],proxy_arithmetic=(d.get('prefill_threshold')or{}).get('direct_proxy_arithmetic'))
            summary['quality'][path.name]=result;continue
        rows=[r for r in d['records']if not r['warmup']]
        if job['name'].endswith('-verify'):
            assert len(rows)==1
            layers=rows[0]['prefill_fast_audit'];assert len(layers)==32 and all(layers)
            checks=[r for layer in layers for r in layer]
            assert all(r['selection_outside_tolerance']==0 for r in checks)
            assert rows[0]['graph_same_cache_checks'] and all(c['max_abs']==0 for c in rows[0]['graph_same_cache_checks'].values())
            summary['verification'][path.name]=dict(length=job['length'],frontiers=len(checks),selection_outside_tolerance=0,
                graph_same_cache_max_abs=0,max_reference_error=max(r['max_abs_output_difference']for r in checks),
                proxy_arithmetic=d['prefill_threshold'].get('direct_proxy_arithmetic'))
            continue
        kind='profiles' if job['name'].endswith('-profiles')else'models'
        assert len(rows)==(1 if kind=='profiles' else 3)
        assert d['dtype']=='bfloat16' and d['batch_size']==1 and d['prefill_mlp_chunk']==8192
        if job['variant']!='dense':assert d['prefill_includes_deferred_cache_completion']
        cfg=d['prefill_threshold']
        if job['variant']not in ('top8','dense'):
            assert cfg['rule']=='region' and cfg['mass_multiplier']==16 and cfg['max_regions']is None
            assert cfg['prefill_block_size']==16384 and cfg['closed_probability_precision']=='scaled-fp16'
        item=dict(file=path.name,variant=job['variant'],length=job['length'],
            median_s=statistics.median(r['prefill_s']for r in rows),samples_s=[r['prefill_s']for r in rows],
            peak_allocated_bytes=max(r['memory_bytes']['peak_allocated']for r in rows),configuration=cfg,
            identity=[d[k]for k in ('manifest_sha256','prompt_token_sha256','checkpoint','dtype','prefill_mlp_chunk')])
        if kind=='profiles':
            g=defaultdict(float)
            for layer in rows[0]['prefill_leaf_profile']:
                for key,value in layer.items():g[key]+=value
            audit=rows[0]['prefill_scaling_audit'];work=[r for layer in audit['layers']for r in layer['rows']]
            query_heads=sum(r['query_heads']for r in work)
            item.update(gpu_seconds=dict(g),
                regions_per_query_head=sum(r['opened_regions']for r in work)/query_heads,
                entries_per_query_head=sum(r['opened_entries']for r in work)/query_heads)
            if job['variant']=='top8':
                item['route_coarse_s']=sum(l['gpu_seconds'].get('producer_coarse_selection',0)for l in audit['layers'])
            else:item['route_coarse_s']=g['fast_produce']+g['fast_select']+g['fast_closed_project']
        summary[kind][path.name]=item
    for n in {v['length']for v in summary['models'].values()}:
        assert len({tuple(v['identity'])for v in summary['models'].values()if v['length']==n})==1
    final={}
    for v in summary['models'].values():
        if v['file'].startswith('threshold-directopt-'+final_label+'-'):
            final.setdefault(str(v['length']),{})[v['variant']]=v
    # Controls are unchanged by selector fixes and are retained from the same
    # frozen-input optimization series; only corrected candidate runs replace it.
    if final_label!='final-v1':
        for v in summary['models'].values():
            if v['file'].startswith('threshold-directopt-final-v1-')and v['variant']in ('baseline','top8','dense'):
                final.setdefault(str(v['length']),{}).setdefault(v['variant'],v)
    summary.update(final=final,hardware=sorted(hardware),
        pending=[j for j in status if j['state']!='finished'],
        failed=[j for j in status if j['state']=='finished' and j['exit_code']!=0])
    summary['stage']='running' if summary['pending']else'complete'
    summary['quality_final']={k:v for k,v in summary['quality'].items()if v['variant']=='select'and v['proxy_arithmetic']=='legacy-rounded-product'}
    verified={v['length']for v in summary['verification'].values()if v['proxy_arithmetic']=='legacy-rounded-product'}
    quality_tasks={task:task_result for run in summary['quality_final'].values()for task,task_result in run['tasks'].items()}
    summary['validated_final']=all({'baseline','select','top8','dense'}<=final.get(str(n),{}).keys()for n in (32768,65536,128000)) and verified=={32768,65536,128000} and all(
        quality_tasks.get(task,{}).get('n')==n and quality_tasks[task]['first']+1.e-9>=reference['tasks'][task]['first']
        for task,n in (('cwe',32),('niah_multikey_2',2),('niah_multikey_3',2),('qa_1',2)))
    lines=['# Direct-summary prefill optimization','',
        'Llama-3.1-8B-Instruct, BF16 model and KV cache, batch one, H100 80GB. '
        'Uncapped equal-per-region c=16, 1,024-entry eligibility, exact initial '
        '16,384-token field, local/sink attention and cache completion are fixed. '
        'GPU experiments use uv and immutable cluster-run snapshots with a serial GPU lock.','',
        '## Implementation','',
        'The selected candidate computes final threshold decisions and the unnormalized closed '
        'weights from the same saved FP32 score pass. It stores exact selection bits, compacts '
        'nonempty bitmap words, and packs only their set bits without rereading routing scores. '
        'The score producer no longer builds conservative prefix candidates. Query scaling '
        'explicitly rounds before adding log-counts, matching the legacy sparse selector; '
        'contracting those operations into an FMA flipped boundary decisions and reduced CWE. '
        'The single FP16 '
        'coarse projection, FP32 accumulation and division after projection are unchanged. '
        'Exact KV attention retains its existing tile and worker settings. No selected-count '
        'CPU readback, selection cap or exact-prefill-size change is introduced.','',
        'Within the existing opt-in direct scaled-FP16 configuration, the corrected selector '
        'is automatic. Set `LOD_H100_DIRECT_FUSE_SELECT=0` to restore the previous selector. '
        'Other prefill paths and serving defaults are unchanged.','',
        '## Uninstrumented full-model timings','',
        'One warmup and three measured repetitions; median wall time includes deferred '
        'cache completion, excludes initial cache allocation and graph capture. All arms '
        'use 8,192-token MLP chunks and the same frozen prompt at each length. '
        'Prefill is not whole-model graph replay. Controls are the completed final-v1 '
        'arms; the corrected selector is rerun as final-v2. The correction does not '
        'affect the control paths, and their input identities are checked.','',
        '| Context tokens | Method | Prefill | Peak PyTorch allocation |','|---:|---|---:|---:|']
    for n,arms in sorted(final.items(),key=lambda x:int(x[0])):
        for variant,v in arms.items():
            label={'baseline':'Previous direct FP16','select':'Optimized direct FP16','top8':'Top-eight','dense':'Dense'}[variant]
            lines.append(f"| {int(n):,} | {label} | {v['median_s']:.3f} s | {v['peak_allocated_bytes']/2**30:.2f} GiB |")
    if not final:lines+=['','Final matched timings are pending. No profile or isolated kernel probe substitutes for them.']
    lines+=['','<details>','<summary>Instrumented 128K trials, including regressions</summary>','',
        'One measured repetition after warmup; these diagnose costs and are not final '
        'timing medians. Score production/selection/coarse is an aggregate of successive '
        'stages, without double-counting their nested timers. For fused-merge trials, '
        'the final merge includes exact-route reduction. Top-eight’s final merge is '
        'not separately instrumented. Regressing trials are retained as experiments, '
        'not promoted implementations.','',
        '| Trial | Wall time | Scores + selection + coarse | Exact KV kernel | Separate exact reduction | Final merge | Mean opened regions | Mean exact entries |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,v in summary['profiles'].items():
        g=v['gpu_seconds'];label=name.removeprefix('threshold-directopt-').removesuffix('-128000-profiles.json')
        if (v['configuration']or{}).get('direct_proxy_arithmetic')=='legacy-rounded-product':label+=' (corrected rounding)'
        elif v['variant']in ('select','combined'):label+=' (rejected: FMA routing)'
        reduction='In final merge' if (v['configuration']or{}).get('direct_fused_merge')else f"{g['reduce']:.3f} s" if 'reduce'in g else 'Not separated'
        merge=f"{g['fast_merge']:.3f} s" if 'fast_merge'in g else 'Not separated'
        lines.append(f"| {label} | {v['median_s']:.3f} s | {v['route_coarse_s']:.3f} s | {g.get('kernel',0):.3f} s | "
            f"{reduction} | {merge} | {v['regions_per_query_head']:.2f} | {v['entries_per_query_head']:.1f} |")
    lines+=['','The flat fused selector, hoisted preparation, fused final reducer, and '
        'alternative exact tiles/worker counts are diagnostic trials. The selected '
        'implementation uses sparse bitmap packing and retains the original exact kernel.','',
        '</details>','','## Correctness and limited quality checks','',
        'Independent frontier checks compare selection and attention against the reference. '
        'Same-cache eager/graph decode logits must match exactly. These sample frontiers; '
        'they do not validate every prefill token independently.','']
    for name,v in summary['verification'].items():
        lines.append(f"- `{name}`: {v['frontiers']} sampled frontiers, zero selection differences outside tolerance, "
            f"exact eager/graph equality; maximum reference error {v['max_reference_error']:.7f}.")
    for name,v in summary['checks'].items():
        if v.get('kind')=='strict-proxy-rounding':
            lines.append(f"- `{name}`: {v['samples']} FP32 cases match independent eager PyTorch exactly; "
                f"the fused-FMA negative control changes {v['fma_differences']} results and flips {v['fma_false_positive_decisions']} strict decisions.")
    for name,v in summary['numerics'].items():
        lines.append(f"- `{name}`: {v['calls']} live calls; "
            f"{'a difference was found' if v['different']else'no differences in producer, selection, coarse or exact outputs'}.")
    lines+=['','Quality uses the frozen 32K cohort and ends at the first EOS/EOT. '
        'CWE is compared against the fresh previous-selector control; retrieval/QA '
        'retains the earlier identical-input reference. A CWE task score or six-example '
        'retrieval/QA check is not a 13-task macro average. The earlier previous-selector '
        'CWE result was 32.81%; the fresh control is 34.38%, with one example changing. '
        'That variation is not credited as a quality improvement from the optimization.','',
        '| Run | Method | Task | Examples | Score |','|---|---|---|---:|---:|']
    for name,v in summary['quality'].items():
        method='Previous selector control' if v['variant']=='baseline' else 'Corrected optimized selector' if v['proxy_arithmetic']=='legacy-rounded-product' else 'Rejected fused-FMA selector'
        for task,s in v['tasks'].items():lines.append(f"| {name} | {method} | {task} | {s['n']} | {s['first']:.2f}% |")
    for name,v in summary['quality'].items():
        lines+=['',f"{v['same_scores']}/{v['n']} individual scores unchanged from the previous FP16 candidate.",'']
    lines+=['','## Provenance and status','',
        f"{len(summary['pending'])} pending; {len(summary['failed'])} failed. Failed and partial results "
        'are excluded. Commands and logs are in provenance/; every executed-source file '
        'is saved by verified SHA-256 in source-objects/. SUMMARY.json hashes analyzed '
        'inputs. No complete-execution subquadratic bound or amortized decode timing is inferred.','']
    if summary['failed']:
        lines+=['The initial three winner-v1 model checks failed before inference because '
            'a configuration import loaded the serving Transformers integration too early. '
            'The dispatch setting was moved to a dependency-free module; the winner-v2 '
            'checks supersede those attempts. No timing or quality result is inferred from them.','']
    (root/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'REPORT.md').write_text('\n'.join(lines))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--final-label',default='final-v2');a=p.parse_args()
    d=publish(a.directory.resolve(),Path(__file__).resolve().parents[1],a.final_label)
    print(json.dumps(dict(stage=d['stage'],pending=len(d['pending']),failed=len(d['failed']),
        validated=d['validated_final'],final={n:{k:v['median_s']for k,v in arms.items()}for n,arms in d['final'].items()})))


if __name__=='__main__':main()
