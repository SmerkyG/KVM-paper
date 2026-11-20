"""Publish matched Qwen3 optimization timings, checks and frozen quality results."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from benchmarks.h100_threshold_next_report import score_files,QTASKS


def selected_recipe(cfg):
    return (cfg['rule'],cfg['mass_multiplier'],cfg['max_regions'],cfg['prefill_block_size'],
        cfg['closed_probability_precision'],cfg['direct_fused_selection'],cfg['closed_reduce_backend'],
        cfg['tc_reduce_rows'],cfg['tc_reduce_parts'],cfg.get('tc_reduce_ranks',32),cfg['fast_tile_queries'])


def publish(root,repo):
    status=json.loads((root/'status.json').read_text());hardware=set();verified=set()
    s=dict(models={},profiles={},verification={},checks={},quality={},inputs_sha256={},invalid_rope=[])
    def remember(p):s['inputs_sha256'][str(p.relative_to(repo))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for job in sorted(status,key=lambda j:j['id']):
        if job['state']!='finished'or job['exit_code']!=0:continue
        p=root/(job['name']+'.json');d=json.loads(p.read_text());remember(p)
        m=root/'provenance'/job['id']/'run.json';meta=json.loads(m.read_text());remember(m);hardware.add(meta['hardware'])
        for name,digest in meta['source_files'].items():
            if digest not in verified:
                assert hashlib.sha256((root/'source-objects'/digest).read_bytes()).hexdigest()==digest,name
                verified.add(digest)
        stage=job['name'].rsplit('-',1)[-1]
        if stage in ('quality','prepare'):continue
        if (d.get('qwen_yarn') or d.get('yarn')) and d.get('rope_max_position_embeddings') != 131072:
            s['invalid_rope'].append(p.name)
            continue
        if stage=='checks':
            assert d['stage']=='complete'
            if job['variant']=='rescale':
                assert all(c['different_bits']==0 for c in d['cases'])
                s['checks']['rescale']=dict(file=p.name,elements=sum(c['elements']for c in d['cases']),different_bits=0)
            else:
                assert len(d['records'])==6 and all(r['reference_token']==r['runtime_token']and r['mean_abs']<=.2 for r in d['records'])
                s['checks'][job['variant']]=dict(file=p.name,steps=6,max_mean_error=max(r['mean_abs']for r in d['records']))
            continue
        assert d['model']=='Qwen/Qwen3-8B'and d['dtype']=='bfloat16'and d['batch_size']==1 and d['prefill_mlp_chunk']==8192
        assert d['qwen_yarn']==(job['length']>32768)
        if job['variant']!='dense':assert d['prefill_includes_deferred_cache_completion']
        rows=[r for r in d['records']if not r['warmup']]
        if stage=='verify':
            assert len(rows)==1
            audit=[c for layer in rows[0]['prefill_fast_audit']for c in layer];graph=rows[0]['graph_same_cache_checks']
            assert audit and all(c['selection_outside_tolerance']==0 for c in audit)
            assert all(c.get('max_bf16_steps_above_absolute_tolerance',0)<=1 for c in audit)
            assert graph and all(c['max_abs']==0 for c in graph.values())
            s['verification'][str(job['length'])]=dict(file=p.name,frontiers=len(audit),selection_outside_tolerance=0,
                max_abs_reference_error=max(c['max_abs_output_difference']for c in audit),
                max_bf16_steps_above_absolute_tolerance=max(c.get('max_bf16_steps_above_absolute_tolerance',0)for c in audit),
                largest_reference_pairs=[c['max_abs_difference_output_reference']for c in audit if c['max_abs_output_difference']>.04],
                graph_same_cache_max_abs=0,configuration=d['prefill_threshold'])
            continue
        assert len(rows)==(1 if stage=='profiles'else 3)
        identity=[d[k]for k in ('manifest_sha256','prompt_token_sha256','checkpoint','dtype','prefill_mlp_chunk','qwen_yarn')]
        cfg=d['prefill_threshold']
        if job['variant']in('baseline','optimized'):
            assert cfg['mass_multiplier']==16 and cfg['max_regions']is None and cfg['prefill_block_size']==16384
        if job['variant']=='optimized':
            assert cfg['direct_fused_selection']and cfg['fast_score_reuse']and cfg['closed_probability_precision']=='scaled-fp16'
            assert cfg['closed_reduce_backend']=='tc-wide'and cfg['tc_reduce_parts']==3
            assert d['cache_update_optimizations']==dict(packed_max=True,fused_rms=False,fused_rescale=True)
            if stage=='models':assert cfg.get('tc_reduce_ranks',32)==32
        v=dict(file=p.name,variant=job['variant'],length=job['length'],identity=identity,configuration=cfg,
            cache_update_optimizations=d['cache_update_optimizations'],median_s=statistics.median(r['prefill_s']for r in rows),
            samples_s=[r['prefill_s']for r in rows],peak_allocated_gib=max(r['memory_bytes']['peak_allocated']for r in rows)/2**30)
        if stage=='profiles':
            gpu=defaultdict(float);aux=defaultdict(float)
            for layer in rows[0]['prefill_leaf_profile']:
                for k,x in layer.items():gpu[k]+=x
            audit=rows[0]['prefill_scaling_audit'];work=[r for layer in audit['layers']for r in layer['rows']];qh=sum(r['query_heads']for r in work)
            for layer in audit['layers']:
                for k,x in layer['gpu_seconds'].items():aux[k]+=x
            v.update(gpu_seconds=dict(gpu),auxiliary_gpu_seconds=dict(aux),
                regions_per_query_head=sum(r['opened_regions']for r in work)/qh,
                entries_per_query_head=sum(r['opened_entries']for r in work)/qh)
            s['profiles'][p.name]=v
        else:s['models'].setdefault(str(job['length']),{})[job['variant']]=v
    assert len(hardware)==1;s['hardware']=sorted(hardware)
    for arms in s['models'].values():assert len({tuple(v['identity'])for v in arms.values()})==1
    for n,check in s['verification'].items():
        assert selected_recipe(check['configuration'])==selected_recipe(s['models'][n]['optimized']['configuration'])
    prior=root.parent/'threshold-next';panel=root.parent/'ruler-32k-panel';tok=prior/'qwen-tokenizer.json';remember(tok)
    references={
        'Dense':panel/'qwen3-ruler-dense-selected.json',
        'Fixed top-eight prefill and decode':panel/'qwen3-ruler-two-tier-selected.json',
        'Threshold decode, top-eight prefill':prior/'threshold-next-qwen-full-v1-qwen-top8-prefill.json',
        'Previous threshold prefill and decode':prior/'threshold-next-qwen-prefill-full-v1-qwen-threshold-prefill.json'}
    expected=None
    for label,p in references.items():
        scored,samples=score_files([p],tok,True);remember(p)
        if expected is None:expected=samples;identity=scored['identity']
        assert scored['identity']==identity and set(samples)==set(expected)
        assert all(all(row[k]==expected[key][k]for k in ('prompt_sha256','effective_prompt_sha256','input_token_sha256','answers'))for key,row in samples.items())
        s['quality'][label]=scored
    for job in sorted(status,key=lambda j:j['id']):
        if not job['name'].endswith('-quality')or job['state']!='finished'or job['exit_code']!=0:continue
        p=root/(job['name']+'.json');scored,samples=score_files([p],tok,True)
        cfg=json.loads(p.read_text())['prefill_threshold']
        assert selected_recipe(cfg)==selected_recipe(s['models']['32768']['optimized']['configuration'])
        assert cfg['state_fused_rescale']and cfg['state_packed_max']and not cfg['state_fused_rms']
        assert scored['identity']==identity
        assert set(samples)==set(expected)and all(all(row[k]==expected[key][k]for k in
            ('prompt_sha256','effective_prompt_sha256','input_token_sha256','answers'))for key,row in samples.items())
        s['quality']['Optimized threshold prefill and decode']=scored
    tasks=s['quality'].get('Optimized threshold prefill and decode',{}).get('tasks',{})
    s['pending']=[j for j in status if j['state']!='finished']
    s['failed']=[j for j in status if j['state']=='finished'and j['exit_code']not in(0,-15,143)]
    s['cancelled']=[j for j in status if j['exit_code']in(-15,143)]
    s['stage']='running'if s['pending']else'complete'
    s['native_32k_validated'] = (
        {'dense','top8','baseline','optimized'} <= s['models'].get('32768', {}).keys()
        and '32768' in s['verification'] and {'rescale','native'} <= s['checks'].keys()
        and set(tasks) == set(QTASKS) and all(v['n'] == 32 for v in tasks.values()))
    s['validated_final']=(all({'dense','top8','baseline','optimized'}<=s['models'].get(str(n),{}).keys()for n in (32768,65536,128000))
        and set(s['verification'])=={'32768','65536','128000'}and set(s['checks'])=={'rescale','native','yarn'}
        and set(tasks)==set(QTASKS)and all(v['n']==32 for v in tasks.values()))
    s['quality_no_task_regression']=(all(v['first']>=s['quality']['Previous threshold prefill and decode']['tasks'][k]['first']
        for k,v in tasks.items())if tasks else None)
    lines=['# Qwen3-8B: Llama optimization transfer on H100','',
        '**Long-context configuration correction.** Earlier 64K and 128,000-token runs '
        'used effective 1.25x YaRN in Transformers 4.51.3. max_position_embeddings remained 40,960 '
        'and overrode the declared factor of four. This report excludes those runs. '
        'The 32K native-RoPE timings and quality panel remain valid. '
        'See the [corrected long-context investigation](../qwen-long/REPORT.md).','',
        'BF16 weights and KV, batch one, one H100 80GB. All GPU experiments use uv and immutable cluster-run snapshots.','',
        '## Full-model prefill','',
        'One warmup and three measured repetitions per arm. Medians include complete cache updates/completion, '
        'local/sink attention and the unchanged 16,384-token initial exact field. Initial cache allocation, loading '
        'and decode graph capture are excluded consistently. Prefill is eager with 8,192-token MLP chunks in every arm. '
        'The single decode step after prefill is a check; no amortized decode timing is reported.','',
        '| Context tokens | Dense | Top-eight | Previous threshold | Optimized threshold | Optimized versus dense |',
        '|---:|---:|---:|---:|---:|---:|']
    for n,arms in sorted(s['models'].items(),key=lambda x:int(x[0])):
        if not {'dense','top8','baseline','optimized'}<=arms.keys():continue
        delta=100*(arms['optimized']['median_s']/arms['dense']['median_s']-1)
        vals=[f'{arms[k]["median_s"]:.3f} s'for k in ('dense','top8','baseline','optimized')]
        lines.append('| '+' | '.join([f'{int(n):,}',*vals,f'{abs(delta):.1f}% '+('more time'if delta>0 else'less time')])+' |')
    lines+=['','Same frozen real-text Qwen-tokenized ProLong prefix across arms. Native Q/K norms are retained. '
        '32K uses native RoPE. Corrected longer runs use factor-four YaRN for every arm. '
        'Threshold routing remains uncapped equal-per-region c=16 '
        'with the existing 1,024-entry eligibility limit; it is not the per-entry bounded rule.','',
        '## Ported implementation','',
        '- Save native QK in FP32; share strict threshold selection and closed-weight preparation; compact and pack only the selected bitmap bits, without sorting or count readbacks.',
        '- Retain explicitly rounded proxy scaling and the globally shifted FP16 coarse projection with FP32 division. Model QK and exact KV remain BF16.',
        '- Use the three-part BF16 tensor-core exact-output reducer with an uncapped scalar fallback for larger frontiers.',
        '- Reuse the adaptive producer stages and scratch policy at 32K/64K; retain 4,096-query scratch throughout the longest context for memory headroom.',
        '- Reuse the exact packed max/index reduction on the original cuBLAS state scores. Fuse Qwen’s own coherence division instead of applying Llama’s cosine key normalization.',
        '- The decode attention kernel already shares the optimized materialized GQA implementation. These prefill times do not update the separate read-only graph or amortized decode tables.','',
        '| Context | Optimized peak PyTorch allocation | Measured samples |','|---:|---:|---|']
    for n,arms in sorted(s['models'].items(),key=lambda x:int(x[0])):
        if 'optimized'in arms:
            a=arms['optimized'];lines.append(f'| {int(n):,} | {a["peak_allocated_gib"]:.2f} GiB | '+', '.join(f'{x:.4f} s'for x in a['samples_s'])+' |')
    lines+=['','Peak allocation includes weights and excludes external driver/library allocations. '
        'Serving defaults and the routing/work policy are unchanged. No complete-execution subquadratic bound is claimed.','',
        '## Quality at 32K','',
        'The same four frozen tasks, 32 examples each, thinking disabled. Scores stop at the first EOS/EOT. '
        'This is not the complete 13-task RULER panel or a new RULER macro average. '
        + ('The [long-context follow-up](../qwen-long/REPORT.md) reports matched 128K quality separately.'
           if (root.parent/'qwen-long/REPORT.md').exists() else 'Quality at 64K/128K has not been rerun.'),'',
        '| Method | Multikey-2 | Multikey-3 | CWE | QA-1 |','|---|---:|---:|---:|---:|']
    for label,d in s['quality'].items():
        lines.append('| '+' | '.join([label,*[f'{d["tasks"][k]["first"]:.2f}%'for k in QTASKS]])+' |')
    if not tasks:lines+=['','The optimized quality panel has not completed; the table contains earlier runs only.']
    lines+=['','## Correctness and scope','',f'Complete measurement and check panel: **{s["validated_final"]}**.','']
    if tasks:
        changes={k:tasks[k]['first']-s['quality']['Previous threshold prefill and decode']['tasks'][k]['first']for k in QTASKS}
        lines+=['Task changes from the previous threshold run: '+', '.join(f'{k} {v:+.2f} pp'for k,v in changes.items())+'.','']
    if 'rescale'in s['checks']:lines.append(f'The Qwen cache rescale matches independent PyTorch arithmetic bit for bit over {s["checks"]["rescale"]["elements"]:,} BF16 outputs, including strided inputs. Native-RoPE model prefill and five decode steps match the Hugging Face reference argmax.')
    for n,v in sorted(s['verification'].items(),key=lambda x:int(x[0])):
        lines.append(f'\nAt {int(n):,} tokens: {v["frontiers"]} sampled frontiers, zero selection differences outside tolerance, '
            f'maximum reference attention error {v["max_abs_reference_error"]:.6f}, '
            f'at most {v["max_bf16_steps_above_absolute_tolerance"]} adjacent BF16 step where error exceeds 0.04, and exact same-cache eager/graph logits.')
    lines+=['','The initial Qwen validator stopped at an absolute difference of 0.0625 and its dependent quality job did not execute. '
        'The follow-up check retains the existing 0.04 absolute tolerance and additionally permits exactly one adjacent '
        'representable BF16 value for larger activations. Raw records retain both absolute errors and BF16 step counts. '
        'Failed or partial runs do not contribute timings or quality scores.','']
    if s['profiles']:
        lines+=['## Reducer tuning','',
            'These are instrumented one-repetition diagnostics. The 64-route tensor-core block was slower than '
            'the 32-route block in the 32K runs. Final timings and quality use 32, with the uncapped fallback.','',
            '| Context | TC routes | Prefill wall time | Exact-output reduction | Regions/query/head | Entries/query/head |',
            '|---:|---:|---:|---:|---:|---:|']
        for v in sorted(s['profiles'].values(),key=lambda v:(v['length'],v['configuration'].get('tc_reduce_ranks',32))):
            lines.append(f'| {v["length"]:,} | {v["configuration"].get("tc_reduce_ranks",32)} | {v["median_s"]:.3f} s | '
                f'{v["gpu_seconds"]["reduce"]:.3f} s | {v["regions_per_query_head"]:.2f} | {v["entries_per_query_head"]:.1f} |')
        lines.append('')
    lines+=['The [32K run commands](qwen-opt-v1-models-runs.json) record the opt-in configuration. '
        'The [archived long-context commands](qwen-opt-scale-v1-models-runs.json) retain the original configuration.','',
        'The [raw summary](SUMMARY.json), commands/logs in provenance/, and content-verified source objects retain all inputs. '
        f'This panel has {len(s["pending"])} pending, {len(s["failed"])} failed and {len(s["cancelled"])} cancelled jobs.','']
    (root/'SUMMARY.json').write_text(json.dumps(s,indent=2)+'\n');(root/'REPORT.md').write_text('\n'.join(lines))
    return s


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    s=publish(a.directory.resolve(),Path(__file__).resolve().parents[1])
    print(json.dumps({k:s[k]for k in ('stage','validated_final','quality_no_task_regression')}))


if __name__=='__main__':main()
