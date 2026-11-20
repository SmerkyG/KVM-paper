"""Publish completed threshold follow-ups, keeping failed and partial runs explicit.

CPU-only analysis. Raw fixed-length outputs are checked against tokenizer
decoding before first-response scoring. No incomplete panel receives an average.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from tokenizers import Tokenizer

TASKS=('niah_single_1','niah_single_2','niah_single_3','niah_multikey_1',
       'niah_multikey_2','niah_multikey_3','niah_multivalue','niah_multiquery','vt','fwe','cwe','qa_1','qa_2')
QTASKS=('niah_multikey_2','niah_multikey_3','cwe','qa_1')


def cleanup(text):
    for src,dst in ((' .','.'),(' ?','?'),(' !','!'),(' ,',','),(" ' ","'"),
                    (" n't","n't"),(" 'm","'m"),(" 's","'s"),(" 've","'ve"),(" 're","'re")):
        text=text.replace(src,dst)
    return text


def metric(task,text,answers):
    hits=[float(a.lower() in text.lower()) for a in answers]
    return max(hits) if task.startswith('qa_') else statistics.mean(hits)


def score_files(paths,tokenizer,qwen=False):
    tok=Tokenizer.from_file(str(tokenizer))
    stops={tok.token_to_id(t) for t in (('<|endoftext|>','<|im_end|>') if qwen else ('<|end_of_text|>','<|eot_id|>'))}
    assert None not in stops
    samples={};groups=defaultdict(list);identity=None
    for path in paths:
        data=json.loads(path.read_text());assert data['stage']=='complete'
        current=tuple(data.get(k,'dataset' if k=='prompt_format' else None) for k in ('model_name','model_revision','tokenizer_revision','model_precision','kv_precision','generation','prompt_format'))
        if identity is None:identity=current
        assert identity==current
        for row in data['records']:
            key=row['task'],row['source_row'];assert key not in samples
            ids=row['output_token_ids'];full=cleanup(tok.decode(ids,skip_special_tokens=True))
            assert full==row['prediction'],f'tokenizer mismatch: {path}/{key}'
            assert abs(metric(row['task'],full,row['answers'])-row['score'])<1.e-10
            end=next((i for i,v in enumerate(ids) if v in stops),len(ids))
            value=metric(row['task'],cleanup(tok.decode(ids[:end],skip_special_tokens=True)),row['answers'])
            samples[key]=dict(score=value,**{k:row.get(k,row['prompt_sha256']) for k in ('prompt_sha256','effective_prompt_sha256','input_token_sha256','answers')})
            groups[row['task']].append(value)
    tasks={task:dict(n=len(vals),first=100*statistics.mean(vals)) for task,vals in groups.items()}
    return dict(tasks=tasks,identity=identity,files=[str(p) for p in paths],
                sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),samples


def latest_direct_results(root,repo,baseline,tokenizer,hardware):
    """Import the completed follow-up, verifying its inputs and frozen quality cohort."""
    directory=root.parent/'closed-offset';path=directory/'SUMMARY.json'
    raw=path.read_bytes();data=json.loads(raw)
    assert data['stage']=='complete' and not data['pending'] and not data['failed']
    assert set(data['hardware'])==hardware,'latest experiments use different hardware'
    for name,digest in data['inputs'].items():
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,f'changed latest input: {name}'
    names=('threshold-offset-scaled-fp16-cwe32-v1-prefix-closed-tuned-ruler.json',
           'threshold-offset-scaled-fp16-panel-v1-prefix-closed-tuned-ruler.json')
    scored,samples=score_files([directory/name for name in names],tokenizer)
    reference,expected=baseline
    assert all(prior is None or actual==prior for actual,prior in zip(scored['identity'],reference['identity']))
    for key,row in samples.items():
        assert key in expected and all(row[k]==expected[key][k] for k in
            ('prompt_sha256','effective_prompt_sha256','input_token_sha256','answers')),'latest quality cohort differs'
    assert set(scored['tasks'])=={'cwe','niah_multikey_2','niah_multikey_3','qa_1'}
    assert all(v['n']==(32 if task=='cwe' else 2) for task,v in scored['tasks'].items())
    assert abs(scored['tasks']['cwe']['first']-data['quality']['scaled-fp16']['percent'])<1.e-9
    assert data['quality']['dense']['percent']==reference['tasks']['cwe']['first']
    retrieval=data['quality']['scaled-fp16-retrieval-qa']
    assert retrieval['n']==6 and abs(retrieval['percent']-
        statistics.mean(v['first'] for task,v in scored['tasks'].items() if task!='cwe'))<1.e-9
    scaling={str(n):data['models'] if n==65536 else data['scaling'][str(n)]
             for n in (32768,65536,128000)}
    verification={**data['scaling_verification'],'65536':data['verification']['scaled-fp16']}
    for n,arms in scaling.items():
        assert {'scaled-fp16','conservative','direct','dense'}<=arms.keys()
        for arm in arms.values():
            assert len(arm['samples_s'])==3 and abs(statistics.median(arm['samples_s'])-arm['median_s'])<1.e-9
        check=verification[n]
        assert check['selection_outside_tolerance']==0 and check['graph_same_cache_max_abs']==0
    return dict(source_summary=str(path.relative_to(repo)),source_sha256=hashlib.sha256(raw).hexdigest(),
        inputs_sha256=data['inputs'],prefill_scaling=scaling,verification=verification,
        cwe={k:data['quality'][k] for k in ('dense','conservative','direct','scaled-bf16','scaled-fp16')},
        quality=scored,retrieval_qa=retrieval,hardware=data['hardware'],
        stage=data['stage'],pending=data['pending'],failed=data['failed'],cancelled=data['cancelled'])


def latest_direct_lines(data,historical=False):
    scaling=data['prefill_scaling'];long=scaling['128000'];candidate=long['scaled-fp16']['median_s']
    lines=['## Previous direct-summary precision panel' if historical else '## Latest threshold prefill results','',
        ('The previous direct closed-summary candidate takes ' if historical else 'The latest direct closed-summary candidate takes ')+
        f"**{scaling['32768']['scaled-fp16']['median_s']:.3f} s at 32K**, "
        f"**{scaling['65536']['scaled-fp16']['median_s']:.3f} s at 64K** and "
        f'**{candidate:.3f} s at 128,000 tokens**. At the longest context, this is '
        f"{100*(1-candidate/long['dense']['median_s']):.1f}% less prefill time than dense and "
        f"{100*(1-candidate/long['conservative']['median_s']):.1f}% less than conservative threshold. "
        'Top-eight remains faster. The candidate is experimental and opt-in; serving defaults are unchanged.','',
        'These Llama runs use two-tier **equal-per-region c=16**, uncapped selection of '
        'eligible regions up to 1,024 entries, and the unchanged 16,384-token initial exact '
        'prefill field, local/sink attention and cache-update schedule. They do not use the '
        'per-entry work-bound rule. Model weights and KV storage remain BF16.','',
        'The candidate saves native QK scores in FP32, uses a conservative prefix candidate '
        'filter followed by exact threshold selection, and computes closed summaries directly. '
        'It forms unnormalized exp(logit − closed maximum) weights in FP32, uses FP16 operands '
        'for one coarse value projection with FP32 accumulation/output, then divides by the '
        'FP32 denominator. Query/KV tensors and exact attention remain BF16, and GQA heads share KV '
        'storage. There is no full-coarse-minus-opened subtraction or selected-count CPU readback.','',
        '### Full-model prefill scaling','',
        'Matched frozen prompts within each context, one warmup and three measured repetitions; '
        'median wall time includes deferred cache completion. Initial cache allocation, model '
        'loading and graph capture are excluded consistently. MLP chunks are 8,192 and '
        'attention scratch tiles are 4,096 queries. These are full-model prefill timings, '
        'not whole-model CUDA graph replay. The two-token decode checks are not amortized '
        'decode measurements. The longest prompt is 128,000, not 131,072 tokens.','',
        '| Context tokens | Dense | Top-eight | Conservative threshold | Previous direct BF16 | New direct, FP16 coarse operands |',
        '|---:|---:|---:|---:|---:|---:|']
    for n,arms in scaling.items():
        vals=[f"{arms[k]['median_s']:.3f} s" if k in arms else '—'
              for k in ('dense','top8','conservative','direct','scaled-fp16')]
        lines.append('| '+' | '.join([f'{int(n):,}',*vals])+' |')
    lines+=['','The 64K row reuses the matched direct-summary precision panel; top-eight was '
        'not rerun at 64K in that series. The 32K/128K conservative controls use automatic '
        'selection/packing dispatch; both direct variants use prefix filtering with the same '
        'producer tile. The precision change adds less than 0.5% time relative to previous '
        'direct BF16 at every length. Threshold remains slower than dense at 32K and 64K.','',
        '| Context tokens | Candidate peak PyTorch allocation | Sampled frontiers across 32 layers | Maximum reference attention error |',
        '|---:|---:|---:|---:|']
    for n,arms in scaling.items():
        check=data['verification'][n]
        lines.append(f"| {int(n):,} | {arms['scaled-fp16']['peak_allocated_bytes']/2**30:.2f} GiB | "
                     f"{check['frontiers']} | {check['max_abs_reference_error']:.7f} |")
    lines+=['','Peak allocation includes model weights and benchmark execution, but excludes '
        'driver/library memory outside PyTorch. All sampled frontiers have zero selection '
        'differences outside tolerance; same-cache eager/graph logits match exactly at every '
        'length. These checks sample frontiers rather than independently validating every token.','',
        '### Latest quality checks at 32K','',
        'CWE uses all 32 examples from the same frozen cohort as the full panel below, scored '
        'only before the first EOS/EOT. Baselines are earlier quality runs on identical inputs; '
        'their speed controls above are rerun. These are task scores, not RULER macro averages.','',
        '| Method | CWE, 32 examples |','|---|---:|']
    for key,label in (('dense','Dense'),('conservative','Conservative prefix threshold'),
                      ('direct','Previous direct BF16'),('scaled-bf16','Unnormalized BF16, FP32 division'),
                      ('scaled-fp16','New direct, FP16 coarse operands')):
        lines.append(f"| {label} | {data['cwe'][key]['percent']:.2f}% |")
    retrieval=data['retrieval_qa']
    lines+=['',f"The new candidate also scores {retrieval['percent']:.0f}% on six retrieval/QA examples: "
        'two each from multikey-2, multikey-3 and QA-1. All six scores are unchanged from '
        'conservative threshold. This small check does not establish full retrieval quality.','',
        '**The new direct-summary candidate has no full 13-task RULER average.** The older '
        '87.52% average below belongs to a different prefill implementation. Its 32.81% CWE '
        'score happens to equal the new candidate’s CWE score; the remaining tasks have not '
        'been rerun for this candidate. No new Qwen or 128K quality panel is implied.','',
        '### Provenance and scope','',
        'The [direct-summary report](../closed-offset/REPORT.md) contains the precision/offset '
        'ablations and numerical audit. Its [machine-readable summary](../closed-offset/SUMMARY.json), '
        'raw inputs, and [run commands and logs](../closed-offset/provenance/) '
        'remain in that directory. '
        'This report verifies their input hashes and rechecks the new quality outputs against '
        'the tokenizer and frozen cohort; SUMMARY.json records the imported panel separately '
        'under latest_direct.','',
        f"Latest panel: {len(data['pending'])} pending, {len(data['failed'])} failed, "
        f"{len(data['cancelled'])} cancelled jobs. The two optional probability-offset expansions "
        'were cancelled while queued; neither contributes a full-task quality score. '
        'Cache completion is included in speed measurements, but these results do not '
        'establish a subquadratic bound for complete prefill or amortized cache updates.','']
    return lines


def latest_optimization_results(root,repo,hardware):
    directory=root.parent/'direct-opt';path=directory/'SUMMARY.json'
    if not path.exists():return None
    raw=path.read_bytes();data=json.loads(raw)
    if not data.get('validated_final') or data['pending']:return None
    assert data['stage']=='complete' and set(data['hardware'])==hardware
    for name,digest in data['inputs_sha256'].items():
        source=repo/name if name.startswith('results/') else directory/name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,f'changed optimization input: {name}'
    return dict(source_summary=str(path.relative_to(repo)),source_sha256=hashlib.sha256(raw).hexdigest(),
        quality=data['quality_final'],quality_reference=data['quality_reference'],
        **{k:data[k]for k in ('final','verification','inputs_sha256','hardware')})


def latest_optimization_lines(data,historical=False):
    final=data['final'];long=final['128000'];new=long['select']['median_s'];old=long['baseline']['median_s']
    lines=['## Previous selector optimization'if historical else'## Latest optimized threshold prefill','',
        f'The optimized direct-FP16 path takes **{new:.3f} s at 128,000 tokens**, '
        f'**{100*(1-new/old):.1f}% less time** than the freshly rerun previous direct-FP16 path. '
        f"It remains {100*(new/long['top8']['median_s']-1):.1f}% slower than top-eight. "
        'These are uninstrumented full-model measurements, including cache completion.','',
        'Final threshold decisions and closed weights now share one saved-FP32-score pass. '
        'The kernel writes exact selection bits; separate word compaction and sparse packing '
        'read those bits without rescanning routing scores. Query scaling explicitly rounds '
        'before adding log-counts to preserve the old selector’s boundary decisions. '
        'The FP16 coarse projection and '
        'FP32 division afterward remain unchanged. The original exact-attention geometry and '
        'worker count were faster than the alternatives tested and are retained.','',
        'Same Llama-3.1-8B-Instruct checkpoint, BF16 model/KV, batch one, H100 80GB, frozen '
        'prompt per length, one warmup and three measured repetitions; medians below. '
        'Uncapped equal-per-region c=16, 1,024-entry region eligibility, the 16,384-token '
        'initial exact field, local/sink attention and cache-update scope are unchanged. '
        'Initial allocation, model loading and graph capture are excluded for every arm; '
        'MLP chunks remain 8,192. The longest prompt is 128,000 tokens.','',
        '| Context tokens | Dense | Top-eight | Previous direct FP16 | Optimized direct FP16 | Threshold time reduction |',
        '|---:|---:|---:|---:|---:|---:|']
    for n,arms in sorted(final.items(),key=lambda x:int(x[0])):
        vals=[f"{arms[k]['median_s']:.3f} s" for k in ('dense','top8','baseline','select')]
        reduction=100*(1-arms['select']['median_s']/arms['baseline']['median_s'])
        lines.append('| '+' | '.join([f'{int(n):,}',*vals,f'{reduction:.1f}%'])+' |')
    quality={t:v for run in data['quality'].values()for t,v in run['tasks'].items()}
    unchanged=sum(run['same_scores']for run in data['quality'].values());total=sum(run['n']for run in data['quality'].values())
    lines+=['',f"The new 32K quality checks score **{quality['cwe']['first']:.2f}% on all 32 CWE examples** "
        'and 100% on two examples each of multikey-2, multikey-3 and QA-1. '
        f'{unchanged}/{total} individual scores are unchanged from the previous FP16 candidate. '
        f"The fresh previous-selector CWE control scores {data['quality_reference']['tasks']['cwe']['first']:.2f}%; "
        'its earlier 32.81% result differs on one example. That variation is not attributed '
        'to the optimization. '
        '**These checks are not a new full 13-task RULER average.**','',
        f"At 128K, peak PyTorch allocation is {long['select']['peak_allocated_bytes']/2**30:.2f} GiB "
        '(including weights; excluding external driver/library allocations). Independent '
        'frontier checks at all three lengths have zero selection differences outside '
        'tolerance and exact same-cache eager/graph logits. Prefill itself is not whole-model '
        'graph replay; two decode tokens are checks rather than amortized decode timings.','',
        'The optimized selector is automatic within the experimental direct scaled-FP16 '
        'path; `LOD_H100_DIRECT_FUSE_SELECT=0` restores the previous selector. Other prefill '
        'paths and serving defaults are unchanged. The '
        '[profiling and optimization report](../direct-opt/REPORT.md) records stage timings, '
        'regressing trials, validation and source provenance; '
        '[SUMMARY.json](../direct-opt/SUMMARY.json) records the matched raw inputs. '
        'No complete-execution subquadratic bound is inferred.','']
    return lines


def latest_short_prefill_results(root,repo,hardware):
    directory=root.parent/'direct-opt-v3';path=directory/'SUMMARY.json'
    if not path.exists():return None
    raw=path.read_bytes();data=json.loads(raw)
    if not data.get('validated_final') or data['pending']:return None
    assert data['stage']=='complete' and set(data['hardware'])==hardware
    for name,digest in data['inputs_sha256'].items():
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,f'changed short-prefill input: {name}'
    return dict(source_summary=str(path.relative_to(repo)),source_sha256=hashlib.sha256(raw).hexdigest(),
        **{k:data[k] for k in ('final','quality','quality_reference','verification','independent_checks','inputs_sha256','hardware')})


def latest_short_prefill_lines(data, *, full_quality=False):
    final=data['final'];long=final.get('128000')
    lines=['## Latest optimized threshold prefill','',
        ('The new matched retime beats dense at 64K and 128,000 tokens and narrows the gap at 32K. 'if long else
         'The new matched retime beats dense at 64K and narrows the gap at 32K. ')+
        'These are eager full-model prefill medians from one warmup and three measured repetitions, '
        'including complete cache updates and completion. Initial cache allocation, loading and '
        'decode graph capture are excluded consistently. These are not attention-only or graph-replay timings.','',
        '| Context tokens | Dense | Top-eight | Previous optimized threshold | New threshold | New versus dense |',
        '|---:|---:|---:|---:|---:|---:|']
    for n,arms in sorted(final.items(),key=lambda x:int(x[0])):
        new=arms['optimized']['median_s'];dense=arms['dense']['median_s'];delta=100*(new/dense-1)
        values=[f'{arms[k]["median_s"]:.3f} s'for k in ('dense','top8','select','optimized')]
        lines.append('| '+' | '.join([f'{int(n):,}',*values,f'{abs(delta):.1f}% '+('more time'if delta>0 else'less time')])+' |')
    tasks={task:row for run in data['quality'].values()for task,row in run['tasks'].items()}
    lines+=['','Same BF16 Llama-3.1-8B-Instruct checkpoint, H100 80GB, batch one and frozen prompts. '
        'Uncapped equal-per-region c=16, 1,024-entry eligibility limit, the initial exact 16,384-token '
        'field, local/sink attention and cache-update schedule are unchanged. This is not the per-entry bounded rule.','',
        'The saved-score selector and globally shifted FP16 coarse projection remain intact. '
        'Small exact frontiers now reduce with tensor cores using three BF16 parts of each FP32 '
        'weight, with the original uncapped reducer above 32 routes. Fused cache-key normalization '
        'reproduces PyTorch 2.5 rounding, and a packed max/index reduction consumes the original '
        'cuBLAS state scores. The fresh top-eight controls use the same cache improvements. '
        'Adaptive scratch tiles use 16,384/8,192/4,096 queries as region counts grow; producer '
        'pipelining uses two stages through 2,048 regions and one above that. '
        'At 128,000 tokens the scratch tile is explicitly 4,096 throughout for memory headroom; '
        'the initial exact field remains 16,384 tokens.','',
        f'The full 32-example CWE check scores **{tasks["cwe"]["first"]:.2f}%**, versus '
        f'**{data["quality_reference"]["tasks"]["cwe"]["first"]:.2f}%** for the fresh previous-selector control. '
        'Two examples each of multikey-2, multikey-3 and QA-1 retain 100%. '
        'Faster online-coarse and tiled-cache variants lost quality and were rejected.','',
        'Independent checks match 4.4 million BF16 normalized values and all tested state max/index results '
        '(NaN payload bits excluded). Sampled model frontiers match selection outside tolerance at all tested '
        'lengths, and same-cache eager/graph decode logits match exactly.','',
        f'Peak PyTorch allocation is {final["32768"]["optimized"]["peak_allocated_gib"]:.2f} GiB at 32K and '
        f'{final["65536"]["optimized"]["peak_allocated_gib"]:.2f} GiB at 64K, including weights and excluding external library/driver memory. '
        'The 32K result remains slower than dense. '+
        (f'The new 128,000-token configuration peaks at {long["optimized"]["peak_allocated_gib"]:.2f} GiB and has not had a new 128K quality panel. '
         'The longest prompt is 128,000, not 131,072 tokens. 'if long else
         'This configuration has not been retimed at 128,000 tokens; the older 128K numbers below use the preceding implementation. ')+
        'Serving defaults are unchanged. '
        'No subquadratic bound for complete prefill or amortized cache updates is claimed.','',
        'See the [optimization retime report](../direct-opt-v3/REPORT.md) for the full configuration, '
        'raw samples, quality checks, rejected trials and immutable source provenance.','']
    if full_quality:
        lines += ['The [full 32K RULER panel](../direct-opt-v3/RULER_32K.md) uses the same source snapshot '
                  'and kernel settings as the accepted prefill timing.', '']
    else:
        lines += ['These targeted checks cover a subset of the 32K RULER panel.', '']
    return lines


def latest_qwen_transfer(root,repo,hardware):
    path=root.parent/'qwen-opt'/'SUMMARY.json'
    if not path.exists():return None
    raw=path.read_bytes();data=json.loads(raw)
    if not data.get('native_32k_validated', data.get('validated_final')) or data['pending']:return None
    assert data['stage']=='complete'and set(data['hardware'])==hardware
    for name,digest in data['inputs_sha256'].items():
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,f'changed Qwen transfer input: {name}'
    return dict(source_summary=str(path.relative_to(repo)),source_sha256=hashlib.sha256(raw).hexdigest(),
        **{k:data[k]for k in ('models','verification','checks','quality','quality_no_task_regression','inputs_sha256','hardware')})


def publish(root,repo):
    status=json.loads((root/'status.json').read_text());by_name={r['name']:r for r in status}
    summary=dict(stage='partial',scaling={},scaling_audit={},quality={},qwen={},graphs={},files_sha256={})
    hardware=set()
    def load(name):
        job=by_name.get(name.removesuffix('.json'))
        if job is None or job['state']!='finished' or job.get('exit_code')!=0 or not (root/name).exists():return None
        metadata=json.loads((root/'provenance'/job['id']/'run.json').read_text())
        hardware.add(metadata['hardware'])
        assert len(hardware)==1,'experiment hardware differs'
        path=root/name;raw=path.read_bytes();summary['files_sha256'][name]=hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    def quality(label,names,tokenizer,qwen=False,baseline=None):
        if any(load(name) is None for name in names):return None
        result,samples=score_files([root/n for n in names],tokenizer,qwen)
        if baseline is not None:
            reference,expected=baseline
            for actual,prior in zip(result['identity'],reference['identity']):
                assert prior is None or actual==prior,'model or prompt metadata differs'
            for key,row in samples.items():
                assert key in expected
                assert all(row[k]==expected[key][k] for k in ('prompt_sha256','effective_prompt_sha256','input_token_sha256','answers'))
        dest=summary['qwen'] if qwen else summary['quality'];dest[label]=result
        if not qwen and set(result['tasks'])==set(TASKS) and all(v['n']==32 for v in result['tasks'].values()):
            result['macro_average']=statistics.mean(t['first'] for t in result['tasks'].values())
        return result,samples

    base=repo/'results/h100-baselines/ruler-32k-panel'
    lt=base/'llama-tokenizer.json';qt=root/'qwen-tokenizer.json'
    llama_dense=score_files([base/n for n in ('ruler-32k-dense-full.json','ruler-32k-dense-missing8.json')],lt)
    llama_retro=score_files([base/n for n in ('ruler-llama-retroinfer-32k-selected.json','ruler-llama-retroinfer-32k-missing5.json')],lt)
    for label,pair in (('Dense',llama_dense),('RetroInfer',llama_retro)):
        d,rows=pair;assert set(d['tasks'])==set(TASKS) and all(v['n']==32 for v in d['tasks'].values())
        d['macro_average']=statistics.mean(t['first'] for t in d['tasks'].values());summary['quality'][label]=d
    quality('Optimized threshold prefill and decode',['threshold-next-llama-full-v1-llama-threshold-prefill.json'],lt,baseline=llama_dense)
    qdense=score_files([base/'qwen3-ruler-dense-selected.json'],qt,True)
    summary['qwen']['Dense']=qdense[0]
    old=score_files([base/'qwen3-ruler-two-tier-selected.json'],qt,True)
    assert old[0]['identity']==qdense[0]['identity'];summary['qwen']['Fixed top-eight prefill and decode']=old[0]
    quality('Threshold decode, top-eight prefill',['threshold-next-qwen-full-v1-qwen-top8-prefill.json'],qt,True,qdense)
    quality('Threshold decode and prefill',['threshold-next-qwen-prefill-full-v1-qwen-threshold-prefill.json'],qt,True,qdense)
    for mode,label in (('multipole','Multipole 1:16'),('squeezed-multipole','Squeezed subset 1:8')):
        for budget in (512,2048):
            name=f'threshold-next-author-full-v1-qwen-{mode}-{budget}.json'
            if load(name) is None:name=f'threshold-next-author-v2-qwen-{mode}-{budget}.json'
            quality(f'{label}, budget {budget}',[name],qt,True,qdense)

    for n in (32768,65536,128000):
        arm={};identity=None
        for impl in ('dense','top8','closed','fast'):
            name=f'threshold-next-scaling-v1-{impl}-m1-{n}.json';data=load(name)
            if n==128000 and impl in ('closed','fast'):
                follow=f'threshold-next-scaling-memory-v2-{impl}-m1-{n}.json';latest=load(follow)
                if latest is not None:name,data=follow,latest
            if n==128000:
                follow=f'threshold-next-scaling-cleanup-v3-{impl}-m1-{n}.json';latest=load(follow)
                if latest is not None:name,data=follow,latest
                follow=f'threshold-next-scaling-allocator-v4-{impl}-m1-{n}.json';latest=load(follow)
                if latest is not None:name,data=follow,latest
            if data is None:continue
            rows=[x for x in data['records'] if not x['warmup']]
            assert len(rows)==3 and data['batch_size']==1 and data['dtype']=='bfloat16'
            assert data['prompt_tokens']==n and data['prefill_mlp_chunk']==8192
            current=(data['manifest_sha256'],data['prompt_token_sha256'],data['checkpoint'],data['dtype'])
            if identity is None:identity=current
            assert identity==current,'prefill scaling inputs differ'
            if impl!='dense':
                assert data['prefill_includes_deferred_cache_completion']
            if impl in ('closed','fast'):
                assert data['prefill_threshold']['prefill_block_size']==16384
                assert data['prefill_threshold']['mass_multiplier']==16 and data['prefill_threshold']['max_regions'] is None
            value=statistics.median(x['prefill_s'] for x in rows)
            assert abs(value-data['median_prefill_s'])<1.e-9
            arm[impl]=dict(seconds=value,file=name,samples=[x['prefill_s'] for x in rows],
                           cache_completion_s=[x.get('prefill_cache_completion_s') for x in rows],
                           cache_initialization_s=[x['cache_init_s'] for x in rows],identity=current,
                           memory_bytes=[x.get('memory_bytes') for x in data['records']],
                           allocator='expandable_segments:True' if 'allocator-v4' in name else 'native default')
        if arm:summary['scaling'][str(n)]=arm

    for n in (32768,128000):
        arm={}
        for impl in ('top8','fast'):
            name=f'threshold-next-scaling-audit-v2-{impl}-m1-{n}.json';data=load(name)
            if data is None:continue
            measured=[x for x in data['records'] if not x['warmup']];assert len(measured)==1
            row=measured[0];audit=row['prefill_scaling_audit'];assert audit['instrumented']
            records=[x for layer in audit['layers'] for x in layer['rows']]
            query_heads=sum(x['query_heads'] for x in records)
            assert query_heads==(n-16384)*32*32
            gpu=defaultdict(float)
            for layer in audit['layers']:
                for key,value in layer['gpu_seconds'].items():gpu[key]+=value
            leaf=defaultdict(float)
            for layer in row['prefill_leaf_profile']:
                for key,value in layer.items():leaf[key]+=value
            arm[impl]=dict(query_heads=query_heads,
                regions_per_query_head=sum(x['opened_regions'] for x in records)/query_heads,
                entries_per_query_head=sum(x['opened_entries'] for x in records)/query_heads,
                maximum_open_regions=max(x['maximum_regions'] for x in records),
                maximum_route_capacity=max(x['route_capacity'] for x in records),
                maximum_leaf_output_capacity_bytes=max(x['query_heads']*x['route_capacity']*128*2 for x in records),
                gpu_seconds=dict(gpu),leaf_gpu_seconds=dict(leaf),file=name)
        if arm:summary['scaling_audit'][str(n)]=arm

    for n in (32768,131072):
        records={}
        for kind in ('author','lod','threshold'):
            parent=load(f'threshold-next-real-v2-real-qwen-{n}.json')
            if parent is None:break
            path=root/f'threshold-next-real-v2-real-qwen-{n}.json.{kind}.json'
            records[kind]=json.loads(path.read_text())
            summary['files_sha256'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        if len(records)!=3:continue
        assert all(records[k]['metadata']==records['author']['metadata'] for k in records)
        if n > 32768 and records['author']['metadata'].get('rope_max_position_embeddings') != 131072:
            summary.setdefault('invalid_rope_graphs', []).append(n)
            continue
        for d in records.values():
            assert all(x['graph_eager_max_abs']==0 and x['replay_max_abs']==0 for x in d['records'])
        graph={}
        for scope in ('remote','local'):
            author=records['author']['records'];lod=records['lod']['records'];threshold=records['threshold']['records']
            pick=lambda seq,**kw:next(x for x in seq if all(x.get(k)==v for k,v in kw.items()))
            t=pick(threshold,scope=scope);assert t['cache_updates'] is False and t['current_token'] is False
            graph[scope]=dict(dense=pick(author,method='dense')['median_ms'],
                multipole=pick(author,method='multipole',tokens_per_centroid=16,budget=512,scope=scope)['median_ms'],
                squeezed=pick(author,method='squeezed-variant',tokens_per_centroid=8,budget=512,scope=scope)['median_ms'],
                top8=pick(lod,mode='two-tier',scope=scope)['median_ms'],threshold=t['median_ms'],
                mean_regions=statistics.mean(t['regions_per_head']),mean_entries=statistics.mean(t['exact_entries_per_head']))
        summary['graphs'][str(n)]=graph
    pending=[r for r in status if r['state']!='finished'];failed=[r for r in status if r['state']=='finished' and r.get('exit_code')!=0]
    summary.update(pending=pending,excluded_runs=failed,hardware=sorted(hardware))
    summary['stage']='running' if pending else 'complete'
    summary['latest_direct']=latest_direct_results(root,repo,llama_dense,lt,hardware)
    optimization=latest_optimization_results(root,repo,hardware)
    if optimization is not None:summary['latest_optimization']=optimization
    short_prefill=latest_short_prefill_results(root,repo,hardware)
    if short_prefill is not None:summary['latest_short_prefill']=short_prefill
    qwen_transfer=latest_qwen_transfer(root,repo,hardware)
    if qwen_transfer is not None:
        summary['latest_qwen_transfer']=qwen_transfer
        label='Optimized threshold prefill and decode'
        summary['qwen'][label]=qwen_transfer['quality'][label]
    full_quality = (root.parent/'direct-opt-v3'/'RULER_32K.md').is_file()
    lines=['# Threshold prefill and real Qwen follow-up','',
        'Llama-3.1-8B-Instruct and Qwen3-8B, BF16, batch one, one H100 80GB. '
        'GPU experiments use uv and source-snapshotting cluster-run; commands, logs and '
        'source hashes are retained in provenance/ and source-objects/. Serving defaults are unchanged.','',
        ('The latest Llama prefill scaling and its 32K RULER panel are presented first. '
         if full_quality else 'The latest Llama prefill scaling and targeted quality checks are presented first. ')
        + 'Earlier Llama and Qwen runs retain their original implementations.','']
    if short_prefill is not None:
        lines+=latest_short_prefill_lines(short_prefill, full_quality=full_quality)
        lines+=['<details>','<summary>Previous selector optimization, including the earlier 128K retime</summary>','']
    if optimization is not None:
        lines+=latest_optimization_lines(optimization,historical=short_prefill is not None)
    if short_prefill is not None:lines+=['</details>','']
    if optimization is not None:
        lines+=['<details>','<summary>Previous direct-summary precision and scaling panel</summary>','']
    lines+=latest_direct_lines(summary['latest_direct'],historical=optimization is not None)
    if optimization is not None:lines+=['</details>','']
    lines+=['<details>','<summary>Historical prefill experiments and stage audits</summary>','',
        'These tables describe earlier implementations, not the direct-summary candidate above. '
        'The subsequent [selection/packing optimization](../selectpack/REPORT.md) '
        'reduces 128,000-token fused threshold prefill from 14.269 s to 13.437 s '
        'in a fresh matched pair, keeping c=16 selection and the exact initial '
        '16,384-token field. The tables below preserve the preceding experiments.','',
        '## Historical prefill optimization','',
        'The exact initial prefill chunk remains 16,384 tokens. Selection remains uncapped '
        'per-region c=16, with native-query exact attention, closed summaries, local and sink fields '
        'and cache completion. There is no selected-count CPU readback. These experiments do not '
        'change the mathematical routing rule or establish subquadratic complete execution.','',
        'Grouped atomic reservations, smaller query/region tiles, a direct closed-summary streaming '
        'pass, and a dimension-split exact-output reduction passed the independent frontier checks '
        'but did not improve full-model prefill. Reducing the closed selector from eight to four '
        'warps gave a small pilot improvement; the matched scaling panel checks its repeatability. '
        'The longer-context fused producer keeps eight warps.','',
        '| Pilot, one warmup + two timed repetitions | 32K prefill | 64K prefill |',
        '|---|---:|---:|']
    pilots=[('Existing control','v1-closed-m1','v1-fast-m1'),
            ('Four selection warps','reduce-v1-closed-w4','reduce-v1-fast-w4'),
            ('Grouped reservations, four queries','v1-closed-m4','v1-fast-m4'),
            ('Tiled selection, 32 queries','tiled-v1-tiled-m32','tiled-v1-tiled-m32'),
            ('Streaming closed-summary pass, 32 queries','stream-v1-stream-m32','stream-v1-stream-m32'),
            ('Dimension-split exact-output reduction','reduce-v2-closed-split','reduce-v2-fast-split')]
    for label,p32,p64 in pilots:
        vals=[]
        for prefix,n in ((p32,32768),(p64,65536)):
            d=load(f'threshold-next-{prefix}-{n}.json');vals.append(f"{d['median_prefill_s']:.3f} s" if d else 'Pending')
        lines.append('| '+' | '.join([label,*vals])+' |')
    lines+=['','## Historical full-model prefill scaling','',
        'Same frozen prompt within each context, same checkpoint and BF16 runtime. One warmup '
        'and three measured repetitions; median wall time, including LoD deferred cache completion '
        'but excluding initial cache allocation for every method. MLP chunks are 8,192 for every '
        'arm at every length. The longest full-model prompt is 128,000 tokens; the Qwen graph '
        'fixture below is separately 131,072. Two-token generation is used only to finalize '
        'prefill and capture; its decode timing is not used as an amortized performance result.','',
        '| Tokens | Dense | Top-eight prefill | Closed threshold | Fused threshold | Best completed threshold / dense |',
        '|---|---:|---:|---:|---:|---:|']
    for n in (32768,65536,128000):
        arm=summary['scaling'].get(str(n),{});vals=[f"{arm[k]['seconds']:.3f} s" if k in arm else
            ('OOM; allocator retry pending' if n==128000 and k=='fast' else 'Pending') for k in ('dense','top8','closed','fast')]
        ratio='Pending'
        completed=[arm[k]['seconds'] for k in ('closed','fast') if k in arm]
        if 'dense' in arm and completed:ratio=f"{min(completed)/arm['dense']['seconds']:.3f}×"
        lines.append('| '+' | '.join([f'{n:,}',*vals,ratio])+' |')
    lines+=['','The final column is a latency ratio: below 1 means faster than dense. '
        'The initial 128,000-token threshold jobs, including the smaller-scratch retries, '
        'completed their warmup prefill but exhausted memory on the second repetition. '
        'Benchmark cleanup inside run_once ran while local pool/engine references still '
        'kept callback cycles reachable. The cleanup rerun collects after that frame returns '
        'and before allocating the next cache, outside timing, and records per-repetition '
        'allocated and peak memory. It restores the original 4,096-query scratch tile; the '
        '16,384-token exact prefill field, routing rule and update schedule are unchanged. '
        'Closed threshold now completes all repetitions. Fused threshold initially failed after '
        'cleanup: a 16 GiB allocation could not fit despite 13.30 GiB of reserved unused '
        'memory, consistent with allocator fragmentation. The separate expandable-segment retry '
        'now completes one warmup and three measured repetitions at 128K; it changes memory '
        'allocation, not attention selection or arithmetic. Native dense/top-eight controls retain '
        'their default allocator, as recorded in SUMMARY.json. Both threshold '
        'implementations are attempted at all lengths; failed runs contribute no time. '
        'Choosing the lower completed time does not change exact-prefill size or routing budget.','',
        '### 128K memory audit','',
        'Allocated PyTorch memory includes model weights and the entire benchmark repetition; '
        'it excludes driver/library allocations outside PyTorch. Peak is the largest of the '
        'three measured repetitions. Collection happens between repetitions, outside prefill '
        'and decode timing. A return to baseline confirms that the previous cache is released.','',
        '| Method | Peak allocated | After collection |','|---|---:|---:|']
    for impl,label in (('dense','Dense'),('top8','Top-eight'),('closed','Closed threshold'),('fast','Fused threshold')):
        arm=summary['scaling'].get('128000',{}).get(impl,{})
        memory=arm.get('memory_bytes',[])
        if len(memory)==4 and all(v is not None for v in memory):
            measured=memory[1:]
            assert max(v['before_cache_init'] for v in measured)-min(v['before_cache_init'] for v in measured)<64*1024**2,'cache memory grows across repetitions'
            peak=max(v['peak_allocated'] for v in measured)/1024**3
            after=max(v['after_run_after_collection'] for v in measured)/1024**3
            lines.append(f'| {label} | {peak:.2f} GiB | {after:.2f} GiB |')
        else:lines.append(f'| {label} | Pending | Pending |')
    lines+=['','### Historical stage and capacity audit','',
        'These diagnostics profile the older fused producer and coarse-correction path. '
        'The new direct-summary candidate has no opened-summary subtraction/correction; '
        'these stage times must not be attributed to it. The instrumented diagnostics '
        'count every remote prefill query head across all 32 '
        'layers, excluding the exact initial field and local/sink. They do not replace '
        'the uninstrumented full-model times above. Counts are accumulated on the GPU '
        'and copied to the CPU after model execution. Both audit arms use expandable segments.','',
        '| Tokens | Method | Mean opened regions | Mean exact entries | Reserved route width | Largest leaf-output capacity |',
        '|---|---|---:|---:|---:|---:|']
    for n in (32768,128000):
        for impl,label in (('top8','Top-eight'),('fast','Fused threshold')):
            item=summary['scaling_audit'].get(str(n),{}).get(impl)
            if item:
                lines.append(f"| {n:,} | {label} | {item['regions_per_query_head']:.2f} | {item['entries_per_query_head']:.1f} | {item['maximum_route_capacity']} | {item['maximum_leaf_output_capacity_bytes']/1024**3:.2f} GiB |")
            else:lines.append(f'| {n:,} | {label} | Pending | Pending | Pending | Pending |')
    lines+=['','The reserved route width is allocation capacity, not the number actually opened. '
        'Leaf dispatch and variable reduction follow actual counts. The existing per-region '
        'c=16 rule has a cutoff proportional to 1/R, so its opened region count may grow with '
        'R; it does not maintain a fixed eight-region budget. The fused implementation also '
        'writes a full saved-score matrix and launches separate threshold packing, exact '
        'result reduction and coarse correction. Top-eight keeps the fixed selection within '
        'its coarse producer and merges eight route results directly.','',
        '| Tokens | Method | Coarse producer + routing | Separate selection | Exact KV kernel | Separate route reduction | Threshold correction/merge |',
        '|---|---|---:|---:|---:|---:|---:|']
    for n in (32768,128000):
        for impl,label in (('top8','Top-eight'),('fast','Fused threshold')):
            item=summary['scaling_audit'].get(str(n),{}).get(impl)
            if item:
                leaf=item['leaf_gpu_seconds'];gpu=item['gpu_seconds']
                producer=gpu.get('producer_coarse_selection',0) if impl=='top8' else leaf.get('fast_prepare',0)+leaf.get('fast_produce',0)
                vals=[f'{producer:.3f} s',f"{leaf.get('fast_select',0):.3f} s" if impl=='fast' else 'Included',
                    f"{leaf.get('kernel',0):.3f} s",f"{leaf.get('reduce',0):.3f} s" if impl=='fast' else 'In final merge',
                    f"{leaf.get('fast_merge',0):.3f} s" if impl=='fast' else 'Not separately measured']
                lines.append('| '+' | '.join([f'{n:,}',label,*vals])+' |')
    lines+=['','GPU-event stage times are summed over all layers. The top-eight final merge '
        'is not separately instrumented, so this is not a complete subtraction of the two '
        'end-to-end times. It separates increased exact KV work from selection, reduction '
        'and correction overhead. Extra work-count reductions slightly perturb the profiles.','',
        '</details>','',
        '## Llama full RULER quality — earlier implementation','',
        'First response ends at the first EOS/EOT token. A full average requires all 13 tasks '
        'and 32 matched examples/task. Historical 88.02% threshold-prefill quality predates the '
        'decoder used in this panel and is not substituted here. The 87.52% result is the '
        'earlier combined threshold prefill/decode implementation, not the new direct-summary '
        'candidate above. A full rerun is required before assigning that candidate an average.','',
        '| Task | Dense | RetroInfer | Earlier threshold prefill + decode |','|---|---:|---:|---:|']
    labels=('Dense','RetroInfer','Optimized threshold prefill and decode')
    for task in TASKS:
        vals=[f"{summary['quality'][k]['tasks'][task]['first']:.2f}" if k in summary['quality'] else 'Pending' for k in labels]
        lines.append('| '+' | '.join([task,*vals])+' |')
    lines.append('| **Average** | '+' | '.join(f"**{summary['quality'][k]['macro_average']:.2f}**" if k in summary['quality'] else 'Pending' for k in labels)+' |')
    if qwen_transfer is not None:
        lines+=['','## Qwen3-8B optimized threshold prefill','',
            'The [Qwen transfer report](../qwen-opt/REPORT.md) records the port, same-cache and numerical checks, '
            'and the separate reducer tuning trials. Qwen keeps its native Q/K norms and coherence-based clustering. '
            'These are three-run eager full-model prefill medians including cache completion, on matched BF16, batch-one prompts. '
            '32K uses native RoPE. The archived 64K and 128,000-token runs used incorrect YaRN scaling and are excluded. '
            'The initial exact field stays 16,384 tokens.','',
            '| Context tokens | Dense | Top-eight | Previous threshold | Optimized threshold | Time reduction versus previous threshold |',
            '|---:|---:|---:|---:|---:|---:|']
        for n,arms in sorted(qwen_transfer['models'].items(),key=lambda x:int(x[0])):
            vals=[f'{arms[k]["median_s"]:.3f} s'for k in ('dense','top8','baseline','optimized')]
            reduction=100*(1-arms['optimized']['median_s']/arms['baseline']['median_s'])
            lines.append('| '+' | '.join([f'{int(n):,}',*vals,f'{reduction:.1f}%'])+' |')
    lines+=['','## Qwen 32K quality','',
        ('The optimized threshold row uses the newly transferred prefill kernels. The other rows use their earlier implementations. '
         if qwen_transfer is not None else 'The new direct-summary candidate has not completed its Qwen quality panel. ')+
        'Four selected tasks on the existing frozen '
        'Qwen cohort, thinking disabled. This is '
        'not the complete 13-task panel and has no full RULER average. Values use first-response '
        'scoring. All LoD configurations here use sparse prefill. Connected author methods use '
        'native dense prefill plus cluster construction, windowed routing position 2048, ten sink '
        'tokens and 128-token cluster updates. Their graph/eager checks must pass exactly. '
        'The Squeezed subset is the Multipole authors’ no-replacement variant at 1:8, not the '
        'separate calibrated Squeezed release. Stable-order selected-index packing is enabled '
        'only in these connected author quality runs; their latency is not substituted for '
        'the untouched paper-component graphs.','',
        '| Method | Examples/task | Multikey-2 | Multikey-3 | CWE | QA-1 |','|---|---:|---:|---:|---:|---:|']
    for label,d in summary['qwen'].items():
        assert set(d['tasks'])==set(QTASKS)
        counts={v['n'] for v in d['tasks'].values()};assert len(counts)==1
        lines.append('| '+' | '.join([label,str(next(iter(counts))),*[f"{d['tasks'][t]['first']:.2f}" for t in QTASKS]])+' |')
    if 'Threshold decode and prefill' not in summary['qwen']:lines+=['','The 32-example/task threshold-prefill Qwen run is pending.']
    lines+=['','## Qwen real-state graph timings','',
        'Same layer-18 model-produced query and KV states, BF16, graph-only paper-component '
        'boundary. Each time is one attention call in milliseconds; model weights and projections, '
        'cache writes, cluster construction and amortized updates are excluded. Local/sink '
        'rows cover the whole prefix. Remote-only LoD excludes one sink and 256 recent entries; '
        'author remote-only fixtures include the full prefix. Multipole lookup/replacement '
        'results are not wired into its timed sparse fixture. Squeezed uses 1:8, Multipole 1:16, '
        'both budget 512. These are not quality-matched whole-model timings. '
        'The archived 128K captures used incorrect YaRN scaling and are excluded. '
        'See the [configuration correction](../qwen-long/REPORT.md).','',
        '| Tokens | Scope | Dense | Multipole | Squeezed subset | LoD top-eight | LoD threshold |',
        '|---|---|---:|---:|---:|---:|---:|']
    for n,scopes in summary['graphs'].items():
        for scope,d in scopes.items():
            lines.append('| '+' | '.join([f'{int(n):,}',scope,*[f'{d[k]:.5f}' for k in ('dense','multipole','squeezed','top8','threshold')]])+' |')
    lines+=['','All threshold graph rows pass full-frontier PyTorch checks across every query '
        'head, routing checks, eager/replay equality and local/sink perturbation checks. '
        'Read-only replay leaves cache lengths unchanged and adds no dummy token.','',
        '## Historical run status and exclusions','',f'The original threshold-next series has '
        f'{len(pending)} queued/running jobs; {len(failed)} excluded failed jobs. '
        'These counts are separate from the latest direct-summary panel above. '
        'No partial or failed generation run contributes a score above. Raw run status and '
        'commands are retained; SUMMARY.json records every scored input SHA-256.','']
    for r in failed:lines.append(f"- `{r['name']}` (exit {r['exit_code']}); inspect its provenance log.")
    lines+=['','Early Qwen attempts used an environment without Qwen3 support; corrected runs '
        'use isolated Transformers 4.51.3. The first read-only graph validation had a reference '
        'adapter error, fixed in v2. The first split reduction checks had a compile-time variable '
        'shape error, fixed and rechecked in v2. Initial connected author runs failed the exact '
        'graph/eager check and are excluded; no quality conclusion is drawn from them.','']
    (root/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'REPORT.md').write_text('\n'.join(lines))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    d=publish(a.directory.resolve(),Path(__file__).resolve().parents[1]);print(json.dumps(dict(stage=d['stage'],pending=len(d['pending']),quality={k:v.get('macro_average') for k,v in d['quality'].items()})))


if __name__=='__main__':main()
