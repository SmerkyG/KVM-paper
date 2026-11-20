"""Audit direct closed-summary scaling and offsets using frozen-cohort scores."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from tokenizers import Tokenizer
from benchmarks.h100_threshold_next_report import cleanup,metric


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    a=p.parse_args();root=a.directory;old=root.parent/'selectpack-64k'
    status=json.loads((root/'status.json').read_text());jobs={j['name']:j for j in status}
    summary=dict(inputs={},models={},quality={},pilots={},numerics={},verification={},scaling={},scaling_verification={})
    hardware=set()

    def read(path):
        summary['inputs'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    def load(stem):
        job=jobs.get('threshold-offset-'+stem);path=root/('threshold-offset-'+stem+'.json')
        if not job or job['state']!='finished' or job.get('exit_code')!=0 or not path.exists():return None
        meta=read(root/'provenance'/job['id']/'run.json');hardware.add(meta['hardware']);assert len(hardware)==1
        return read(path)

    tokpath=root.parent/'ruler-32k-panel/llama-tokenizer.json'
    summary['inputs'][str(tokpath)]=hashlib.sha256(tokpath.read_bytes()).hexdigest()
    tok=Tokenizer.from_file(str(tokpath));stops={tok.token_to_id(s)for s in ('<|end_of_text|>','<|eot_id|>')}
    assert None not in stops
    quality_identity=None

    def score(data,tasks=None):
        nonlocal quality_identity
        if data is None:return None
        assert data['stage']=='complete';rows={}
        fields=('model_name','model_revision','tokenizer_revision','model_precision','generation','prompt_format')
        current={k:data.get(k,'dataset'if k=='prompt_format'else None)for k in fields}
        if quality_identity is None:quality_identity=current
        assert all(current[k]is None or quality_identity[k]is None or current[k]==quality_identity[k]for k in fields)
        for r in data['records']:
            if tasks and r['task']not in tasks:continue
            key=r['task'],r['source_row'];assert key not in rows
            ids=r['output_token_ids'];end=next((i for i,v in enumerate(ids)if v in stops),len(ids))
            full=cleanup(tok.decode(ids,skip_special_tokens=True));assert full==r['prediction']
            assert abs(metric(r['task'],full,r['answers'])-r['score'])<1.e-9
            first=cleanup(tok.decode(ids[:end],skip_special_tokens=True))
            rows[key]=dict(score=metric(r['task'],first,r['answers']),input=r['input_token_sha256'],
                answers=r['answers'],tokens=ids[:end])
        return rows

    baseline=score(read(old/'threshold-select64-combine-cwe32-safe-v1-prefix-ruler.json'))
    direct=score(read(old/'threshold-select64-combine-cwe32-direct-v1-prefix-closed-tuned-ruler.json'))
    panel=score(read(old/'threshold-select64-prefix-safe-v1-prefix-ruler.json'))
    assert len(baseline)==len(direct)==32 and len(panel)==8
    dense={}
    for name in ('ruler-32k-dense-full.json','ruler-32k-dense-missing8.json'):
        vals=score(read(root.parent/'ruler-32k-panel'/name),tasks={'cwe'})
        assert not (dense.keys()&vals.keys());dense.update(vals)
    assert len(dense)==32

    def compare(rows,ref):
        assert all(k in ref and rows[k]['input']==ref[k]['input'] and rows[k]['answers']==ref[k]['answers']for k in rows)
        return dict(n=len(rows),percent=100*statistics.mean(v['score']for v in rows.values()),
            same_scores=sum(rows[k]['score']==ref[k]['score']for k in rows),
            same_tokens=sum(rows[k]['tokens']==ref[k]['tokens']for k in rows),
            changes=[dict(task=k[0],source_row=k[1],before=ref[k]['score'],after=v['score'])for k,v in rows.items()if v['score']!=ref[k]['score']])

    summary['quality']['conservative']=compare(baseline,baseline)
    summary['quality']['direct']=compare(direct,baseline)
    summary['quality']['dense']=compare(dense,baseline)
    labels={'scaled-bf16':'Unnormalized BF16 weights, FP32 division',
            'scaled-fp16':'Unnormalized FP16 weights, FP32 division',
            'centered-fp16':'FP16 with centered values','offset-bf16':'BF16 with uniform probability offset',
            'offset-fp16':'FP16 with uniform probability offset'}
    for key in labels:
        rows=score(load(key+'-v1-prefix-closed-tuned-ruler'))
        if rows is not None:summary['pilots'][key]=compare(rows,baseline)
        rows=score(load(key+'-cwe32-v1-prefix-closed-tuned-ruler'))
        if rows is not None:
            assert len(rows)==32;summary['quality'][key]=compare(rows,baseline)
        rows=score(load(key+'-panel-v1-prefix-closed-tuned-ruler'))
        if rows is not None:
            assert len(rows)==6 and {k[0]for k in rows}=={'niah_multikey_2','niah_multikey_3','qa_1'}
            summary['quality'][key+'-retrieval-qa']=compare(rows,panel)

    identity=None
    for key,stem in (('conservative','controls-v1-prefix-65536-models'),
        ('direct','controls-v1-prefix-closed-tuned-65536-models'),('dense','controls-v1-dense-65536-models'),
        ('scaled-bf16','scaled-bf16-v1-prefix-closed-tuned-65536-models'),
        ('scaled-fp16','scaled-fp16-v1-prefix-closed-tuned-65536-models'),
        ('offset-bf16','offset-bf16-v1-prefix-closed-tuned-65536-models')):
        d=load(stem)
        if d is None:continue
        current=[d[k]for k in ('manifest_sha256','prompt_token_sha256','checkpoint','dtype','prefill_mlp_chunk')]
        if identity is None:identity=current
        assert identity==current
        rows=[r for r in d['records']if not r['warmup']];assert len(rows)==3
        if key!='dense':
            assert d['prefill_includes_deferred_cache_completion']
            assert d['prefill_threshold']['mass_multiplier']==16 and d['prefill_threshold']['max_regions']is None
            assert d['prefill_threshold']['prefill_block_size']==16384
        summary['models'][key]=dict(median_s=statistics.median(r['prefill_s']for r in rows),
            samples_s=[r['prefill_s']for r in rows],peak_allocated_bytes=max(r['memory_bytes']['peak_allocated']for r in rows))
    for key in ('scaled-bf16','scaled-fp16','offset-bf16'):
        d=load(key+'-v1-prefix-closed-tuned-65536-verify')
        if d is None:continue
        rows=[r for r in d['records']if not r['warmup']];assert len(rows)==1
        layers=rows[0]['prefill_fast_audit'];assert len(layers)==32 and all(layers)
        checks=[r for l in layers for r in l]
        assert all(r['selection_outside_tolerance']==0 for r in checks)
        assert all(r['max_abs']==0 for r in rows[0]['graph_same_cache_checks'].values())
        summary['verification'][key]=dict(frontiers=len(checks),selection_outside_tolerance=0,
            max_abs_reference_error=max(r['max_abs_output_difference']for r in checks),graph_same_cache_max_abs=0)
    numerics=load('numerics-v2-coarse-numerics-65536')
    if numerics is not None:
        assert len(numerics['records'])==4 and all(r['score_max_abs']==0 for r in numerics['records'])
        summary['numerics']=numerics

    for length in (32768,128000):
        identity=None;arms={}
        for key,stem in (
            ('scaled-fp16',f'scaling-fp16-v1-prefix-closed-tuned-{length}-models'),
            ('conservative',f'scaling-controls-v1-auto-{length}-models'),
            ('direct',f'scaling-controls-v1-prefix-closed-tuned-{length}-models'),
            ('dense',f'scaling-controls-v1-dense-{length}-models'),
            ('top8',f'scaling-controls-v1-top8-{length}-models')):
            d=load(stem)
            if d is None:continue
            rows=[r for r in d['records']if not r['warmup']];assert len(rows)==3
            current=[d[k]for k in ('manifest_sha256','prompt_token_sha256','checkpoint','dtype','prefill_mlp_chunk')]
            if identity is None:identity=current
            assert identity==current and d['batch_size']==1 and d['prompt_tokens']==length
            if key!='dense':assert d['prefill_includes_deferred_cache_completion']
            if key not in ('dense','top8'):
                cfg=d['prefill_threshold']
                assert cfg['mass_multiplier']==16 and cfg['max_regions']is None and cfg['prefill_block_size']==16384
                if key=='scaled-fp16':assert cfg['closed_probability_precision']=='scaled-fp16'
            arms[key]=dict(median_s=statistics.median(r['prefill_s']for r in rows),
                samples_s=[r['prefill_s']for r in rows],
                peak_allocated_bytes=max(r['memory_bytes']['peak_allocated']for r in rows))
        if arms:summary['scaling'][str(length)]=arms
        d=load(f'scaling-fp16-v1-prefix-closed-tuned-{length}-verify')
        if d is not None:
            rows=[r for r in d['records']if not r['warmup']];assert len(rows)==1
            layers=rows[0]['prefill_fast_audit'];assert len(layers)==32 and all(layers)
            checks=[r for layer in layers for r in layer]
            assert all(r['selection_outside_tolerance']==0 for r in checks)
            assert all(r['max_abs']==0 for r in rows[0]['graph_same_cache_checks'].values())
            summary['scaling_verification'][str(length)]=dict(frontiers=len(checks),selection_outside_tolerance=0,
                graph_same_cache_max_abs=0,max_abs_reference_error=max(r['max_abs_output_difference']for r in checks))

    lines=['# Direct closed-summary attention: scaling and offsets','',
        'Llama-3.1-8B-Instruct, BF16 model and KV cache, batch one, H100 80GB. '
        'The c=16 per-region threshold opens every qualifying eligible region. '
        'The 16,384-token initial exact field, local/sink fields, 1,024-entry '
        'region eligibility and full cache completion are unchanged. All GPU '
        'experiments use uv and immutable cluster-run snapshots with one serial GPU lock.','',
        '## Formulations','',
        'All versions compute only the closed-summary frontier, without '
        'subtracting an opened-summary attention field. For closed native summary '
        'logits z, the scaled versions compute m=max(z), w=exp(z-m), d=sum(w) '
        'in FP32, project w in one BF16 or FP16 tensor-core GEMM with FP32 output, '
        'and divide that output by d in FP32. The closed log partition remains '
        'm+log(d). This defers normalization until after value accumulation.','',
        'FP16 variants convert only coarse projection operands; model weights, '
        'KV storage, routing and exact attention stay BF16. Values are shared '
        'across GQA heads. No selected-count CPU readback is introduced.','',
        'The value-offset trial uses v0+sum(alpha*(v-v0)). The probability-offset '
        'trial uses mean(v)+sum((alpha-1/R)*v) across all R nonempty summaries, '
        'with alpha=0 on opened summaries. Its common mean is shared across '
        'queries, so it also requires only one GEMM. These offsets are exact '
        'identities before rounding; they need not improve finite-precision error.','',
        '## Matched 64K speed and 32K CWE quality','',
        'Prefill timings are fresh medians of three measured runs after one '
        'warmup, including deferred cache completion. They use identical frozen '
        '65,536-token prompts, model/MLP chunks and scope. CWE quality uses all '
        '32 frozen examples at 32K and is scored only before the first EOS/EOT. '
        'Baseline quality comes from the earlier identical frozen cohort; '
        'baseline timings are rerun here. These CWE scores are not RULER macro averages.','',
        '| Method | 64K prefill | CWE, 32 examples at 32K |','|---|---:|---:|']
    if all(k in summary['models']for k in ('scaled-fp16','direct','conservative','dense')) and 'scaled-fp16'in summary['quality']:
        speed=summary['models']['scaled-fp16']['median_s'];direct_s=summary['models']['direct']['median_s'];dense_s=summary['models']['dense']['median_s']
        quality=summary['quality']['scaled-fp16']['percent']
        lines[4:4]=[f'The single-projection FP16 candidate takes **{speed:.3f} s at 64K**, '
            f'{100*(speed/direct_s-1):.2f}% above the previous direct BF16 path, and '
            f'scores **{quality:.2f}% on 32K CWE** versus 30.00% conservative and '
            f'27.19% previous direct BF16. It remains {100*(speed/dense_s-1):.1f}% '
            'slower than dense at 64K. It is an experimental direct-summary '
            'candidate; the full 13-task RULER panel has not been rerun.','']
    for key,label in (('conservative','Conservative prefix selection'),('direct','Previous direct closed BF16'),
        ('scaled-bf16',labels['scaled-bf16']),('scaled-fp16',labels['scaled-fp16']),('offset-bf16',labels['offset-bf16']),('dense','Dense')):
        model=summary['models'].get(key);quality=summary['quality'].get(key)
        timing=f'{model["median_s"]:.3f} s'if model else'Pending'
        accuracy=f'{quality["percent"]:.2f}%'if quality else'Pilot only'if key=='offset-bf16'else'Pending'
        lines.append(f'| {label} | {timing} | {accuracy} |')
    for key in ('scaled-bf16','scaled-fp16','offset-bf16'):
        d=summary['quality'].get(key+'-retrieval-qa')
        if d:lines+=['',f'{labels[key]}: {d["same_scores"]}/{d["n"]} first-response scores '
            'unchanged on two examples each of mk2, mk3 and QA1.']
    if summary['scaling']:
        lines+=['','## Context-length scaling','',
            'Fresh 32,768- and 128,000-token matched runs, one warmup and three '
            'measured repetitions. The 65,536-token row reuses the matched '
            'runs above. Prefill includes full cache completion, local/sink '
            'attention and the unchanged 16,384-token initial exact field. '
            'The new 32K/128K conservative controls use automatic packing dispatch (prefix '
            'filter at 2,048–4,095 slots; prefix scan from 4,096). Both direct '
            'variants use prefix filtering with the same producer tile so '
            'their projection costs can be compared.','',
            '| Context tokens | Conservative | Previous direct BF16 | New direct FP16 | Dense | Top eight |',
            '|---:|---:|---:|---:|---:|---:|']
        for length in (32768,65536,128000):
            arms=summary['models']if length==65536 else summary['scaling'].get(str(length),{})
            values=[]
            for key in ('conservative','direct','scaled-fp16','dense','top8'):
                if key not in arms:values.append('—');continue
                if key=='scaled-fp16'and length!=65536 and str(length)not in summary['scaling_verification']:
                    values.append('Awaiting verification');continue
                values.append(f'{arms[key]["median_s"]:.3f} s')
            lines.append('| '+f'{length:,}'+' | '+' | '.join(values)+' |')
        for length,arms in summary['scaling'].items():
            if length not in summary['scaling_verification']or not {'scaled-fp16','dense','conservative','direct'}<=arms.keys():continue
            speed=arms['scaled-fp16']['median_s'];dense_s=arms['dense']['median_s'];conservative=arms['conservative']['median_s'];direct_s=arms['direct']['median_s']
            v=summary['scaling_verification'][length]
            lines+=['',f'At {int(length):,} tokens, new direct FP16 uses '
                f'{abs(100*(speed/dense_s-1)):.1f}% {"more"if speed>dense_s else"less"} '
                f'prefill time than dense, {abs(100*(speed/conservative-1)):.1f}% '
                f'{"more"if speed>conservative else"less"} than conservative '
                f'and {100*(speed/direct_s-1):+.2f}% relative to previous direct BF16. '
                f'Peak PyTorch allocation is {arms["scaled-fp16"]["peak_allocated_bytes"]/2**30:.2f} GiB. '
                f'{v["frontiers"]} sampled frontiers across 32 layers pass; '
                'selection has zero differences outside tolerance and same-cache '
                'eager/graph logits match exactly.']
        lines+=['','Top-eight was not rerun at 64K in this series. These are prefill '
            'scaling results, not additional quality panels or amortized decode '
            'measurements. No complete-execution subquadratic bound is inferred.']
    lines+=['','## Two-example CWE ablations','',
        'These pilots locate the previously changed example; they are separate '
        'from full-task quality and must not be treated as aggregate estimates.','',
        '| Variant | CWE row 9 | CWE row 14 |','|---|---:|---:|']
    for key,label in labels.items():
        d=score(load(key+'-v1-prefix-closed-tuned-ruler'))
        if d:lines.append(f'| {label} | {100*d[("cwe",9)]["score"]:.0f}% | {100*d[("cwe",14)]["score"]:.0f}% |')
    if numerics is not None:
        lines+=['','## Identical-input numerical audit','',
            'Four live 64K prefill calls, 32 queries per head per call, checked '
            'against FP32 PyTorch on the same scores, selected bits, sizes and '
            'values. This diagnostic is not a model timing or quality estimate.','',
            '| Call | Direct BF16 RMSE | Scaled BF16 RMSE | Scaled FP16 RMSE | FP16 value offset RMSE | BF16 probability offset RMSE |',
            '|---:|---:|---:|---:|---:|---:|']
        for r in numerics['records']:
            values=[r[k]['rmse']for k in ('direct-bf16','direct-scaled-bf16','direct-scaled-fp16','direct-scaled-fp16-centered','direct-offset-bf16')]
            lines.append('| '+str(r['call'])+' | '+' | '.join(f'{v:.3g}'for v in values)+' |')
        lost=max(r['fp16_max_underflow_mass_fraction']for r in numerics['records'])
        lines+=['',f'The maximum discarded FP16 exponential tail mass on these '
            f'fixtures is {lost:.3g} of the closed mass. Maximum absolute coarse '
            f'value is {max(r["mean_values_max_abs"]for r in numerics["records"]):.3g}. '
            'The split-BF16 direct reference remains more arithmetically accurate '
            'but did not recover the original CWE example in the previous '
            'investigation. Numerical accuracy and task recall must be checked separately.']
    lines+=['','## Verification and provenance','']
    for key,d in summary['verification'].items():
        lines.append(f'- {key}: {d["frontiers"]} sampled frontiers across 32 layers; '
            'zero selection differences outside tolerance; same-cache eager/graph '
            f'logits match exactly; maximum reference attention error {d["max_abs_reference_error"]:.6f}.')
    pending=[j['name']for j in status if j['state']!='finished']
    failed=[j['name']for j in status if j['state']=='finished'and j.get('exit_code')not in (0,-15)]
    cancelled=[j['name']for j in status if j['state']=='finished'and j.get('exit_code')==-15]
    summary.update(stage='running'if pending else'complete',pending=pending,failed=failed,cancelled=cancelled,hardware=sorted(hardware))
    summary['quality_identity']=quality_identity
    lines+=['',f'{len(pending)} pending; {len(failed)} failed; {len(cancelled)} cancelled jobs. Only completed '
        'successful runs enter the tables. Raw commands, logs and executed-source '
        'hashes are retained in provenance/ and analysis-commands.json. SUMMARY.json '
        'hashes analyzed inputs. Full-model timings include GPU preparation, '
        'conversion and output restoration; standalone projection timers exclude '
        'preparation and are not used for the headline.','',
        'The optional full-CWE and retrieval expansions for the probability-offset '
        'BF16 prototype were cancelled while still queued after the scaled FP16 '
        'candidate completed its quality checks. Its quality results remain '
        'limited to the two-example pilot; no full-task score is inferred.','',
        'These remain experimental formulations. No serving default or exact '
        'prefill size changes, and no complete-execution subquadratic claim is made.','']
    (root/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(stage=summary['stage'],models=summary['models'],
        quality={k:v['percent']for k,v in summary['quality'].items()},pending=len(pending),failed=failed)))


if __name__=='__main__':main()
