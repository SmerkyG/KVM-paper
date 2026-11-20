"""Report controlled coarse-arithmetic ablations, preserving first-EOS scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from tokenizers import Tokenizer
from benchmarks.h100_threshold_next_report import cleanup,metric


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    a=p.parse_args();root=a.directory
    jobs={x['name']:x for x in json.loads((root/'status.json').read_text())}
    summary=dict(inputs={},models={},quality={},numerics={},verification={})
    hardware=set()

    def load(stem):
        name='threshold-select64-'+stem;job=jobs.get(name);path=root/(name+'.json')
        if not job or job['state']!='finished' or job.get('exit_code')!=0 or not path.exists():return None
        meta=json.loads((root/'provenance'/job['id']/'run.json').read_text())
        hardware.add(meta['hardware']);assert len(hardware)==1
        summary['inputs'][path.name]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),run=job['id'])
        return json.loads(path.read_text())

    tokpath=root.parent/'ruler-32k-panel/llama-tokenizer.json';tok=Tokenizer.from_file(str(tokpath))
    stops={tok.token_to_id(t)for t in ('<|end_of_text|>','<|eot_id|>')};assert None not in stops
    def quality(stem):
        data=load(stem)
        if data is None:return None
        assert data['stage']=='complete'
        rows={}
        for r in data['records']:
            ids=r['output_token_ids'];end=next((i for i,v in enumerate(ids)if v in stops),len(ids))
            full=cleanup(tok.decode(ids,skip_special_tokens=True));assert full==r['prediction']
            assert abs(metric(r['task'],full,r['answers'])-r['score'])<1.e-9
            text=cleanup(tok.decode(ids[:end],skip_special_tokens=True))
            rows[(r['task'],r['source_row'])]=dict(score=metric(r['task'],text,r['answers']),
                input=r['input_token_sha256'],answers=r['answers'],tokens=ids[:end])
        return rows

    lines=['# H100 coarse-arithmetic regression and combinations','',
        'Llama-3.1-8B-Instruct BF16, batch one. All GPU experiments use uv and '
        'immutable cluster-run snapshots on the same H100. The c=16 region threshold '
        'is uncapped; the 16,384-token initial exact field, local/sink fields, '
        '1,024-entry region eligibility and cache-completion scope remain fixed.','',
        '## Controlled ablations','',
        'The aggressive candidate already includes the conservative prefix selection '
        'improvement. These ablations isolate its larger producer tile, closed-only '
        'coarse attention, and BF16 probability rounding. These first checks contain '
        'only two CWE examples; they are not RULER averages.','',
        '| Change from conservative prefix selection | CWE row 9 | CWE row 14 |',
        '|---|---:|---:|']
    controls=quality('prefix-safe-v1-prefix-ruler')
    quality_arms=(('Conservative prefix selection','prefix-safe-v1-prefix-ruler'),
        ('Larger producer tile only','combine-tile-v1-prefix-tuned-ruler'),
        ('Direct closed projection, old producer tile','combine-closed64-v1-prefix-closed-ruler'),
        ('Direct closed projection, split BF16 probabilities','combine-split-v1-prefix-closed-tuned-ruler'),
        ('Online coarse/correction coefficients through cuBLAS','combine-online-v1-prefix-closed-tuned-ruler'),
        ('Online coefficients, single BF16 projection','combine-online-single-v1-prefix-closed-tuned-ruler'),
        ('Shared log-counts, original coarse/correction','combine-log-v1-prefix-ruler'))
    for label,stem in quality_arms:
        rows=quality(stem)
        if rows is None:continue
        keys=[('cwe',9),('cwe',14)];assert all(rows[k]['input']==controls[k]['input']for k in keys)
        summary['quality'][stem]={str(k):v for k,v in rows.items()}
        lines.append(f'| {label} | {100*rows[keys[0]]["score"]:.0f}% | {100*rows[keys[1]]["score"]:.0f}% |')
    lines+=['','A higher-precision projection encodes each FP32 probability as two '
        'BF16 values and sums two FP32 GEMM outputs. It isolates probability '
        'quantization from the change in coarse-attention arithmetic. The online '
        'combination instead reconstructs the original 64-region online weights, '
        'including their BF16 rounding, before subtracting unrounded opened-summary '
        'weights. It projects these effective coefficients with shared-KV cuBLAS. '
        'This approximates the original arithmetic, rather than giving bitwise '
        'equivalence: the full normalizer and GEMM accumulation order also differ.','',
        '## Full-model 64K timing','',
        'Median of three measured prefills after one warmup. All include deferred '
        'cache completion. The original matched controls are retained as dated '
        'controls; an implementation combination receives its own fresh run.','',
        '| Implementation | Prefill seconds |','|---|---:|']
    identity=None
    for label,stem in (
        ('Original threshold','final-v1-control-65536-models'),
        ('Conservative prefix selection','prefix-safe-v1-prefix-65536-models'),
        ('Aggressive direct closed projection','final-v1-prefix-closed-tuned-65536-models'),
        ('Conservative with larger producer tile','combine-tile-v1-prefix-tuned-65536-models'),
        ('Conservative with shared log-counts','combine-log-v1-prefix-65536-models'),
        ('Online-compatible cuBLAS combination','combine-online-v1-prefix-closed-tuned-65536-models'),
        ('Online coefficients, single BF16 projection','combine-online-single-v1-prefix-closed-tuned-65536-models'),
        ('Dense','final-v1-dense-65536-models'),('Top eight','final-v1-top8-65536-models')):
        data=load(stem)
        if data is None:continue
        current=[data[k]for k in ('manifest_sha256','prompt_token_sha256','checkpoint','dtype','prefill_mlp_chunk')]
        if identity is None:identity=current
        assert identity==current
        rows=[r for r in data['records']if not r['warmup']];assert len(rows)==3
        if label!='Dense':assert data['prefill_includes_deferred_cache_completion']
        samples=[r['prefill_s']for r in rows];value=statistics.median(samples)
        summary['models'][stem]=dict(median_s=value,samples_s=samples)
        lines.append(f'| {label} | {value:.3f} |')

    safe=quality('combine-cwe32-safe-v1-prefix-ruler')
    direct=quality('combine-cwe32-direct-v1-prefix-closed-tuned-ruler')
    if safe is not None and direct is not None:
        assert safe.keys()==direct.keys() and len(safe)==32 and all(k[0]=='cwe'for k in safe)
        assert all(safe[k]['input']==direct[k]['input'] and safe[k]['answers']==direct[k]['answers']for k in safe)
        means=[100*statistics.mean(rows[k]['score']for k in rows)for rows in (safe,direct)]
        changes=[dict(source_row=k[1],before=safe[k]['score'],after=direct[k]['score'])for k in safe if safe[k]['score']!=direct[k]['score']]
        summary['cwe32']=dict(n=32,conservative_pct=means[0],aggressive_pct=means[1],changes=changes)
        lines+=['','## All 32 frozen CWE examples at 32K','',
            f'First-response CWE recall: conservative **{means[0]:.4f}%**, aggressive '
            f'**{means[1]:.4f}%**. {len(changes)}/32 examples change score. This is the '
            'complete frozen CWE task, not a new 13-task RULER average.','',
            '| Source row | Conservative | Aggressive |','|---:|---:|---:|']
        for c in changes:lines.append(f'| {c["source_row"]} | {100*c["before"]:.0f}% | {100*c["after"]:.0f}% |')

    panel=quality('combine-online-panel-v1-prefix-closed-tuned-ruler')
    if panel is not None:
        assert panel.keys()==controls.keys() and len(panel)==8
        assert all(panel[k]['input']==controls[k]['input']for k in panel)
        same=sum(panel[k]['score']==controls[k]['score']for k in panel)
        summary['online_panel']=dict(n=8,same_score=same)
        lines+=['',f'The online-compatible combination preserves **{same}/8** first-response '
            'scores on the separate mk2/mk3/CWE/QA correctness panel.']
    online=quality('combine-cwe32-online-v1-prefix-closed-tuned-ruler')
    if safe is not None and online is not None:
        assert safe.keys()==online.keys() and len(online)==32
        assert all(safe[k]['input']==online[k]['input']for k in safe)
        average=100*statistics.mean(online[k]['score']for k in online)
        changes=[dict(source_row=k[1],before=safe[k]['score'],after=online[k]['score'])for k in safe if safe[k]['score']!=online[k]['score']]
        summary['online_cwe32']=dict(n=32,percent=average,changes=changes)
        lines+=['',f'On all 32 CWE examples, the two-projection online-compatible '
            f'combination scores **{average:.4f}%**, with {len(changes)}/32 scores '
            'changed relative to conservative prefix selection.']

    data=load('combine-numerics-v3-coarse-numerics-65536')
    if data is not None:
        assert data['stage']=='complete';summary['numerics']=data
        lines+=['','## Identical-input numerical audit','',
            'Four live prefill calls, 32 queries per head per call. These compare '
            'the same saved FP32 QK scores, region sizes, selected bits and BF16 '
            'values. PyTorch computes an FP32 closed-frontier reference. This '
            'diagnostic is not model timing or a quality estimate.','',
            '| Call | Regions | Original vs FP32 RMSE | Direct BF16 vs FP32 RMSE | Split direct vs FP32 RMSE | Online combination vs original max error |',
            '|---:|---:|---:|---:|---:|---:|']
        for r in data['records']:
            assert r['score_max_abs']<1.e-4
            lines.append(f'| {r["call"]} | {r["regions"]} | {r["original_subtraction_rmse"]:.3g} | {r["direct-bf16"]["rmse"]:.3g} | {r["direct-split-bf16"]["rmse"]:.3g} | {r["online64-split-bf16"]["vs_original_max_abs"]:.3g} |')
    lines+=['','## Frontier and graph checks','']
    for stem in ('combine-tile-v1-prefix-tuned-65536-verify','combine-log-v1-prefix-65536-verify',
                 'combine-online-v1-prefix-closed-tuned-65536-verify',
                 'combine-online-single-v1-prefix-closed-tuned-65536-verify'):
        data=load(stem)
        if data is None:continue
        rows=[r for r in data['records']if not r['warmup']];assert len(rows)==1
        layers=rows[0]['prefill_fast_audit'];assert len(layers)==32 and all(layers)
        checks=[r for l in layers for r in l]
        mismatches=sum(r['selection_outside_tolerance']for r in checks);assert mismatches==0
        graph_error=max(r['max_abs']for r in rows[0]['graph_same_cache_checks'].values());assert graph_error==0
        value=dict(frontiers=len(checks),selection_mismatches=mismatches,graph_max_abs=graph_error,
            reference_max_abs=max(r['max_abs_output_difference']for r in checks))
        summary['verification'][stem]=value
        lines.append(f'- `{stem}`: {len(checks)} sampled frontiers across 32 layers; '
            f'zero selection mismatches outside tolerance; reference attention '
            f'maximum absolute error {value["reference_max_abs"]:.6f}; same-cache '
            'eager/graph logits match exactly.')
    followup_path=root.parent/'closed-offset'/'SUMMARY.json'
    if followup_path.exists():
        latest=json.loads(followup_path.read_text())
        if latest.get('stage')=='complete' and 'scaled-fp16'in latest['models'] and 'scaled-fp16'in latest['quality']:
            value=latest['models']['scaled-fp16']['median_s'];quality=latest['quality']['scaled-fp16']['percent']
            lines[4:4]=['**Subsequent result:** the [normalization and offset follow-up](../closed-offset/REPORT.md) '
                f'keeps one direct closed-summary projection, using FP16 operands and '
                f'FP32 normalization afterward: **{value:.3f} s at 64K**, '
                f'**{quality:.2f}% CWE recall on the 32 frozen examples at 32K**. '
                'Model weights and KV storage remain BF16. This supersedes the '
                'two-projection combination\'s speed/quality tradeoff below.','']
            summary['normalization_followup']=dict(file=str(followup_path),sha256=hashlib.sha256(followup_path.read_bytes()).hexdigest())
    pending=[x['name']for x in jobs.values()if 'combine-'in x['name'] and x['state']!='finished']
    failed=[x['name']for x in jobs.values()if 'combine-'in x['name'] and x['state']=='finished'and x.get('exit_code')not in (0,-15)]
    cancelled=[x['name']for x in jobs.values()if 'combine-'in x['name'] and x['state']=='finished'and x.get('exit_code')==-15]
    summary.update(pending=pending,failed=failed,cancelled=cancelled,hardware=sorted(hardware))
    lines+=['','No complete-execution subquadratic claim is made. Selection has no '
        'cap, CPU count readback or change in exact prefill size. Failed and partial '
        f'runs are excluded. {len(pending)} pending, {len(failed)} failed and '
        f'{len(cancelled)} cancelled combination jobs. '
        'Commands, logs and content-verified executed sources are in the adjacent '
        'provenance directory and analysis command ledger.','',
        'The first numerical probe was rejected before measurement because its '
        'prompt-replay option required token capture. The second, still-queued '
        'snapshot had the same issue and was cancelled. The corrected third probe '
        'adds token capture; neither earlier job contributes a measurement.','']
    (root/'COMBINATION.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'COMBINATION.md').write_text('\n'.join(lines))
    print(json.dumps(dict(models=summary['models'],cwe32=summary.get('cwe32'),pending=pending,failed=failed)))


if __name__=='__main__':main()
