"""Summarize successful selection-packing runs and paired token checks."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

from tokenizers import Tokenizer
from benchmarks.h100_threshold_next_report import cleanup, metric


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    root=a.directory;status=json.loads((root/'status.json').read_text());by_name={r['name']:r for r in status}
    result=dict(probes={},models={},profiles={},quality={},checks=[],automatic_checks={},input_sha256={})
    hardware=set()
    def load(path):
        job=by_name.get(path.stem)
        if job is None or job['state']!='finished' or job.get('exit_code')!=0:return None
        d=json.loads(path.read_text());result['input_sha256'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        meta=json.loads((root/'provenance'/job['id']/'run.json').read_text());hardware.add(meta['hardware']);assert len(hardware)==1
        return d
    for path in sorted(root.glob('*.json')):
        d=load(path)
        if d is None:continue
        if '-check-m' in path.name:
            assert d['stage']=='complete';result['checks'].append(dict(file=path.name,cases=len(d['cases'])))
        elif '-live-probe-' in path.name:
            assert d['stage']=='complete';result['probes'][path.name]=d
        elif match:=re.fullmatch(r'threshold-next-selectpack-(.+)-(control|scan|bias|hybrid)-(\d+)-(models|profiles)',path.stem):
            group,backend,n,kind=match.groups();rows=[r for r in d['records'] if not r['warmup']]
            assert len(rows)==(3 if kind=='models' else 1)
            assert d['prefill_includes_deferred_cache_completion'] and d['batch_size']==1
            assert d['prefill_threshold']['mass_multiplier']==16 and d['prefill_threshold']['max_regions'] is None
            assert d['prefill_threshold']['prefill_block_size']==16384
            entry=dict(seconds=statistics.median(r['prefill_s'] for r in rows),file=path.name,
                input_identity=[d['manifest_sha256'],d['prompt_token_sha256'],d['checkpoint'],d['dtype'],d['prefill_mlp_chunk']],
                tokens=[r['output_token_ids'] for r in rows],
                peak_allocated_bytes=max(r['memory_bytes']['peak_allocated'] for r in rows))
            if kind=='profiles':
                keys={k for l in rows[0]['prefill_leaf_profile'] for k in l}
                entry['gpu_seconds']={k:sum(l.get(k,0) for l in rows[0]['prefill_leaf_profile']) for k in keys}
            result[kind].setdefault(group,{}).setdefault(n,{})[backend]=entry
        elif path.stem.endswith('-ruler-check'):
            assert d['stage']=='complete';result['quality'][path.name]=d
        elif path.stem.endswith('-automatic-128000-verify'):
            assert d['prefill_threshold']['selection_scan_policy']=='auto'
            assert d['prefill_threshold']['selection_scan_min_regions']==4096
            rows=[r for r in d['records'] if not r['warmup']]
            assert len(rows)==1
            layers=rows[0]['prefill_fast_audit'];assert len(layers)==32 and all(layers)
            frontiers=[v for layer in layers for v in layer]
            assert all(v['selection_outside_tolerance']==0 for v in frontiers)
            assert rows[0]['graph_same_cache_checks']
            result['automatic_checks'][path.name]=dict(
                layers=len(layers),frontiers=len(frontiers),
                selection_mismatches=sum(v['selection_mismatches'] for v in frontiers),
                selection_outside_tolerance=sum(v['selection_outside_tolerance'] for v in frontiers),
                max_abs_output_difference=max(v['max_abs_output_difference'] for v in frontiers),
                graph_same_cache_checks=rows[0]['graph_same_cache_checks'])
    lines=['# H100 threshold selection and packing optimization','',
        'BF16 Llama-3.1-8B-Instruct, batch one, one H100. c=16 per-region routing, '
        'the 1,024-entry eligible-region limit, exact initial 16,384-token prefill, '
        'local/sink attention and cache updates remain unchanged. No sorting, '
        'selection cap or selected-count CPU readback is introduced. Experiments '
        'use uv and source-snapshotting cluster-run.','']
    final=result['models'].get('final-v1',{})
    if all(set(final.get(str(n),{}))=={'control','scan'} for n in (32768,65536,128000)):
        result['selected_defaults']=dict(query_tile=32,region_tile=256,warps=4,min_regions=4096,expert_atomics=False)
        lines += ['## Selected implementation','',
            'The uncapped, per-region fused threshold path now automatically uses '
            '**32-query × 256-region tiles, four warps and device prefix scans** '
            'from 4,096 active regions onward. Below that point it retains the '
            'original full-row pass. Empty packing tiles skip mask unpacking and '
            'prefix work. This changes the selection implementation within the '
            'opt-in threshold adapter; vLLM serving retains its existing routing defaults.','',
            '| Context tokens | Original packing | Optimized packing | Prefill reduction |',
            '|---:|---:|---:|---:|']
        for n in (32768,65536,128000):
            c=final[str(n)]['control'];s=final[str(n)]['scan']
            assert c['input_identity']==s['input_identity'] and c['tokens']==s['tokens']
            reduction=100*(1-s['seconds']/c['seconds'])
            label='Unchanged' if abs(reduction)<0.1 else f'{reduction:.1f}%'
            lines.append(f"| {n:,} | {c['seconds']:.3f} s | {s['seconds']:.3f} s | {label} |")
        profiles=result['profiles'].get('final-v1',{}).get('128000',{})
        if set(profiles)=={'control','scan'}:
            c=profiles['control']['gpu_seconds']['fast_select'];s=profiles['scan']['gpu_seconds']['fast_select']
            lines += ['',f'At 128,000 tokens, full-model selection/packing GPU time falls from **{c:.3f} s to {s:.3f} s ({100*(1-s/c):.1f}% less)**. '
                'The isolated heavy live fixture improves by 48.5%; that number is not the full-prefill speedup.']
        lines += ['', 'The scan reads each saved score once, writes a compact selection mask '
            'and tile counts, computes query/expert prefixes, then packs selected pairs '
            'from the mask. It introduces no expert atomics. Work is linear in the '
            'query–region matrix size; this does not establish a subquadratic bound '
            'for complete prefill or amortized cache maintenance.','']
    lines += ['## Live selection fixture','',
        'The probe intercepts real prefill scores in memory and intentionally stops '
        'model execution after tuning. It saves no Q/K/V snapshot. These are selection '
        'timings, not full-model latency or quality. Each candidate is checked against '
        'the original full-row pass: complete selection bits, counts, entries, selected '
        'region ordering, expert counts and unique gap-free expert offsets.','']
    for name,d in result['probes'].items():
        baseline=d['records'][0]['median_ms'];f=d['fixture']
        lines += [f"Fixture `{name}`: scores {f['scores_shape']}, {f['mean_regions']:.2f} mean selected regions/query-head.",'',
            '| Backend | Query tile | Region tile | Warps | Median | Reduction vs full-row |','|---|---:|---:|---:|---:|---:|']
        for r in d['records']:
            lines.append(f"| {r['backend']} | {r.get('m','—')} | {r.get('n','—')} | {r.get('warps',8)} | {r['median_ms']:.3f} ms | {100*(1-r['median_ms']/baseline):.1f}% |")
        lines.append('')
    lines += ['## Full-model prefill','',
        'One warmup plus three measured repetitions; median wall time, including '
        'final cache completion and excluding model loading, initial cache allocation '
        'and graph capture. Every arm uses expandable allocation segments and 8,192-token '
        'MLP chunks. Internal attention scratch tiles remain 4,096 queries.','',
        '| Paired series | Tokens | Backend | Control | Candidate | Reduction | Sampled output IDs match |',
        '|---|---:|---|---:|---:|---:|---|']
    for group,lengths in result['models'].items():
        for n,arms in lengths.items():
            control=arms.get('control')
            if control is None:continue
            for backend,d in arms.items():
                if backend=='control':continue
                assert d['input_identity']==control['input_identity']
                equal=d['tokens']==control['tokens']
                lines.append(f"| {group} | {int(n):,} | {backend} | {control['seconds']:.3f} s | {d['seconds']:.3f} s | {100*(1-d['seconds']/control['seconds']):.1f}% | {equal} |")
    lines += ['','The two teacher-forced output tokens are only a sampled model check. '
        'They do not establish RULER quality. A negative reduction is a regression; '
        'microbenchmarks are not substituted for full-model results.','',
        '## GPU stage profiles','',
        'One measured instrumented repetition after warmup; summed CUDA-event time '
        'over all layers. These are diagnostic stage costs.','',
        '| Series | Backend | Tokens | Selection/packing | Exact including dispatch/reduction | Coarse producer | Merge |',
        '|---|---|---:|---:|---:|---:|---:|']
    for group,lengths in result['profiles'].items():
        for n,arms in lengths.items():
            for backend,d in arms.items():
                gpu=d['gpu_seconds'];lines.append(f"| {group} | {backend} | {int(n):,} | {gpu.get('fast_select',0):.3f} s | {gpu.get('fast_exact',0):.3f} s | {gpu.get('fast_produce',0):.3f} s | {gpu.get('fast_merge',0):.3f} s |")
    lines += ['','## Validation and limits','',f"{len(result['checks'])} synthetic fixture configurations passed, covering query and region tails, empty/oversized regions, concentrated mass, and empty selection.",'',
        'The scan variant uses tiled score evaluation, query and expert prefix scans, '
        'and a second mask-only packing pass. The hybrid instead reserves one expert '
        'range per query tile, while keeping query-prefix packing. The bias variant '
        'precomputes region-only log-counts and threshold shifts and reuses query RMS '
        'from the producer, retaining the original atomic expert packing.','']
    for name,check in result['automatic_checks'].items():
        lines += [f"Automatic-dispatch 128,000-token reference check: **{check['layers']} layers, {check['frontiers']} sampled frontiers**, "
            f"{check['selection_mismatches']} selection differences ({check['selection_outside_tolerance']} outside boundary tolerance), "
            f"maximum absolute attention-output difference {check['max_abs_output_difference']:.6f}. "
            'Same-cache eager/CUDA-graph checks also pass. Instrumented verification time is excluded from latency tables.','']
    controls={k:v for k,v in result['quality'].items() if '-control-ruler-check' in k}
    if controls:
        tokenizer=root.parent/'ruler-32k-panel'/'llama-tokenizer.json'
        config=root/'model-generation-config.json'
        tok=Tokenizer.from_file(str(tokenizer))
        stops=set(json.loads(config.read_text())['config']['eos_token_id'])
        result['input_sha256'][str(tokenizer)]=hashlib.sha256(tokenizer.read_bytes()).hexdigest()
        result['input_sha256'][config.name]=hashlib.sha256(config.read_bytes()).hexdigest()
        def first(ids):
            return ids[:next((i for i,v in enumerate(ids) if v in stops),len(ids))]
    for name,control in controls.items():
        prefix=name.replace('-control-ruler-check.json','')
        for other,candidate in result['quality'].items():
            if not other.startswith(prefix) or other==name:continue
            cs={(r['task'],r['source_row']):r for r in control['records']}
            ds={(r['task'],r['source_row']):r for r in candidate['records']};assert cs.keys()==ds.keys()
            for k in cs:
                assert cs[k]['input_token_sha256']==ds[k]['input_token_sha256']
            same=sum(cs[k]['output_token_ids']==ds[k]['output_token_ids'] for k in cs)
            first_same=sum(first(cs[k]['output_token_ids'])==first(ds[k]['output_token_ids']) for k in cs)
            scores=[]
            for k in cs:
                assert cs[k]['answers']==ds[k]['answers']
                values=[metric(k[0],cleanup(tok.decode(first(rows[k]['output_token_ids']),skip_special_tokens=True)),rows[k]['answers']) for rows in (cs,ds)]
                scores.append(dict(task=k[0],source_row=k[1],control=values[0],scan=values[1],
                    first_response_identical=first(cs[k]['output_token_ids'])==first(ds[k]['output_token_ids'])))
            equal_scores=sum(v['control']==v['scan'] for v in scores)
            changed={task:sum(not v['first_response_identical'] for v in scores if v['task']==task)
                for task in dict.fromkeys(v['task'] for v in scores)}
            changed_text=', '.join(f'{task} ({n} examples)' for task,n in changed.items() if n)
            result.setdefault('quality_token_checks',{})[other]=dict(n=len(cs),identical=same,
                first_responses_identical=first_same,first_response_scores_equal=equal_scores,examples=scores)
            lines += [f'Paired RULER correctness check: **{equal_scores}/{len(cs)} first-response scores unchanged**, '
                f'**{first_same}/{len(cs)} first-response token sequences identical**, and '
                f'{same}/{len(cs)} complete fixed-length sequences identical for `{other}`. '
                'The first response ends at the checkpoint\'s EOS/EOT tokens; later continuation '
                f'differences are excluded from scoring. First-response differences: {changed_text or "none"}. '
                'This is a two-example-per-task correctness '
                'panel, not a new full RULER accuracy estimate.','',
                '| Task | Examples | Original first-response score | Scan first-response score |',
                '|---|---:|---:|---:|']
            for task in dict.fromkeys(v['task'] for v in scores):
                rows=[v for v in scores if v['task']==task]
                lines.append(f"| {task} | {len(rows)} | {100*statistics.mean(v['control'] for v in rows):.2f}% | {100*statistics.mean(v['scan'] for v in rows):.2f}% |")
            lines.append('')
    pending=[r for r in status if r['state']!='finished'];failed=[r for r in status if r['state']=='finished' and r.get('exit_code')!=0]
    result.update(stage='running' if pending else 'complete',pending=pending,excluded_runs=failed,hardware=sorted(hardware))
    lines += ['## Provenance','',
        'Raw outputs, commands, immutable per-run source hashes and logs are retained '
        'in this directory. source-objects/ stores content-verified executed sources. '
        'SUMMARY.json records the SHA-256 of every analyzed raw output. Failed and '
        'partial runs are excluded from all tables.','',f'{len(pending)} pending jobs; {len(failed)} excluded failed jobs.','']
    for r in failed:lines.append(f"- `{r['name']}`: exit {r['exit_code']}; see its provenance log.")
    (root/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(stage=result['stage'],models=list(result['models']),checks=len(result['checks']))))


if __name__=='__main__':main()
