"""Audit complete H100 three-tier experiments and render their measured report."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    a = p.parse_args()
    root = a.directory
    phases = {phase:read(root/f'three-tier-v2-{phase}-runs.json') for phase in ('tune','model','quality','bracket','diagnostic')}
    jobs = [job for phase in phases.values() for job in phase['jobs']]
    hardware = set()
    source_hashes = collections.defaultdict(set)
    data = {}
    for job in jobs:
        meta = read(root/'provenance'/job['id']/'run.json')
        if meta['state'] != 'finished' or meta.get('exit_code') != 0:
            raise ValueError('experiment not successful: '+job['name'])
        hardware.add(meta['hardware'])
        for path, digest in meta['source_files'].items():
            if path.startswith(('lod_attention/','integrations/')) or path in (
                'benchmarks/retro_lod_adapter.py', 'benchmarks/h100_comparison.py',
                'benchmarks/h100_threshold_three_tier.py', 'benchmarks/amd_threshold_three_tier.py',
                'benchmarks/amd_threshold_merge.py', 'benchmarks/amd_threshold_decode.py',
                'benchmarks/retro_ruler_run.py'):
                source_hashes[path].add(digest)
        data[job['name']] = read(root/Path(job['output']).name)
    if len(hardware) != 1 or any(len(v)!=1 for v in source_hashes.values()):
        raise ValueError('hardware or executed code differs between comparison arms')
    for path, digests in source_hashes.items():
        if hashlib.sha256((root/'source'/path).read_bytes()).hexdigest() not in digests:
            raise ValueError('collected source differs from executed source: '+path)
    frontier = read(root/'three-tier-h100-frontier-v2.json')
    if frontier['stage'] != 'complete' or len(frontier['records']) != 24:
        raise ValueError('independent frontier suite incomplete')
    summary = dict(hardware=list(hardware), source_audit_files=len(source_hashes),
                   frontier_cases=len(frontier['records']),
                   frontier_max_abs=max(r['max_abs'] for r in frontier['records']),
                   model=[], attention=[], checks=[], quality=[], settings=phases['model']['chosen'])
    table = ['| Context | Dense | Top-8 BF16 | Threshold BF16 | Top-8 INT4 | Threshold INT4 |',
             '|---|---:|---:|---:|---:|---:|']
    prefill = ['| Context | Dense | Top-8 BF16 | Threshold BF16 | Top-8 INT4 | Threshold INT4 |',
               '|---|---:|---:|---:|---:|---:|']
    attention_table = ['| Context | KV | Top-eight | Threshold, uncached | Threshold, cached and tuned |',
                       '|---|---|---:|---:|---:|']
    caching_table = ['| Context | KV | Threshold without caching | Threshold with caching | Reduction |',
                     '|---|---|---:|---:|---:|']
    for length in (32768,65536):
        dense = data[f'three-tier-v2-dense-{length//1024}k']
        if (not dense['native_cuda_graph'] or dense['timed_decode_steps']!=1024
                or dense['decode_warmup_steps']!=512 or len([r for r in dense['records'] if not r['warmup']])!=2):
            raise ValueError('dense window differs from LoD')
        ms, seconds = [dense['median_decode_ms_per_token']], [dense['median_prefill_s']]
        for mode in ('three-tier-bf16','three-tier-int4'):
            stem = f'three-tier-v2-{mode}-{length//1024}k'
            arms = {arm:data[stem+'-'+arm] for arm in ('top8','threshold','uncached','top8-after')}
            for arm, result in arms.items():
                measured = [r for r in result['records'] if not r['warmup']]
                if (len(measured)!=2 or result['timed_decode_steps']!=1024
                        or result['decode_warmup_steps']!=512 or not result['lod_cuda_graph']
                        or any(r['lod_layer_catch_up_events_timed']!=128 for r in measured)):
                    raise ValueError('incomplete amortized window: '+stem+'-'+arm)
            before, after = (arms[x]['median_decode_ms_per_token'] for x in ('top8','top8-after'))
            control = statistics.mean((before,after))
            candidate, uncached = (arms[x]['median_decode_ms_per_token'] for x in ('threshold','uncached'))
            ms += [control,candidate]
            seconds += [arms[x]['median_prefill_s'] for x in ('top8','threshold')]
            summary['model'].append(dict(length=length,mode=mode,top8_ms=control,top8_before_ms=before,top8_after_ms=after,threshold_ms=candidate,
                uncached_ms=uncached,reduction_pct=100*(1-candidate/control),dense_speedup=ms[0]/candidate))
            caching_table.append(f'| {length//1024}K | {"INT4" if mode.endswith("int4") else "BF16"} | '+
                f'{uncached:.3f} | {candidate:.3f} | {100*(1-candidate/uncached):.1f}% |')
            check = data[stem+'-check']['records'][0]['graph_same_cache_checks']
            if len(check)<15 or any(r['max_abs']!=0 or r['graph_argmax']!=r['eager_argmax'] for r in check.values()):
                raise ValueError('cached/uncached graph check failed: '+stem)
            summary['checks'].append(dict(length=length,mode=mode,steps=len(check),max_logit_abs=0))
            probe = data[stem+'-tune']
            if probe['stage']!='complete' or len(probe['records'])!=45:
                raise ValueError('incomplete tuning: '+stem)
            selected = phases['model']['chosen'][mode+'/'+str(length)]
            means = {}
            for arm in ('top8','three-tier-reference','three-tier'):
                rows = [r for r in probe['records'] if r['implementation']==arm and
                        (arm!='three-tier' or all(r[k]==v for k,v in selected.items()))]
                if len(rows)!=3:
                    raise ValueError('missing selected real-layer trials')
                means[arm]=statistics.mean(r['microseconds'] for r in rows)
            summary['attention'].append(dict(length=length,mode=mode,**means))
            attention_table.append(f'| {length//1024}K | {"INT4" if mode.endswith("int4") else "BF16"} | '+
                ' | '.join(f'{means[k]:.2f}' for k in ('top8','three-tier-reference','three-tier'))+' |')
        table.append(f'| {length//1024}K | '+' | '.join(f'{x:.3f}' for x in ms)+' |')
        prefill.append(f'| {length//1024}K | '+' | '.join(f'{x:.3f}' for x in seconds)+' |')
    dense_quality = read(Path('results/h100-baselines/ruler-32k-panel/ruler-32k-dense-full.json'))
    dense_rows = {(r['task'],r['source_row']):r for r in dense_quality['records']}
    quality_table = ['| Retrieval task | Examples | Dense | Top-8 BF16 | Threshold BF16 | Top-8 INT4 | Threshold INT4 |',
                     '|---|---:|---:|---:|---:|---:|---:|']
    qdata = {}
    for mode in ('three-tier-bf16','three-tier-int4'):
        for arm in ('top8','threshold'):
            q = data[f'three-tier-v2-{mode}-32k-quality-{arm}']
            if q['stage']!='complete' or len(q['records'])!=32:
                raise ValueError('retrieval panel incomplete')
            qdata[mode,arm]={(r['task'],r['source_row']):r for r in q['records']}
    keys=set(next(iter(qdata.values())))
    if any(set(rows)!=keys for rows in qdata.values()):
        raise ValueError('retrieval samples differ')
    for key in keys:
        group=[dense_rows[key],*(rows[key] for rows in qdata.values())]
        if (len({r['input_token_sha256'] for r in group})!=1
                or len({json.dumps(r['answers']) for r in group})!=1):
            raise ValueError('retrieval input/gold mismatch')
    summary['retrieval_pairs'] = {}
    for mode in ('three-tier-bf16','three-tier-int4'):
        control, candidate = qdata[mode,'top8'], qdata[mode,'threshold']
        if any(control[k]['output_token_ids'][0]!=candidate[k]['output_token_ids'][0] for k in keys):
            raise ValueError('initial prefill prediction changed between routing variants')
        summary['retrieval_pairs'][mode] = dict(
            improved=sum(candidate[k]['score']>control[k]['score'] for k in keys),
            regressed=sum(candidate[k]['score']<control[k]['score'] for k in keys),
            unchanged=sum(candidate[k]['score']==control[k]['score'] for k in keys),
            first_prefill_token_disagreements=0)
    summary['miss_diagnostic'] = []
    for mode in ('three-tier-bf16','three-tier-int4'):
        probe=data[f'three-tier-v2-{mode}-32k-quality-uncached-misses']
        if probe['stage']!='complete' or len(probe['records'])!=2:
            raise ValueError('uncached retrieval diagnostic incomplete')
        for row in probe['records']:
            key=(row['task'],row['source_row'])
            cached=qdata[mode,'threshold'][key]
            if row['input_token_sha256']!=cached['input_token_sha256'] or row['output_token_ids']!=cached['output_token_ids']:
                raise ValueError('cached/uncached retrieval predictions disagree')
            summary['miss_diagnostic'].append(dict(mode=mode,task=key[0],source_row=key[1],score=row['score'],identical_output_tokens=True))
    for task in ('niah_multikey_3','niah_multivalue'):
        taskkeys=[k for k in keys if k[0]==task]
        scores=[100*statistics.mean(dense_rows[k]['score'] for k in taskkeys)]
        for mode in ('three-tier-bf16','three-tier-int4'):
            for arm in ('top8','threshold'):
                scores.append(100*statistics.mean(qdata[mode,arm][k]['score'] for k in taskkeys))
        summary['quality'].append(dict(task=task,examples=len(taskkeys),scores=scores))
        quality_table.append(f'| {task} | {len(taskkeys)} | '+' | '.join(f'{x:.2f}' for x in scores)+' |')
    audit=read(root/'first-response-audit.json')
    if audit['stage']!='complete':
        raise ValueError('first-response scoring audit incomplete')
    for path in root.glob('three-tier-v2-*-32k-quality-*.json'):
        if path.name in audit['input_sha256'] and hashlib.sha256(path.read_bytes()).hexdigest()!=audit['input_sha256'][path.name]:
            raise ValueError('scoring audit input changed')
    summary['quality_scoring']='quality stores full fixed-length output; quality_first_response stops at EOS/EOT'
    summary['quality_first_response']=audit['summary']
    summary['retrieval_pairs_first_response']=audit['paired_outcomes']
    first_table=['| Retrieval task | Examples | Dense | RetroInfer | Top-8 BF16 | Threshold BF16 | Top-8 INT4 | Threshold INT4 |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|']
    lookup={(r['mode'],r['variant'],r['task']):r['first_response'] for r in audit['summary']}
    for task in ('niah_multikey_3','niah_multivalue'):
        values=[lookup[m,v,task] for m,v in (('dense','reference'),('retroinfer','reference'),('bf16','top8'),('bf16','threshold'),('int4','top8'),('int4','threshold'))]
        first_table.append(f'| {task} | 16 | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    (root/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    text = '''# AMD-derived three-tier threshold decode on H100

Llama-3.1-8B-Instruct, BF16 weights, batch one, H100 80GB HBM3, RetroInfer runtime. All GPU experiments use uv and self-documenting cluster-run snapshots. These runs use the corrected cache-update kernel.

## Complete model, including amortized cache updates

Latency in ms/token. All arms graph the complete model decode step. Each run has one warmup repetition and two measured repetitions; each repetition has 512 untimed steps followed by 1,024 timed steps. Top-eight decode rows average the medians of fresh controls before and after the candidate; the separate controls remain in SUMMARY.json. LoD timed windows each include 128 layer cache-update events, including the cached-mean refresh when enabled. Prompts and teacher-forced decode inputs are identical across arms; fresh dense controls use the same GPU.

'''+ '\n'.join(table)+'''

The threshold rule is normalized-query proxy mass > 16 / number of nonempty regions. Regions above the existing 1,024-entry eligibility limit stay summarized. Region counts are uncapped. Each opened region contributes its top two exact 16-token pages plus one disjoint residual summary. This differs from opening the fixed top eight regions; it is a routing-policy comparison as well as a performance comparison.

## Prefill and maintenance

Full-model prefill in seconds, including explicit final cache catch-up. Prefill retains top-eight region routing and opens all leaves in those regions. Its exact initial 16,384-token chunk, local attention and sink attention are unchanged. Top-two page selection applies to decode. Cache allocation and graph capture are outside prefill and decode timings, consistently with the existing harness; raw records retain their separate costs.

'''+ '\n'.join(prefill)+'''

Persistent BF16-rounded region means are refreshed at the serving pool's existing install/catch-up boundaries. No leaf-index rebuild, exact-KV copy, GQA expansion, selected-count host readback or selected-region cap is added. The uncached controls in SUMMARY.json use the same selected settings and frontier and include the same amortized state updates.

The following full-model ms/token comparison isolates mean caching: region selections, routing tiles, page workers, precision and all other attention settings match between each pair. The cost of maintaining the cached means is included.

'''+ '\n'.join(caching_table)+'''

## Validation and retrieval

The independent PyTorch frontier suite passed 24 BF16/INT4 cases covering uneven region sizes, overflow page tables, partial pages, selections opening every eligible region or none, empty remote context, ties, graph replay and current-KV appends. Both cached and uncached H100 arithmetic were checked. Real-query tuning uses layers 0, 15 and 31; every threshold candidate selects the same regions as the uncached control. Complete-model graph checks compare cached graph logits with uncached eager logits on the identical cache, including steps around cache updates. All sampled logits agree exactly.

The limited 32K retrieval panel uses 16 examples per task, 32 examples per precision, with greedy generation and the existing task token limits. Top-eight controls were rerun. Dense and RetroInfer scores reuse matching examples from the frozen panels; token hashes and answers are checked. Within each precision, the initial prefill prediction matches between top-eight and threshold arms on every example. This is not a 13-task RULER average or a 64K quality evaluation.

### First response

These scores stop at the first EOS/EOT token, matching the main 13-task quality comparisons. The saved generations were rescored on CPU; `first-response-audit.json` records input and tokenizer hashes, per-example scores and stop positions, with the analysis in `first-response-audit.py`.

'''+ '\n'.join(first_table)+'''

BF16 improves eight examples with no regressions; INT4 improves ten and regresses one. The earlier INT4 multikey score of 93.75% counted all generated text; first-response accuracy is 87.50%. The BF16 top-eight baseline similarly changes from 50.00% over full output to 43.75% on the first response. Full-output scores remain below for transparency.

### Full fixed-length output

These scores can award credit for text generated after the first EOS/EOT token.

'''+ '\n'.join(quality_table)+'''

For full-output scoring, BF16 improves seven examples and regresses none. INT4 improves eleven examples and regresses one (`niah_multikey_3`, source row 187), for a net gain of ten. BF16's remaining full-output miss is source row 59. Both source rows were rerun in both precisions without mean caching; every generated token matches the corresponding cached run. Mean caching leaves these mistakes and the INT4 regression unchanged. The full panel and this single regression are retained explicitly; thresholds remain opt-in.

## Implementation and reproduction

`benchmarks/h100_threshold_three_tier.py` adapts the shared AMD threshold producer and page consumer to Llama GQA (32 query / 8 KV heads, dimension 128). The producer computes native and RMS-normalized proxy logits, combines tile normalizers, selects regions while computing closed-summary attention with a probability-residual precision correction, and packs region IDs with prefix scans. Persistent page workers consume device-resident actual counts. A dimension-blocked merge combines coarse, page/residual and local/sink fields and appends the current KV exactly once.

Enable `LOD_H100_THRESHOLD_MULTIPLIER=16`, `LOD_H100_THRESHOLD_IMPLEMENTATION=three-tier`, and `LOD_H100_THREE_TIER_CACHED_MEANS=1` before creating the cache. Use `three-tier-bf16` or `three-tier-int4` mode. The measured tile/worker/warp/merge settings are in SUMMARY.json and three-tier-v2-model-runs.json. They correspond to `LOD_H100_THRESHOLD_TILE`, `LOD_H100_THREE_TIER_WORKERS`, `LOD_H100_THREE_TIER_PAGE_WARPS`, and `LOD_H100_THRESHOLD_MERGE_BLOCK`. This path supports batch one and per-region thresholds. Serving defaults are unchanged.

The threshold page-summary scan uses `engine.leaf_block_n`: 32 pages for BF16 and 16 for INT4. The raw `lod_route_geometry.page_block` field records the unchanged production `decode_block_n` attribute. The batch adapter explicitly rejects H100 threshold settings; its configuration guard was checked before GPU setup and is outside the timed batch-one path.

Fixed-query graph timings cover the complete attention call plus the same local-length rollback for every arm; they exclude model layers and cache maintenance. Their raw tuning trials and selected averages are kept separately from complete-model results.

Attention-call latency below is in microseconds, averaged over real queries from layers 0, 15 and 31. The uncached column uses the original 64-region-tile / 64-worker setting; the cached column also uses its selected tuning settings. Full-model uncached controls instead use exactly the candidate settings to isolate mean caching. Raw `mean_opened_region_entries` describes the total sizes of selected regions; exact remote attention reads at most two 16-token pages per selected region.

'''+ '\n'.join(attention_table)+'''

Packing uses linear prefix scans. The current coarse producer combines all tile normalizers in each coarse tile, so that normalization step still has a quadratic-in-tile-count term. Prefill and existing cache construction/update paths have no new subquadratic-work guarantee. This report claims measured latency only.

Commands, runtime flags, GPU identity, exit states and source hashes are retained in provenance/. The report audits identical executed LoD/adapter code and hardware across final arms and verifies the collected source/ snapshot. The first frontier attempt failed on the old Transformers import facade; unfinished initial tuning runs were superseded to expand page-worker counts. Failed/cancelled/partial trials remain recorded and are excluded from the final comparison.
'''
    (root/'REPORT.md').write_text(text)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
