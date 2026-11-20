"""Validate and summarize paired AMD threshold-routing screening results."""
import argparse
import json
import statistics
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    tasks=('niah_multikey_2','niah_multikey_3')
    configs=('top8','8','16','32','64')
    lines=['# AMD two-tier threshold routing: targeted 64K screening','',
           'Qwen3.8-27B-FP8 and K2-Horizon-32B-FP8, batch one, BF16 KV, '
           'two-tier LoD and greedy decoding. For each task, the eight fixed examples '
           'include four historical two-tier failures and four successes. This deliberately '
           'difficult, nonrepresentative subset is useful for screening but is not a '
           'full-panel accuracy estimate. Fresh fixed-eight controls and all threshold '
           'settings use identical tokenized prompts and answers within each model.', '',
           'Threshold routing opens eligible regions with normalized-query proxy mass '
           'greater than c times the mean over nonempty remote regions; regions over '
           '1,024 entries remain summaries. The existing AMD fixed-eight router uses native '
           'query scores, so the comparison changes query normalization as well as the '
           'selection rule. Prefill uses the existing fixed-eight LoD path. '
           'The last-step region counts below average across the sampled requests, '
           'layers, and query heads; they are not averages over all generated tokens.', '']
    for model in ('qwen','k2'):
        runs={c:json.loads((args.results/f'{model}-ruler-difficult-{c}.json').read_text()) for c in configs}
        union_path=args.results/'k2-ruler-difficult-32-union.json'
        if model=='k2' and union_path.exists():
            union_data=json.loads(union_path.read_text())
            if union_data['stage']=='complete':
                if not union_data.get('threshold_union') or union_data.get('threshold_multiplier')!=32:
                    raise ValueError('incorrect union quality metadata')
                runs['32-union']=union_data
        if any(d['stage']!='complete' or d['limit_per_task']!=8 or d['selection']!='half-failures' for d in runs.values()):
            raise ValueError(f'incomplete or mismatched selection for {model}')
        rows={c:{(r['task'],r['doc_id']):r for r in d['records']} for c,d in runs.items()}
        base=rows['top8']
        if len(base)!=16 or any(set(v)!=set(base) for v in rows.values()):
            raise ValueError(f'cohort mismatch for {model}')
        for c,group in rows.items():
            for key,row in group.items():
                if any(row[field]!=base[key][field] for field in ('prompt_hash','input_token_sha256','target_hash','targets')):
                    raise ValueError(f'prompt or target mismatch {model}/{c}/{key}')
        lines += [f'## {model.upper()} quality screening','',
                  '| Routing | Multi-key 2 | Multi-key 3 | Mean regions per head at last step | Paired examples improved / declined |',
                  '|---|---:|---:|---:|---:|']
        for c,d in runs.items():
            group=rows[c]
            scores=[100*sum(r['score'] for r in group.values() if r['task']==task)/8 for task in tasks]
            counts=[n for r in group.values() for layer in r.get('last_decode_routing',[]) for n in layer['regions']]
            if c not in ('top8','32-union') and not counts:raise ValueError(f'no routing counts for {model}/{c}')
            mean=f'{statistics.mean(counts):.2f}' if counts else ('fixed 8 maximum' if c=='top8' else 'not recorded')
            improved=sum(group[k]['score']>base[k]['score'] for k in base)
            declined=sum(group[k]['score']<base[k]['score'] for k in base)
            label='Fixed eight' if c=='top8' else ('c=32 GQA union' if c=='32-union' else f'c={c}')
            lines.append(f'| {label} | {scores[0]:.2f}% | {scores[1]:.2f}% | {mean} | {improved} / {declined} |')
        lines += ['', 'A single answer changes a task score by 12.5 points in this screen. '
                  'The fresh fixed-eight controls reproduced historical failures as follows:', '',
                  '| Task | Historical failures still failing | Historical successes newly failing |',
                  '|---|---:|---:|']
        for task in tasks:
            subset=[r for r in base.values() if r['task']==task]
            failed=[r for r in subset if r['historical_two_tier_score']<1]
            new_failures=sum(r['historical_two_tier_score']==1 and r['score']<1 for r in subset)
            lines.append(f'| {task} | {sum(r["score"]<1 for r in failed)} / {len(failed)} | {new_failures} / {8-len(failed)} |')
        lines += ['', 'Historical and fresh runs differ in runtime and source revisions; '
                  'the cause of changed outcomes has not been isolated. Qwen and K2 '
                  'subsets were selected independently.', '']
    prefix_paths={
        'qwen':[args.results/'qwen-ruler-prefix32-top8.json', args.results/'qwen-ruler-prefix32-16.json'],
        'k2':[args.results/'k2-ruler-prefix32-top8.json', args.results/'k2-ruler-prefix32-32.json',
              args.results/'k2-ruler-prefix32-32-union.json'],
    }
    if all(path.exists() for paths in prefix_paths.values() for path in paths):
        cohorts={model:[json.loads(path.read_text()) for path in paths] for model,paths in prefix_paths.items()}
        if all(d['stage']=='complete' for values in cohorts.values() for d in values):
            lines += ['## Paired 32-example-per-task follow-up', '',
                      'These are the first 32 archived examples of each multi-key task, '
                      'rather than the failure-enriched screen. Every row uses the same '
                      'tokenized inputs and reference answers within each model.', '',
                      '| Model | Routing | Multi-key 2 | Multi-key 3 | Paired examples improved / declined |',
                      '|---|---|---:|---:|---:|']
            for model,values in cohorts.items():
                labels=('fixed eight','c=16') if model=='qwen' else ('fixed eight','c=32','c=32 GQA union')
                groups=[{(r['task'],r['doc_id']):r for r in d['records']} for d in values]
                base=groups[0]
                if len(base)!=64 or any(len(g)!=64 or set(g)!=set(base) for g in groups):
                    raise ValueError(f'32-example follow-up cohort mismatch: {model}')
                for label,group in zip(labels,groups):
                    for key,row in group.items():
                        if any(row[field]!=base[key][field] for field in ('prompt_hash','input_token_sha256','target_hash','targets')):
                            raise ValueError(f'follow-up input mismatch: {model}/{label}/{key}')
                    scores=[100*sum(r['score'] for r in group.values() if r['task']==task)/32 for task in tasks]
                    improved=sum(group[k]['score']>base[k]['score'] for k in base)
                    declined=sum(group[k]['score']<base[k]['score'] for k in base)
                    lines.append(f'| {model.upper()} | {label} | {scores[0]:.2f}% | {scores[1]:.2f}% | {improved} / {declined} |')
            lines += ['', 'A single answer changes a task score by 3.125 points. '
                      'These two tasks are not a full-panel average.', '']
    lines += ['## 64K end-to-end decode timing','',
              'The same batch-one ProLong prompt, 512 generated decode steps, '
              'one warmup and two timed repetitions on an MI325X. Timed decode '
              'includes the serving path and cache updates; prefill is excluded. '
              'Generation is autoregressive, so different methods can follow different '
              'token trajectories and open different regions; the timings are not quality matched.', '',
              '| Model | Fixed eight ms/token | c=16 ms/token | c=32 ms/token |',
              '|---|---:|---:|---:|']
    instability=[]
    for model in ('qwen','k2'):
        paths=[args.results/f'{model}-top8-64k-speed-pilot.json']+[args.results/f'{model}-c{c}-64k-speed.json' for c in (16,32)]
        data=[json.loads(p.read_text()) for p in paths]
        hashes=[d['measurements']['65536']['prompts'][0]['token_sha256'] for d in data]
        if len(set(hashes))!=1 or any(d['batch_size']!=1 or d['decode_tokens']!=513 for d in data):
            raise ValueError(f'timing inputs or geometry mismatch for {model}')
        values=[d['measurements']['65536']['decode_ms_per_batch_step'] for d in data]
        lines.append(f'| {model.upper()} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} |')
        for label,d in zip(('fixed eight','c=16','c=32'),data):
            if not d['measurements']['65536']['greedy_output_identical']:
                instability.append(f'{model.upper()} {label}')
    union=json.loads((args.results/'k2-c32-union-64k-speed.json').read_text())
    standard=json.loads((args.results/'k2-c32-64k-speed.json').read_text())
    control=json.loads((args.results/'k2-top8-64k-speed-pilot.json').read_text())
    um=union['measurements']['65536'];sm=standard['measurements']['65536'];cm=control['measurements']['65536']
    if not union.get('decode_threshold_union') or um['prompts'][0]['token_sha256']!=sm['prompts'][0]['token_sha256']:
        raise ValueError('K2 union timing metadata or prompt mismatch')
    lines += ['', 'Repeat-to-repeat greedy outputs differed in: '+(', '.join(instability) if instability else 'none of these runs')+'. '
              'This affects interpretation of small quality differences; the fixed-cache synthetic '
              'kernel checks were repeatable. The selected-region work differs across methods.', '',
              'An experimental K2 c=32 GQA-union worker measured '
              f'{um["decode_ms_per_batch_step"]:.3f} ms/token on the same prompt, '
              f'versus {sm["decode_ms_per_batch_step"]:.3f} for the first threshold '
              f'worker and {cm["decode_ms_per_batch_step"]:.3f} for fixed eight. '
              'This worker shares pages across '
              'query heads and uses BF16 attention weights; repeated greedy output '
              'differed and it remains experimental.', '',
              'The threshold path is opt-in and currently supports batch-one two-tier BF16. '
              'It is not a serving default, and this screening does not establish a best multiplier.', '',
              '[Synthetic correctness](synthetic-determinism.json), '
              '[K2 kernel tuning](k2-synthetic-tuning.json), '
              '[Qwen kernel tuning](qwen-synthetic-tuning.json), '
              '[K2 GQA-union tuning](k2-union-synthetic-tuning.json), and '
              '[Qwen GQA-union tuning](qwen-union-synthetic-tuning.json).', '',
              '[K2 union timing](k2-c32-union-64k-speed.json), '
              '[Qwen c=16 timing](qwen-c16-64k-speed.json), '
              '[Qwen c=32 timing](qwen-c32-64k-speed.json), '
              '[K2 c=16 timing](k2-c16-64k-speed.json), '
              '[K2 c=32 timing](k2-c32-64k-speed.json).', '']
    for model in ('qwen','k2'):
        lines.append(f'{model.upper()} screening outputs: ' + ', '.join(
            f'[{"fixed eight" if c=="top8" else "c="+c}]({model}-ruler-difficult-{c}.json)' for c in configs) + '.')
    if (args.results/'k2-ruler-difficult-32-union.json').exists():
        lines.append('[K2 c=32 GQA-union screening outputs](k2-ruler-difficult-32-union.json).')
    if all(path.exists() for paths in prefix_paths.values() for path in paths):
        for model,paths in prefix_paths.items():
            lines.append(f'{model.upper()} 32-example-per-task outputs: ' + ', '.join(
                f'[{path.stem}]({path.name})' for path in paths) + '.')
    lines.append('')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__=='__main__':
    main()
