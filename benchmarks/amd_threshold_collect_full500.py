"""Validate and report complete 500-example paired AMD RULER tasks."""
import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.amd_threshold_ruler import score


def dense_scores(archive, frozen, model, task, cohort):
    paths=list((archive/model/'dense'/task).rglob('samples_*.jsonl'))
    if len(paths)!=1:
        raise ValueError(f'expected one dense archive: {model}/{task}: {paths}')
    digest=hashlib.sha256()
    dense={}
    with paths[0].open('rb') as stream:
        for line in stream:
            digest.update(line)
            row=json.loads(line)
            if row['doc_id'] in dense:
                raise ValueError(f'duplicate dense example: {model}/{task}/{row["doc_id"]}')
            if abs(score(row,row['filtered_resps'][0])-row['65536'])>1e-12:
                raise ValueError(f'dense scorer mismatch: {model}/{task}/{row["doc_id"]}')
            dense[row['doc_id']]=row
    if len(dense)!=500:
        raise ValueError(f'expected 500 dense examples: {model}/{task}')
    seen=set()
    for shard in cohort['shards']:
        with (frozen/shard['path']).open() as stream:
            for line in stream:
                source=json.loads(line)
                row=dense.get(source['doc_id'])
                if row is None or source['doc_id'] in seen:
                    raise ValueError(f'dense cohort mismatch: {model}/{task}/{source["doc_id"]}')
                seen.add(source['doc_id'])
                if (any(source[k]!=row[k] for k in ('doc_hash','prompt_hash','target_hash','arguments'))
                        or source['doc']['outputs']!=row['doc']['outputs']):
                    raise ValueError(f'dense input mismatch: {model}/{task}/{source["doc_id"]}')
    if seen!=set(dense):
        raise ValueError(f'dense cohort incomplete: {model}/{task}')
    return 100*sum(row['65536'] for row in dense.values())/500, str(paths[0]), digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dense-archive',type=Path,
                   help='Directory containing model/dense/task archived RULER logs')
    args=p.parse_args()
    manifest=json.loads((args.frozen/'manifest.json').read_text())
    if manifest['stage']!='complete' or manifest['samples_per_task']!=500:
        raise ValueError('incomplete frozen manifest')
    lines=['# AMD two-tier threshold: paired 500-example RULER tasks','',
           'Qwen3.8-27B-FP8 and K2-Horizon-32B-FP8, batch one, BF16 KV, '
           '64K archived RULER inputs, greedy generation, and two-tier sparse '
           'prefill. Each task has all 500 examples, with matched input tokens and '
           'answers across fresh fixed-eight and c=16 threshold runs. The threshold '
           'uses normalized-query mass; the existing fixed-eight AMD router uses native '
           'query scores, so both the routing score and selection rule change.', '',
           ('The archived dense and fixed-eight scores are included for context; those historical ' if args.dense_archive
            else 'The archived fixed-eight score is included for context; the historical ')+
           'runs used a different runtime and code revision. Paired changes compare '
           'only the two fresh runs. QA and multi-value use the original RULER '
           'substring-recall metric, so an example can have a fractional score.', '']
    if args.dense_archive:
        lines += ['Archived dense and the fresh runs use the same 500 requests and answers per task: '
                  'document, prompt, target, and request arguments were validated against the frozen cohort. '
                  'The dense comparison remains descriptive because runtime and scheduling differ.', '',
                  '| Model | Task | Archived dense | Archived fixed eight | Fresh fixed eight | Threshold c=16 | vs fresh top-8, points | vs dense, points | Improved / declined vs fresh top-8 |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    else:
        lines += ['| Model | Task | Archived fixed eight | Fresh fixed eight | Threshold c=16 | Difference, points | Paired examples improved / declined |',
                  '|---|---|---:|---:|---:|---:|---:|']
    summaries=[]
    for model,tasks in manifest['models'].items():
        for task,cohort in tasks.items():
            groups={}
            for variant in ('top8','c16'):
                rows={}
                for shard in cohort['shards']:
                    path=args.results/model/task/variant/f'shard-{shard["index"]:02d}.json'
                    if not path.exists():raise ValueError(f'missing shard: {path}')
                    data=json.loads(path.read_text())
                    if data['stage']!='complete' or data['mode']!='two-tier' or data['tasks']!=[task] or data['batch_size']!=1:
                        raise ValueError(f'invalid shard metadata: {path}')
                    if variant=='top8' and data['threshold_multiplier'] is not None:raise ValueError(f'wrong control: {path}')
                    if variant=='c16' and (data['threshold_multiplier']!=16 or data.get('threshold_union',False)):
                        raise ValueError(f'wrong threshold: {path}')
                    ids=[r['doc_id'] for r in data['records']]
                    if len(ids)!=shard['count'] or set(ids)!=set(shard['doc_ids']):raise ValueError(f'invalid shard examples: {path}')
                    for row in data['records']:
                        if row['doc_id'] in rows or row['task']!=task:raise ValueError(f'duplicate or wrong task: {path}')
                        rows[row['doc_id']]=row
                if len(rows)!=500:raise ValueError(f'expected 500 rows: {model}/{task}/{variant}')
                groups[variant]=rows
            base=groups['top8'];candidate=groups['c16']
            if set(base)!=set(candidate):raise ValueError(f'cohort mismatch: {model}/{task}')
            for key,row in base.items():
                new=candidate[key]
                if any(row[field]!=new[field] for field in ('prompt_hash','doc_hash','target_hash','input_token_sha256','targets','historical_two_tier_score')):
                    raise ValueError(f'paired input mismatch: {model}/{task}/{key}')
            old=100*sum(r['historical_two_tier_score'] for r in base.values())/500
            a=100*sum(r['score'] for r in base.values())/500
            b=100*sum(r['score'] for r in candidate.values())/500
            improved=sum(candidate[k]['score']>base[k]['score'] for k in base)
            declined=sum(candidate[k]['score']<base[k]['score'] for k in base)
            row={'model':model,'task':task,'examples':500,'archived_fixed8':old,'fresh_fixed8':a,
                 'threshold_c16':b,'delta_points':b-a,'improved':improved,'declined':declined}
            if args.dense_archive:
                dense,source,digest=dense_scores(args.dense_archive,args.frozen,model,task,cohort)
                row.update(archived_dense=dense,dense_source=source,dense_source_sha256=digest,
                           threshold_minus_dense_points=b-dense)
                lines.append(f'| {model.upper()} | {task} | {dense:.2f}% | {old:.2f}% | {a:.2f}% | {b:.2f}% | {b-a:+.2f} | {b-dense:+.2f} | {improved} / {declined} |')
            else:
                lines.append(f'| {model.upper()} | {task} | {old:.2f}% | {a:.2f}% | {b:.2f}% | {b-a:+.2f} | {improved} / {declined} |')
            summaries.append(row)
    lines += ['', 'These are selected tasks, not a 13-task RULER average. '
              'Both fresh LoD paths use fixed-eight sparse prefill, so the experiment does not '
              'measure a threshold-prefill implementation.', '',
              f'Frozen cohort manifest: `{args.frozen.resolve() / "manifest.json"}`.', '']
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text('\n'.join(lines))
    args.output.with_suffix('.json').write_text(json.dumps({'stage':'complete','tasks':summaries},indent=2)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
