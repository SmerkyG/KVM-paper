"""Freeze and shard the archived 500-example AMD RULER request logs."""
import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.amd_threshold_ruler import score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shard-size',type=int,default=50)
    args=p.parse_args()
    if args.shard_size<1 or 500%args.shard_size:
        p.error('shard size must divide 500')
    tasks={'qwen':['niah_multikey_2','niah_multikey_3','ruler_qa_hotpot','ruler_qa_squad'],
           'k2':['niah_multikey_2','niah_multikey_3','niah_multivalue','ruler_qa_hotpot','ruler_qa_squad']}
    manifest={'stage':'preparing','samples_per_task':500,'shard_size':args.shard_size,'models':{}}
    args.output.mkdir(parents=True,exist_ok=True)
    for model,names in tasks.items():
        manifest['models'][model]={}
        for task in names:
            paths=list((args.archive/model/'two-tier'/task).rglob('samples_*.jsonl'))
            if len(paths)!=1:raise ValueError(f'expected one archived file: {model}/{task}: {paths}')
            raw=paths[0].read_bytes(); rows=[json.loads(line) for line in raw.splitlines()]
            rows.sort(key=lambda r:r['doc_id'])
            if len(rows)!=500 or len({r['doc_id'] for r in rows})!=500:
                raise ValueError(f'expected 500 unique archived examples: {model}/{task}')
            for r in rows:
                if abs(score(r,r['filtered_resps'][0])-r['65536'])>1e-12:
                    raise ValueError(f'archived score mismatch: {model}/{task}/{r["doc_id"]}')
            shards=[]
            for start in range(0,500,args.shard_size):
                part=rows[start:start+args.shard_size]
                path=args.output/model/f'shard-{start//args.shard_size:02d}'/task/'samples_frozen.jsonl'
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('w') as stream:
                    for r in part:
                        compact={k:r[k] for k in ('doc_id','arguments','filtered_resps','doc_hash','prompt_hash','target_hash','65536')}
                        compact['doc']={'outputs':r['doc']['outputs']}
                        stream.write(json.dumps(compact,ensure_ascii=False)+'\n')
                shards.append({'index':start//args.shard_size,'count':len(part),
                               'doc_ids':[r['doc_id'] for r in part],
                               'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                               'path':str(path.relative_to(args.output))})
            manifest['models'][model][task]={'source':str(paths[0]),'source_sha256':hashlib.sha256(raw).hexdigest(),
                                              'shards':shards}
            print(json.dumps({'model':model,'task':task,'samples':len(rows),'shards':len(shards)}),flush=True)
    manifest['stage']='complete'
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
