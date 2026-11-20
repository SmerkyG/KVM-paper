"""Run RULER using page-threshold geometry chosen by a completed timing probe."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import runpy
import statistics
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',type=Path,required=True)
    a,rest=p.parse_known_args();raw=a.probe.read_bytes();data=json.loads(raw);assert data['stage']=='complete' and data['length']==32768
    assert data.get('page_threshold_rule','page')==os.environ.get('LOD_H100_PAGE_THRESHOLD_RULE','page')
    assert data.get('page_query','native')==os.environ.get('LOD_H100_PAGE_QUERY','native')
    assert data.get('region_multiplier',16)==float(os.environ['LOD_H100_THRESHOLD_MULTIPLIER'])
    c=float(os.environ['LOD_H100_PAGE_THRESHOLD']);groups=collections.defaultdict(list)
    for row in data['records']:
        if row['c']==c and row.get('residual','region')==os.environ.get('LOD_H100_PAGE_RESIDUAL','region'):
            groups[row['workers'],row['warps']].append(row)
    assert groups and all({r['layer'] for r in rows}=={0,15,31} and len(rows)==3 for rows in groups.values())
    workers,warps=min(groups,key=lambda k:statistics.mean(r['microseconds'] for r in groups[k]))
    os.environ['LOD_H100_THREE_TIER_WORKERS']=str(workers);os.environ['LOD_H100_PAGE_THRESHOLD_WARPS']=str(warps)
    os.environ['LOD_H100_PAGE_PROBE_SHA256']=hashlib.sha256(raw).hexdigest()
    print(json.dumps(dict(page_multiplier=c,workers=workers,warps=warps,probe_sha256=os.environ['LOD_H100_PAGE_PROBE_SHA256'])),flush=True)
    sys.argv=['benchmarks.retro_ruler_run',*rest];runpy.run_module('benchmarks.retro_ruler_run',run_name='__main__')


if __name__=='__main__':main()
