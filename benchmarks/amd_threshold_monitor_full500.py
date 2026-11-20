"""Summarize completion and failures for submitted AMD RULER shards."""
import argparse
import collections
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--finished',type=Path,default=Path('/home/dan/subusers/agent/.cluster-run/agent/finished'))
    args=p.parse_args();groups=collections.defaultdict(lambda:{'submitted':0,'finished':0,'running_records':0,'failed':[]})
    for submission in sorted(args.output.glob('*-submissions.json')):
        log=json.loads(submission.read_text())
        for job in log['jobs']:
            key=(job['model'],job['task'],job['variant']);g=groups[key];g['submitted']+=1
            finished=args.finished/job['id']
            if finished.exists():
                fields=finished.read_text().split('\t')
                if len(fields)<9 or fields[7]!='finished' or fields[8]!='0':
                    g['failed'].append(job['id']);continue
            path=Path(job['output'])
            if path.exists():
                try:
                    data=json.loads(path.read_text())
                    g['running_records']+=len(data['records'])
                    if data['stage']=='complete':g['finished']+=1
                except (OSError,json.JSONDecodeError,KeyError):pass
            elif finished.exists():g['failed'].append(job['id']+' (missing output)')
    for key,g in sorted(groups.items()):print(json.dumps({'model':key[0],'task':key[1],'variant':key[2],**g}))


if __name__=='__main__':main()
