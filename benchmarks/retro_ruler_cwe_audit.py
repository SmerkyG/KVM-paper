"""Audit CWE references against the numbered input list and inspect predictions."""
import argparse
import collections
import gzip
import json
import re
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    with gzip.open(a.cohort, 'rt') as f:
        cohort = json.load(f)
    data = json.loads(a.result.read_text())
    samples = {(s['task'], s['source_row']): s for s in cohort['samples']}
    rows = [r for r in data['records'] if r['task'] == 'cwe']
    if data['stage'] != 'complete' or len(rows) != 32:
        raise ValueError('requires the 32 completed CWE rows')
    records = []
    for row in rows:
        sample = samples[(row['task'], row['source_row'])]
        if sample['answers'] != row['answers'] or sample['prompt_sha256'] != row['prompt_sha256']:
            raise ValueError('row and frozen prompt mismatch')
        user_marker = '<|start_header_id|>user<|end_header_id|>\n\n'
        list_text = sample['prompt'].rsplit(user_marker, 1)[1].split('\nQuestion:', 1)[0]
        entries = re.findall(r'(?:^|\s)(\d+)\.\s+(\S+)', list_text)
        if [int(n) for n, _ in entries] != list(range(1, len(entries) + 1)):
            raise ValueError(f'nonconsecutive list: {row["index"]}')
        counts = collections.Counter(word.lower() for _, word in entries)
        references = [word.lower() for word in sample['answers']]
        refs = [counts[word] for word in references]
        other = [count for word, count in counts.items() if word not in references]
        starts_the = bool(re.match(r'^\s*1\.\s*the\b', row['prediction'], re.I))
        records.append({'index': row['index'], 'entries': len(entries),
                        'reference_counts': refs, 'other_max_count': max(other),
                        'references_strict_top10': len(references) == 10 and len(set(references)) == 10
                            and min(refs) > max(other),
                        'starts_with_the': starts_the, 'score': row['score'],
                        'prediction': row['prediction']})
    output = {'n': len(records), 'mode': data['mode'],
              'references_strict_top10': sum(r['references_strict_top10'] for r in records),
              'starts_with_the': sum(r['starts_with_the'] for r in records),
              'zero_score': sum(r['score'] == 0 for r in records), 'records': records}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k:v for k,v in output.items() if k != 'records'}))
    print('reference count ranges:', min(min(r['reference_counts']) for r in records),
          max(max(r['reference_counts']) for r in records))
    print('max distractor count:', max(r['other_max_count'] for r in records))
    print('entry count range:', min(r['entries'] for r in records), max(r['entries'] for r in records))


if __name__ == '__main__':
    main()
