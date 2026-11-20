"""Require the optimized decode to reproduce the prior and reference answers."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text())
    if candidate['stage'] != 'complete' or len(candidate['records']) != 32:
        raise ValueError('expected 32 completed candidate examples')
    rows = {(r['task'], r['row']): r for r in candidate['records']}
    if len(rows) != 32:
        raise ValueError('duplicate candidate examples')
    for path in args.reference:
        reference = json.loads(path.read_text())
        if reference['stage'] != 'complete' or reference['cohort_sha256'] != candidate['cohort_sha256']:
            raise ValueError(f'reference provenance mismatch: {path}')
        original = {(r['task'], r['row']): r for r in reference['records']}
        if set(original) != set(rows):
            raise ValueError(f'reference examples mismatch: {path}')
        for key, row in rows.items():
            for field in ('input_token_sha256', 'answers', 'output_token_ids'):
                if row[field] != original[key][field]:
                    raise ValueError(f'{field} differs for {key} against {path}')
    result = {'stage': 'complete', 'examples': len(rows), 'references': [str(p) for p in args.reference]}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
