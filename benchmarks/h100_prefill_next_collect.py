"""Collect completed H100 prefill experiments and their run provenance.

This read-only monitor does not launch experiments. Each source snapshot and
command is identified by the accompanying cluster-run metadata.
"""
import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

REMOTE = r'''
import glob, json, os, pathlib
state = pathlib.Path('/home/ubuntu/.local/state/lod-cluster-run')
result = {}
for manifest in glob.glob('/home/ubuntu/comparison/results/prefill-next-*-runs.json'):
    data = json.loads(pathlib.Path(manifest).read_text())
    def visit(value):
        if isinstance(value, dict):
            for child in value.values(): visit(child)
        elif isinstance(value, str):
            path = state / value / 'run.json'
            if not path.exists(): return
            metadata = json.loads(path.read_text())
            row = {'metadata': metadata}
            if metadata.get('state') == 'finished':
                log = path.parent / 'output.log'
                row['log'] = log.read_text(errors='replace') if log.exists() else ''
                cmd = metadata['command']
                if metadata.get('exit_code') == 0 and '--output' in cmd:
                    output = pathlib.Path(cmd[cmd.index('--output') + 1])
                    if output.is_file():
                        row['filename'] = output.name
                        row['output'] = json.loads(output.read_text())
            result[value] = row
    visit(data)
print(json.dumps(result))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = args.output / 'provenance'
    provenance.mkdir(exist_ok=True)
    while True:
        try:
            raw = subprocess.check_output(['ssh', '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=8',
                                           args.host, 'python3', '-c', shlex.quote(REMOTE)], timeout=45, text=True)
            rows = json.loads(raw)
            pending = []
            for run_id, row in rows.items():
                metadata = row['metadata']
                if metadata.get('state') != 'finished':
                    pending.append(run_id)
                    continue
                dest = provenance / run_id
                dest.mkdir(exist_ok=True)
                (dest / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
                (dest / 'output.log').write_text(row.get('log', ''))
                if 'output' in row:
                    (args.output / row['filename']).write_text(json.dumps(row['output'], indent=2) + '\n')
                elif metadata.get('exit_code') == 0 and '--output' in metadata['command']:
                    pending.append(run_id)
            (args.output / 'status.json').write_text(json.dumps({
                'pending': pending,
                'finished': {run_id: row['metadata'].get('exit_code') for run_id, row in rows.items()
                             if row['metadata'].get('state') == 'finished'},
            }, indent=2) + '\n')
            if rows and not pending:
                print('All tracked prefill runs finished.', flush=True)
                return
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            print(f'Collection will retry: {exc}', flush=True)
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
