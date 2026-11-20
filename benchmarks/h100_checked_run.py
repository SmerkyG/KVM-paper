"""Run a queued experiment only after its prerequisite cluster runs succeed."""
import argparse
import json
import math
import os
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--require-run', type=Path, action='append', default=[])
p.add_argument('--require-smoke', type=Path)
p.add_argument('--require-graph-check', type=Path, action='append', default=[])
p.add_argument('--graph-max-abs', type=float, default=0.0,
               help='Maximum allowed sampled graph/eager logit difference (default exact)')
p.add_argument('--require-check-step', type=int, action='append', default=[])
p.add_argument('--require-retro-updates', type=int, default=0)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
for path in a.require_run:
    run = json.loads(path.read_text())
    if run.get('state') != 'finished' or run.get('exit_code') != 0:
        raise SystemExit(f'Prerequisite did not pass: {path}')
for path in a.require_graph_check:
    result = json.loads(path.read_text())
    checks = result.get('records', [])[-1].get('graph_same_cache_checks', {})
    if any(str(step) not in checks for step in a.require_check_step):
        raise SystemExit(f'Required graph/eager boundary check missing: {path}')
    if result['records'][-1].get('retro_index_updates', 0) < a.require_retro_updates:
        raise SystemExit(f'Required RetroInfer index updates missing: {path}')
    if len(checks) < 5 or any(
        not math.isfinite(row['max_abs']) or row['max_abs'] > a.graph_max_abs
        or (row['graph_argmax_by_request'] != row['eager_argmax_by_request']
            if 'graph_argmax_by_request' in row else row['graph_argmax'] != row['eager_argmax'])
        for row in checks.values()
    ):
        raise SystemExit(f'Graph/eager same-cache check failed: {path}')
if a.require_smoke:
    result = json.loads(a.require_smoke.read_text())
    answers = result.get('smoke_responses', [])
    if result.get('stage') != 'complete' or len(answers) != 2:
        raise SystemExit('vLLM smoke did not finish')
    if 'paris' not in answers[0].lower() or '42' not in answers[1]:
        raise SystemExit(f'vLLM smoke answers were unexpected: {answers!r}')
command = a.command[1:] if a.command[:1] == ['--'] else a.command
if not command:
    raise SystemExit('No command given')
print('Prerequisites passed; starting', command, flush=True)
os.execvp(command[0], command)
