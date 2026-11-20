"""Copy completed H100 threshold results into the workspace as runs finish.

This is a read-only monitor, not an experiment. It copies only outputs from
successful runs and records failed prerequisites rather than treating a
partially written result as complete.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

RUNS = {
    '1794151494564042494-entry-24-cap16-ruler32k-retrieval': ('32k', 'entry-24-cap16-ruler32k-retrieval.json'),
    '1794151498110214424-entry-32-cap16-ruler32k-retrieval': ('32k', 'entry-32-cap16-ruler32k-retrieval.json'),
    **{f'{run}-entry-{bound}-cap16-{length}-amortized': ('32k', f'entry-{bound}-cap16-{length}-amortized.json')
       for run, bound, length in [
           ('1794151507857981761', 24, 32768), ('1794151511407204545', 32, 32768),
           ('1794151515026814044', 24, 65536), ('1794151518665677537', 32, 65536)]},
    '1794151534350782327-threshold-region-ruler-64k-4x16-v3': ('64k', 'threshold-region-ruler-64k-4x16.json'),
    '1794151537859868307-threshold-entry-ruler-64k-4x16-v3': ('64k', 'threshold-entry-ruler-64k-4x16.json'),
    '1794151541428644282-dense-ruler-64k-retrieval-4x16-v3': ('64k', 'dense-ruler-64k-retrieval-4x16.json'),
    '1794151545017680236-threshold-entry-24-ruler64k-retrieval-v2': ('64k', 'threshold-entry-24-ruler64k-retrieval.json'),
    '1794151548568975269-threshold-entry-32-ruler64k-retrieval-v2': ('64k', 'threshold-entry-32-ruler64k-retrieval.json'),
    '1794151557307254222-threshold-entry-64k-retrieval-report-v4': ('64k', 'threshold-entry-64k-retrieval-report.md'),
    '1794151560927532876-entry-multipliers-64k-report-v2': ('64k', 'threshold-entry-multipliers-64k-report.md'),
    '1794151626793885020-entry-24-cap16-ruler64k-retrieval': ('64k', 'entry-24-cap16-ruler64k-retrieval.json'),
    '1794151630299829676-entry-32-cap16-ruler64k-retrieval': ('64k', 'entry-32-cap16-ruler64k-retrieval.json'),
    '1794151748850327470-entry-cap16-64k-report': ('64k', 'threshold-entry-capped-multipliers-64k-report.md'),
    '1794151748915396037-entry-cap16-32k-report': ('32k', 'threshold-entry-cap16-32k-report.md'),
    '1794152514236977587-threshold-prefill-frontier-correctness': ('32k', 'threshold-prefill-frontier-correctness.json'),
    '1794153149729268191-threshold-prefill-frontier-bucket-correctness': ('32k', 'threshold-prefill-frontier-bucket-correctness.json'),
    '1794152514295819100-prefill-region-16-cap16-ruler32k-pilot': ('32k', 'threshold-prefill-region-16-cap16-ruler32k-pilot.json'),
    '1794152514353308585-prefill-entry-16-cap16-ruler32k-pilot': ('32k', 'threshold-prefill-entry-16-cap16-ruler32k-pilot.json'),
    '1794152514410994884-prefill-entry-24-cap16-ruler32k-pilot': ('32k', 'threshold-prefill-entry-24-cap16-ruler32k-pilot.json'),
    '1794152514467224126-prefill-region-16-cap16-32768-timing-pilot': ('32k', 'threshold-prefill-region-16-cap16-32768-timing-pilot.json'),
    '1794152514523431526-prefill-entry-16-cap16-32768-timing-pilot': ('32k', 'threshold-prefill-entry-16-cap16-32768-timing-pilot.json'),
    '1794152514577821623-prefill-entry-24-cap16-32768-timing-pilot': ('32k', 'threshold-prefill-entry-24-cap16-32768-timing-pilot.json'),
    '1794153285758740219-threshold-prefill-frontier-audit-correctness': ('32k', 'threshold-prefill-frontier-audit-correctness.json'),
    '1794153285816681968-prefill-region-16-cap16-ruler32k-audit8': ('32k', 'threshold-prefill-region-16-cap16-ruler32k-audit8.json'),
    '1794153285875037083-prefill-entry-16-cap16-ruler32k-audit8': ('32k', 'threshold-prefill-entry-16-cap16-ruler32k-audit8.json'),
    '1794153285931434572-prefill-entry-24-cap16-ruler32k-audit8': ('32k', 'threshold-prefill-entry-24-cap16-ruler32k-audit8.json'),
    '1794153485840080736-prefill-region-16-cap16-32768-timing': ('32k', 'threshold-prefill-region-16-cap16-32768-timing.json'),
    '1794153485895483547-prefill-entry-16-cap16-32768-timing': ('32k', 'threshold-prefill-entry-16-cap16-32768-timing.json'),
    '1794153485951666099-prefill-entry-24-cap16-32768-timing': ('32k', 'threshold-prefill-entry-24-cap16-32768-timing.json'),
    '1794154863654376416-threshold-prefill-fused-frontier-correctness': ('32k', 'threshold-prefill-fused-frontier-correctness.json'),
    '1794154863711111176-prefill-fused-region-16-ruler32k-pilot': ('32k', 'threshold-prefill-fused-region-16-ruler32k-pilot.json'),
    '1794154863765902240-prefill-fused-entry-16-ruler32k-pilot': ('32k', 'threshold-prefill-fused-entry-16-ruler32k-pilot.json'),
    '1794154863819959659-prefill-fused-entry-24-ruler32k-pilot': ('32k', 'threshold-prefill-fused-entry-24-ruler32k-pilot.json'),
    '1794154863873161680-prefill-fused-region-16-tile1024-32768-timing': ('32k', 'threshold-prefill-fused-region-16-tile1024-32768-timing.json'),
    '1794154863926462344-prefill-fused-entry-16-tile1024-32768-timing': ('32k', 'threshold-prefill-fused-entry-16-tile1024-32768-timing.json'),
    '1794154863982668280-prefill-fused-entry-16-tile2048-32768-timing': ('32k', 'threshold-prefill-fused-entry-16-tile2048-32768-timing.json'),
    '1794154864038408311-prefill-fused-entry-24-tile1024-32768-timing': ('32k', 'threshold-prefill-fused-entry-24-tile1024-32768-timing.json'),
    '1794154864093876888-prefill-fused-entry-24-tile2048-32768-timing': ('32k', 'threshold-prefill-fused-entry-24-tile2048-32768-timing.json'),
    '1794155228271863068-threshold-prefill-fused-shared-frontier-correctness': ('32k', 'threshold-prefill-fused-shared-frontier-correctness.json'),
    '1794155228332569992-prefill-fused-shared-region-16-ruler32k-verify': ('32k', 'threshold-prefill-fused-shared-region-16-ruler32k-verify.json'),
    '1794155228391053608-prefill-fused-shared-entry-16-ruler32k-verify': ('32k', 'threshold-prefill-fused-shared-entry-16-ruler32k-verify.json'),
    '1794155228448652798-prefill-fused-shared-entry-24-ruler32k-verify': ('32k', 'threshold-prefill-fused-shared-entry-24-ruler32k-verify.json'),
    '1794155228507353223-prefill-fused-shared-region-16-tile1024-32768-timing': ('32k', 'threshold-prefill-fused-shared-region-16-tile1024-32768-timing.json'),
    '1794155228565945758-prefill-fused-shared-entry-16-tile512-32768-timing': ('32k', 'threshold-prefill-fused-shared-entry-16-tile512-32768-timing.json'),
    '1794155228623913146-prefill-fused-shared-entry-16-tile1024-32768-timing': ('32k', 'threshold-prefill-fused-shared-entry-16-tile1024-32768-timing.json'),
    '1794155228679977622-prefill-fused-shared-entry-16-tile2048-32768-timing': ('32k', 'threshold-prefill-fused-shared-entry-16-tile2048-32768-timing.json'),
    '1794155228735562959-prefill-fused-shared-entry-24-tile512-32768-timing': ('32k', 'threshold-prefill-fused-shared-entry-24-tile512-32768-timing.json'),
    '1794155228792797804-prefill-fused-shared-entry-24-tile1024-32768-timing': ('32k', 'threshold-prefill-fused-shared-entry-24-tile1024-32768-timing.json'),
    '1794155228849229800-prefill-fused-shared-entry-24-tile2048-32768-timing': ('32k', 'threshold-prefill-fused-shared-entry-24-tile2048-32768-timing.json'),
    '1794155344504850922-threshold-prefill-fused-grouped-frontier-correctness': ('32k', 'threshold-prefill-fused-grouped-frontier-correctness.json'),
    '1794155344562528793-prefill-fused-grouped-region-16-ruler32k-verify': ('32k', 'threshold-prefill-fused-grouped-region-16-ruler32k-verify.json'),
    '1794155344621157777-prefill-fused-grouped-entry-16-ruler32k-verify': ('32k', 'threshold-prefill-fused-grouped-entry-16-ruler32k-verify.json'),
    '1794155344678986911-prefill-fused-grouped-entry-24-ruler32k-verify': ('32k', 'threshold-prefill-fused-grouped-entry-24-ruler32k-verify.json'),
    '1794155505196853108-threshold-prefill-fused-matched-frontier-correctness': ('32k', 'threshold-prefill-fused-matched-frontier-correctness.json'),
    '1794155505256041868-prefill-fused-matched-region-16-ruler32k-verify': ('32k', 'threshold-prefill-fused-matched-region-16-ruler32k-verify.json'),
    '1794155505315534597-prefill-fused-matched-entry-16-ruler32k-verify': ('32k', 'threshold-prefill-fused-matched-entry-16-ruler32k-verify.json'),
    '1794155505373862849-prefill-fused-matched-entry-24-ruler32k-verify': ('32k', 'threshold-prefill-fused-matched-entry-24-ruler32k-verify.json'),
    **{f'{run}-prefill-fused-matched-{rule}-{bound}-tile{tile}-32768-timing': ('32k', f'threshold-prefill-fused-matched-{rule}-{bound}-tile{tile}-32768-timing.json')
       for run, rule, bound, tile in [('1794155505432253408', 'region', 16, 1024),
           ('1794155505490395516', 'entry', 16, 512), ('1794155505547624617', 'entry', 16, 1024),
           ('1794155505603759059', 'entry', 16, 2048), ('1794155505662331407', 'entry', 24, 512),
           ('1794155505720334338', 'entry', 24, 1024), ('1794155505778039796', 'entry', 24, 2048)]},
    '1794155860968008782-threshold-prefill-fused-compact-frontier-stress': ('32k', 'threshold-prefill-fused-compact-frontier-stress.json'),
    '1794155861025851266-prefill-fused-compact-region-16-ruler32k-verify': ('32k', 'threshold-prefill-fused-compact-region-16-ruler32k-verify.json'),
    '1794155861083756270-prefill-fused-compact-entry-16-ruler32k-verify': ('32k', 'threshold-prefill-fused-compact-entry-16-ruler32k-verify.json'),
    '1794155861142200879-prefill-fused-compact-entry-24-ruler32k-verify': ('32k', 'threshold-prefill-fused-compact-entry-24-ruler32k-verify.json'),
    **{f'{run}-prefill-fused-compact-{rule}-{bound}-tile{tile}-32768-timing': ('32k', f'threshold-prefill-fused-compact-{rule}-{bound}-tile{tile}-32768-timing.json')
       for run, rule, bound, tile in [('1794155861201369630', 'region', 16, 1024),
           ('1794155861258854423', 'entry', 16, 512), ('1794155861317728216', 'entry', 16, 1024),
           ('1794155861375545020', 'entry', 16, 2048), ('1794155861433623832', 'entry', 24, 512),
           ('1794155861490747327', 'entry', 24, 1024), ('1794155861546337591', 'entry', 24, 2048)]},
    '1794155861602557508-prefill-fused-compact-entry16-profile': ('32k', 'threshold-prefill-fused-compact-entry16-profile.json'),
    '1794156874160572810-prefill-fused-compact-entry16-leaf-profile': ('32k', 'threshold-prefill-fused-compact-entry16-leaf-profile.json'),
    '1794156969648593284-prefill-direct-frontier-stress': ('32k', 'threshold-prefill-direct-frontier-stress.json'),
    '1794156969705362295-prefill-direct-entry-16-ruler32k-verify': ('32k', 'threshold-prefill-direct-entry-16-ruler32k-verify.json'),
    '1794156969761426470-prefill-direct-entry-16-tile512-32768-timing': ('32k', 'threshold-prefill-direct-entry-16-tile512-32768-timing.json'),
    '1794157278099357850-prefill-stream-frontier-stress': ('32k', 'threshold-prefill-stream-frontier-stress.json'),
    '1794157278162650510-prefill-stream-entry-16-ruler32k-verify': ('32k', 'threshold-prefill-stream-entry-16-ruler32k-verify.json'),
    '1794157278226580405-prefill-stream-entry-16-tile512-32768-timing': ('32k', 'threshold-prefill-stream-entry-16-tile512-32768-timing.json'),
    '1794157278283374349-prefill-stream-entry-24-ruler32k-verify': ('32k', 'threshold-prefill-stream-entry-24-ruler32k-verify.json'),
    '1794157278340257364-prefill-stream-entry-24-tile512-32768-timing': ('32k', 'threshold-prefill-stream-entry-24-tile512-32768-timing.json'),
    '1794157278397108782-prefill-warp8-frontier-stress': ('32k', 'threshold-prefill-warp8-frontier-stress.json'),
    '1794157278455385633-prefill-warp8-entry-16-ruler32k-verify': ('32k', 'threshold-prefill-warp8-entry-16-ruler32k-verify.json'),
    '1794157278513125581-prefill-warp8-entry-16-tile512-32768-timing': ('32k', 'threshold-prefill-warp8-entry-16-tile512-32768-timing.json'),
    '1794157393340215717-prefill-direct-entry-24-ruler32k-verify': ('32k', 'threshold-prefill-direct-entry-24-ruler32k-verify.json'),
    '1794157393398560174-prefill-direct-entry-24-tile512-32768-timing': ('32k', 'threshold-prefill-direct-entry-24-tile512-32768-timing.json'),
    '1794157681059325988-prefill-indexed-b64-w4-frontier-stress': ('32k', 'threshold-prefill-indexed-b64-w4-frontier-stress.json'),
    '1794157681116802133-prefill-indexed-b64-w4-entry-16-ruler32k-verify': ('32k', 'threshold-prefill-indexed-b64-w4-entry-16-ruler32k-verify.json'),
    '1794157681174601026-prefill-indexed-b64-w4-entry-16-tile512-32768-timing': ('32k', 'threshold-prefill-indexed-b64-w4-entry-16-tile512-32768-timing.json'),
    '1794157983146404798-prefill-indexed-nofill-b64-w4-frontier-stress': ('32k', 'threshold-prefill-indexed-nofill-b64-w4-frontier-stress.json'),
    '1794157983203881239-prefill-indexed-nofill-b64-w4-entry-16-ruler32k-verify': ('32k', 'threshold-prefill-indexed-nofill-b64-w4-entry-16-ruler32k-verify.json'),
    '1794157983259923380-prefill-indexed-nofill-b64-w4-entry-16-tile512-32768-timing': ('32k', 'threshold-prefill-indexed-nofill-b64-w4-entry-16-tile512-32768-timing.json'),
    '1794158326824052473-prefill-indexed-nofill-b128-w4-entry-24-ruler32k-verify': ('32k', 'threshold-prefill-indexed-nofill-b128-w4-entry-24-ruler32k-verify.json'),
    '1794158326881757898-prefill-indexed-nofill-b128-w4-entry-24-tile512-32768-timing': ('32k', 'threshold-prefill-indexed-nofill-b128-w4-entry-24-tile512-32768-timing.json'),
    '1794158326940041062-prefill-indexed-nofill-b128-w4-warm-profile': ('32k', 'threshold-prefill-indexed-nofill-b128-w4-warm-profile.json'),
    '1794158699095857373-prefill-default-frontier-stress': ('32k', 'threshold-prefill-default-frontier-stress.json'),
    '1794158699152646347-prefill-default-entry-16-ruler32k-verify': ('32k', 'threshold-prefill-default-entry-16-ruler32k-verify.json'),
    '1794158699211019239-prefill-default-entry-16-tile512-32768-timing': ('32k', 'threshold-prefill-default-entry-16-tile512-32768-timing.json'),
    '1794158699269158647-prefill-default-entry-24-ruler32k-verify': ('32k', 'threshold-prefill-default-entry-24-ruler32k-verify.json'),
    '1794158699328416800-prefill-default-entry-24-tile512-32768-timing': ('32k', 'threshold-prefill-default-entry-24-tile512-32768-timing.json'),
    '1794159156174045685-prefill-default-entry-16-ruler32k-audit8': ('32k', 'threshold-prefill-default-entry-16-cap16-ruler32k-audit8.json'),
    '1794159156231686036-prefill-default-entry-24-ruler32k-audit8': ('32k', 'threshold-prefill-default-entry-24-cap16-ruler32k-audit8.json'),
    '1794158920427103328-prefill-select128-frontier-stress': ('32k', 'threshold-prefill-select128-frontier-stress.json'),
    '1794158920483632881-prefill-select128-entry-16-ruler32k-verify': ('32k', 'threshold-prefill-select128-entry-16-ruler32k-verify.json'),
    '1794158920538649444-prefill-select128-entry-16-tile512-32768-timing': ('32k', 'threshold-prefill-select128-entry-16-tile512-32768-timing.json'),
    '1794157834059014246-prefill-indexed-b64-w4-profile': ('32k', 'threshold-prefill-indexed-b64-w4-profile.json'),
    '1794157834112976422-prefill-optimized-entry-16-ruler32k-audit8': ('32k', 'threshold-prefill-optimized-entry-16-cap16-ruler32k-audit8.json'),
    '1794157834167629721-prefill-optimized-entry-24-ruler32k-audit8': ('32k', 'threshold-prefill-optimized-entry-24-cap16-ruler32k-audit8.json'),
    **{f'{run}-prefill-{tag}-{suffix}': ('32k', f'threshold-prefill-{tag}-{suffix}.json')
       for run, tag, suffix in [
           ('1794157833720660947', 'indexed-b32-w4', 'frontier-stress'),
           ('1794157833779934480', 'indexed-b32-w4', 'entry-16-ruler32k-verify'),
           ('1794157833838868996', 'indexed-b32-w4', 'entry-16-tile512-32768-timing'),
           ('1794157833895073017', 'indexed-b128-w4', 'frontier-stress'),
           ('1794157833950335272', 'indexed-b128-w4', 'entry-16-ruler32k-verify'),
           ('1794157834005068596', 'indexed-b128-w4', 'entry-16-tile512-32768-timing')]},
    **{f'{run}-prefill-{tag}-{suffix}': ('32k', f'threshold-prefill-{tag}-{suffix}.json')
       for run, tag, suffix in [
           ('1794157393456345867', 'direct-b64-w4', 'frontier-stress'),
           ('1794157393513343292', 'direct-b64-w4', 'entry-16-ruler32k-verify'),
           ('1794157393570541171', 'direct-b64-w4', 'entry-16-tile512-32768-timing'),
           ('1794157393627703352', 'direct-b64-w8', 'frontier-stress'),
           ('1794157393684383096', 'direct-b64-w8', 'entry-16-ruler32k-verify'),
           ('1794157393741719907', 'direct-b64-w8', 'entry-16-tile512-32768-timing'),
           ('1794157393800054861', 'direct-b128-w8', 'frontier-stress'),
           ('1794157393856737953', 'direct-b128-w8', 'entry-16-ruler32k-verify'),
           ('1794157393914291064', 'direct-b128-w8', 'entry-16-tile512-32768-timing')]},
}


def update_prefill_report(directory):
    """Refresh the report only when the complete input set is available."""
    paths = {
        'baseline': 'ruler-32k-llama-two-tier-threshold-c16-fused-full13.json',
        'region': 'threshold-prefill-region-16-cap16-ruler32k-pilot.json',
        'entry16': 'threshold-prefill-entry-16-cap16-ruler32k-pilot.json',
        'entry24': 'threshold-prefill-entry-24-cap16-ruler32k-pilot.json',
        'baseline-timing': 'threshold-region-32768-guard-amortized.json',
    }
    for name, rule, bound in (('region-timing', 'region', 16), ('entry16-timing', 'entry', 16), ('entry24-timing', 'entry', 24)):
        warm = f'threshold-prefill-{rule}-{bound}-cap16-32768-timing.json'
        pilot = f'threshold-prefill-{rule}-{bound}-cap16-32768-timing-pilot.json'
        paths[name] = warm if (directory / warm).exists() else pilot
    # Do not mix warm and single-run variants in a timing table.
    timing_names = ('region-timing', 'entry16-timing', 'entry24-timing')
    if not all('-pilot.json' not in paths[name] for name in timing_names):
        for name in timing_names:
            paths[name] = paths[name].replace('-timing.json', '-timing-pilot.json')
    if not all((directory / name).exists() for name in paths.values()):
        return
    tokenizer = Path('/tmp/lod-llama31-tokenizer.json')
    if not tokenizer.exists():
        return
    audit = {'region-audit': 'threshold-prefill-region-16-cap16-ruler32k-audit8.json',
             'entry16-audit': 'threshold-prefill-entry-16-cap16-ruler32k-audit8.json',
             'entry24-audit': 'threshold-prefill-entry-24-cap16-ruler32k-audit8.json'}
    if all((directory / name).exists() for name in audit.values()):
        paths.update(audit)
    command = [sys.executable, '-m', 'benchmarks.h100_threshold_prefill_report']
    for name, path in paths.items():
        command.extend(('--' + name, str(directory / path)))
    command.extend(('--tokenizer', str(tokenizer), '--output', str(directory / 'threshold-prefill-progress.md')))
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, timeout=120)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=int, default=60)
    args = parser.parse_args()
    dirs = {key: args.results_root / f'ruler-{key}-panel' for key in ('32k', '64k')}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    remap_path = dirs['32k'] / 'threshold-requeue-map.json'
    remap = json.loads(remap_path.read_text()) if remap_path.exists() else {}
    pending = {remap.get(run_id, run_id): value for run_id, value in RUNS.items()
               if not (dirs[value[0]] / value[1]).exists()}
    done = {}
    state = dirs['32k'] / 'h100-threshold-sync-status.json'
    python = ('import json,pathlib; p=pathlib.Path.home()/".local/state/lod-cluster-run"; '
              'rows=((d.name,json.loads((d/"run.json").read_text())) '
              'for d in p.iterdir() if (d/"run.json").is_file()); '
              'print(json.dumps({name:{"state":r.get("state"),"exit_code":r.get("exit_code")} '
              'for name,r in rows}))')
    while pending:
        try:
            response = subprocess.check_output(['ssh', '-o', 'StrictHostKeyChecking=no',
                                                '-o', 'ConnectTimeout=8', args.host,
                                                'python3', '-c', repr(python)], text=True, timeout=40)
            remote = json.loads(response)
            for run_id, (panel, filename) in list(pending.items()):
                info = remote.get(run_id, {})
                if info.get('state') != 'finished':
                    continue
                if info.get('exit_code') != 0:
                    done[run_id] = {'status': 'failed', 'exit_code': info.get('exit_code')}
                    del pending[run_id]
                    continue
                output = dirs[panel] / filename
                tmp = output.with_name(output.name + '.partial-sync')
                subprocess.run(['scp', '-q', '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=8',
                                f'{args.host}:/home/ubuntu/comparison/results/{filename}', str(tmp)],
                               check=True, timeout=300)
                tmp.replace(output)
                done[run_id] = {'status': 'copied', 'file': str(output)}
                del pending[run_id]
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            print(f'Sync will retry: {exc}', flush=True)
        try:
            update_prefill_report(dirs['32k'])
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            print(f'Prefill report will retry: {exc}', flush=True)
        state.write_text(json.dumps({'pending': list(pending), 'finished': done}, indent=2) + '\n')
        if pending:
            time.sleep(args.poll_seconds)
    print('All tracked runs finished; see', state, flush=True)


if __name__ == '__main__':
    main()
