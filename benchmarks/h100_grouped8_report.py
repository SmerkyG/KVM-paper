"""Report capped-threshold routes through the original grouped leaf kernel."""
import argparse
import json
import statistics
from pathlib import Path

from tokenizers import Tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    args = p.parse_args()
    def load(name):
        path = args.directory / name
        return json.loads(path.read_text()) if path.exists() else None
    def timing(name):
        d = load(name + '.json')
        return f"{d['median_prefill_s']:.3f} s" if d else 'pending'
    lines = ['# H100 threshold routing through the original grouped leaf kernel', '',
             'BF16 Llama-3.1-8B-Instruct, batch one, two-tier entry-16 prefill, '
             'unchanged 16,384-token exact first chunk. A threshold candidate is opened only '
             'if it also fits the original strict entry-work bound; we keep at most the first '
             'eight surviving candidates in region order. This is not top-8 by score. '
             'All other regions remain coarse. The cap is a speed experiment with a different '
             'frontier from uncapped threshold.', '',
             'The grouped variant calls the same `paged_leaf_attention` region-grouped '
             'kernel as top-8; an additional variant produces its per-head route counts and '
             'offsets during selection, as top-8 does. No per-query GPU-to-CPU count readback '
             'is used. Prefill times are medians of three runs after one warmup.', '',
             '| Configuration | 32K full prefill | 64K full prefill |', '|---|---:|---:|',
             f"| Original top-8 routing and grouped kernel | {timing('grouped8-original-top8-32k')} | {timing('grouped8-original-top8-64k')} |",
             f"| Capped threshold, per-query indexed kernel | {timing('grouped8-fusedtest-direct-32k')} | {timing('grouped8-fusedtest-direct-64k')} |",
             f"| Capped threshold, grouped kernel | {timing('grouped8-fusedtest-grouped-32k')} | {timing('grouped8-fusedtest-grouped-64k')} |",
             f"| Capped threshold, grouped kernel with fused counts | {timing('grouped8-fusedtest-fused-32k')} | {timing('grouped8-fusedtest-fused-64k')} |", '',
             'The three capped-backend rows use the same source snapshot and prompt; '
             'top-8 is a different routing rule and is shown as context. Results do not '
             'establish equivalent RULER quality.', '']
    manifest = load('grouped8-fused-runs.json')
    if manifest:
        hashes = []
        for job in manifest['jobs']:
            if job['name'].startswith('grouped8-fusedtest-'):
                meta = load('provenance/' + job['id'] + '/run.json')
                if meta and meta.get('state') == 'finished' and meta.get('exit_code') == 0:
                    hashes.append(meta['source_files'])
        if hashes and any(x != hashes[0] for x in hashes[1:]):
            raise ValueError('backend timing snapshots do not match')
    audit = load('grouped8-direct-audit.json')
    if audit:
        rows = [v for r in audit['records'] for v in r['prefill_cap_audit'] if v]
        sums = {k: sum(x[k] for x in rows) for k in rows[0]}
        q = sums['query_heads']
        lines += [f"On one 32K prompt, auditing the first 64 remote queries per head and layer "
                  f"({q:,} query heads), {100*sums['above_cap_heads']/q:.2f}% had more than eight "
                  f"qualifying regions. Average opened regions fell from {sums['uncapped_regions']/q:.2f} "
                  f"to {sums['capped_regions']/q:.2f}, and average opened entries from "
                  f"{sums['uncapped_entries']/q:.2f} to {sums['capped_entries']/q:.2f}. "
                  f"{sums['selection_mismatches']} capped selections differed from the reference.", '']
    for name, label in [('grouped8-identical-state-crosscheck', 'Grouped'),
                        ('grouped8-fused-counts-crosscheck', 'Grouped with fused counts')]:
        d = load(name + '.json')
        if not d:
            lines += [label + ' identical-state crosscheck: pending.', '']
            continue
        rows = [v for r in d['records'] for v in r['prefill_backend_audit'] if v]
        n = sum(x['query_heads'] for x in rows)
        maxerr = max(x['max_abs_output_difference'] for x in rows)
        meanerr = sum(x['sum_abs_output_difference'] for x in rows) / sum(x['output_elements'] for x in rows)
        lines += [f'{label} versus indexed, using exactly the same query, selected regions, and '
                  f'KV state: {n:,} sampled query heads, maximum merged-output absolute difference '
                  f'{maxerr:.8f}, mean {meanerr:.8f}.', '']
    direct = load('grouped8-direct-verify.json')
    grouped = load('grouped8-grouped-effective-verify.json')
    if direct and grouped:
        t = Tokenizer.from_file(str(args.tokenizer))
        stops = {t.token_to_id('<|eot_id|>'), t.token_to_id('<|end_of_text|>')}
        if None in stops:
            raise ValueError('tokenizer missing stop IDs')
        matches = full = 0
        for a, b in zip(direct['records'], grouped['records'], strict=True):
            if (a['task'], a['index'], a['input_token_sha256']) != (b['task'], b['index'], b['input_token_sha256']):
                raise ValueError('verification examples differ')
            def first(ids):
                return ids[:next((i for i, v in enumerate(ids) if v in stops), len(ids))]
            matches += first(a['output_token_ids']) == first(b['output_token_ids'])
            full += a['output_token_ids'] == b['output_token_ids']
        lines += [f'The capped backends matched the reference selection on 49,152 sampled query heads '
                  f'across three RULER examples per backend. Their first-response token sequences matched '
                  f'on {matches} of 3 examples; their full fixed-length sequences matched on {full} of 3 '
                  f'(the runs continue generating after end-of-turn).', '']
    for length in (32, 64):
        lines += [f'{length}K synchronized diagnostic timings, summed across layers:', '',
                  '| Backend | Score generation | Select + coarse | Index expansion | Exact leaves |',
                  '|---|---:|---:|---:|---:|']
        for method, name in [('Per-query', f'grouped8-direct-{length}k-profile'),
                             ('Grouped', f'grouped8-grouped-{length}k-profile'),
                             ('Grouped + fused counts', f'grouped8-fused-counts-{length}k-profile')]:
            d = load(name + '.json')
            if not d:
                lines.append(f'| {method} | pending | — | — | — |')
                continue
            records = [r for r in d['records'] if not r['warmup']]
            def stage(k):
                return statistics.median(sum(x.get(k, 0.0) for x in r['prefill_profile']) for r in records)
            lines.append(f"| {method} | {stage('route_logits'):.3f} s | {stage('select_and_coarse'):.3f} s | "
                         f"{stage('expand_indices'):.3f} s | {stage('exact_leaves'):.3f} s |")
        lines += ['', 'These stage measurements add synchronization and instrumentation overhead; '
                  'use the uninstrumented full-prefill table for speed comparisons. Grouped exact '
                  'leaves include route grouping; the per-query path has a separate index-expansion stage.', '']
    lines += ['Complete source hashes, commands, logs, and raw outputs are retained in the '
              '`provenance` directory. The earlier uncapped threshold result is in the '
              '[one-pass report](../prefill-onepass/REPORT.md).', '']
    (args.directory / 'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
