"""Report matched H100 score-reuse speed, frontier, and first-response checks."""
import argparse
import inspect
import json
from pathlib import Path

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase
from benchmarks.retro_ruler_prompt_compare import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    args = p.parse_args()
    load = lambda name: json.loads((args.directory / name).read_text())
    tok = Tokenizer.from_file(str(args.tokenizer))
    stops = {tok.token_to_id('<|end_of_text|>'), tok.token_to_id('<|eot_id|>')}
    if None in stops:
        raise ValueError('missing Llama stop token')
    cleanup = PreTrainedTokenizerBase.clean_up_tokenization

    def first(r):
        ids = r['output_token_ids']
        end = next((i for i, token in enumerate(ids) if token in stops), len(ids))
        text = tok.decode(ids[:end], skip_special_tokens=True)
        text = cleanup(None, text) if len(inspect.signature(cleanup).parameters) == 2 else cleanup(text)
        return score(r['task'], text, r['answers'])

    lines = ['# H100 threshold-prefill score reuse', '',
             'BF16 Llama-3.1-8B-Instruct, batch 1, entry-16 two-tier prefill and decode. '
             'Score reuse computes native BF16 query–mean-key products once and scales them by inverse query RMS for routing. '
             'The reference separately computes products from a BF16-normalized query. '
             'This experiment is opt-in; the exact initial prefill chunk stays 16,384 tokens.', '',
             '| Context | Baseline prefill | Score reuse prefill | Reduction |',
             '|---|---:|---:|---:|']
    for length in (32, 64):
        old = load(f'prefill-score-baseline-{length}k.json')['median_prefill_s']
        new = load(f'prefill-score-reuse-{length}k.json')['median_prefill_s']
        lines.append(f'| {length}K | {old:.3f} s | {new:.3f} s | {(old-new)/old*100:.2f}% |')
    lines += ['', 'Full-model, uninstrumented prefill latency; each is the median of three runs after one warmup '
              'on the same prompt and H100. Diagnostic instrumentation is disabled for these timings.', '']
    paired = load('prefill-score-reuse-paired.json')
    lines += ['Synthetic paired check (eight queries, 32 heads, 48 deliberately imbalanced regions):', '']
    for r in paired['records']:
        if r['implementation'] == 'reuse_vs_fused':
            lines.append(f"- Entry bound {r['entry_bound']}, strict work bound {r['work_bound']}: "
                         f"{r['changed_regions']:,} / {r['query_region_pairs']:,} region decisions changed; "
                         f"maximum output difference {r['max_abs']:.6f}.")
    lines += ['', 'The initial reuse-only frontier check failed its 0.025 maximum-error tolerance; '
              'the paired check confirmed that the original fused path passed on those inputs.']
    d = load('prefill-score-reuse-audit.json')
    rows = [a for r in d['records'] for a in r.get('prefill_reuse_audit', []) if a]
    if len(rows) != 32:
        raise ValueError(f'expected one audit per layer, got {len(rows)}')
    sums = {k: sum(r[k] for r in rows) for k in rows[0] if not k.startswith('max_')}
    maximum = max(r['max_abs_route_logit_difference'] for r in rows)
    lines += ['', 'Real-state selection audit on one 32K prompt: the first 64 later queries per head and layer, '
              'all 32 layers. It compares against the original normalized-query matmul using the same tile shape.', '',
              f"- {sums['changed_regions']:,} / {sums['region_comparisons']:,} query–region decisions changed "
              f"({100*sums['changed_regions']/sums['region_comparisons']:.4f}%).",
              f"- {sums['changed_query_heads']:,} / {sums['query_heads']:,} query heads had at least one changed region "
              f"({100*sums['changed_query_heads']/sums['query_heads']:.2f}%).",
              f"- Mean opened entries per query head: {sums['old_opened_entries']/sums['query_heads']:.3f} "
              f"before and {sums['new_opened_entries']/sums['query_heads']:.3f} after. "
              f"Maximum absolute routing-logit difference: {maximum:.6f}.", '',
              '| RULER task | Examples | Baseline first-response score | Reuse first-response score | Improved | Declined |',
              '|---|---:|---:|---:|---:|---:|']
    for task, group in [('qa_1', 'qa1'), ('niah_multikey_2', 'retrieval')]:
        old = json.loads((args.matrix / f'ruler32k-prefill-matrix-entry-entry-{group}.json').read_text())
        new = load('prefill-score-reuse-' + task.replace('_', '-') + '.json')
        if old['stage'] != 'complete' or new['stage'] != 'complete':
            raise ValueError('incomplete quality run')
        if any(old.get(k) != new.get(k) for k in ('model_name', 'model_revision', 'cohort_sha256', 'generation')):
            raise ValueError('quality metadata differs')
        aa = {r['index']: r for r in old['records'] if r['task'] == task}
        bb = {r['index']: r for r in new['records'] if r['task'] == task}
        if len(aa) != 32 or aa.keys() != bb.keys():
            raise ValueError('quality samples differ')
        for i in aa:
            if any(aa[i][k] != bb[i][k] for k in ('input_token_sha256', 'answers')):
                raise ValueError(f'quality input mismatch {task} {i}')
        before, after = [first(aa[i]) for i in aa], [first(bb[i]) for i in aa]
        improved = sum(y > x for x, y in zip(before, after))
        declined = sum(y < x for x, y in zip(before, after))
        lines.append(f'| {task} | 32 | {100*sum(before)/32:.2f}% | {100*sum(after)/32:.2f}% | {improved} | {declined} |')
    lines += ['', 'Quality is measured on paired frozen RULER examples, scoring only the first response through the first '
              'end-of-turn or end-of-text token. This limited panel does not establish equivalence on the full RULER suite. '
              'One net multikey regression in 32 examples is an observed sample result, not a precise estimate of the '
              'full-suite effect. Run metadata and individual outputs are preserved alongside this report. '
              'Baseline quality outputs are in the [matched prefill matrix](../ruler-32k-prefill-matrix/REPORT.md).', '']
    (args.directory / 'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
