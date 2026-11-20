"""Validate and summarize full-model AMD-to-H100 decode optimization results."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--label', default='transfer-v1')
    a = p.parse_args()
    root = a.directory
    prefix = 'threshold-' + a.label
    manifest = json.loads((root / (prefix + '-runs.json')).read_text())
    metadata = [json.loads((root / 'provenance' / j['id'] / 'run.json').read_text()) for j in manifest['jobs']]
    if any(d['state'] != 'finished' or d.get('exit_code') != 0 for d in metadata):
        raise ValueError('model run failed or remains incomplete')
    for key in ('source_files', 'hardware'):
        if any(m[key] != metadata[0][key] for m in metadata):
            raise ValueError('model experiment source or hardware differs')
    for name, sha in metadata[0]['source_files'].items():
        if hashlib.sha256((root / 'source' / name).read_bytes()).hexdigest() != sha:
            raise ValueError('frozen source hash mismatch: ' + name)
    critical = ['benchmarks/h100_threshold_decode.py', 'benchmarks/h100_threshold_decode_fast.py',
                'benchmarks/h100_threshold_decode_materialized.py', 'benchmarks/amd_threshold_frontier.py',
                'benchmarks/amd_threshold_shared.py', 'benchmarks/amd_threshold_materialized.py',
                'benchmarks/amd_threshold_merge.py', 'integrations/vllm_lod/vllm_lod_plugin/pool.py',
                'lod_attention/kernels/paged_decode_kernels.py']
    tuning_jobs = json.loads((root / 'threshold-materialized-v2-runs.json').read_text())['jobs']
    for job in tuning_jobs:
        meta = json.loads((root / 'provenance' / job['id'] / 'run.json').read_text())
        if meta.get('exit_code') != 0 or meta['state'] != 'finished':
            raise ValueError('selected tuning prerequisite failed')
        if any(meta['source_files'][name] != metadata[0]['source_files'][name] for name in critical):
            raise ValueError('tuning used a different attention implementation')
    fixture = json.loads((root / 'threshold-materialized-v2-check.json').read_text())
    if fixture['stage'] != 'complete' or len(fixture['records']) != 6:
        raise ValueError('missing independent frontier cases')
    for length in (32, 64):
        tune = json.loads((root / f'threshold-materialized-v2-tune-{length}k.json').read_text())
        if tune['stage'] != 'complete' or len(tune['records']) != 42 or any(
            r['selection_mismatches'] or r['max_abs_vs_indexed'] > .02 for r in tune['records']):
            raise ValueError('real-query checks failed')
    summary = {}
    lines = ['# AMD-derived two-tier threshold decode on H100', '',
        'Llama-3.1-8B-Instruct, BF16 weights and KV, batch one, RetroInfer runtime, H100 80GB HBM3. '
        'Both arms use the same normalized-query per-region c=16 threshold, including the existing '
        '1,024-entry eligibility limit; selections remain uncapped. All GPU experiments use uv and cluster-run.', '',
        '## Full model with amortized cache updates', '',
        'Complete-model CUDA graphs, 512 untimed decode steps followed by 1,024 measured steps, '
        'one warmup repetition and two measured repetitions per run. Each candidate is bracketed '
        'by fresh controls on the same GPU. Decode time includes normal KV appends, cache catch-up, '
        'persistent-index rebuilds and mean refreshes. Each measured repetition includes 128 layer '
        'cache-update events. Initial prefill and graph capture are outside decode time.', '',
        '| Initial context | Current optimized H100 | AMD-derived pipeline | Latency reduction |',
        '|---|---:|---:|---:|']
    model_data = {}
    for length in (32, 64):
        data = {}
        for mode in ('control', 'candidate', 'control-after'):
            name = f'{prefix}-{mode}-{length}k'
            d = json.loads((root / (name + '.json')).read_text())
            rows = [r for r in d['records'] if not r['warmup']]
            if (not d['lod_cuda_graph'] or not d['replay_prompt_tokens'] or d['batch_size'] != 1
                or d['timed_decode_steps'] != 1024 or d['decode_warmup_steps'] != 512 or len(rows) != 2
                or [r['lod_layer_catch_up_events_timed'] for r in rows] != [128, 128]):
                raise ValueError('measurement scope differs: ' + name)
            if d['prefill_threshold'] is not None:
                raise ValueError('prefill routing was changed')
            if d['decode_threshold']['rule'] != 'region' or d['decode_threshold']['multiplier'] != 16:
                raise ValueError('routing rule differs')
            data[mode] = d
        for mode in ('candidate', 'control-after'):
            for key in ('model', 'checkpoint', 'dtype', 'manifest', 'batch_size', 'prompt_tokens', 'output_tokens'):
                if data[mode][key] != data['control'][key]:
                    raise ValueError('model inputs differ: ' + key)
        controls = [r['decode_ms_per_token'] for mode in ('control', 'control-after') for r in data[mode]['records'] if not r['warmup']]
        current = statistics.median(controls)
        candidate = data['candidate']['median_decode_ms_per_token']
        disagreements = {}
        for mode in ('control', 'control-after'):
            pairs = []
            for left, right in zip(data[mode]['records'], data['candidate']['records']):
                if len(left['output_token_ids']) != data[mode]['output_tokens'] or len(right['output_token_ids']) != len(left['output_token_ids']):
                    raise ValueError('predicted-token sequence lengths differ')
                pairs.append(sum(x != y for x, y in zip(left['output_token_ids'], right['output_token_ids'])))
            disagreements[mode] = pairs
        summary[str(length)] = dict(control_ms=current, candidate_ms=candidate,
            reduction_pct=100 * (1 - candidate / current), controls_ms=controls,
            candidate_repetitions_ms=[r['decode_ms_per_token'] for r in data['candidate']['records'] if not r['warmup']],
            prefill_s={mode: d['median_prefill_s'] for mode, d in data.items()},
            settings=manifest['chosen'][str(length * 1024)], predicted_token_disagreements=disagreements)
        lines.append(f'| {length}K | {current:.3f} ms/token | {candidate:.3f} ms/token | {summary[str(length)]["reduction_pct"]:.1f}% |')
        model_data[length] = data
    lines += ['', 'Both arms retain top-eight sparse prefill and the unchanged 16,384-token exact initial '
        'chunk. The candidate also prepares its persistent index during prefill; this cost is included below.', '',
        '| Context | Control prefill before / after | Candidate prefill | Control decode before / after |',
        '|---|---:|---:|---:|']
    for length, data in model_data.items():
        lines.append(f'| {length}K | {data["control"]["median_prefill_s"]:.3f} / {data["control-after"]["median_prefill_s"]:.3f} s | '
                     f'{data["candidate"]["median_prefill_s"]:.3f} s | {data["control"]["median_decode_ms_per_token"]:.3f} / '
                     f'{data["control-after"]["median_decode_ms_per_token"]:.3f} ms/token |')
    lines += ['', '## Implementation', '',
        'The selected implementation reuses the AMD materialized four-stage pipeline: (1) shared-GQA '
        'proxy scores and partial normalizers; (2) threshold selection, closed-summary attention and '
        'per-tile union packing; (3) compact physical indices plus per-head membership bits; (4) tensor-core '
        'attention over that frontier, including local/current/sink entries, followed by a dimension-blocked '
        'final merge. K/V storage is shared directly; only indices and membership bits are compacted.', '',
        'Region means and the region-ordered leaf index are maintained at normal cache-update boundaries. '
        'The H100 adapter explicitly retains the old BF16-rounded count division for means. No selected-count '
        'CPU readback, selected-region cap, GQA expansion or smaller exact-prefill chunk is introduced. '
        'The entry-16 rule is outside this candidate; it continues to use its existing implementation.', '',
        '## Checks and fixed-query diagnostics', '',
        'The independent PyTorch frontier suite passed six per-region cases, including empty selections, '
        'uneven region sizes, all-open stress and the 1,024-entry boundary. The winning settings were '
        'selected from identical-query graph measurements on layers 0, 15 and 31. These diagnostics time '
        'the complete attention call plus identical local-length rollback and exclude model layers and cache updates.', '',
        'All 84 real-query comparisons selected the same regions as the indexed control. In the raw '
        'diagnostics, `mean_union_entries` includes local/current/sink entries for the materialized '
        'pipeline; the earlier direct shared-entry prototype records remote entries only. Their complete '
        'attention timings both include local and sink attention.', '',
        '| Context | Current attention | Selected candidate attention | Sampled model steps | Different next-token choices | Maximum logit difference |',
        '|---|---:|---:|---:|---:|---:|']
    for length in (32, 64):
        check = json.loads((root / f'{prefix}-candidate-{length}k-check.json').read_text())
        checks = check['records'][0]['graph_same_cache_checks']
        changed = sum(c['graph_argmax'] != c['eager_argmax'] for c in checks.values())
        if len(checks) != 17 or changed:
            raise ValueError('same-cache model checks failed')
        maximum = max(c['max_abs'] for c in checks.values())
        chosen = manifest['chosen'][str(length * 1024)]
        lines.append(f'| {length}K | {chosen["control_fixed_query_us"]:.2f} µs | {chosen["fixed_query_us"]:.2f} µs | '
                     f'{len(checks)} | {changed} | {maximum:.6f} |')
        summary[str(length)]['same_cache_check_max_abs'] = maximum
    lines += ['', 'Checks compare the candidate graph to legacy eager decode on the identical cache, '
        'including steps around cache updates. Floating-point reductions differ. These component and '
        'next-token checks do not establish full RULER quality equivalence; RULER has not been rerun.', '',
        '| Context | Predicted-token disagreements vs control, warmup / measured 1 / measured 2 |', '|---|---|']
    for length in (32, 64):
        lines.append(f'| {length}K | {summary[str(length)]["predicted_token_disagreements"]["control"]} out of 1,537 per repetition |')
    lines += ['', '## Reproduction and limits', '',
        'Enable `LOD_H100_THRESHOLD_IMPLEMENTATION=materialized` before cache allocation, together with '
        '`LOD_H100_THRESHOLD_MULTIPLIER=16` and `LOD_H100_THRESHOLD_RULE=region`. Exact measured settings, '
        'commands, GPU identity and source hashes are in `threshold-transfer-v1-runs.json`, `SUMMARY.json` '
        'and `provenance/`. The reusable operators retain their `LOD_AMD_THRESHOLD_*` tuning options; '
        'the measured runs explicitly set them. Serving defaults remain unchanged.', '',
        'Both lengths select 32 exact-worker splits, 128-entry attention tiles, four attention warps, '
        '32-dimension merge blocks and the probability-residual precision correction. Set '
        '`LOD_H100_THRESHOLD_SPLITS=32`, `LOD_AMD_THRESHOLD_ENTRY_BLOCK=128`, '
        '`LOD_AMD_THRESHOLD_ENTRY_WARPS=4`, `LOD_AMD_THRESHOLD_MERGE_BLOCK=32` and '
        '`LOD_AMD_THRESHOLD_PRECISE=1` with the enable flags above. The physical-index compactor '
        'uses its tested defaults of four shards and one warp. In raw full-model JSON, '
        '`decode_threshold.entry_block` describes the indexed implementation and the `shared_*` '
        'fields describe the direct shared-entry prototype; the materialized implementation reads '
        'the `LOD_AMD_THRESHOLD_*` values recorded in each run command.', '',
        'Per-step packing uses linear prefix scans with fixed worker counts. The persistent index refresh '
        'visits the remote cache at update boundaries. Partial routing normalizers are currently combined '
        'in each coarse tile; complete prefill, routing and cache maintenance have no new subquadratic '
        'complexity guarantee. Timings report measured costs, including maintenance.', '',
        'A direct port of shared exact-entry attention alone was approximately tied at 32K and 64K and '
        'slower for entry-16. Dimension-blocking the old final merge alone gave only a small improvement. '
        'The measured winner combines cached means, the persistent index, fused tiled selection and '
        'shared tensor-core attention. The first materialized fixture run failed on the old Transformers '
        'import facade; its dependent probes were skipped. The corrected fixture uses the existing '
        'RetroInfer namespace adapter. Failed and partial runs are retained and excluded from comparisons.', '',
        'All final full-model controls, candidates and checks use identical frozen source files and GPU '
        'identity. `source/` is validated against the recorded hashes. Raw runs and all tuning trials '
        'remain in this directory.']
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    (root / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
