"""Report Qwen3 long-context quality and matched decode-only speed."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics

from tokenizers import Tokenizer

from benchmarks.h100_threshold_next_report import QTASKS, cleanup, metric, score_files
from benchmarks.retro_ruler_prepare import ALL_TASKS


NAMES = {'lod': 'Threshold LoD', 'dense': 'Dense', 'squeezed-p90': 'Released Squeezed port, p90',
         'squeezed-p80': 'Released Squeezed port, p80', 'subset-512': 'Squeezed subset 1:8, 512',
         'subset-2048': 'Squeezed subset 1:8, 2,048', 'multipole-512': 'Multipole 1:16, 512',
         'subset-squeezed-512': 'Multipole clustering with released Squeezed kernels',
         'squeezed-1531-p90': 'Released Squeezed p90, 1,531 centroids',
         'squeezed-2900-p90': 'Released Squeezed p90, 2,900 centroids',
         'multipole-p90': 'Squeezed p90 on Multipole index'}
QUALITY_SECTIONS = {65536: 'quality_64k', 128000: 'quality'}


def quality_table(results, timing):
    lines = ['| Method | Examples/task | Multikey-2 | Multikey-3 | CWE | QA-1 | Four-task mean | Decode ms/token |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method, tasks in results.items():
        n = {v['n'] for v in tasks.values()}
        assert len(n) == 1
        vals = [f'{tasks[k]["first"]:.2f}%' if k in tasks else '—' for k in QTASKS]
        mean = f'{statistics.mean(tasks[k]["first"] for k in QTASKS):.2f}%' if all(k in tasks for k in QTASKS) else '—'
        latency = f'{timing["decoders"][method]["median_decode_ms"]:.3f}' if timing else '—'
        lines.append('| ' + ' | '.join([NAMES[method], str(next(iter(n))), *vals, mean, latency]) + ' |')
    return lines


def publish(root, repo):
    status = json.loads((root / 'status.json').read_text())
    tokpath = root.parent / 'threshold-next/qwen-tokenizer.json'
    tok = Tokenizer.from_file(str(tokpath))
    stops = {tok.token_to_id(t) for t in ('<|endoftext|>', '<|im_end|>')}
    assert None not in stops
    summary = dict(quality={}, quality_64k={}, cohorts={}, shared_quality={}, quality_runtime={}, decode={}, checks={},
                   inputs_sha256={}, invalid_rope=[])
    verified = set()
    identities = {}
    prefill_tokens = {}
    paired_scores = {}
    paired_tokens = {}
    cohorts_by_digest = {}
    hardware = set()
    def remember(path):
        summary['inputs_sha256'][str(path.relative_to(repo))] = hashlib.sha256(path.read_bytes()).hexdigest()
    remember(tokpath)
    for job in sorted(status, key=lambda j: j['id']):
        if job['state'] != 'finished' or job['exit_code'] != 0:
            continue
        name = job['name']
        file = root / (name + '.json')
        data = json.loads(file.read_text())
        meta_path = root / 'provenance' / job['id'] / 'run.json'
        meta = json.loads(meta_path.read_text())
        remember(file)
        remember(meta_path)
        hardware.add(meta['hardware'])
        for digest in meta['source_files'].values():
            if digest not in verified:
                assert hashlib.sha256((root / 'source-objects' / digest).read_bytes()).hexdigest() == digest
                verified.add(digest)
        if job['variant'] == 'cohort':
            depth = []
            for row in data['samples']:
                prompt = row['prompt']
                assert hashlib.sha256(prompt.encode()).hexdigest() == row['prompt_sha256']
                if row['task'] not in ('niah_multikey_2', 'niah_multikey_3'):
                    continue
                match = re.search(r'What is the special magic (?:number|uuid) for (.+?) mentioned in the provided text\?', prompt)
                assert match and len(row['answers']) == 1
                tag = f' for {match[1]} is: '
                assert prompt.count(tag) == 1
                at = prompt.index(tag + row['answers'][0] + '.')
                assert at < match.start()
                depth.append(at / len(prompt))
            summary['cohorts'][str(job['length'])] = dict(file=file.name, n=len(data['samples']),
                context_limit=data['context_limit'], retrieval_labels_checked=len(depth),
                retrieval_character_depth=[min(depth), max(depth)])
            cohorts_by_digest[summary['inputs_sha256'][str(file.relative_to(repo))]] = file.name
            continue
        assert data['stage'] == 'complete'
        is_long_model_run = job['length'] > 32768 and name.endswith(('-quality', '-shared', '-reference'))
        expected_positions = 65536 if job['length'] <= 65536 else 131072
        if is_long_model_run and data.get('rope_max_position_embeddings') != expected_positions:
            summary['invalid_rope'].append(dict(name=name, file=file.name,
                reason='Transformers 4.51.3 used effective YaRN 1.25x because max_position_embeddings stayed 40960'))
            continue
        if name.endswith(('-index-check', '-reference', '-pack-check', '-rope-check')) or job['length'] < 32768:
            summary['checks'][name] = data
            continue
        if name.endswith('-quality'):
            assert data['model_name'] == 'Qwen/Qwen3-8B' and data['qwen_yarn']
            scored, samples = score_files([file], tokpath, True)
            assert {r['task'] for r in data['records'] if r['graph_check'] is not None} == set(scored['tasks'])
            for r in data['records']:
                if check := r['graph_check']:
                    assert check['max_abs'] == 0 and check['graph_argmax'] == check['eager_argmax']
            panel = dict(file=file.name, tasks=scored['tasks'],
                tokens=[min(r['input_tokens'] for r in data['records']), max(r['input_tokens'] for r in data['records'])],
                cohort_sha256=data['cohort_sha256'], model_identity=scored['identity'])
            panel['cohort_file'] = cohorts_by_digest[panel['cohort_sha256']]
            summary[QUALITY_SECTIONS[job['length']]][job['variant']] = panel
            arm = job['length'], job['variant']
            identities[arm] = {(r['task'], r['source_row']):
                               (r['input_token_sha256'], r['prompt_sha256'], r['answers'])
                               for r in data['records']}
            paired_scores[arm] = {k: v['score'] for k, v in samples.items()}
            paired_tokens[arm] = {(r['task'], r['source_row']): r['output_token_ids'] for r in data['records']}
            continue
        assert data['prefill'] == 'shared optimized two-tier threshold LoD'
        assert data['model_name'] == 'Qwen/Qwen3-8B' and data['dtype'] == 'bfloat16' and data['batch_size'] == 1
        assert data['records'] and all(set(r['decoders']) == set(data['methods']) for r in data['records'])
        for row in data['records']:
            for d in row['decoders'].values():
                assert d['graph_check']['max_abs'] == 0
        options = sorted(x for x in meta['command'][1:] if '=' in x and x.startswith(
            ('LOD_', 'SQUEEZED_', 'MULTIPOLE_', 'RETRO_PREFILL_MLP_CHUNK=', 'TORCHINDUCTOR_COMPILE_THREADS='))
            and not x.startswith('MULTIPOLE_PROFILE_UPDATES='))
        recipe_files = (
            'h100_qwen_shared_decode.py', 'retro_qwen3_adapter.py', 'retro_lod_adapter.py',
            'retro_native_graph.py', 'retro_squeezed_adapter.py', 'retro_multipole_adapter.py',
            'h100_squeezed_index.py', 'squeezed_stable_pack.py', 'qwen3_rope.py',
        )
        recipe_sources = [meta['source_files'].get(f'benchmarks/{f}') for f in recipe_files]
        recipe = hashlib.sha256(json.dumps([data['model_revision'], options, recipe_sources]).encode()).hexdigest()
        prefill_options = [x for x in options if x.startswith(('LOD_', 'RETRO_PREFILL_MLP_CHUNK='))]
        prefill_sources = sorted((path, digest) for path, digest in meta['source_files'].items()
                                 if path.startswith(('lod_attention/', 'integrations/vllm_lod/',
                                                     'benchmarks/h100_prefill', 'benchmarks/h100_threshold_prefill'))
                                 or path in ('benchmarks/retro_qwen3_adapter.py', 'benchmarks/retro_lod_adapter.py',
                                             'benchmarks/qwen3_rope.py', 'benchmarks/h100_comparison.py'))
        prefill_recipe = hashlib.sha256(json.dumps([
            data['model_revision'], data['rope_scaling'], data['prefill'], prefill_options, prefill_sources
        ]).encode()).hexdigest()
        if data['cohort_sha256'] is not None:
            cohort_file = cohorts_by_digest.get(data['cohort_sha256'])
            if job['length'] > 32768:
                assert cohort_file is not None
            resumed = data.get('resumed_from')
            if resumed:
                checkpoint = root / resumed['file']
                original_meta = root / 'provenance' / resumed['run_id'] / 'run.json'
                remember(checkpoint)
                remember(original_meta)
                original = json.loads(original_meta.read_text())
                assert resumed['sha256'] == summary['inputs_sha256'][str(checkpoint.relative_to(repo))]
                assert original['hardware'] == resumed['hardware']
                resumed = dict(resumed, prefill_tile=next(
                    (x.split('=', 1)[1] for x in original['command'] if x.startswith('LOD_H100_FAST_TILE=')), None))
                prior_rows = json.loads(checkpoint.read_text())['records']
                assert len(prior_rows) == resumed['records']
                current = {(r['task'], r['source_row']): r for r in data['records']}
                assert all(current[(r['task'], r['source_row'])] == r for r in prior_rows)
            groups = {method: defaultdict(list) for method in data['methods']}
            for r in data['records']:
                for method, d in r['decoders'].items():
                    ids = d['output_token_ids']
                    assert ids[0] == r['prefill_token_id']
                    end = next((i for i, t in enumerate(ids) if t in stops), len(ids))
                    value = metric(r['task'], cleanup(tok.decode(ids[:end], skip_special_tokens=True)), r['answers'])
                    assert abs(value - d['score']) < 1e-10
                    groups[method][r['task']].append(value)
            if '--profile-quality' in meta['command']:
                summary['quality_runtime'][name] = dict(
                    file=file.name, dense_kv_source=data.get('dense_kv_source', 'host'),
                    length=job['length'], hardware=meta['hardware'], prefill_recipe=prefill_recipe,
                    captures_host_kv=data.get('captures_host_kv', True),
                    multipole_handoff=data.get('multipole_handoff'),
                    multipole_weights=data.get('multipole_weights'), rows=[
                    dict(task=r['task'], source_row=r['source_row'], elapsed_s=r['elapsed_s'],
                         prefill=r['quality_profile'], decoders={
                             method: dict(d['quality_profile'], cache_setup_s=d.get('cache_setup_s'),
                                          generated_tokens=len(d['output_token_ids']))
                             for method, d in r['decoders'].items()})
                    for r in data['records']])
                continue
            summary['shared_quality'][name] = dict(file=file.name, length=job['length'],
                cohort_sha256=data['cohort_sha256'], cohort_file=cohort_file, recipe=recipe,
                hardware=meta['hardware'], prefill_recipe=prefill_recipe,
                dense_kv_source=data.get('dense_kv_source'), captures_host_kv=data.get('captures_host_kv'),
                resumed_from=resumed,
                stop_at_eos=data.get('stop_at_eos', False),
                prefill_tile=next((x.split('=', 1)[1] for x in prefill_options
                                   if x.startswith('LOD_H100_FAST_TILE=')), None),
                prompt_seconds=sum(r['elapsed_s'] for r in data['records'] if 'elapsed_s' in r),
                tokens=[min(r['input_tokens'] for r in data['records']), max(r['input_tokens'] for r in data['records'])],
                tasks={method: {k: dict(n=len(v), first=100 * statistics.mean(v)) for k, v in rows.items()}
                       for method, rows in groups.items()})
            if 'multipole-p90' in data['methods']:
                widths = data['multipole_fixed_centroids_per_kv_head']
                p90 = [r['decoders']['multipole-p90'] for r in data['records']]
                multipole = [r['decoders']['multipole-512'] for r in data['records']]
                assert all(d['reused_initial_clusters'] and d['squeezed_index']['clusters'] == widths
                           and d['squeezed_index']['calibration_tokens'] == 100 for d in p90)
                assert all(d['centroids_per_kv_head'] == widths for d in multipole)
                summary['shared_quality'][name]['shared_multipole_index'] = dict(
                    centroids_per_kv_head=widths,
                    median_prefill_s=statistics.median(r['prefill_s'] for r in data['records']),
                    median_build_s=statistics.median(d['cache_setup_s'] for d in multipole),
                    median_calibration_s=statistics.median(d['squeezed_index']['seconds'] for d in p90),
                    median_prompt_s=statistics.median(r['elapsed_s'] for r in data['records']))
            identities[name] = {(r['task'], r['source_row']): (r['input_token_sha256'], r['prompt_sha256'], r['answers'])
                                for r in data['records']}
            prefill_tokens[name] = {(r['task'], r['source_row']): r['prefill_token_id'] for r in data['records']}
        else:
            assert len(data['records']) == 1
            row = data['records'][0]
            prefix_tokens = data.get('timing_prefix_length', job['length'])
            assert row['input_tokens'] == prefix_tokens <= job['length']
            windows = [tuple(v['steps'] for v in d['windows']) for d in row['decoders'].values()]
            assert len(set(windows)) == 1
            warmup = data.get('warmup_steps')
            if prefix_tokens < job['length']:
                assert prefix_tokens + warmup + sum(windows[0]) <= expected_positions
            for d in row['decoders'].values():
                assert d['median_decode_ms'] == statistics.median(1000*v['seconds']/v['steps'] for v in d['windows'])
            summary['decode'][name] = dict(file=file.name, length=job['length'], windows=windows[0],
                 prefix_tokens=prefix_tokens, warmup_steps=warmup, hardware=meta['hardware'],
                 prompt_token_sha256=row['input_token_sha256'], decoders=row['decoders'], recipe=recipe,
                 prefill_recipe=prefill_recipe,
                 squeezed_index=row.get('squeezed_index'))
            if 'multipole_fixed_centroids_per_kv_head' in data:
                summary['decode'][name]['fixed_centroids'] = data['multipole_fixed_centroids_per_kv_head']
    summary['hardware'] = sorted(hardware)
    for length, section in QUALITY_SECTIONS.items():
        dense_arm, lod_arm = (length, 'dense'), (length, 'lod')
        if {dense_arm, lod_arm} <= identities.keys():
            assert identities[dense_arm] == identities[lod_arm]
            dense, lod = (summary[section][method] for method in ('dense', 'lod'))
            assert dense['model_identity'] == lod['model_identity']
            assert dense['cohort_sha256'] == lod['cohort_sha256'] and dense['tokens'] == lod['tokens']
            x, y = paired_scores[dense_arm], paired_scores[lod_arm]
            key = 'paired_examples_64k' if length == 65536 else 'paired_examples'
            summary[key] = dict(ties=sum(x[k] == y[k] for k in x),
                lod_higher=sum(y[k] > x[k] for k in x), dense_higher=sum(x[k] > y[k] for k in x),
                identical_generations=sum(paired_tokens[dense_arm][k] == paired_tokens[lod_arm][k] for k in x))
    for name, panel in summary['shared_quality'].items():
        section = QUALITY_SECTIONS.get(panel['length'])
        lod = summary[section].get('lod') if section else None
        if lod and panel['cohort_sha256'] == lod['cohort_sha256']:
            assert all(identities[(panel['length'], 'lod')].get(k) == v for k, v in identities[name].items())
        dense_arm = panel['length'], 'dense'
        if (dense_arm in identities
                and panel['cohort_sha256'] == summary[section]['dense']['cohort_sha256']):
            assert all(identities[dense_arm].get(k) == v for k, v in identities[name].items())
            groups = defaultdict(list)
            for key in identities[name]:
                groups[key[0]].append(paired_scores[dense_arm][key])
            panel['full_dense_reference'] = dict(
                file=summary[section]['dense']['file'],
                tasks={task: 100 * statistics.mean(scores) for task, scores in groups.items()})
    summary['pending'] = [j for j in status if j['state'] != 'finished']
    summary['failed'] = [j for j in status if j['state'] == 'finished' and j['exit_code'] not in (None, 0, -15, 143)]
    summary['interrupted'] = [j for j in status if j['state'] == 'finished' and j['exit_code'] is None]
    summary['cancelled'] = [j for j in status if j['exit_code'] in (-15, 143)]
    lines = ['# Qwen3-8B long-context quality and decode comparison', '',
             'BF16, batch one, one H100 80GB per run. All GPU runs use uv and immutable cluster-run snapshots. '
             'Corrected runs use 2x YaRN with max_position_embeddings=65536 at 64K and '
             '4x YaRN with max_position_embeddings=131072 near 128K.', '',
             '**Correction to the previous long-context results.** The earlier 128K runs left '
             'max_position_embeddings at 40,960. Transformers 4.51.3 consequently used 1.25x YaRN '
             'despite factor=4 in the configuration. The native runtime and independent Transformers '
             'control shared this error. Those results are quarantined. The 32K native-RoPE panel remains valid.', '',
             'Qwen documents 32,768 native tokens and 131,072 tokens with YaRN. '
             'The correction follows its [Transformers long-context configuration]'
             '(https://github.com/QwenLM/Qwen3/blob/main/docs/source/inference/transformers.md).', '']
    lines += ['## Full-model quality', '',
              'Dense and LoD share the frozen prompts at each length. Each length uses a separate cohort '
              'with the same task generators, seed and Qwen chat format. Thinking is disabled and scoring '
              'stops at the first EOS/EOT. LoD uses uncapped per-region c=16 threshold attention in prefill '
              'and decode with a 16,384-token initial exact field.', '']
    for length, section in QUALITY_SECTIONS.items():
        context = '64K' if length == 65536 else '128,000 tokens'
        lines += [f'### Near {context}', '']
        dense = summary[section].get('dense')
        lod = summary[section].get('lod')
        if dense is None:
            lines += ['The matched dense and LoD quality panel is pending.', '']
            continue
        lines += [f'Prompts span {dense["tokens"][0]:,}–{dense["tokens"][1]:,} tokens.', '',
                  '| Task | Examples | Dense | Optimized LoD |', '|---|---:|---:|---:|']
        for task in QTASKS:
            row = dense['tasks'][task]
            value = 'Pending'
            if lod:
                assert row['n'] == lod['tasks'][task]['n']
                value = f'{lod["tasks"][task]["first"]:.2f}%'
            lines.append(f'| {task} | {row["n"]} | {row["first"]:.2f}% | {value} |')
        if lod:
            key = 'paired_examples_64k' if length == 65536 else 'paired_examples'
            pairs = summary[key]
            lines += ['', f'On paired examples, {pairs["ties"]} scores tie, {pairs["lod_higher"]} favor LoD, '
                      f'and {pairs["dense_higher"]} favor dense. '
                      f'Fixed-length generations match exactly on {pairs["identical_generations"]} of the paired prompts. '
                      f'[Dense outputs]({dense["file"]}), [LoD outputs]({lod["file"]}).']
        lines.append('')
    for name, check in summary['checks'].items():
        if name.endswith('-rope-check'):
            lines += ['', f'The [frequency regression check]({name}.json) evaluates the installed '
                      'Transformers implementation directly.', '',
                      '| Configuration | max_position_embeddings | Declared factor | Effective factor | Attention scaling |',
                      '|---|---:|---:|---:|---:|']
            for row in check['cases']:
                label = 'Corrected' if row['corrected'] else 'Archived'
                lines.append(f'| {label} | {row["max_position_embeddings"]:,} | {row["declared_factor"]:.2f} | '
                             f'{row["effective_factor"]:.2f} | {row["attention_scaling"]:.9f} |')
        if name.endswith('-reference'):
            lines += ['', f'The independent [Transformers reference]({name}.json) uses the same YaRN and prompts '
                      f'with a {check["rope_max_position_embeddings"]:,}-token context limit.', '',
                      '| Task | Source row | Input tokens | RetroInfer dense score | Transformers dense score | First three tokens agree |',
                      '|---|---:|---:|---:|---:|---|']
            for row in check['records']:
                lines.append(f'| {row["task"]} | {row["source_row"]} | {row["input_tokens"]:,} | '
                             f'{row["native_score"]:.0%} | {row["hf_score"]:.0%} | {row["agree"]} |')
    panels = summary['shared_quality']
    methods_32k = ('lod', 'dense', 'multipole-512')
    expected_32k = {(task, row) for task in ALL_TASKS for row in range(32)}
    ruler13 = next((name for name, p in reversed(panels.items()) if p['length'] == 32768
                    and set(p['tasks']) == set(methods_32k)
                    and set(identities[name]) == expected_32k), None)
    if ruler13:
        panel = panels[ruler13]
        decode = next((d for d in reversed(summary['decode'].values())
                       if (d['length'], d['recipe'], d['hardware'])
                       == (panel['length'], panel['recipe'], panel['hardware'])
                       and d.get('fixed_centroids') is None and set(methods_32k) <= set(d['decoders'])), None)
        summary['ruler13_32k'] = dict(quality=ruler13, timing=decode['file'] if decode else None)
        lines += ['', '## Full 32K RULER panel', '',
                  'Qwen3-8B BF16 with native RoPE and thinking disabled. All three decoders share '
                  'one optimized LoD prefill per prompt. The cohort has 32 examples per task. '
                  'The mean weights the 13 tasks equally.', '',
                  f'Measured prompt time sums to {panel["prompt_seconds"] / 60:.1f} minutes on one H100. '
                  f'[Generations]({panel["file"]}), [frozen cohort]({panel["cohort_file"]}).', '',
                  ]
        if panel['stop_at_eos']:
            lines += ['Scoring stops at the first EOS/EOT. Generation checks for EOS/EOT every eight '
                      'tokens after the initial fixed-length segment. All runs use the same task caps.', '']
        if resumed := panel['resumed_from']:
            lines += [f'The run preserved {resumed["records"]} prompts from the '
                      f'[earlier checkpoint]({resumed["file"]}).', '']
            if resumed['prefill_tile'] != panel['prefill_tile']:
                lines += [f'After the retry, the shared prefill used {panel["prefill_tile"]}-query '
                          'workspace tiles. The context limit, exact prefill field, and routing policy '
                          'were the same throughout the panel.', '']
        lines += ['| RULER task | Threshold LoD | Dense | Multipole 1:16, 512 |',
                  '|---|---:|---:|---:|']
        for task in ALL_TASKS:
            scores = [panel['tasks'][method][task]['first'] for method in methods_32k]
            assert all(panel['tasks'][method][task]['n'] == 32 for method in methods_32k)
            lines.append(f'| {task} | ' + ' | '.join(f'{score:.2f}%' for score in scores) + ' |')
        means = [statistics.mean(panel['tasks'][method][t]['first'] for t in ALL_TASKS) for method in methods_32k]
        lines += ['| **13-task mean** | ' + ' | '.join(f'**{value:.2f}%**' for value in means) + ' |', '']
        if decode:
            lines += ['Whole-model decode includes periodic cache updates. '
                      f'[Matched 32K timing data]({decode["file"]}).', '',
                      '| Decoder | Median ms/token |', '|---|---:|']
            lines += [f'| {NAMES[m]} | {decode["decoders"][m]["median_decode_ms"]:.3f} |' for m in methods_32k]
            lines.append('')
    index_methods = ('lod', 'dense', 'multipole-512', 'multipole-p90')
    shared_index_32k = next((name for name, p in reversed(panels.items())
                            if p['length'] == 32768 and set(p['tasks']) == set(index_methods)
                            and p.get('shared_multipole_index', {}).get('centroids_per_kv_head') == 2900
                            and set(identities[name]) == expected_32k), None)
    if shared_index_32k:
        panel = panels[shared_index_32k]
        setup = panel['shared_multipole_index']
        summary['ruler13_32k_matched_centroids'] = dict(quality=shared_index_32k)
        if ruler13:
            assert identities[shared_index_32k] == identities[ruler13]
        lines += ['', '## 32K RULER with a shared 2,900-centroid index', '',
                  'Qwen3-8B BF16 on H100, with 32 prompts per task and batch-one decoding. All four '
                  'decoders share one optimized LoD prefill per prompt. Multipole uses a 512-token exact '
                  'budget, value-centroid replacement, and 128-token online refinement at fixed centroid '
                  'count. Squeezed p90 reuses the same initial Multipole centroids and assignments, '
                  'calibrates on the final 100 prefill queries at routing position 2,048, and holds its '
                  'prefix index fixed. The sink, unclustered prefix tail, and generated tokens are exact. '
                  f'[Generations]({panel["file"]}), [frozen cohort]({panel["cohort_file"]}).', '']
        if resumed := panel['resumed_from']:
            lines += [f'The final run resumed from the [{resumed["records"]}-prompt checkpoint]({resumed["file"]}). '
                      'Only fully scored prompts carried forward.', '']
        lines += ['| RULER task | Examples | Threshold LoD | Dense | Multipole, 512 | Squeezed p90 |',
                  '|---|---:|---:|---:|---:|---:|']
        for task in ALL_TASKS:
            assert all(panel['tasks'][m][task]['n'] == 32 for m in index_methods)
            scores = [panel['tasks'][m][task]['first'] for m in index_methods]
            lines.append(f'| {task} | 32 | ' + ' | '.join(f'{score:.2f}%' for score in scores) + ' |')
        means = [statistics.mean(panel['tasks'][m][t]['first'] for t in ALL_TASKS) for m in index_methods]
        lines += ['| **13-task mean** | **416** | ' + ' | '.join(f'**{value:.2f}%**' for value in means) + ' |', '',
                  f'The panel took {panel["prompt_seconds"] / 60:.1f} minutes of prompt processing. '
                  f'Median prompt time was {setup["median_prompt_s"]:.2f} s, including '
                  f'{setup["median_prefill_s"]:.2f} s for shared prefill, '
                  f'{setup["median_build_s"]:.2f} s for Multipole cache setup and fitting, and '
                  f'{setup["median_calibration_s"]:.2f} s for p90 calibration.', '']
    full_64k = {name: p for name, p in panels.items()
                if p['length'] == 65536 and set(identities[name]) == {(task, i) for task in QTASKS for i in range(32)}}
    matched = [(a, b) for a, lod_dense in full_64k.items() for b, multipole in full_64k.items()
               if set(lod_dense['tasks']) == {'lod', 'dense'} and set(multipole['tasks']) == {'multipole-512'}
               and lod_dense['prefill_recipe'] == multipole['prefill_recipe']
               and lod_dense['cohort_sha256'] == multipole['cohort_sha256']
               and identities[a] == identities[b]]
    compared = matched[-1] if matched else ()
    matched_decode = None
    if compared:
        lod_dense, multipole = (panels[name] for name in compared)
        methods = lod_dense['tasks'] | multipole['tasks']
        matched_decode = next((d for d in reversed(summary['decode'].values())
                               if (d['length'], d['recipe'], d['hardware'])
                               == (multipole['length'], multipole['recipe'], multipole['hardware'])
                               and set(methods) <= set(d['decoders'])), None)
        first, second = (prefill_tokens[name] for name in compared)
        same_first = sum(first[k] == second[k] for k in first)
        summary['matched_prompts_64k'] = dict(quality=list(compared),
                                             prefill_first_token_matches=same_first,
                                             timing=matched_decode['file'] if matched_decode else None)
        lines += ['', '## 64K comparison on the same 128 prompts', '',
                  'LoD and dense share a prefill execution within each prompt. Multipole recomputes '
                  'the same optimized LoD prefill recipe in a separate run. Prompt tokens, task labels, '
                  'and the 2× YaRN configuration match. Multipole uses the author’s 1:16 clustering, '
                  'centroid replacement, and 512-token exact budget.', '',
                  f'The independently executed prefills produced the same first token on '
                  f'{same_first} of {len(first)} prompts.', '',
                  f'[LoD and dense generations]({lod_dense["file"]}), '
                  f'[Multipole generations]({multipole["file"]}).', '']
        if resumed := lod_dense['resumed_from']:
            lines += [f'The LoD and dense run reused {resumed["records"]} prompts from the '
                      f'[pre-restart checkpoint]({resumed["file"]}). The replacement H100 finished '
                      'that panel and ran all Multipole prompts.', '']
        if matched_decode:
            lines += [f'All decode latencies come from a separate matched three-way run on the '
                      f'replacement H100, including periodic updates. [Timing data]({matched_decode["file"]}).', '']
        lines += quality_table(methods, matched_decode) + ['']
    lines += ['', '## Same-prefill decode quality', '',
              'All rows within a panel start with the same optimized LoD prefill, original per-layer '
              'activations, and first output token. Dense decode therefore diagnoses what the shared '
              'prefill retained. The released Squeezed port uses its cuML fit, global percentile calibration, '
              'and released lookup, sparse attention, and reduction kernels with shared GQA KV. '
              'Each physical KV head is fitted once, then its centers and labels are shared across its query heads. '
              'The port uses deterministic two-pass packing of the same selected set for graph/eager parity. '
              'Multipole uses one centroid per 16 tokens, centroid replacement, and a 512-token exact budget. '
              'The Multipole authors’ Squeezed subset uses 1:8 clustering '
              'and a token budget. It uses the authors’ windowed routing, with centroid replacement disabled. '
              'Each completed arm passes an exact graph/eager logit check at the same cache state.', '']
    preferred_panels = {}
    for name, panel in summary['shared_quality'].items():
        if name in compared or name in (ruler13, shared_index_32k):
            continue
        key = panel['length'], panel['recipe'], tuple(sorted(panel['tasks']))
        previous = preferred_panels.get(key)
        if previous is None or len(identities[name]) >= len(identities[previous]):
            preferred_panels[key] = name
    for key, name in sorted(preferred_panels.items()):
        panel = summary['shared_quality'][name]
        matching = [d for d in summary['decode'].values()
                    if (d['length'], d['recipe'], d['hardware']) == (panel['length'], panel['recipe'], panel['hardware'])
                    and set(panel['tasks']) <= set(d['decoders'])]
        timing = matching[-1] if matching else None
        lines += [f'[{name}]({panel["file"]}), context limit {panel["length"]:,}. '
                  f'Actual input lengths {panel["tokens"][0]:,}–{panel["tokens"][1]:,}.', '']
        if panel['dense_kv_source'] == 'device' and not panel['captures_host_kv']:
            lines += ['Dense decode rebuilds its cache on the GPU from the exact BF16 LoD entries.', '']
        if resumed := panel['resumed_from']:
            count = resumed['records']
            remaining = len(identities[name]) - count
            first_gpu = resumed['hardware'].split(', ')[1]
            second_gpu = panel['hardware'].split(', ')[1]
            lines += [f'This panel reused {count} matched prompts from the [pre-restart checkpoint]({resumed["file"]}) '
                      f'on {first_gpu}. The resumed run completed {remaining} matched prompts on {second_gpu}.', '']
        lines += quality_table(panel['tasks'], timing) + ['']
        if reference := panel.get('full_dense_reference'):
            values = ', '.join(f'{task} {reference["tasks"][task]:.2f}%' for task in QTASKS)
            lines += [f'Full dense prefill and decode on these same prompts scores {values}. '
                      f'[Reference generations]({reference["file"]}).', '']
    if not summary['shared_quality']:
        lines += ['No complete same-prefill quality panel yet.', '']
    if summary['quality_runtime']:
        lines += ['<a id="quality-run-wall-time-at-64k"></a>', '', '## Quality-run wall time', '',
                  'These short profiles use frozen RULER prompts with shared LoD prefill. '
                  'Scoring stops at the first EOS/EOT. The run commands record the generation limits. '
                  'The full prompt time includes cache setup, activation handoff, and graph checks. '
                  'It excludes loading the main model once and cleanup after saving the prompt.', '',
                  'The GPU-resident path assembles native dense KV from the LoD sink, chronological BF16 '
                  'archive, and live exact tail. The verification profile compares every layer with the '
                  'original model activations. These short profiles measure runtime.', '']
        transport_profiles = [p for p in summary['quality_runtime'].values()
                              if p['multipole_handoff'] and 'multipole-512' in p['rows'][0]['decoders']]
        if transport_profiles:
            lines += ['### Multipole worker transport', '',
                      'CUDA IPC shares GPU allocations across the two Triton runtimes. The worker '
                      'uses the same compiled author clustering, attention kernels and 128-token updates. '
                      'The prompt time covers shared prefill and all three decoders. '
                      'The Multipole arm includes worker launch, index construction, graph validation and generation.', '',
                      '| KV transfer | Worker weights | Prompts | Median prompt (s) | CPU KV export (s) | Multipole arm (s) | Worker setup (s) | Index setup (s) |',
                      '|---|---|---:|---:|---:|---:|---:|---:|']
            for panel in sorted(transport_profiles,
                                key=lambda p: (p['multipole_handoff'] == 'cuda-ipc', p['multipole_weights'] == 'cuda-ipc')):
                rows = panel['rows']
                arms = [r['decoders']['multipole-512'] for r in rows]
                times = [statistics.median(r['elapsed_s'] for r in rows),
                         statistics.median(r['prefill']['prefill_capture_s'] for r in rows),
                         *(statistics.median(d[k] for d in arms)
                           for k in ('arm_setup_s', 'worker_setup_s', 'cache_setup_s'))]
                route = 'CUDA IPC' if panel['multipole_handoff'] == 'cuda-ipc' else 'CPU and tmpfs'
                weights = 'Shared on GPU' if panel['multipole_weights'] == 'cuda-ipc' else 'Loaded per prompt'
                lines.append(f'| [{route}]({panel["file"]}) | {weights} | {len(rows)} | '
                             + ' | '.join(f'{t:.2f}' for t in times) + ' |')
            lines.append('')
        for panel in summary['quality_runtime'].values():
            route = {
                'device': 'GPU-resident dense KV',
                'host': 'Dense KV via CPU',
                'verify': 'GPU-resident dense KV checked against the original activations',
            }.get(panel['dense_kv_source'], 'Quality runtime')
            lines += [f'{panel["length"]:,}-token limit. [{route}]({panel["file"]})', '',
                      '| Task | Source row | Full prompt (s) | Prefill with export (s) | Export (s) | '
                      'LoD attention and state update (s) | Dense cache setup (s) | Generation across arms (s) |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|']
            for row in panel['rows']:
                p, d = row['prefill'], row['decoders']
                export = f'{p["prefill_capture_s"]:.2f}' if 'prefill_capture_s' in p else '—'
                attention = f'{p["prefill_attention_and_state_s"]:.2f}' if 'prefill_attention_and_state_s' in p else '—'
                generation = sum(arm['generation_s'] for arm in d.values())
                dense_setup = f'{d["dense"]["arm_setup_s"]:.2f}' if 'dense' in d else '—'
                lines.append(f'| {row["task"]} | {row["source_row"]} | {row["elapsed_s"]:.2f} | '
                             f'{p["prefill_and_capture_s"]:.2f} | {export} | {attention} | '
                             f'{dense_setup} | {generation:.2f} |')
            lines.append('')
            if 'multipole-512' in panel['rows'][0]['decoders']:
                lines += ['| Source row | Multipole arm (s) | Worker model setup (s) | Index setup (s) | '
                          'Graph and check (s) | Generation (s) | Handoff and other overhead (s) |',
                          '|---:|---:|---:|---:|---:|---:|---:|']
                for row in panel['rows']:
                    d = row['decoders']['multipole-512']
                    phases = [d[k] for k in ('worker_setup_s', 'cache_setup_s', 'graph_and_check_s', 'generation_s')]
                    times = [d['arm_setup_s'], *phases, d['arm_setup_s'] - sum(phases)]
                    lines.append(f'| {row["source_row"]} | ' + ' | '.join(f'{value:.2f}' for value in times) + ' |')
                lines.append('')
    lines += ['## Decode-only speed', '',
              'Full-model graph replay on one shared real-text prefill per context, with identical '
              'teacher-forced tokens. Timed windows include per-step KV writes, host launch and '
              'required ongoing cache/index updates. Prefill, initial index construction, cache '
              'allocation, graph capture, and warmup are excluded. Consecutive windows are not independent prefills. '
                  'Quality is scored on the frozen RULER prompts. The timing workload uses a separate real-text prefix.', '']
    fixed = next((d for d in reversed(summary['decode'].values()) if d['length'] == 32768
                  and d.get('fixed_centroids') is not None
                  and {'lod', 'dense', 'subset-512', 'multipole-512'} <= set(d['decoders'])), None)
    if fixed:
        count = fixed['fixed_centroids']
        lines += ['### Matched centroid count at 32K', '',
                  f'Qwen3-8B BF16, batch one, native RoPE on one H100. The methods share '
                  f'a {fixed["prefix_tokens"]:,}-token LoD prefill, then decode '
                  f'{fixed["warmup_steps"]:,} warmup and {sum(fixed["windows"]):,} timed tokens. '
                  f'[Timing data]({fixed["file"]}).', '',
                  f'The two Multipole-based methods use {count:,} centroid slots per KV head and '
                  'a 512-token exact budget per KV head. Both refine their final clustering block '
                  'every 128 tokens with the released sequential K-means kernel and zero new '
                  'centroid seeds. Multipole also uses value-centroid replacement for the unselected '
                  'clusters. LoD keeps its production centroid schedule.', '',
                  '| Method | Centroid slots per KV head | Median ms/token | Window ms/token | Index updates (ms/token) |',
                  '|---|---:|---:|---|---:|']
        for method, label in (('lod', 'Threshold LoD'), ('dense', 'Dense'),
                              ('subset-512', 'Squeezed with Multipole clustering'), ('multipole-512', 'Multipole')):
            d = fixed['decoders'][method]
            slots = [n for w in d['windows'] for n in w.get('last_layer_centroid_slots_per_kv_head', ())]
            if method.startswith(('subset-', 'multipole-')):
                assert slots and set(slots) == {count}
            span = f'{min(slots):,}–{max(slots):,}' if slots and min(slots) != max(slots) else (
                f'{slots[0]:,}' if slots else '—')
            times = ', '.join(f'{1000 * w["seconds"] / w["steps"]:.3f}' for w in d['windows'])
            profiled = all('cache_update_seconds' in w for w in d['windows'])
            update = (f'{statistics.median(1000 * sum(w["cache_update_seconds"]) / w["steps"] for w in d["windows"]):.3f}'
                      if profiled else '—')
            lines.append(f'| {label} | {span} | {d["median_decode_ms"]:.3f} | {times} | {update} |')
        lines += ['', 'Centroid counts are from the last layer at the measured window boundaries. '
                  'Total latency includes all required cache updates. The update column shows '
                  'the separately instrumented time for the two Multipole-based methods.', '']
    kernel_swaps = [d for d in summary['decode'].values() if d['length'] == 32768
                    and d.get('fixed_centroids') == 2900
                    and {'subset-512', 'subset-squeezed-512'} <= set(d['decoders'])
                    and 'kernel_profile' in d['decoders']['subset-squeezed-512']]
    if kernel_swaps:
        labels = {'subset-512': 'Multipole', 'subset-squeezed-512': 'Squeezed'}
        lines += ['### Kernel swap at matched centroid count', '',
                  'Both subset arms start from the same 2,900-centroid index and use the '
                  'same grouped 512-token selection and 128-token update interval. The swap uses '
                  'the released Squeezed lookup, sparse attention, and reduction with shared '
                  'head addressing. Its per-query-head scores are summed before the common '
                  'budget sort and index packing.', '',
                  '| First kernel set | Kernels | Median ms/token | Index updates (ms/token) | Decode excluding profiled updates (ms/token) |',
                  '|---|---|---:|---:|---:|']
        profiles = []
        for run in kernel_swaps:
            arms = run['decoders']
            first = next(m for m in arms if m in labels)
            assert not arms[first]['reused_initial_clusters']
            for method, d in arms.items():
                if method not in labels:
                    continue
                assert d['reused_initial_clusters'] == (method != first)
                windows = d['windows']
                update = statistics.median(1000 * sum(w['cache_update_seconds']) / w['steps'] for w in windows)
                rest = statistics.median(1000 * (w['seconds'] - sum(w['cache_update_seconds'])) / w['steps']
                                         for w in windows)
                lines.append(f'| [{labels[first]}]({run["file"]}) | {labels[method]} | '
                             f'{d["median_decode_ms"]:.3f} | {update:.3f} | {rest:.3f} |')
                profiles.append((labels[first], labels[method], d['kernel_profile']))
        lines += ['', 'The final-layer diagnostic replays each phase in a separate CUDA graph. '
                  'For the alignment check, lookup uses the first 2,816 slots or pads '
                  'the 2,900-slot index with empty entries to reach 2,944.', '',
                  '| First kernel set | Kernels | Lookup at 2,816 (µs) | Lookup at 2,900 (µs) | Lookup at 2,944 (µs) | Selection and pack (µs) | Sparse attention and merge (µs) |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for first, name, d in profiles:
            lookups = ' | '.join(f'{d["lookup_us_by_centroids"][n]:.2f}' for n in ('2816', '2900', '2944'))
            lines.append(f'| {first} | {name} | {lookups} | {d["selection_and_pack_us"]:.2f} | {d["sparse_us"]:.2f} |')
        if all(not any(d['alternate_lookup_token_differences_by_kv_head']) for _, _, d in profiles):
            lines += ['', 'Both lookups selected identical tokens on each profiled final-layer input.']
        lines.append('')
    count_methods = ('squeezed-1531-p90', 'squeezed-2900-p90')
    count_panel = next((d for d in reversed(summary['decode'].values()) if d['length'] == 32768
                        and set(count_methods) <= set(d['decoders'])), None)
    if count_panel:
        decoders = count_panel['decoders']
        lookup = decoders[count_methods[-1]]['lookup_only_ms_per_token']
        lines += ['### Released Squeezed centroid count at 32K', '',
                  'Qwen3-8B BF16 at batch one on H100, using shared GQA centroid storage. '
                  f'Both indexes use the same {count_panel["prefix_tokens"]:,}-token prefill and '
                  f'{count_panel["warmup_steps"]:,} warmup and {sum(count_panel["windows"]):,} timed '
                  'teacher-forced decode tokens. Each index has its own p90 threshold fitted on '
                  f'the same 100 observation tokens. [Timing data]({count_panel["file"]}).', '',
                  '| Centroids per KV head | Decode ms/token | Window ms/token | Lookup only (ms/token) | Mean exact prefix tokens per query head | Build and calibrate (s) |',
                  '|---|---:|---|---:|---:|---:|']
        for method, width in zip(count_methods, (1531, 2900), strict=True):
            d = decoders[method]
            index = d['squeezed_index']
            assert index['clusters'] == width and index['observation_window'] == 100
            windows = d['windows']
            assert all(w['last_layer_centroid_slots_per_kv_head'] == [width, width] for w in windows)
            times = ', '.join(f'{1000 * w["seconds"] / w["steps"]:.3f}' for w in windows)
            mean = statistics.mean(w['selected_prefix_tokens_per_query_head']['mean'] for w in windows)
            lines.append(f'| {width:,} | {d["median_decode_ms"]:.3f} | {times} | {lookup[str(width)]:.3f} | '
                         f'{mean:,.0f} | {index["seconds"]:.1f} |')
        lines += ['', 'The exact prefix count averages heads and layers at the end of each timed window. '
                  'The lookup-only diagnostic times one graph-captured centroid lookup per layer. '
                  'Both widths use the same observation queries, with the smaller width taking the first '
                  '1,531 centroids of the larger index. Build and calibration are outside decode timing.', '']
    online = next((d for d in reversed(summary['decode'].values()) if d['length'] == 32768
                   and d.get('fixed_centroids') is None and 'subset-512' in d['decoders']), None)
    if online:
        match_fields = ('length', 'prefill_recipe', 'hardware', 'prompt_token_sha256',
                        'prefix_tokens', 'warmup_steps', 'windows')
        calibrated = next((d for d in reversed(summary['decode'].values())
                           if all(d[k] == online[k] for k in match_fields)
                           and 'squeezed-p90' in d['decoders']), None)
        subset = online['decoders']['subset-512']
        lines += ['### Squeezed with Multipole clustering at 32K', '',
                  'The online variant uses Multipole’s block clustering at one centroid per eight '
                  'tokens, a 512-token exact budget, and centroid replacement disabled. '
                  'It updates the index every 128 tokens. The released port uses cuML clustering '
                  'and percentile calibration to select from a fixed prefix index.', '',
                  '| Decoder | Initial index and cache setup (s) | Decode ms/token |',
                  '|---|---:|---:|',
                  f'| [Squeezed subset 1:8, 512]({online["file"]}) | {subset["cache_setup_s"]:.2f} | '
                  f'{subset["median_decode_ms"]:.3f} |']
        if calibrated:
            released = calibrated['decoders']['squeezed-p90']
            lines += [f'| [Released Squeezed port, p90]({calibrated["file"]}) | '
                      f'{calibrated["squeezed_index"]["seconds"]:.2f} | {released["median_decode_ms"]:.3f} |']
        lines += ['', 'The online setup measurement includes cache installation and clustering. '
                  'The released-port setup measurement includes clustering and calibration.', '']
        if all('cache_update_seconds' in w for w in subset['windows']):
            updates = statistics.median(1000 * sum(w['cache_update_seconds']) / w['steps'] for w in subset['windows'])
            remaining = statistics.median(1000 * (w['seconds'] - sum(w['cache_update_seconds'])) / w['steps']
                                         for w in subset['windows'])
            lines += [f'Online index updates account for {updates:.2f} ms/token. '
                      f'The remaining decode step takes {remaining:.2f} ms/token.', '']
        for methods, label in (({'subset-512'}, 'Online Squeezed alone'),
                               ({'lod', 'dense', 'subset-512'}, 'LoD, dense, and online Squeezed together')):
            profile = next((p for p in reversed(summary['quality_runtime'].values())
                            if all(p[k] == online[k] for k in ('length', 'hardware', 'prefill_recipe'))
                            and set(p['rows'][0]['decoders']) == methods), None)
            if not profile:
                continue
            rows = profile['rows']
            build = statistics.median(r['decoders']['subset-512']['cache_setup_s'] for r in rows)
            total = statistics.median(r['elapsed_s'] for r in rows)
            lines += [f'{label} took a median {total:.2f} seconds per prompt across {len(rows)} RULER '
                      f'pilot prompts. Median online setup was {build:.2f} seconds. '
                      f'[Pilot records]({profile["file"]}).', '']
    if matched_decode:
        match_fields = ('length', 'prefill_recipe', 'hardware', 'prompt_token_sha256',
                        'prefix_tokens', 'warmup_steps', 'windows')
        squeezed = next((d for d in reversed(summary['decode'].values())
                         if all(d[k] == matched_decode[k] for k in match_fields)
                         and 'squeezed-p90' in d['decoders'] and {'lod', 'dense'} <= set(d['decoders'])), None)
        lines += ['### Matched 64K whole-model decode', '',
                  f'Each run prefills {matched_decode["prefix_tokens"]:,} tokens, warms up for '
                  f'{matched_decode["warmup_steps"]} decode steps, and measures '
                  f'{sum(matched_decode["windows"]):,} teacher-forced steps on one H100. '
                  f'[LoD, dense, and Multipole timing data]({matched_decode["file"]}).', '']
        arms = dict(matched_decode['decoders'])
        if squeezed:
            arms['squeezed-p90'] = squeezed['decoders']['squeezed-p90']
            summary['matched_prompts_64k']['squeezed_timing'] = squeezed['file']
            control = squeezed['decoders']
            lines += [f'The Squeezed run used the same prompt, teacher-forced tokens, and prefill recipe. '
                      f'Its LoD and dense controls measured {control["lod"]["median_decode_ms"]:.2f} '
                      f'and {control["dense"]["median_decode_ms"]:.2f} ms/token. '
                      f'[Squeezed timing data]({squeezed["file"]}).', '']
        lines += ['| Decoder | Median ms/token | Tokens/s | 32-per-task quality panel |',
                  '|---|---:|---:|---|']
        for method, d in arms.items():
            ms = d['median_decode_ms']
            status_label = 'Speed only' if method.startswith('squeezed-') else 'Complete'
            lines.append(f'| {NAMES[method]} | {ms:.3f} | {1000/ms:.1f} | {status_label} |')
        lines += ['', '### Initial index setup outside the decode timer', '',
                  '| Method | Seconds | Measured work |', '|---|---:|---|',
                  f'| Multipole | {arms["multipole-512"]["cache_setup_s"]:.2f} | Cache installation and initial clustering |']
        if squeezed:
            lines.append(f'| Released Squeezed port | {squeezed["squeezed_index"]["seconds"]:.2f} | '
                         'Initial clustering and percentile calibration |')
        lines += ['', 'LoD builds its region state during the common prefill.', '']
    for name, panel in sorted(summary['decode'].items(), key=lambda item: item[1]['length']):
        if panel is fixed or panel is count_panel or panel in kernel_swaps:
            continue
        steps = ', '.join(str(n) for n in panel['windows'])
        description = (f'{panel["prefix_tokens"]:,} prefix tokens. '
                       f'The {len(panel["windows"])} consecutive windows measure {steps} decode steps.')
        if panel['warmup_steps'] is not None:
            description += f' Warmup uses {panel["warmup_steps"]} decode steps.'
        lines += [f'{description} GPU {panel["hardware"].split(", ")[1]}. [Timing data]({panel["file"]}).', '',
                  '| Method | Median ms/token | Window ms/token | Update events by window |', '|---|---:|---|---|']
        for method, d in panel['decoders'].items():
            times = ', '.join(f'{1000*v["seconds"]/v["steps"]:.3f}' for v in d['windows'])
            updates = ', '.join(str(v['cache_update_events']) for v in d['windows'])
            lines.append(f'| {NAMES[method]} | {d["median_decode_ms"]:.3f} | {times} | {updates} |')
        lines.append('')
        if d := panel['decoders'].get('multipole-512'):
            if all('cache_update_seconds' in w for w in d['windows']):
                update = statistics.median(1000 * sum(w['cache_update_seconds']) / w['steps'] for w in d['windows'])
                rest = statistics.median(1000 * (w['seconds'] - sum(w['cache_update_seconds'])) / w['steps']
                                         for w in d['windows'])
                lines += [f'Multipole’s profiled online updates account for {update:.3f} ms/token. '
                          f'The remaining whole-model decode time is {rest:.3f} ms/token.', '']
        if index := panel['squeezed_index']:
            lines += [f'The released port built {index["clusters"]:,} centroids per physical KV head '
                      f'with a {index["observation_window"]}-token observation window. '
                      f'Index build and calibration took {index["seconds"]:.1f} seconds outside the decode timer.', '']
    if not summary['decode']:
        lines += ['No complete matched decode timing panel yet.', '']
    lines += ['LoD counts cache catch-up events per layer. Multipole and its Squeezed subset count index-update '
              'boundaries for the whole model. Calibrated Squeezed keeps the prefix index fixed '
              'and attends exactly to generated tokens. The percentiles set score thresholds calibrated on the '
              'observation window. The subset runs its compiled '
              'clustering and decode graph in its compatible Triton environment, using the exact saved prefill '
              'activations. Each subset budget builds its own initial index using the worker’s seed-0 RNG stream. '
              'The one-time host handoff is outside the decode window.', '',
              '## Validation and archived attempts', '',
              f'The archive contains {len(summary["invalid_rope"])} completed model runs with the incorrect '
              'long-context RoPE configuration. SUMMARY.json retains their files and exclusion reasons. '
              'The prior 64K/128K prefill and 128K real-state graph reports also require corrected reruns.', '',
              'The released port’s original atomic compaction produced graph/eager logit differences. '
              'Deterministic packing retains the same selected entries and gives exact graph/eager logits '
              'in the reported runs. The packing check also matches PyTorch selected indices at 4K, 32K, and 128K.', '',
              'Earlier setup attempts encountered missing generator dependencies, incompatible Inductor/Triton '
              'versions, and an unavailable offline wheel requirement. The completed reruns use the installed '
              'compatible clustering environment and retain the authors’ compilation.', '',
              f'The run archive has {len(summary["pending"])} pending, {len(summary["failed"])} failed, '
              f'{len(summary["interrupted"])} interrupted, and {len(summary["cancelled"])} cancelled jobs. '
              'Tables use completed panels. '
              'See [SUMMARY.json](SUMMARY.json) and provenance/ for hashes, commands and logs.', '']
    (root / 'REPORT.md').write_text('\n'.join(lines))
    (root / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    a = p.parse_args()
    s = publish(a.directory.resolve(), Path(__file__).resolve().parents[1])
    print(json.dumps({k: list(s[k]) for k in ('quality_64k', 'quality', 'shared_quality', 'decode')}))


if __name__ == '__main__':
    main()
