"""Run a frozen RULER cohort with dense, RetroInfer, or LoD attention."""
import argparse
import collections
import gc
import hashlib
import json
import os
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.environ.get('RETROINFER_ROOT', '/home/ubuntu/comparison/RetroInfer'))

from benchmarks.retro_lod_adapter import attach_lod, attach_lod_sparse_prefill_dense_decode
from config import generate_config
from model_hub import load_model, load_tokenizer


LIMITS = {**{f'niah_single_{i}': 128 for i in (1, 2, 3)},
          **{f'niah_multikey_{i}': 128 for i in (1, 2, 3)},
          'niah_multivalue': 128, 'niah_multiquery': 128,
          'vt': 30, 'fwe': 50, 'cwe': 120, 'qa_1': 32, 'qa_2': 32}


def score(task, prediction, answers):
    matches = [float(answer.lower() in prediction.lower()) for answer in answers]
    return max(matches) if task in ('qa_1', 'qa_2') else sum(matches) / len(matches)


def format_prompt(prompt, prompt_format, tokenizer=None):
    if prompt_format in ('dataset', 'qwen3-preformatted'):
        return prompt
    if prompt_format == 'qwen3-chat':
        if tokenizer is None:
            raise ValueError('Qwen3 prompt formatting needs its tokenizer')
        headers = list(re.finditer(r'<\|start_header_id\|>(system|user|assistant)<\|end_header_id\|>\n\n', prompt))
        messages = []
        for index, header in enumerate(headers):
            content = prompt[header.end():headers[index + 1].start() if index + 1 < len(headers) else len(prompt)]
            messages.append({'role': header.group(1), 'content': content.removesuffix('<|eot_id|>')})
        if not messages or messages[-1]['role'] != 'assistant' or messages[-2]['role'] != 'user':
            raise ValueError('expected final user turn and assistant answer prefix')
        prefix = messages.pop()['content']
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False) + prefix
    user_header = '<|start_header_id|>user<|end_header_id|>\n\n'
    assistant_header = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    turns = prompt.split(user_header)[1:]
    if not turns or len(turns) != prompt.count(assistant_header):
        raise ValueError('expected alternating Llama 3 user and assistant messages')
    formatted = []
    for index, turn in enumerate(turns):
        user, assistant = turn.split(assistant_header, 1)
        if index < len(turns) - 1:
            if not assistant.endswith('<|eot_id|>'):
                raise ValueError('expected a completed assistant example')
            assistant = assistant.removesuffix('<|eot_id|>')
        formatted.append('[INST] ' + user + ' [/INST]' + assistant)
    return '\n'.join(formatted)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def threshold_route_audit(cache):
    """Inspect the final decode step, without adding work to measured kernels."""
    rows=[]
    for layer,pool in enumerate(cache.pools):
        scratch=pool._threshold_scratch
        route=scratch['route'].detach().cpu().float()
        selected=scratch['selected'].detach().cpu().long()
        selected_count=scratch['count'].detach().cpu().long()
        counts=pool.state['counts'][0,:,:,0].detach().cpu().float()
        lengths=pool.state['page_cache']['slot_lengths'][0].detach().cpu().long()
        heads=route.shape[0];group=heads//counts.shape[0];slots=counts.shape[1]
        kh=torch.arange(heads)//group
        valid=torch.arange(slots)[None,:]<selected_count[:,None]
        idx=selected.clamp(0,slots-1)
        chosen_counts=counts[kh].gather(1,idx)
        chosen_lengths=lengths[kh].gather(1,idx)
        if not torch.equal(chosen_counts[valid].long(),chosen_lengths[valid]):
            raise RuntimeError('selected region count and exact leaf count differ')
        entries=torch.where(valid,chosen_lengths,0).sum(1)
        remote=lengths[kh].masked_fill(~torch.isfinite(route),0).sum(1)
        mass=route.softmax(1).gather(1,idx).masked_fill(~valid,0).sum(1)
        bound=cache.threshold_work_bound if cache.threshold_rule=='entry' else 16
        ratio=entries.float()/(bound*remote.sqrt())
        if cache.threshold_rule=='entry' and not bool((entries*entries<bound*bound*remote).all()):
            raise RuntimeError('per-entry work bound violated')
        rows.append({'layer':layer,'heads':heads,'entries_sum':int(entries.sum()),
                     'regions_sum':int(selected_count.sum()),'proxy_mass_sum':float(mass.sum()),
                     'empty_heads':int((selected_count==0).sum()),
                     'max_entries':int(entries.max()),'max_bound_fraction':float(ratio.max()),
                     'min_proxy_mass':float(mass.min()),'remote_min':int(remote.min()),
                     'remote_max':int(remote.max())})
    return {'scope':'final decode step, every layer and query head','layers':rows}


def main():
    from benchmarks.h100_prefill_direct_config import fused_selection_enabled
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--reference-cohort', type=Path, help='Require shared tasks to match this earlier cohort exactly')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('dense', 'retroinfer', 'two-tier', 'three-tier-bf16', 'three-tier-int4','multipole','squeezed-multipole','clusterkv'), required=True)
    parser.add_argument('--dtype', choices=('bfloat16', 'float16'), default='bfloat16')
    parser.add_argument('--clusterkv-centroids', type=int, default=400)
    parser.add_argument('--clusterkv-budget', type=int, default=1024)
    parser.add_argument('--eager', action='store_true', help='Run eager decode for FP16 dense or ClusterKV')
    parser.add_argument('--multipole-budget',type=int,default=512,help='Exact remote token budget in the authors Multipole harness')
    parser.add_argument('--tasks', nargs='+', choices=tuple(LIMITS), help='Run only these tasks')
    parser.add_argument('--prompt-format', choices=('dataset', 'retroinfer-meta-chat', 'qwen3-chat', 'qwen3-preformatted'), default='dataset',
                        help='Keep the same task context and references while changing the chat wrapper')
    parser.add_argument('--limit-per-task', type=int, default=None)
    parser.add_argument('--resume-from', type=Path, help='Continue a saved run with identical benchmark settings')
    parser.add_argument('--source-rows',nargs='+',type=int,
                        help='Diagnostic subset of frozen source rows; does not change cohort/reference validation')
    parser.add_argument('--check-graph', action='store_true', help='Check first graph step against eager on each task')
    parser.add_argument('--audit-threshold-routing', action='store_true', help='Collect final-step per-layer threshold work and proxy coverage')
    parser.add_argument('--lod-dense-prefill', action='store_true',
                        help='Diagnostic: build the LoD cache but use dense prefill outputs')
    parser.add_argument('--lod-dense-decode', action='store_true',
                        help='Diagnostic: LoD sparse prefill with full-KV dense decode')
    parser.add_argument('--validate-prefill-against', type=Path,
                        help='Require matching inputs and first token from a reference run')
    parser.add_argument('--lod-native-route-four-sink', action='store_true',
                        help='Diagnostic: native-query routing and four exact sink tokens')
    parser.add_argument('--model-name', default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--qwen-yarn', action='store_true', help='Enable YaRN for Qwen3')
    parser.add_argument('--qwen-yarn-factor', type=float, choices=(2.0, 4.0), default=4.0)
    args = parser.parse_args()
    if args.qwen_yarn and args.model_name != 'Qwen/Qwen3-8B':
        parser.error('--qwen-yarn requires Qwen/Qwen3-8B')
    author_mode=args.mode in ('multipole','squeezed-multipole')
    clusterkv = args.mode == 'clusterkv'
    eager = clusterkv or args.eager
    if eager and (args.mode not in ('clusterkv', 'dense') or args.dtype != 'float16' or args.check_graph):
        parser.error('eager quality supports FP16 ClusterKV or dense without graph checks')
    if args.resume_from and not eager:
        parser.error('--resume-from requires eager FP16 quality')
    if args.dtype != 'bfloat16' and (args.model_name != 'meta-llama/Llama-3.1-8B-Instruct'
            or args.mode not in ('clusterkv', 'dense')):
        parser.error('FP16 quality supports Llama dense and ClusterKV')
    dtype = getattr(torch, args.dtype)
    threshold_multiplier = os.environ.get('LOD_H100_THRESHOLD_MULTIPLIER')
    if args.audit_threshold_routing and threshold_multiplier is None:
        parser.error('--audit-threshold-routing requires threshold decode')
    if args.audit_threshold_routing:
        torch.set_num_threads(1)
    if threshold_multiplier is not None and (args.mode not in ('two-tier', 'three-tier-bf16', 'three-tier-int4') or args.lod_dense_decode):
        parser.error('LOD_H100_THRESHOLD_MULTIPLIER requires LoD decode')
    if args.audit_threshold_routing and args.mode != 'two-tier':
        parser.error('exact-entry routing audit currently requires two-tier')
    if args.lod_native_route_four_sink:
        if args.mode != 'three-tier-bf16' or args.model_name != 'meta-llama/Llama-3.1-8B-Instruct':
            parser.error('native routing/four sinks requires Llama three-tier BF16')
        os.environ['LOD_RETRO_NATIVE_ROUTE_FOUR_SINK'] = '1'
    if args.limit_per_task is not None and args.limit_per_task < 1:
        parser.error('--limit-per-task must be positive')
    if (args.lod_dense_prefill or args.lod_dense_decode) and args.mode not in (
            'two-tier', 'three-tier-bf16', 'three-tier-int4'):
        parser.error('prefill/decode diagnostics apply only to LoD')
    if args.lod_dense_prefill and args.lod_dense_decode:
        parser.error('choose one LoD diagnostic')
    if args.validate_prefill_against and not (args.lod_dense_prefill or args.lod_dense_decode or author_mode):
        parser.error('--validate-prefill-against requires a LoD diagnostic')
    reference_prefill = {}
    if args.validate_prefill_against:
        reference_data = json.loads(args.validate_prefill_against.read_text())
        expected_reference_mode = 'dense' if args.lod_dense_prefill or author_mode else args.mode
        if reference_data['stage'] != 'complete' or reference_data['mode'] != expected_reference_mode:
            raise ValueError('prefill reference must be a complete matching run')
        reference_prefill = {(r['task'], r['source_row']): r for r in reference_data['records']}
    raw = args.input.read_bytes()
    cohort = json.loads(raw)
    reference_hash = None
    if args.reference_cohort is not None:
        reference_raw = args.reference_cohort.read_bytes()
        reference = json.loads(reference_raw)
        reference_hash = hashlib.sha256(reference_raw).hexdigest()
        if any(reference[k] != cohort[k] for k in ('dataset', 'revision', 'seed', 'samples_per_task')):
            raise ValueError('reference cohort provenance does not match')
        current = {(s['task'], s['source_row']): s for s in cohort['samples']}
        for sample in reference['samples']:
            key = (sample['task'], sample['source_row'])
            if key not in current or current[key] != sample:
                raise ValueError(f'reference cohort sample differs: {key}')
        for task, source in reference['source_files'].items():
            if cohort['source_files'].get(task) != source:
                raise ValueError(f'reference cohort source differs: {task}')
    counts = collections.Counter()
    samples = []
    for sample in cohort['samples']:
        task = sample['task']
        if task not in LIMITS:
            raise ValueError(f'unsupported task {task}')
        if args.tasks is not None and task not in args.tasks:
            continue
        if args.source_rows is not None and sample['source_row']not in args.source_rows:
            continue
        if args.limit_per_task is not None and counts[task] >= args.limit_per_task:
            continue
        if hashlib.sha256(sample['prompt'].encode()).hexdigest() != sample['prompt_sha256']:
            raise ValueError(f'prompt hash mismatch: {task}/{sample["index"]}')
        counts[task] += 1
        samples.append(sample)
    if args.tasks is not None and set(counts) != set(args.tasks):
        raise ValueError('requested task is absent from the cohort')
    torch.set_grad_enabled(False)
    tokenizer = load_tokenizer(args.model_name)
    is_qwen3 = args.model_name == 'Qwen/Qwen3-8B'
    if is_qwen3:
        from benchmarks.retro_qwen3_adapter import RetroQwen3Model
        rope = {'rope_type':'yarn','factor':args.qwen_yarn_factor,
                'original_max_position_embeddings':32768} if args.qwen_yarn else None
        model = RetroQwen3Model(args.model_name, int(32768 * args.qwen_yarn_factor) if rope else 40960,
                               torch.bfloat16, 'cuda:0', tokenizer, rope_scaling=rope)
    else:
        model = load_model(args.model_name, 131072 if cohort.get('context_label') == '64k' else 65536,
                           dtype, 'cuda:0', tokenizer)
    from benchmarks.h100_comparison import configure_prefill_mlp_chunk
    configure_prefill_mlp_chunk(model)
    if clusterkv:
        from benchmarks.retro_clusterkv_adapter import attach_clusterkv
        attach_clusterkv(model, clusters=args.clusterkv_centroids, budget=args.clusterkv_budget)
    elif author_mode:
        from benchmarks.retro_multipole_adapter import attach_multipole
        attach_multipole(model,budget=args.multipole_budget,replacement=args.mode=='multipole',
            tokens_per_centroid=16 if args.mode=='multipole' else 8)
    elif args.mode not in ('dense', 'retroinfer'):
        if args.lod_dense_decode:
            attach_lod_sparse_prefill_dense_decode(model, args.mode)
        else:
            attach_lod(model, args.mode, dense_prefill=args.lod_dense_prefill)
    if os.environ.get('LOD_H100_PREFILL_RULE'):
        # The adapter installs the LoD namespace before importing its config.
        from lod_attention._config import PREFILL_CHUNK_SIZE
    adaptive_page = None
    if threshold_multiplier is not None and (page_multiplier := os.environ.get('LOD_H100_PAGE_THRESHOLD')):
        from benchmarks.h100_threshold_three_tier import (
            adaptive_page_options, adaptive_page_routing_options, threshold_launch_options, use_tuned_adaptive_pages,
        )
        tuned = use_tuned_adaptive_pages('cuda')
        route = adaptive_page_routing_options(tuned=tuned)
        launch = threshold_launch_options(
            tile=int(os.environ.get('LOD_H100_THRESHOLD_TILE', '64')),
            splits=int(os.environ.get('LOD_H100_THRESHOLD_SPLITS', '64')),
            tuned_mode='three-tier-adaptive-bf16' if tuned else None)
        adaptive_page = {'rule': route['rule'], 'multiplier': float(page_multiplier),
                         'workers': launch['workers'], 'warps': route['page_threshold_warps'],
                         'calibration_probe_sha256': os.environ.get('LOD_H100_PAGE_PROBE_SHA256'),
                         'residual': route['residual'], 'query': route['query_score'],
                         'single_page_regions': 'always exact', 'cap': None}
        if tuned:
            adaptive_page['kernel'] = adaptive_page_options(tuned=True)
            adaptive_page['tuned_defaults'] = True

    result = {'stage': 'running', 'mode': args.mode, 'model_name': args.model_name, 'qwen_yarn': args.qwen_yarn,
              'rope_max_position_embeddings': model.config.max_position_embeddings,
              'rope_scaling': model.config.rope_scaling if is_qwen3 else None,
              'source_rows_filter':args.source_rows,
              'tokenizer_revision': tokenizer.init_kwargs.get('_commit_hash'),
              'model_precision': 'bf16' if dtype == torch.bfloat16 else 'fp16',
              'kv_precision': 'int4' if args.mode.endswith('int4') else 'bf16' if dtype == torch.bfloat16 else 'fp16',
              'cohort_sha256': hashlib.sha256(raw).hexdigest(), 'dataset': cohort['dataset'],
              'reference_cohort_sha256': reference_hash,
              'dataset_revision': cohort['revision'], 'seed': cohort['seed'],
              'expected': dict(counts), 'generation': 'greedy, fixed task limits, ignore eos',
              'task_token_limits': LIMITS, 'graph_decode': not eager,
              'model_revision': getattr(model.config, '_commit_hash', None),
              'tokenizer_special_tokens': ('tokenizer default; Qwen chat template applied explicitly'
                                           if args.prompt_format in ('qwen3-chat', 'qwen3-preformatted') else
                                           'tokenizer default; no added chat template'),
              'prompt_format': args.prompt_format,
              'prefill_attention': ('dense with author cluster construction' if author_mode or clusterkv else
                                   'dense with LoD cache construction' if args.lod_dense_prefill else
                                    'dense' if args.mode in ('dense', 'retroinfer') else 'LoD sparse'),
              'prefill_threshold': ({'rule': os.environ['LOD_H100_PREFILL_RULE'],
                                     'mass_multiplier': float(os.environ.get('LOD_H100_PREFILL_MULTIPLIER', '16')),
                                     'entry_bound': int(os.environ.get('LOD_H100_PREFILL_ENTRY_BOUND', '16')),
                                     'max_regions': int(os.environ.get('LOD_H100_PREFILL_MAX_REGIONS', '0')) or None,
                                     'leaf_backend': ('grouped-eight' if os.environ.get('LOD_H100_PREFILL_GROUPED8') == '1' else 'threshold-default'),
                                     'prefill_block_size': PREFILL_CHUNK_SIZE,
                                     'reuse_mean': os.environ.get('LOD_H100_PREFILL_REUSE_MEAN') == '1',
                                     'no_host_sync': (os.environ.get('LOD_H100_PREFILL_NO_HOST_SYNC') == '1'
                                                      or os.environ.get('LOD_H100_PREFILL_PACKED') == '1'
                                                      or os.environ.get('LOD_H100_PREFILL_IMPLEMENTATION')in('closed','fast')),
                                     'packed': os.environ.get('LOD_H100_PREFILL_PACKED') == '1',
                                     'fuse_pack': os.environ.get('LOD_H100_PREFILL_FUSE_PACK') == '1',
                                     'pair_reuse': os.environ.get('LOD_H100_PREFILL_PAIR_REUSE') == '1',
                                     'work_bound': int(os.environ.get('LOD_H100_PREFILL_WORK_BOUND', os.environ.get('LOD_H100_PREFILL_ENTRY_BOUND', '16'))),
                                     'tile_queries': int(os.environ.get('LOD_H100_PREFILL_TILE_QUERIES', '512')),
                                      'aggregate_metadata': os.environ.get('LOD_H100_PREFILL_AGGREGATE') == '1',
                                      'selection_query_group': int(os.environ.get('LOD_H100_CLOSED_SELECT_QM','1')),
                                      'selection_scan_policy':os.environ.get('LOD_H100_FAST_SCAN_SELECT','auto'),
                                      'selection_scan_min_regions':int(os.environ.get('LOD_H100_SCAN_MIN_REGIONS','4096' if os.environ.get('LOD_H100_FAST_SCAN_SELECT','auto')=='auto' else '0')),
                                      'selection_scan_tile':[int(os.environ.get('LOD_H100_SCAN_M','32')),int(os.environ.get('LOD_H100_SCAN_N','256'))],
                                      'prefix_filter':os.environ.get('LOD_H100_FAST_PREFIX_SELECT')=='1' and not fused_selection_enabled(),
                                      'prefix_filter_policy':os.environ.get('LOD_H100_FAST_PREFIX_SELECT','auto'),
                                      'direct_closed':os.environ.get('LOD_H100_FAST_PREFIX_CLOSED')=='1',
                                      'closed_blas':os.environ.get('LOD_H100_FAST_CLOSED_BLAS')=='1',
                                      'closed_probability_precision':os.environ.get('LOD_H100_CLOSED_PRECISION','bf16'),
                                      'closed_arithmetic':os.environ.get('LOD_H100_CLOSED_ARITHMETIC','direct'),
                                      'closed_value_center':os.environ.get('LOD_H100_CLOSED_VALUE_CENTER')=='1',
                                      'direct_hoisted_operands':os.environ.get('LOD_H100_DIRECT_HOIST')=='1',
                                      'direct_fused_selection':fused_selection_enabled(),
                                      'closed_reduce_backend':os.environ.get('LOD_H100_CLOSED_REDUCE','wide'),
                                      'state_fused_rms':os.environ.get('LOD_H100_STATE_FUSED_RMS')=='1',
                                      'state_fused_rescale':os.environ.get('LOD_H100_STATE_FUSED_RESCALE')=='1',
                                      'state_packed_max':os.environ.get('LOD_H100_STATE_PACKED_MAX')=='1',
                                      'fast_tile_queries':os.environ.get('LOD_H100_FAST_TILE','4096'),
                                      'tc_reduce_rows':int(os.environ.get('LOD_H100_TC_REDUCE_ROWS','4')),
                                      'tc_reduce_parts':int(os.environ.get('LOD_H100_TC_REDUCE_PARTS','3')),
                                      'tc_reduce_ranks':int(os.environ.get('LOD_H100_TC_REDUCE_RANKS','32')),
                                      'direct_proxy_arithmetic':'legacy-rounded-product' if fused_selection_enabled()else None,
                                      'direct_selection_mode':os.environ.get('LOD_H100_DIRECT_SELECT_MODE','bitmap'),
                                      'direct_select_warps':int(os.environ.get('LOD_H100_DIRECT_SELECT_WARPS','8')),
                                      'direct_split_word_compaction':os.environ.get('LOD_H100_DIRECT_WORDS_SPLIT','1')=='1',
                                      'fast_exact_geometry':[int(os.environ.get('LOD_H100_CLOSED_LEAF_M','64')),int(os.environ.get('LOD_H100_CLOSED_LEAF_N','32')),int(os.environ.get('LOD_H100_CLOSED_LEAF_WARPS','4'))],
                                      'fast_persistent_workers':int(os.environ.get('LOD_H100_CLOSED_WORKERS','264')),
                                      'direct_fused_merge':os.environ.get('LOD_H100_DIRECT_FUSE_MERGE')=='1',
                                      'shared_log_counts':os.environ.get('LOD_H100_FAST_SHARED_LOG')=='1',
                                      'fused_merge':os.environ.get('LOD_H100_FAST_FUSED_MERGE')=='1',
                                      'producer_tile':[int(os.environ.get('LOD_H100_FAST_PRODUCE_M','64')),int(os.environ.get('LOD_H100_FAST_PRODUCE_N','64')),int(os.environ.get('LOD_H100_FAST_PRODUCE_WARPS','4')),(os.environ.get('LOD_H100_FAST_PRODUCE_STAGES','2') if os.environ.get('LOD_H100_FAST_PRODUCE_STAGES')=='auto' else int(os.environ.get('LOD_H100_FAST_PRODUCE_STAGES','2')))],
                                      'tiled_selection': os.environ.get('LOD_H100_FAST_TILED_SELECT') == '1',
                                      'selection_tile': [int(os.environ.get('LOD_H100_FAST_TILED_M','16')),int(os.environ.get('LOD_H100_FAST_TILED_N','128'))],
                                      'implementation': os.environ.get('LOD_H100_PREFILL_IMPLEMENTATION', 'reference')}
                                    if os.environ.get('LOD_H100_PREFILL_RULE') else None),
              'decode_attention': ('dense with full KV' if args.lod_dense_decode else args.mode),
              'threshold_routing': ({'mass_multiplier': float(threshold_multiplier),
                                     'rule': os.environ.get('LOD_H100_THRESHOLD_RULE', 'region'),
                                     'entry_bound': int(os.environ.get('LOD_H100_ENTRY_BOUND','16')) if os.environ.get('LOD_H100_THRESHOLD_RULE') == 'entry' else None,
                                     'work_bound': int(os.environ.get('LOD_H100_ENTRY_WORK_BOUND',os.environ.get('LOD_H100_ENTRY_BOUND','16'))) if os.environ.get('LOD_H100_THRESHOLD_RULE') == 'entry' else None,
                                     'query': 'RMS-normalized for region ranking',
                                     'maximum_open_region_entries': (None if os.environ.get('LOD_H100_THRESHOLD_RULE') == 'entry' else 1024),
                                     'implementation': os.environ.get('LOD_H100_THRESHOLD_IMPLEMENTATION', 'legacy'),
                                     'pages_per_region': int(os.environ.get('LOD_H100_THREE_TIER_PAGES', '2')) if args.mode.startswith('three-tier') and not os.environ.get('LOD_H100_PAGE_THRESHOLD') else None,
                                     'page_threshold': adaptive_page,
                                     'exact_attention': ('native query; selected regions, local and sink' if args.mode == 'two-tier' else
                                                         ('native query; threshold pages and disjoint residual per opened region, local and sink' if os.environ.get('LOD_H100_PAGE_THRESHOLD') else
                                                         f"native query; top-{int(os.environ.get('LOD_H100_THREE_TIER_PAGES', '2'))} pages and disjoint residual per opened region, local and sink"))}
                                    if threshold_multiplier is not None else None),
              'retroinfer_settings': ({'retrieval_budget': 0.018, 'estimation_budget': 0.232,
                                      'gpu_only': True, 'sink_tokens': 4, 'initial_local_tokens': 64,
                                      'prefill': 'dense', 'cache_rebuild': 'author GPU-only policy',
                                      'qwen_config': 'Llama-3.1 geometry-matched configuration' if is_qwen3 else None}
                                     if args.mode == 'retroinfer' else None),
              'lod_native_route_four_sink': args.lod_native_route_four_sink,
              'multipole_settings': ({'budget_tokens':args.multipole_budget,'tokens_per_centroid':16 if args.mode=='multipole' else 8,
                  'replacement':args.mode=='multipole','sink_tokens':10,'update_interval':128,'routing_position':2048,
                  'stable_compaction_order':os.environ.get('MULTIPOLE_STABLE_PACK')=='1',
                  'implementation':'authors connected Multipole generation adapter; Squeezed subset disables replacement'} if author_mode else None),
              'records': []}
    if clusterkv:
        result['clusterkv_settings'] = dict(prefill_centroids=args.clusterkv_centroids, budget=args.clusterkv_budget,
                                            dense_layers=2, sink=16, window=320, clusters_per_update=8,
                                            kmeans_iterations=20, offload=False, seed=1234)
    if args.resume_from:
        previous = json.loads(args.resume_from.read_text())
        expected = {(sample['task'], sample['source_row'], sample['prompt_sha256']) for sample in samples}
        seen = {(row['task'], row['source_row'], row['prompt_sha256']) for row in previous['records']}
        if (previous | {'records': []}) != result or not seen <= expected or len(seen) != len(previous['records']):
            raise ValueError('resume requires matching settings and unique, matching prompt records')
        result['records'] = previous['records']
        samples = [s for s in samples if (s['task'], s['source_row'], s['prompt_sha256']) not in seen]
    save(args.output, result)
    checked = set()
    for sample in samples:
        task = sample['task']
        start = time.monotonic()
        prompt = format_prompt(sample['prompt'], args.prompt_format, tokenizer)
        tokenized = tokenizer(prompt, return_tensors='pt')
        inputs = tokenized['input_ids'].to('cuda')
        input_length = inputs.shape[1]
        tokens = LIMITS[task]
        if input_length + tokens > model.max_length:
            raise ValueError(f'input exceeds model max length: {task}/{sample["index"]}: {input_length}')
        token_bytes = struct.pack(f'<{input_length}I', *inputs[0].tolist())
        model.attention_type = ('RetroClusterKV' if clusterkv else 'RetroMultipole' if author_mode else
                                'Full_Flash_Attn' if args.mode == 'dense' or args.lod_dense_decode else
                                'RetroInfer' if args.mode == 'retroinfer' else 'LoD')
        model.batch_size = 1
        model.input_length = input_length
        model.max_new_length = tokens
        model.prefill_bsz = 1
        model.prefill_method = 'full'
        if args.mode == 'retroinfer':
            config_name = 'meta-llama/Llama-3.1-8B-Instruct' if is_qwen3 else args.model_name
            config = generate_config(config_name, input_length, 'RetroInfer', 0.018, 0.232,
                                     0.0, use_cuda_graph=True, gpu_only=True)
        else:
            config = {} if is_qwen3 else generate_config(args.model_name, input_length, 'Full_Flash_Attn')
        if clusterkv:
            torch.manual_seed(1234)
        model.init_kv_cache(np.zeros(1, dtype=int), config)
        token = model.sampling(model.prefill_forward(inputs_ids=inputs))
        if args.validate_prefill_against:
            prior = reference_prefill[(task, sample['source_row'])]
            if (prior['input_token_sha256'] != hashlib.sha256(token_bytes).hexdigest()
                    or prior['output_token_ids'][0] != int(token.item())):
                raise RuntimeError(f'prefill disagrees with reference: {task}/{sample["index"]}')
        if args.lod_dense_decode:
            torch.cuda.synchronize()
            del model._lod_prefill_cache
        model.move()
        generated = [token.clone()]
        graph = None
        if eager:
            pass
        elif args.mode in ('dense', 'retroinfer') or args.lod_dense_decode:
            from benchmarks.retro_native_graph import NativeDecodeGraph
            graph = NativeDecodeGraph(model, token, model.attention_type)
        else:
            model.kv_cache.capture_decode(model, token)
        check = None
        for step in range(tokens - 1):
            if args.check_graph and task not in checked and step == 0:
                if graph is not None:
                    token, check = graph.checked_step(token)
                else:
                    token, check = model.kv_cache.checked_graph_step(model, token)
                if check['graph_argmax'] != check['eager_argmax'] or check['max_abs'] != 0.0:
                    raise RuntimeError(f'graph/eager mismatch on {task}/{sample["index"]}: {check}')
                checked.add(task)
            elif eager:
                token = model.sampling(model.decode_forward(inputs_ids=token))
            elif graph is not None:
                token = graph.step(token)
            else:
                token = model.kv_cache.graph_step(token)
            generated.append(token.clone())
        output_ids = torch.cat(generated, dim=-1).tolist()[0]
        prediction = tokenizer.decode(output_ids, skip_special_tokens=True)
        record = {'task': task, 'index': sample['index'], 'source_row': sample['source_row'],
                  'prompt_sha256': sample['prompt_sha256'], 'effective_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                  'input_token_sha256': hashlib.sha256(token_bytes).hexdigest(),
                  'input_tokens': input_length, 'answers': sample['answers'], 'prediction': prediction,
                  'output_token_ids': output_ids, 'score': score(task, prediction, sample['answers']),
                  'graph_check': check, 'elapsed_s': round(time.monotonic() - start, 3)}
        if args.mode == 'retroinfer':
            cache = model.kv_cache
            record['retroinfer_state'] = {name: getattr(cache, name) for name in
                                         ('n_centroids', 'nprobe', 'es_cluster_num', 'static_pattern_total', 'THRESHOLD')}
            record['retroinfer_state']['index_updates_during_decode'] = graph.index_update_count
        if args.audit_threshold_routing:
            record['threshold_route_audit'] = threshold_route_audit(model.kv_cache)
        if os.environ.get('LOD_H100_PAGE_THRESHOLD'):
            record['page_threshold_work'] = {'scope':'final decode step; every layer and query head',
                'fields':['opened_regions','candidate_pages','exact_pages','exact_entries'],
                'layers':[pool._h100_page_threshold_stats.sum(1).cpu().tolist() for pool in model.kv_cache.pools]}
        if os.environ.get('LOD_H100_PREFILL_AUDIT') == '1':
            layers = []
            for layer, pool in enumerate(model.kv_cache.pools):
                stats = getattr(pool.engine, '_lod_prefill_threshold_stats', None)
                if stats is None:
                    raise RuntimeError('threshold prefill audit did not observe a remote query')
                layers.append({'layer': layer, 'query_heads': stats['query_heads'],
                               'entries_sum': int(stats['entries_sum'].item()),
                               'regions_sum': int(stats['regions_sum'].item()),
                               'proxy_mass_sum': float(stats['proxy_mass_sum'].item()),
                               'max_bound_fraction': float(stats['max_bound_fraction'].item())})
            record['threshold_prefill_audit'] = {
                'scope': 'all remote prefill query heads; exact initial field and local/sink excluded',
                'layers': layers}
        if os.environ.get('LOD_H100_FAST_VERIFY') == '1':
            record['prefill_fast_audit'] = [
                getattr(pool.engine, '_lod_fast_prefill_audit', None)
                for pool in model.kv_cache.pools]
            if not all(record['prefill_fast_audit']):
                raise RuntimeError('fast threshold prefill verification did not run on every layer')
        if os.environ.get('LOD_H100_PREFILL_VERIFY') == '1':
            record['prefill_fused_validation'] = [
                getattr(pool.engine, '_lod_prefill_fused_validation', None)
                for pool in model.kv_cache.pools]
            if any(row is None for row in record['prefill_fused_validation']):
                raise RuntimeError('fused prefill validation did not run on every layer')
        result['records'].append(record)
        save(args.output, result)
        print(json.dumps({k: record[k] for k in ('task', 'index', 'input_tokens', 'score', 'elapsed_s')}), flush=True)
        if graph is not None:
            graph.close()
            del graph
        del model.kv_cache
        del inputs, token, generated
        gc.collect()
    result['stage'] = 'complete'
    result['task_scores'] = {task: 100 * sum(row['score'] for row in result['records'] if row['task'] == task) / n
                             for task, n in counts.items()}
    save(args.output, result)
    print(json.dumps({'stage': 'complete', 'task_scores': result['task_scores']}), flush=True)


if __name__ == '__main__':
    main()
