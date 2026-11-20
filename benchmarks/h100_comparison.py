"""Run batch-one author implementations on the frozen H100 LoD prompts.

Use via h100_cluster_run; author source and Python environments are external.
"""
import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path


def configure_prefill_mlp_chunk(model):
    """Use the same bounded MLP workspace across full-model benchmark arms."""
    if os.environ.get('RETRO_PREFILL_MLP_CHUNK'):
        # The author's 64K-token MLP chunks can exhaust GPU memory at
        # large batches even when the decode KV cache itself fits.
        # Keep the same MLP and attention; only reduce the prefill chunk.
        import inspect
        import textwrap
        import types
        chunk = int(os.environ['RETRO_PREFILL_MLP_CHUNK'])
        if chunk <= 0:
            raise ValueError('RETRO_PREFILL_MLP_CHUNK must be positive')
        original = model.layer_prefill.__func__
        source = textwrap.dedent(inspect.getsource(original))
        old_range = 'range(0, seq_len, 65536)'
        old_end = 'start_idx + 65536'
        if source.count(old_range) != 1 or source.count(old_end) != 1:
            raise RuntimeError('Unexpected RetroInfer prefill implementation')
        source = source.replace(old_range, f'range(0, seq_len, {chunk})')
        source = source.replace(old_end, f'start_idx + {chunk}')
        namespace = dict(original.__globals__)
        exec(compile(source, inspect.getsourcefile(original), 'exec'), namespace)
        model.layer_prefill = types.MethodType(namespace['layer_prefill'], model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("quest", "retro", "retro-full", "retro-quest", "retro-quest-author", "retro-clusterkv", "retro-clusterkv-full", "retro-squeezed", "retro-multipole", "retro-multipole-squeezed", "retro-lod-bf16", "retro-lod-int4"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument('--qwen-yarn',action='store_true',help='Use explicit factor-four YaRN for the long-context Qwen3 comparison, for every arm')
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1, help="Synchronous equal-length requests")
    parser.add_argument("--tokens", type=int, default=1025)
    parser.add_argument("--decode-warmup-steps", type=int, default=0,
                        help="Decode steps before the timed window, after prefill and graph capture")
    parser.add_argument("--expect-retro-index-updates", type=int,
                        help="Fail if the timed window contains a different number of RetroInfer index updates")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--quest-budget-ratio", type=float, default=0.25)
    parser.add_argument("--quest-budget-tokens", type=int,
                        help="Fixed QUEST budget, including the current page; overrides the ratio")
    parser.add_argument("--quest-shared-kv", action="store_true",
                        help="Use the patched author kernels with shared GQA KV storage")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="float16",
                        help="QUEST dtype; the released kernels support float16")
    parser.add_argument("--retro-dtype", choices=("bfloat16", "float16"), default="bfloat16",
                        help="Model dtype in the RetroInfer runtime")
    parser.add_argument("--capture-token-ids", action="store_true",
                        help="Save output token IDs for a correctness run")
    parser.add_argument("--lod-graph", action="store_true",
                        help="Capture the entire LoD model decode step")
    parser.add_argument("--quest-graph", action="store_true",
                        help="Capture the entire QUEST-style RetroInfer model decode step")
    parser.add_argument("--squeezed-graph", action="store_true", help="Capture Squeezed Attention model decode")
    parser.add_argument("--multipole-graph", action="store_true", help="Capture Multipole model decode")
    parser.add_argument("--clusterkv-centroids", type=int, default=400)
    parser.add_argument("--clusterkv-budget", type=int, default=1024)
    clusterkv_execution = parser.add_mutually_exclusive_group()
    clusterkv_execution.add_argument('--clusterkv-graph', action='store_true', help='Capture full ClusterKV decode')
    clusterkv_execution.add_argument('--clusterkv-device-eager', action='store_true',
                                     help='Run ClusterKV with device metadata and eager launches')
    parser.add_argument("--multipole-budget", type=int, default=512, help="Exact-token cluster budget")
    parser.add_argument("--squeezed-index", type=Path, help="Offline fixed-prefix cluster index")
    parser.add_argument("--squeezed-percentile", type=float, default=0.9)
    parser.add_argument("--native-graph", action="store_true",
                        help="Capture the author-framework dense or RetroInfer model step")
    parser.add_argument('--finish-lod-prefill-cache', action='store_true',
                        help='Include deferred LoD catch-up in prefill and record its separate cost')
    parser.add_argument("--replay-prompt-tokens", action="store_true",
                        help="Teacher-force a shared sequence from the prompt for correctness checks")
    parser.add_argument("--logits-file", type=Path,
                        help="Save selected decode logits for correctness checks")
    parser.add_argument("--check-graph-same-cache", action="store_true",
                        help="Compare graph and eager logits on identical LoD cache snapshots")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.qwen_yarn and args.model_name!='Qwen/Qwen3-8B':
        parser.error('--qwen-yarn requires Qwen/Qwen3-8B')
    if not 0 <= args.decode_warmup_steps < args.tokens - 1:
        parser.error('--decode-warmup-steps must be nonnegative and less than tokens minus one')
    if args.method == 'quest' and args.decode_warmup_steps:
        parser.error('--decode-warmup-steps requires a RetroInfer-runtime method')
    if args.expect_retro_index_updates is not None and args.method != 'retro':
        parser.error('--expect-retro-index-updates requires RetroInfer')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.batch_size > 1 and args.method not in ('retro-full', 'retro', 'retro-multipole', 'retro-multipole-squeezed', 'retro-lod-bf16', 'retro-lod-int4', 'retro-quest-author', 'retro-squeezed'):
        parser.error('This method has no common-runtime batched adapter')
    if args.batch_size > 1 and args.method in ('retro-full', 'retro') and not args.native_graph:
        parser.error('Batched native comparison requires --native-graph')
    if args.method.startswith('retro-clusterkv') and (args.retro_dtype != 'float16' or args.batch_size != 1):
        parser.error('ClusterKV requires batch-one FP16')
    clusterkv_device = args.clusterkv_graph or args.clusterkv_device_eager
    if clusterkv_device and not args.method.startswith('retro-clusterkv'):
        parser.error('ClusterKV execution flags require a ClusterKV method')
    if args.method.startswith('retro-multipole') and (not args.multipole_graph or args.retro_dtype != 'bfloat16'):
        parser.error('Multipole requires --multipole-graph and BF16')
    if args.multipole_graph and not args.method.startswith('retro-multipole'):
        parser.error('--multipole-graph requires a Multipole method')
    if args.method == 'retro-squeezed' and (not args.squeezed_graph or args.squeezed_index is None or args.retro_dtype != 'bfloat16'):
        parser.error('retro-squeezed requires --squeezed-graph, --squeezed-index and BF16')
    if args.squeezed_graph and args.method != 'retro-squeezed':
        parser.error('--squeezed-graph requires retro-squeezed')
    if args.quest_shared_kv and args.method != "retro-quest-author":
        parser.error("--quest-shared-kv requires --method retro-quest-author")
    if args.quest_budget_tokens is not None and (args.quest_budget_tokens < 16 or args.quest_budget_tokens % 16):
        parser.error("--quest-budget-tokens must be a positive multiple of the 16-token page size")
    if args.lod_graph and not args.method.startswith("retro-lod-"):
        parser.error("--lod-graph requires a RetroInfer LoD method")
    if args.quest_graph and args.method not in ("retro-quest", "retro-quest-author"):
        parser.error("--quest-graph requires a RetroInfer QUEST method")
    if args.method == "retro-quest-author" and (args.retro_dtype != "float16" or not args.quest_graph):
        parser.error("author QUEST requires --retro-dtype float16 and --quest-graph")
    if args.native_graph and args.method not in ("retro", "retro-full"):
        parser.error("--native-graph requires retro or retro-full")
    if args.check_graph_same_cache and not ((args.lod_graph or args.quest_graph or args.squeezed_graph or args.multipole_graph or args.native_graph or args.clusterkv_graph) and args.replay_prompt_tokens):
        parser.error("--check-graph-same-cache requires a graph and --replay-prompt-tokens")
    if args.replay_prompt_tokens and not args.capture_token_ids:
        parser.error("--replay-prompt-tokens requires --capture-token-ids")

    import torch
    from benchmarks.h100_prefill_direct_config import fused_selection_enabled

    torch.set_grad_enabled(False)
    manifest = json.loads(Path(args.manifest).read_text())
    ids = manifest["contexts"][str(args.length)]["prompts"][0]["prompt_token_ids"]
    if args.prompt_limit:
        ids = ids[:args.prompt_limit]
    assert len(ids) == (args.prompt_limit or args.length)
    inputs = torch.tensor([ids] * args.batch_size, dtype=torch.long, device="cuda")
    if args.replay_prompt_tokens and args.tokens - 1 > len(ids):
        raise ValueError("replay sequence is longer than the prompt")
    records = []
    budget = None
    squeezed_index = None
    index_read_s = None
    if args.method == 'retro-squeezed':
        start = time.perf_counter()
        squeezed_index = torch.load(args.squeezed_index, map_location='cpu', weights_only=True)
        index_read_s = time.perf_counter() - start
        prompt_hash = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
        if squeezed_index['prompt_hash'] != prompt_hash or squeezed_index['checkpoint'] != args.checkpoint:
            raise ValueError('Squeezed index does not match the prompt and checkpoint')
        if args.squeezed_percentile not in squeezed_index['threshold']:
            raise ValueError('Squeezed index does not have the requested percentile')

    if args.method == "quest":
        import inspect
        import importlib
        import textwrap
        import flashinfer.rope
        import quest.utils
        from quest import LlamaForCausalLM
        from quest.models.QuestAttention import QuestAttention
        quest_attention_module = importlib.import_module("quest.models.QuestAttention")

        # The original fast path only supports unscaled/linear RoPE. Preserve
        # Llama 3.1's exact frequency scaling using FlashInfer's fused kernel.
        def init_rope(self):
            self.rope_scale = 1.0

        rope_args = {}

        def llama31_rope(q, k, past_kv_len, rope_scale=None, rope_theta=None):
            key = (q.shape[0], past_kv_len)
            if key not in rope_args:
                rope_args[key] = (
                    torch.tensor([0, q.shape[0]], dtype=torch.int32, device=q.device),
                    torch.tensor([past_kv_len], dtype=torch.int32, device=q.device),
                )
            indptr, offsets = rope_args[key]
            flashinfer.rope.apply_llama31_rope_inplace(q, k, indptr, offsets)

        QuestAttention._init_rope = init_rope
        quest.utils.apply_rope_in_place = llama31_rope
        if args.dtype == "bfloat16":
            # The released RAFT radix selector supports FP16 but not BF16.
            # Preserve the top-k policy with PyTorch's BF16 implementation.
            def bf16_topk(scores, controller):
                count = controller.inference_page_budget - 1
                indices = torch.topk(scores, count, dim=-1, sorted=False).indices
                controller.topk_dindices_buffer.copy_(
                    controller.kv_indices_without_last.gather(1, indices))
            quest.utils.decode_topk = bf16_topk
        # The fast controller allocates one cache head per query head. Expand
        # Llama 3.1's GQA keys after RoPE so that its kernels see that layout.
        source = textwrap.dedent(inspect.getsource(QuestAttention.forward))
        marker = '    torch.cuda.nvtx.range_push("append_kv")\n'
        assert source.count(marker) == 1
        source = source.replace(marker,
            "    if self.num_key_value_groups > 1:\n"
            "        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)\n"
            "        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)\n\n" + marker)
        generated = {}
        exec(source, vars(quest_attention_module), generated)
        QuestAttention.forward = generated["forward"]
        dtype = getattr(torch, args.dtype)
        model = LlamaForCausalLM.from_pretrained(
            args.checkpoint, torch_dtype=dtype, device_map="cuda:0"
        ).eval()
        # Quest's attention uses its own causal kernel and ignores this mask.
        # Avoid allocating a quadratic mask at 32K/64K.
        model.model._prepare_decoder_attention_mask = lambda *unused, **kw: None
        if args.quest_budget_tokens is not None:
            budget = args.quest_budget_tokens
        elif args.quest_budget_ratio >= 1:
            # A full-attention reference must cover generated tokens too.
            budget = ((len(ids) + args.tokens + 31) // 16) * 16
        else:
            budget = max(16, int(len(ids) * args.quest_budget_ratio) // 16 * 16)
        model.quest_init(page_size=16, max_seq_len=len(ids) + args.tokens + 16,
                         token_budget=budget, dtype=dtype, device=torch.device("cuda:0"))

        def run_once():
            model.quest_clear()
            torch.cuda.synchronize()
            start = time.perf_counter()
            hidden = model.model(input_ids=inputs, use_cache=True).last_hidden_state
            token = model.lm_head(hidden[:, -1:, :]).argmax(-1)
            torch.cuda.synchronize()
            prefill = time.perf_counter() - start
            start = time.perf_counter()
            for _ in range(args.tokens - 1):
                hidden = model.model(input_ids=token, use_cache=True).last_hidden_state
                token = model.lm_head(hidden).argmax(-1)
            torch.cuda.synchronize()
            return {"prefill_s": prefill, "decode_s": time.perf_counter() - start}

    else:
        from config import generate_config
        from model_hub import load_model, load_tokenizer

        tokenizer = load_tokenizer(args.model_name)
        is_qwen3=args.model_name=='Qwen/Qwen3-8B'
        if is_qwen3:
            from benchmarks.retro_qwen3_adapter import RetroQwen3Model
            rope={'rope_type':'yarn','factor':4.0,'original_max_position_embeddings':32768}if args.qwen_yarn else None
            model=RetroQwen3Model(args.checkpoint,len(ids)+args.tokens,getattr(torch,args.retro_dtype),'cuda:0',tokenizer,rope_scaling=rope)
        else:
            model = load_model(args.model_name, len(ids) + args.tokens, getattr(torch, args.retro_dtype),
                               "cuda:0", tokenizer)
        configure_prefill_mlp_chunk(model)
        if args.method.startswith('retro-multipole'):
            if args.batch_size == 1:
                from benchmarks.retro_multipole_adapter import attach_multipole
            else:
                from benchmarks.retro_multipole_batch_adapter import attach_multipole_batch as attach_multipole
            attach_multipole(model, budget=args.multipole_budget, replacement=args.method == 'retro-multipole')
            attention = 'RetroMultipole'
        elif args.method.startswith('retro-clusterkv'):
            from benchmarks.retro_clusterkv_adapter import attach_clusterkv
            attach_clusterkv(model, clusters=args.clusterkv_centroids, budget=args.clusterkv_budget,
                             full=args.method == 'retro-clusterkv-full')
            attention = 'RetroClusterKV'
        elif args.method == 'retro-squeezed':
            if args.batch_size == 1:
                from benchmarks.retro_squeezed_adapter import attach_squeezed
            else:
                from benchmarks.retro_squeezed_batch_adapter import attach_squeezed_batch as attach_squeezed
            attach_squeezed(model, squeezed_index, args.squeezed_percentile)
            attention = 'RetroSqueezed'
        elif args.method in ("retro-quest", "retro-quest-author"):
            budget = (args.quest_budget_tokens if args.quest_budget_tokens is not None else
                      len(ids) + args.tokens if args.quest_budget_ratio >= 1
                      else max(16, int(len(ids) * args.quest_budget_ratio) // 16 * 16))
            if args.method == "retro-quest-author":
                if args.batch_size == 1:
                    from benchmarks.retro_quest_author_adapter import attach_author_quest
                else:
                    from benchmarks.retro_quest_author_batch_adapter import attach_author_quest_batch as attach_author_quest
                attach_author_quest(model, budget, shared_kv=args.quest_shared_kv)
            else:
                from benchmarks.retro_quest_adapter import attach_quest
                attach_quest(model, budget)
            attention = "RetroQuest"
        elif args.method.startswith("retro-lod-"):
            if args.batch_size == 1:
                from benchmarks.retro_lod_adapter import attach_lod
            else:
                from benchmarks.retro_lod_batch_adapter import attach_lod_batch as attach_lod
            attach_lod(model, "two-tier" if os.environ.get("LOD_RETRO_TWO_TIER") == "1" else "three-tier-int4" if args.method.endswith("int4") else "three-tier-bf16")
            attention = "LoD"
        else:
            attention = "RetroInfer" if args.method == "retro" else "Full_Flash_Attn"
        dense_attention_events = []
        if os.environ.get('LOD_H100_PREFILL_SCALING_AUDIT')=='1':
            if attention!='LoD':raise ValueError('prefill work audit requires LoD')
            from benchmarks.h100_prefill_work_audit import install
            install(model)
        if os.environ.get('LOD_H100_DENSE_PREFILL_GPU_EVENTS') == '1':
            if args.method != 'retro-full' or args.model_name != 'meta-llama/Llama-3.1-8B-Instruct':
                raise ValueError('dense attention instrumentation requires the Llama RetroInfer full-attention path')
            import model_hub.llama as llama_module
            original_dense_prefill = llama_module.full_prefill_attn
            def timed_dense_prefill(*call_args, **call_kwargs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                value = original_dense_prefill(*call_args, **call_kwargs)
                end.record()
                dense_attention_events.append((start, end))
                return value
            llama_module.full_prefill_attn = timed_dense_prefill
        config = {} if is_qwen3 and attention!='RetroInfer' else generate_config('meta-llama/Llama-3.1-8B-Instruct'if is_qwen3 else args.model_name, len(ids),
                                 "Full_Flash_Attn" if attention in ("LoD", "RetroQuest", "RetroSqueezed", "RetroMultipole", "RetroClusterKV") else attention, 0.018, 0.232,
                                 0.0, use_cuda_graph=True, gpu_only=True)

        def run_once():
            if attention in ("RetroMultipole", "RetroClusterKV"):
                torch.manual_seed(1234)
            model.attention_type = attention
            model.batch_size = args.batch_size
            model.input_length = len(ids)
            model.max_new_length = args.tokens
            model.prefill_bsz = 1
            model.prefill_method = "full"
            import numpy as np
            if attention == 'RetroClusterKV':
                model.kv_cache = None
            torch.cuda.synchronize()
            setup_start = time.perf_counter()
            model.init_kv_cache(np.zeros(args.batch_size, dtype=int), config)
            torch.cuda.synchronize()
            cache_init_s = time.perf_counter() - setup_start
            dense_attention_events.clear()
            start = time.perf_counter()
            token = model.sampling(model.prefill_forward(inputs_ids=inputs))
            token_ids = [token] if args.capture_token_ids else None
            saved_logits = {}
            graph_checks = {}
            prefill_cache_completion_s = 0.0
            if args.finish_lod_prefill_cache:
                if attention != 'LoD':
                    raise ValueError('prefill cache completion applies only to LoD')
                torch.cuda.synchronize()
                completion_start = time.perf_counter()
                pending = [(slot, model.kv_cache.context) for slot in range(args.batch_size)]
                for pool in model.kv_cache.pools:
                    if args.batch_size == 1:
                        pool.catch_up(*pending[0])
                    else:
                        pool.catch_up_many(pending)
                torch.cuda.synchronize()
                prefill_cache_completion_s = time.perf_counter() - completion_start
            model.move()
            torch.cuda.synchronize()
            prefill = time.perf_counter() - start
            dense_attention_profile = ({'calls': len(dense_attention_events),
                                        'gpu_seconds': sum(a.elapsed_time(b) for a, b in dense_attention_events) / 1000}
                                       if os.environ.get('LOD_H100_DENSE_PREFILL_GPU_EVENTS') == '1' else None)
            capture = 0.0
            native_graph = None
            if clusterkv_device:
                start = time.perf_counter()
                model.kv_cache.init_device_decode(model, token, capture=args.clusterkv_graph)
                torch.cuda.synchronize()
                capture = time.perf_counter() - start
            elif args.native_graph:
                from benchmarks.retro_native_graph import NativeDecodeGraph
                start = time.perf_counter()
                native_graph = NativeDecodeGraph(model, token, attention)
                torch.cuda.synchronize()
                capture = time.perf_counter() - start
            elif attention == "RetroInfer":
                start = time.perf_counter()
                model.kv_cache.capture_cuda_graph()
                torch.cuda.synchronize()
                capture = time.perf_counter() - start
            elif args.lod_graph or args.quest_graph or args.squeezed_graph or args.multipole_graph:
                start = time.perf_counter()
                model.kv_cache.capture_decode(model, token)
                torch.cuda.synchronize()
                capture = time.perf_counter() - start
            torch.cuda.synchronize()
            start = time.perf_counter()
            if os.environ.get("MULTIPOLE_NSYS") == "1":
                torch.cuda.cudart().cudaProfilerStart()
            decode_warmup_s = 0.0
            updates_before_timing = 0
            eager_retro_updates = 0
            check_steps = {0, 1, 14, 15, 16, 17, 30, 31, 32, 33,
                           254, 255, 256, 257, 510, 511, 512, 513,
                           766, 767, 768, 769, 1022, 1023, 1024, 1025,
                           2046, 2047, 2048, 2049, args.tokens - 2}
            if clusterkv_device:
                window = model.kv_cache.controller.window
                check_steps.update(i for end in range(window, args.tokens, window)
                                   for i in (end - 2, end - 1, end))
            for step in range(args.tokens - 1):
                if step == args.decode_warmup_steps and step:
                    torch.cuda.synchronize()
                    decode_warmup_s = time.perf_counter() - start
                    start = time.perf_counter()
                    if native_graph is not None:
                        updates_before_timing = native_graph.index_update_count
                    elif attention == 'RetroInfer':
                        updates_before_timing = eager_retro_updates
                    elif attention == 'LoD' and os.environ.get('LOD_TRACK_UPDATES') == '1':
                        updates_before_timing = (model.kv_cache.catch_up_events if args.batch_size == 1
                                                 else sum(pool.catch_up_rows for pool in model.kv_cache.pools))
                    elif attention == 'RetroMultipole' or args.method == 'retro-clusterkv':
                        updates_before_timing = model.kv_cache.update_count
                if args.replay_prompt_tokens:
                    offset = len(ids) - (args.tokens - 1) + step
                    token = inputs[:, offset:offset + 1]
                if clusterkv_device:
                    checked = args.check_graph_same_cache and step in check_steps
                    token, logits, stats = model.kv_cache.device_step(model, token, check=checked)
                    if checked:
                        graph_checks[step] = stats
                elif args.lod_graph or args.quest_graph or args.squeezed_graph or args.multipole_graph:
                    if args.check_graph_same_cache and step in check_steps:
                        token, graph_checks[step] = model.kv_cache.checked_graph_step(model, token)
                    else:
                        token = model.kv_cache.graph_step(token)
                    logits = model.kv_cache.graph_logits
                elif native_graph is not None:
                    if args.check_graph_same_cache and step in check_steps:
                        token, graph_checks[step] = native_graph.checked_step(token)
                    else:
                        token = native_graph.step(token)
                    logits = native_graph.logits
                else:
                    if attention == 'RetroInfer':
                        cache = model.kv_cache
                        eager_retro_updates += cache.static_pattern_total == cache.THRESHOLD
                    logits = model.decode_forward(inputs_ids=token)
                    token = model.sampling(logits)
                if args.logits_file and step in check_steps:
                    saved_logits[step] = logits.detach().clone()
                if token_ids is not None:
                    token_ids.append(token.clone() if (args.lod_graph or args.quest_graph or args.squeezed_graph or args.multipole_graph or args.clusterkv_graph or native_graph is not None) else token)
            torch.cuda.synchronize()
            decode = time.perf_counter() - start
            if os.environ.get("MULTIPOLE_NSYS") == "1":
                torch.cuda.cudart().cudaProfilerStop()
            if native_graph is not None:
                native_graph.close()
            route_geometry = None
            threshold_launch = None
            two_tier_launch = None
            squeezed_geometry = None
            lod_updates = None
            if attention == "LoD":
                if os.environ.get('LOD_TRACK_UPDATES') == '1':
                    updates = (model.kv_cache.catch_up_events if args.batch_size == 1
                               else sum(pool.catch_up_rows for pool in model.kv_cache.pools))
                    lod_updates = updates - updates_before_timing
                engine = model.kv_cache.pools[0].engine
                route_geometry = {
                    "group_size": int(engine.decode_route_group_size),
                    "warps": int(engine.decode_route_num_warps),
                    "reduce_warps": int(engine.decode_route_reduce_num_warps),
                    "page_block": int(engine.decode_block_n),
                }
                if os.environ.get('LOD_H100_THRESHOLD_IMPLEMENTATION') in ('three-tier', 'three-tier-reference'):
                    threshold_launch = getattr(model.kv_cache, 'threshold_launch', None)
                    if threshold_launch is not None:
                        page_block = threshold_launch['page_block_n'] or int(engine.leaf_block_n)
                    else:
                        page_block = int(os.environ.get('LOD_H100_THREE_TIER_PAGE_BLOCK', engine.leaf_block_n))
                    route_geometry['threshold_page_block'] = page_block
                if getattr(model.kv_cache, 'two_tier_optimized', None) is not None:
                    two_tier_launch = dict(tile=model.kv_cache.threshold_tile, splits=model.kv_cache.threshold_splits,
                                           two_tier_optimized=model.kv_cache.two_tier_optimized,
                                           **model.kv_cache.two_tier_kernel_options)
            elif attention == 'RetroSqueezed':
                squeezed_geometry = {'block_m': model.kv_cache.block_m,
                                     'num_warps': model.kv_cache.num_warps}
                selection_counts = (model.kv_cache.selection_counts.cpu().tolist()
                                    if model.kv_cache.selection_counts is not None else None)
            multipole_updates = model.kv_cache.update_count if attention == "RetroMultipole" else None
            multipole_update_times = model.kv_cache.update_times if attention == "RetroMultipole" else None
            clusterkv_updates = (model.kv_cache.update_count
                                 if args.method == 'retro-clusterkv' else None)
            prefill_profile = ([getattr(pool.engine, '_lod_prefill_threshold_profile', {}) for pool in model.kv_cache.pools]
                               if attention == 'LoD' and os.environ.get('LOD_H100_PREFILL_PROFILE') == '1' else None)
            leaf_profile = ([{name: sum(start.elapsed_time(end) for start, end in pairs) / 1000
                              for name, pairs in getattr(pool.engine, '_lod_leaf_timing_events', {}).items()}
                             for pool in model.kv_cache.pools]
                            if prefill_profile is not None else None)
            reuse_audit = ([getattr(pool.engine, '_lod_prefill_reuse_audit', None) for pool in model.kv_cache.pools]
                           if attention == 'LoD' and os.environ.get('LOD_H100_PREFILL_REUSE_AUDIT') == '1' else None)
            cap_audit = ([getattr(pool.engine, '_lod_prefill_cap_audit', None) for pool in model.kv_cache.pools]
                         if attention == 'LoD' and os.environ.get('LOD_H100_PREFILL_CAP_AUDIT') == '1' else None)
            backend_audit = ([getattr(pool.engine, '_lod_grouped8_backend_audit', None) for pool in model.kv_cache.pools]
                             if attention == 'LoD' and os.environ.get('LOD_H100_PREFILL_GROUPED8_CROSSCHECK') == '1' else None)
            top8_backend_audit = ([getattr(pool.engine, '_lod_top8_query_audit', None) for pool in model.kv_cache.pools]
                                  if attention == 'LoD' and os.environ.get('LOD_H100_TOP8_QUERY_AUDIT') == '1' else None)
            overlap_audit = ([getattr(pool.engine, '_lod_prefill_overlap_audit', None) for pool in model.kv_cache.pools]
                             if attention == 'LoD' and os.environ.get('LOD_H100_OVERLAP_AUDIT') == '1' else None)
            fast_audit = ([getattr(pool.engine, '_lod_fast_prefill_audit', None) for pool in model.kv_cache.pools]
                          if attention == 'LoD' and os.environ.get('LOD_H100_FAST_VERIFY') == '1' else None)
            closed_audit = ([getattr(pool.engine, '_lod_closed_prefill_audit', None) for pool in model.kv_cache.pools]
                            if attention == 'LoD' and os.environ.get('LOD_H100_CLOSED_VERIFY') == '1' else None)
            scaling_audit=None
            if os.environ.get('LOD_H100_PREFILL_SCALING_AUDIT')=='1':
                from benchmarks.h100_prefill_work_audit import collect
                scaling_audit=collect(model.kv_cache)
            del model.kv_cache
            gc.collect()
            result = {"prefill_s": prefill, "decode_s": decode, "capture_s": capture}
            if args.finish_lod_prefill_cache:
                result['prefill_cache_completion_s'] = prefill_cache_completion_s
                result['prefill_before_cache_completion_s'] = prefill - prefill_cache_completion_s
            if dense_attention_profile is not None:
                result['dense_prefill_attention_profile'] = dense_attention_profile
            result['cache_init_s'] = cache_init_s
            result['decode_warmup_s'] = decode_warmup_s
            if prefill_profile is not None:
                result['prefill_profile'] = prefill_profile
                result['prefill_leaf_profile'] = leaf_profile
            if reuse_audit is not None:
                result['prefill_reuse_audit'] = reuse_audit
            if cap_audit is not None:
                result['prefill_cap_audit'] = cap_audit
            if backend_audit is not None:
                result['prefill_backend_audit'] = backend_audit
            if top8_backend_audit is not None:
                result['prefill_top8_query_audit'] = top8_backend_audit
            if overlap_audit is not None:
                result['prefill_overlap_audit'] = overlap_audit
            if fast_audit is not None:
                result['prefill_fast_audit'] = fast_audit
            if closed_audit is not None:
                result['prefill_closed_audit'] = closed_audit
            if scaling_audit is not None:
                result['prefill_scaling_audit']=scaling_audit
            if attention == 'RetroInfer':
                updates = native_graph.index_update_count if native_graph is not None else eager_retro_updates
                result['retro_index_updates'] = updates
                result['retro_index_updates_timed'] = updates - updates_before_timing
                if (args.expect_retro_index_updates is not None and
                        result['retro_index_updates_timed'] != args.expect_retro_index_updates):
                    raise RuntimeError(f"Expected {args.expect_retro_index_updates} timed index updates, got {result['retro_index_updates_timed']}")
            if multipole_updates is not None:
                result["multipole_cluster_updates"] = multipole_updates
                result["multipole_cluster_updates_timed"] = multipole_updates - updates_before_timing
                if multipole_update_times:
                    result["multipole_update_times_s"] = multipole_update_times
            if clusterkv_updates is not None:
                result['clusterkv_updates_total'] = clusterkv_updates
                result['clusterkv_updates_timed'] = clusterkv_updates - updates_before_timing
                result['clusterkv_centroids_end'] = args.clusterkv_centroids + 8 * clusterkv_updates
            if squeezed_geometry is not None:
                result['squeezed_geometry'] = squeezed_geometry
                if selection_counts is not None:
                    result['squeezed_selected_tokens_by_layer_and_head'] = selection_counts
            if route_geometry is not None:
                result["lod_route_geometry"] = route_geometry
            if threshold_launch is not None:
                result["lod_threshold_launch"] = threshold_launch
            if two_tier_launch is not None:
                result["lod_two_tier_launch"] = two_tier_launch
            if lod_updates is not None:
                result['lod_layer_catch_up_events_timed'] = lod_updates
            if token_ids is not None:
                tokens_by_request = torch.cat(token_ids, dim=-1).tolist()
                result["output_token_ids"] = tokens_by_request[0] if args.batch_size == 1 else tokens_by_request
            if args.logits_file:
                args.logits_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save({k: v.cpu() for k, v in saved_logits.items()}, args.logits_file)
            if graph_checks:
                result["graph_same_cache_checks"] = graph_checks
            return result

    if os.environ.get('LOD_H100_PREFILL_RULE'):
        # The RetroInfer adapter installs the LoD namespace for its older
        # transformers version before this import.
        from lod_attention._config import PREFILL_CHUNK_SIZE
    for iteration in range(args.warmups + args.repeats):
        # run_once's local pool/engine references can keep the old LoD cache's
        # callback cycles reachable until its frame returns. Collect here,
        # outside both that frame and every timed interval, before allocating
        # the next cache. Otherwise the following prefill can retain two caches.
        gc.collect()
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = run_once()
        memory_after_run = torch.cuda.memory_allocated()
        peak_allocated = torch.cuda.max_memory_allocated()
        gc.collect()
        result['memory_bytes'] = {'before_cache_init': memory_before,
                                'after_run_before_collection': memory_after_run,
                                'after_run_after_collection': torch.cuda.memory_allocated(),
                                'peak_allocated': peak_allocated}
        timed_steps = args.tokens - 1 - args.decode_warmup_steps
        result.update(iteration=iteration, warmup=iteration < args.warmups,
                      decode_ms_per_batch_step=result["decode_s"] * 1000 / timed_steps,
                      decode_ms_per_token=result["decode_s"] * 1000 / (timed_steps * args.batch_size),
                      decode_tokens_per_second=timed_steps * args.batch_size / result["decode_s"])
        records.append(result)
        payload = {"measurement_scope": "full-model inference", "method": args.method,
                   "model": args.model_name, "checkpoint": args.checkpoint,
                   "qwen_yarn":args.qwen_yarn,
                   "rope_max_position_embeddings":model.config.max_position_embeddings if args.qwen_yarn else None,
                   "cache_update_optimizations":{"packed_max":os.environ.get('LOD_H100_STATE_PACKED_MAX')=='1',
                       "fused_rms":os.environ.get('LOD_H100_STATE_FUSED_RMS')=='1',
                       "fused_rescale":os.environ.get('LOD_H100_STATE_FUSED_RESCALE')=='1'},
                   "lod_cuda_graph": args.lod_graph,
                   "quest_cuda_graph": args.quest_graph,
                   "squeezed_cuda_graph": args.squeezed_graph,
                   "squeezed_index": str(args.squeezed_index) if args.squeezed_index else None,
                   "squeezed_index_read_s": index_read_s,
                   "squeezed_offline_clustering_s": squeezed_index['offline_clustering_s'] if squeezed_index else None,
                   "squeezed_threshold_percentile": args.squeezed_percentile if squeezed_index else None,
                   "squeezed_cluster_fraction_pct": squeezed_index['percent_clusters'] if squeezed_index else None,
                   "squeezed_observation_window": squeezed_index['observation_window'] if squeezed_index else None,
                   "squeezed_batch_implementation": (
                       "serial-independent-author-controllers" if args.method == "retro-squeezed" and args.batch_size > 1
                       else None),
                   "clusterkv_cuda_graph": args.clusterkv_graph,
                   "clusterkv_device_metadata": clusterkv_device,
                   "clusterkv_settings": ({"prefill_centroids": args.clusterkv_centroids,
                                           "token_budget": args.clusterkv_budget,
                                           "dense_layers": 2, "sink": 16, "window": 320,
                                           "clusters_per_update": 8, "kmeans_iterations": 20,
                                           "offload": False, "cuda_graph": args.clusterkv_graph, "seed": 1234}
                                          if args.method == 'retro-clusterkv' else None),
                   "multipole_cuda_graph": args.multipole_graph,
                   "multipole_budget": args.multipole_budget if args.multipole_graph else None,
                   "multipole_cluster_seed": 1234 if args.multipole_graph else None,
                   "native_cuda_graph": args.native_graph,
                   "replay_prompt_tokens": args.replay_prompt_tokens,
                   "dtype": args.dtype if args.method == "quest" else args.retro_dtype,
                   "batch_size": args.batch_size,
                   "decode_threshold": ({
                       "implementation": os.environ.get('LOD_H100_THRESHOLD_IMPLEMENTATION', 'legacy'),
                       "rule": os.environ.get('LOD_H100_THRESHOLD_RULE', 'region'),
                       "multiplier": float(os.environ['LOD_H100_THRESHOLD_MULTIPLIER']),
                       "entry_bound": int(os.environ.get('LOD_H100_ENTRY_BOUND', '16')),
                       "work_bound": int(os.environ.get('LOD_H100_ENTRY_WORK_BOUND', os.environ.get('LOD_H100_ENTRY_BOUND', '16'))),
                       "tile": int(os.environ.get('LOD_H100_THRESHOLD_TILE', '64')),
                       "splits": int(os.environ.get('LOD_H100_THRESHOLD_SPLITS', '64')),
                       "entry_block": int(os.environ.get('LOD_H100_THRESHOLD_ENTRY_BLOCK', '32')),
                       "shared_entry_block": int(os.environ.get('LOD_H100_SHARED_ENTRY_BLOCK', '64')),
                       "shared_pack_workers": int(os.environ.get('LOD_H100_SHARED_PACK_WORKERS', '16')),
                       "shared_pack_shards": int(os.environ.get('LOD_H100_SHARED_PACK_SHARDS', '4')),
                       "shared_precise": os.environ.get('LOD_H100_SHARED_PRECISE', '1') == '1',
                       "merge_block": int(os.environ.get('LOD_H100_THRESHOLD_MERGE_BLOCK', '0')),
                       "three_tier_workers": int(os.environ.get('LOD_H100_THREE_TIER_WORKERS', os.environ.get('LOD_H100_THRESHOLD_SPLITS', '64'))),
                       "three_tier_page_warps": int(os.environ.get('LOD_H100_THREE_TIER_PAGE_WARPS', '1')),
                       "three_tier_page_block": int(os.environ['LOD_H100_THREE_TIER_PAGE_BLOCK']) if 'LOD_H100_THREE_TIER_PAGE_BLOCK' in os.environ else None,
                       "three_tier_trim_merge": os.environ.get('LOD_H100_THREE_TIER_TRIM_MERGE') == '1',
                       "three_tier_gqa_local": os.environ.get('LOD_H100_THREE_TIER_GQA_LOCAL') == '1',
                       "three_tier_fused_pack": os.environ.get('LOD_H100_THREE_TIER_FUSED_PACK') == '1',
                       "three_tier_route_warps": int(os.environ.get('LOD_H100_THREE_TIER_ROUTE_WARPS', '4')),
                       "three_tier_local_tile": int(os.environ['LOD_H100_THREE_TIER_LOCAL_TILE']) if 'LOD_H100_THREE_TIER_LOCAL_TILE' in os.environ else None,
                       "fixed_page_scan": int(os.environ['LOD_H100_FIXED_PAGE_SCAN']) if 'LOD_H100_FIXED_PAGE_SCAN' in os.environ else None,
                       "materialize_cutoffs": os.environ.get('LOD_H100_THREE_TIER_MATERIALIZE_CUTOFFS') == '1',
                       "three_tier_cached_means": os.environ.get('LOD_H100_THREE_TIER_CACHED_MEANS', '1') == '1',
                       "three_tier_pages": int(os.environ.get('LOD_H100_THREE_TIER_PAGES', '2')) if not os.environ.get('LOD_H100_PAGE_THRESHOLD') else None,
                       "page_threshold_multiplier": float(os.environ['LOD_H100_PAGE_THRESHOLD']) if os.environ.get('LOD_H100_PAGE_THRESHOLD') else None,
                       "page_threshold_rule": os.environ.get('LOD_H100_PAGE_THRESHOLD_RULE','page') if os.environ.get('LOD_H100_PAGE_THRESHOLD') else None,
                       "page_query": os.environ.get('LOD_H100_PAGE_QUERY','native') if os.environ.get('LOD_H100_PAGE_THRESHOLD') else None,
                       "page_residual": os.environ.get('LOD_H100_PAGE_RESIDUAL','region') if os.environ.get('LOD_H100_PAGE_THRESHOLD') else None,
                       "page_threshold_warps": int(os.environ.get('LOD_H100_PAGE_THRESHOLD_WARPS', '4')),
                       "selected_region_cap": None,
                   } if os.environ.get('LOD_H100_THRESHOLD_MULTIPLIER') else None),
                   "prefill_threshold": ({"rule": os.environ['LOD_H100_PREFILL_RULE'],
                       "closed_options": ({"matmul": os.environ.get('LOD_H100_CLOSED_MATMUL', 'triton'),
                            "query_tile": int(os.environ.get('LOD_H100_CLOSED_TILE', '512' if os.environ['LOD_H100_PREFILL_RULE']=='entry' or os.environ.get('LOD_H100_CLOSED_EXACT')=='indexed' else '4096')),
                            "exact": os.environ.get('LOD_H100_CLOSED_EXACT', 'grouped'),
                            "workers": int(os.environ.get('LOD_H100_CLOSED_WORKERS', '264')),
                            "reduce": os.environ.get('LOD_H100_CLOSED_REDUCE', 'wide'), "variable_dispatch": True,
                            "metadata": os.environ.get('LOD_H100_CLOSED_METADATA', 'scan'),
                            "select_warps": int(os.environ.get('LOD_H100_CLOSED_SELECT_WARPS', '8')),
                            "select_qm": int(os.environ.get('LOD_H100_CLOSED_SELECT_QM', '1')),
                            "reduce_qm": int(os.environ.get('LOD_H100_CLOSED_REDUCE_QM', '4')),
                            "packed_output": os.environ.get('LOD_H100_CLOSED_PACKED_OUTPUT') == '1',
                            "leaf_m": int(os.environ.get('LOD_H100_CLOSED_LEAF_M', '64')),
                            "leaf_n": int(os.environ.get('LOD_H100_CLOSED_LEAF_N', '32')),
                            "leaf_warps": int(os.environ.get('LOD_H100_CLOSED_LEAF_WARPS', '4'))}
                            if os.environ.get('LOD_H100_PREFILL_IMPLEMENTATION')=='closed' else None),
                       "fast_tile_queries": ('auto' if os.environ.get('LOD_H100_FAST_TILE')=='auto' else int(os.environ.get('LOD_H100_FAST_TILE', '512' if os.environ['LOD_H100_PREFILL_RULE']=='entry' and os.environ.get('LOD_H100_FAST_LEAF')=='native' else '4096')))
                           if os.environ.get('LOD_H100_PREFILL_IMPLEMENTATION') == 'fast' else None,
                       "fast_leaf_backend": os.environ.get('LOD_H100_FAST_LEAF', 'grouped')
                           if os.environ.get('LOD_H100_PREFILL_IMPLEMENTATION') == 'fast' else None,
                       "fast_score_reuse": os.environ.get('LOD_H100_PREFILL_IMPLEMENTATION') == 'fast' and os.environ.get('LOD_H100_FAST_RECOMPUTE_CLOSED')!='1',
                       "state_fused_rms":os.environ.get('LOD_H100_STATE_FUSED_RMS')=='1',
                       "state_fused_rescale":os.environ.get('LOD_H100_STATE_FUSED_RESCALE')=='1',
                       "state_packed_max":os.environ.get('LOD_H100_STATE_PACKED_MAX')=='1',
                       "tc_reduce_rows":int(os.environ.get('LOD_H100_TC_REDUCE_ROWS','4')),
                       "tc_reduce_parts":int(os.environ.get('LOD_H100_TC_REDUCE_PARTS','3')),
                       "tc_reduce_ranks":int(os.environ.get('LOD_H100_TC_REDUCE_RANKS','32')),
                       "fast_recomputed_closed": os.environ.get('LOD_H100_FAST_RECOMPUTE_CLOSED')=='1',
                       "closed_normalize_query_group":int(os.environ.get('LOD_H100_CLOSED_NORMALIZE_M','0')),
                       "closed_reduce_backend":os.environ.get('LOD_H100_CLOSED_REDUCE','wide'),
                       "fast_select_m": int(os.environ.get('LOD_H100_FAST_SELECT_M', '1')),
                       "fast_full_select": os.environ.get('LOD_H100_FAST_FULL_SELECT') == '1',
                       "fast_saved_correction": os.environ.get('LOD_H100_FAST_SAVED_CORRECTION') == '1',
                       "fast_sparse_packing": os.environ.get('LOD_H100_FAST_SPARSE_PACK') == '1',
                       "fast_bitmap_packing": os.environ.get('LOD_H100_FAST_BITMAP_PACK') == '1',
                       "fast_prefix_filter": os.environ.get('LOD_H100_FAST_PREFIX_SELECT','auto')!='0' and not fused_selection_enabled(),
                       "prefix_filter_policy":os.environ.get('LOD_H100_FAST_PREFIX_SELECT','auto'),
                       "fast_prefix_full": os.environ.get('LOD_H100_FAST_PREFIX_FULL') == '1',
                       "fast_direct_closed": os.environ.get('LOD_H100_FAST_PREFIX_CLOSED') == '1',
                       "fast_closed_blas": os.environ.get('LOD_H100_FAST_CLOSED_BLAS') == '1',
                       "closed_probability_precision": os.environ.get('LOD_H100_CLOSED_PRECISION','bf16'),
                       "closed_arithmetic": os.environ.get('LOD_H100_CLOSED_ARITHMETIC','direct'),
                       "closed_value_center": os.environ.get('LOD_H100_CLOSED_VALUE_CENTER') == '1',
                       "direct_hoisted_operands": os.environ.get('LOD_H100_DIRECT_HOIST') == '1',
                       "direct_fused_selection": fused_selection_enabled(),
                       "direct_proxy_arithmetic": 'legacy-rounded-product' if fused_selection_enabled() else None,
                       "direct_selection_mode": os.environ.get('LOD_H100_DIRECT_SELECT_MODE','bitmap'),
                       "direct_select_warps": int(os.environ.get('LOD_H100_DIRECT_SELECT_WARPS','8')),
                       "direct_split_word_compaction": os.environ.get('LOD_H100_DIRECT_WORDS_SPLIT','1')=='1',
                       "fast_exact_geometry": [int(os.environ.get('LOD_H100_CLOSED_LEAF_M','64')),int(os.environ.get('LOD_H100_CLOSED_LEAF_N','32')),int(os.environ.get('LOD_H100_CLOSED_LEAF_WARPS','4'))],
                       "fast_persistent_workers": int(os.environ.get('LOD_H100_CLOSED_WORKERS','264')),
                       "direct_fused_merge": os.environ.get('LOD_H100_DIRECT_FUSE_MERGE') == '1',
                       "fast_shared_log": os.environ.get('LOD_H100_FAST_SHARED_LOG') == '1',
                       "fast_producer_tile": [int(os.environ.get('LOD_H100_FAST_PRODUCE_M','64')),int(os.environ.get('LOD_H100_FAST_PRODUCE_N','64')),int(os.environ.get('LOD_H100_FAST_PRODUCE_WARPS','4')),(os.environ.get('LOD_H100_FAST_PRODUCE_STAGES','2') if os.environ.get('LOD_H100_FAST_PRODUCE_STAGES')=='auto' else int(os.environ.get('LOD_H100_FAST_PRODUCE_STAGES','2')))],
                       "fast_fused_route_merge": os.environ.get('LOD_H100_FAST_FUSED_MERGE') == '1',
                       "fast_fused_merge_tile": [int(os.environ.get('LOD_H100_FUSED_MERGE_M','4')),int(os.environ.get('LOD_H100_FUSED_MERGE_R','16'))],
                       "fast_route_capacity": int(os.environ.get('LOD_H100_FAST_ROUTE_CAPACITY', '128')),
                       "fast_cap_early_exit": os.environ.get('LOD_H100_FAST_CAP_EARLY_EXIT', '1') == '1',
                       "mass_multiplier": float(os.environ.get('LOD_H100_PREFILL_MULTIPLIER', '16')),
                       "entry_bound": int(os.environ.get('LOD_H100_PREFILL_ENTRY_BOUND', '16')),
                       "max_regions": int(os.environ.get('LOD_H100_PREFILL_MAX_REGIONS', '0')) or None,
                       "leaf_backend": ('grouped-eight' if os.environ.get('LOD_H100_PREFILL_GROUPED8') == '1' else 'threshold-default'),
                       "grouped_fused_counts": os.environ.get('LOD_H100_PREFILL_GROUPED8_FUSED_COUNTS') == '1',
                       "work_bound": int(os.environ.get('LOD_H100_PREFILL_WORK_BOUND', os.environ.get('LOD_H100_PREFILL_ENTRY_BOUND', '16'))),
                       "tile_queries": int(os.environ.get('LOD_H100_PREFILL_TILE_QUERIES', '512')),
                       "prefill_block_size": PREFILL_CHUNK_SIZE,
                       "reuse_mean": os.environ.get('LOD_H100_PREFILL_REUSE_MEAN') == '1',
                       "no_host_sync": (os.environ.get('LOD_H100_PREFILL_NO_HOST_SYNC') == '1'
                                        or os.environ.get('LOD_H100_PREFILL_PACKED') == '1'
                                        or os.environ.get('LOD_H100_PREFILL_IMPLEMENTATION') in ('closed', 'fast')),
                       "packed": os.environ.get('LOD_H100_PREFILL_PACKED') == '1',
                       "fuse_pack": os.environ.get('LOD_H100_PREFILL_FUSE_PACK') == '1',
                       "pair_reuse": os.environ.get('LOD_H100_PREFILL_PAIR_REUSE') == '1',
                       "aggregate_metadata": os.environ.get('LOD_H100_PREFILL_AGGREGATE') == '1',
                       "selection_query_group": int(os.environ.get('LOD_H100_CLOSED_SELECT_QM','1')),
                       "selection_prefix_scan":os.environ.get('LOD_H100_PREFILL_IMPLEMENTATION')=='fast' and os.environ.get('LOD_H100_FAST_SCAN_SELECT','auto')!='0',
                       "selection_scan_policy":os.environ.get('LOD_H100_FAST_SCAN_SELECT','auto'),
                       "selection_scan_min_regions":int(os.environ.get('LOD_H100_SCAN_MIN_REGIONS','4096' if os.environ.get('LOD_H100_FAST_SCAN_SELECT','auto')=='auto' else '0')),
                       "selection_scan_tile":[int(os.environ.get('LOD_H100_SCAN_M','32')),int(os.environ.get('LOD_H100_SCAN_N','256'))],
                       "selection_scan_warps":int(os.environ.get('LOD_H100_SCAN_WARPS','4')),
                       "selection_scan_atomic_experts":os.environ.get('LOD_H100_SCAN_ATOMIC_EXPERTS')=='1',
                       "selection_precomputed_bias":os.environ.get('LOD_H100_FAST_BIASED_SELECT')=='1',
                       "tiled_selection": os.environ.get('LOD_H100_FAST_TILED_SELECT') == '1',
                       "selection_tile": [int(os.environ.get('LOD_H100_FAST_TILED_M','16')),int(os.environ.get('LOD_H100_FAST_TILED_N','128'))],
                       "implementation": os.environ.get('LOD_H100_PREFILL_IMPLEMENTATION', 'reference')}
                       if os.environ.get('LOD_H100_PREFILL_RULE') else None),
                   "prompt_tokens": len(ids), "manifest": args.manifest,
                   "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                   "prompt_token_sha256": hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
                   "output_tokens": args.tokens,
                   "prefill_includes_deferred_cache_completion": args.finish_lod_prefill_cache,
                   "decode_warmup_steps": args.decode_warmup_steps, "timed_decode_steps": timed_steps,
                   "quest_budget_ratio": args.quest_budget_ratio if budget is not None and args.quest_budget_tokens is None else None,
                   "quest_budget_tokens": budget,
                   "quest_shared_kv": args.quest_shared_kv,
                   "quest_batch_implementation": (
                       "serial-independent-author-controllers" if args.method == "retro-quest-author" and args.batch_size > 1
                       else None),
                   "prefill_mlp_chunk": int(os.environ['RETRO_PREFILL_MLP_CHUNK']) if 'RETRO_PREFILL_MLP_CHUNK' in os.environ else 65536,
                   "records": records}
        if launch := result.get("lod_threshold_launch"):
            payload["decode_threshold"].update(
                tile=launch['tile'], three_tier_workers=launch['workers'],
                merge_block=launch['merge_block'],
                three_tier_page_warps=launch['page_warps'], three_tier_page_block=launch['page_block_n'],
                three_tier_trim_merge=launch['trim_idle_merge'], three_tier_gqa_local=launch['gqa_local'],
                three_tier_fused_pack=launch['fused_pack'], three_tier_route_warps=launch['route_warps'],
                three_tier_local_tile=launch['local_tile'], materialize_cutoffs=launch['materialize_cutoffs'],
                fixed_page_scan=launch['fixed_page_scan'],
                factor_int4_page_scales=args.method == 'retro-lod-int4' and launch['factor_int4_page_scales'],
            )
        if launch := result.get("lod_two_tier_launch"):
            payload["decode_threshold"].update(launch)
        if payload["decode_threshold"] and os.environ.get('LOD_H100_PAGE_THRESHOLD'):
            from benchmarks.h100_threshold_three_tier import (
                adaptive_page_options, adaptive_page_routing_options, threshold_launch_options,
                use_tuned_adaptive_pages,
            )

            tuned = use_tuned_adaptive_pages('cuda')
            route = adaptive_page_routing_options(tuned=tuned)
            payload["decode_threshold"].update(
                page_query=route['query_score'], page_threshold_warps=route['page_threshold_warps'],
                adaptive_page_kernel=adaptive_page_options(tuned=tuned))
            if tuned and args.batch_size == 1:
                launch = threshold_launch_options(tuned_mode='three-tier-adaptive-bf16')
                payload["decode_threshold"].update(
                    tile=launch['tile'], three_tier_workers=launch['workers'],
                    three_tier_trim_merge=launch['trim_idle_merge'], three_tier_gqa_local=launch['gqa_local'],
                    three_tier_fused_pack=launch['fused_pack'], three_tier_local_tile=launch['local_tile'],
                    materialize_cutoffs=launch['materialize_cutoffs'])
        measured = [x for x in records if not x["warmup"]]
        if measured:
            payload["median_decode_ms_per_batch_step"] = statistics.median(x["decode_ms_per_batch_step"] for x in measured)
            payload["median_decode_tokens_per_second"] = statistics.median(x["decode_tokens_per_second"] for x in measured)
            payload["median_decode_ms_per_token"] = statistics.median(x["decode_ms_per_token"] for x in measured)
            payload["median_prefill_s"] = statistics.median(x["prefill_s"] for x in measured)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
