"""Four-layer GLM5.3-Flash FP8 integration fixture (not trained-model quality).

Real attention geometry, KDA, mHC and native learned sparse indexing; reduced
vocabulary and dense FFN widths keep this runnable on one development GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
from pathlib import Path


def audit_worker(worker):
    from collections import Counter
    model = worker.model_runner.model
    dtypes = Counter(str(p.dtype) for p in model.parameters())
    modules = Counter(type(m).__name__ for m in model.modules())
    mla = [m for m in model.modules() if type(m).__name__ == "MLAAttention"]
    pools = [m._vllm_lod_pool for m in mla if hasattr(m, "_vllm_lod_pool")]
    from vllm.models.glm5next.common import kda
    kda_prefill = kda.chunk_kda_with_fused_gate
    return dict(parameter_dtypes=dict(dtypes), modules=dict(modules),
        kda_gluon_prefill=bool(getattr(kda_prefill, "_lod_glm_g8", False)),
        kda_gluon_prefill_calls=getattr(kda_prefill, "calls", 0),
        mla=[dict(heads=m.num_heads, key_dim=m.head_size, latent_dim=m.kv_lora_rank,
                  direct_dim=m.qk_rope_head_dim, expanded_key_dim=m.qk_nope_head_dim,
                  expanded_value_dim=m.v_head_dim, backend=type(m.impl).__name__,
                  prefill_backend=type(m.prefill_backend).__name__,
                  sparse=m.use_sparse or m.impl.is_sparse,
                  indexer=m.indexer is not None,
                  native_prefix_calls=getattr(m, "_vllm_lod_native_prefix_calls", 0),
                  native_prefix_tokens=getattr(m, "_vllm_lod_native_prefix_tokens", 0),
                  deferred_query_tokens=getattr(m, "_vllm_lod_deferred_query_tokens", 0),
                  exact_prefill_calls=getattr(m, "_glm53_exact_prefill_calls", 0),
                  exact_prefill_tokens=getattr(m, "_glm53_exact_prefill_tokens", 0),
                  exact_prefill_histories=len(getattr(m, "_glm53_exact_prefill_history", {})),
                  lod_pool=hasattr(m, "_vllm_lod_pool")) for m in mla],
        pools=[dict(local_lens=p.local_lens.tolist(), state_lens=p.state_lens.tolist(),
                    leaf_lens=p.leaf_lens.tolist(), metadata=p.metadata,
                    exact_first_chunk=p.engine.prefill_exact_first_chunk,
                    max_open_centroid_leaves=p.engine.max_open_centroid_leaves,
                    prefill_topk=p.engine.prefill_two_level_topk,
                    decode_topk=p.engine.two_level_topk,
                    head_tiled_decode=p.kimi_head_tiled_decode,
                    projected_clustering_heads=(getattr(p.engine, "_glm53_projected_clustering_weight").size(0)
                        if hasattr(p.engine, "_glm53_projected_clustering_weight") else 0),
                    projected_clustering_calls=getattr(p.engine, "_glm53_projected_clustering_calls", 0),
                    projected_leaf_calls=getattr(p.engine, "_lod_glm_projected_leaf_calls", 0),
                    count_sums=p.state["counts"].sum(dim=(1, 2, 3)).tolist()) for p in pools])


def enable_worker_trace(worker):
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    return os.getpid()


def disable_native_prefix(worker):
    """Diagnostic control: change only initial attention, not cache building."""
    for layer in worker.model_runner.model.modules():
        if getattr(layer, "_vllm_lod_glm53", False):
            layer._vllm_lod_native_prefix = False


def disable_exact_prefix(worker):
    """Diagnostic only: keep the ordinary two-chunk front, not the 16K bypass."""
    for layer in worker.model_runner.model.modules():
        pool = getattr(layer, "_vllm_lod_pool", None)
        if getattr(layer, "_vllm_lod_glm53", False) and pool is not None:
            pool.engine.prefill_exact_first_chunk = False


def use_vector_page_lookup(worker):
    """Fixture-only control: repeat directory lookup for every leaf lane."""
    import lod_attention._core as core
    original = core.paged_leaf_attention
    def vector_lookup(*args, **kwargs):
        kwargs["scalar_page_lookup"] = False
        return original(*args, **kwargs)
    core.paged_leaf_attention = vector_lookup


def use_absorbed_coarse(worker):
    """Matched diagnostic control for the pre-projection prefill path."""
    for layer in worker.model_runner.model.modules():
        if getattr(layer, "_vllm_lod_glm53", False):
            layer._vllm_lod_absorbed_coarse = True


def use_latent_local(worker):
    for layer in worker.model_runner.model.modules():
        if getattr(layer, "_vllm_lod_glm53", False):
            layer._vllm_lod_pool.engine._lod_glm_project_local = False


def trace_native_indexer(worker):
    """Log the first native decode call without modifying its computation."""
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops
    original = ops.rocm_fp8_paged_mqa_logits
    seen = set()

    def traced(q, cache, weights, lengths, blocks, schedule, max_model_len, **kwargs):
        shape = (tuple(q.shape), tuple(cache.shape), max_model_len)
        if shape not in seen:
            seen.add(shape)
            print("GLM53_NATIVE_INDEXER " + json.dumps(dict(
                query_shape=list(q.shape), cache_shape=list(cache.shape),
                cache_stride=list(cache.stride()), lengths_shape=list(lengths.shape),
                blocks_shape=list(blocks.shape), max_model_len=max_model_len)), flush=True)
        return original(q, cache, weights, lengths, blocks, schedule, max_model_len, **kwargs)

    ops.rocm_fp8_paged_mqa_logits = traced


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="tests/fixtures/glm53-flash-mixed4-fp8")
    p.add_argument("--mode", choices=("full", "two-tier", "three-tier-bf16", "three-tier-int4"), required=True)
    p.add_argument("--exact-all-history", action="store_true",
                   help="Benchmark-only dense MLA control, with no sparse indexer")
    exact_prefill = p.add_mutually_exclusive_group()
    exact_prefill.add_argument("--exact-prefill", action="store_true")
    exact_prefill.add_argument("--exact-final-row", action="store_true")
    p.add_argument("--uncapped", action="store_true")
    p.add_argument("--projected-clustering", action="store_true")
    p.add_argument("--length", type=int, default=4096)
    p.add_argument("--decode-tokens", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--kv-cache-mib", type=int, default=512,
                   help="Native vLLM cache budget per worker; increase symmetrically for B8 to avoid preemption")
    p.add_argument("--tp", type=int, choices=(1, 2, 4, 8), default=1,
                   help="Tensor-parallel workers; attention geometry is partitioned by vLLM")
    p.add_argument("--ffn-width", type=int,
                   help="Optional random-fixture width; use 1024 for matched TP4/TP8 tests")
    p.add_argument("--profile-prefill", action="store_true",
                   help="Diagnostic worker trace; use --decode-tokens 1 to isolate prefill")
    p.add_argument("--vector-page-lookup", action="store_true",
                   help="Diagnostic control: repeat page lookup for each leaf lane rather than once per page")
    p.add_argument("--absorbed-coarse", action="store_true",
                   help="Diagnostic control: retain latent512 centroid attention, rather than projected256")
    p.add_argument("--latent-local", action="store_true",
                   help="Diagnostic control: retain absorbed512 exact local attention")
    prefix = p.add_mutually_exclusive_group()
    prefix.add_argument("--native-prefix", action="store_true",
                   help="Experimental native-sparse initial chunk; later chunks/decode remain LoD")
    prefix.add_argument("--exact-prefix", action="store_true",
                   help="Diagnostic LoD control: exact instead of native-sparse initial chunk")
    prefix.add_argument("--no-exact-prefix", action="store_true",
                   help="Diagnostic only: disable the exact 16K front; preserve the usual 512-token front")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if min(args.length, args.decode_tokens, args.batch_size, args.kv_cache_mib) <= 0:
        p.error("length, decode tokens, batch size and cache budget must be positive")
    if args.exact_all_history and args.mode != "full":
        p.error("--exact-all-history requires --mode full")
    if (args.exact_prefill or args.uncapped or args.exact_final_row or args.projected_clustering) and args.mode != "two-tier":
        p.error("LoD ablations require --mode two-tier")
    if args.exact_final_row and args.uncapped:
        p.error("--exact-final-row keeps the release cap unchanged")
    if (args.exact_prefix or args.native_prefix or args.no_exact_prefix) and args.mode == "full":
        p.error("prefix ablations require --mode two-tier")
    if args.vector_page_lookup and args.mode == "full":
        p.error("page lookup control requires --mode two-tier")
    if args.absorbed_coarse and args.mode == "full":
        p.error("absorbed coarse control requires --mode two-tier")
    if args.latent_local and args.mode == "full":
        p.error("latent local requires two-tier prefill")
    fixture_config = json.loads((Path(args.checkpoint) / "config.json").read_text())
    ffn_width = args.ffn_width or int(fixture_config["intermediate_size"])
    if ffn_width <= 0 or ffn_width % (128 * args.tp):
        p.error("FP8 fixture FFN width must be a multiple of 128 * TP (use --ffn-width 1024 for TP8)")
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"  # local trusted fixture RPC
    # AITER's implicit model-table merge writes a root-owned shared /tmp
    # directory in this image. Explicit shipped CSVs bypass that merge without
    # changing the GEMM or its quantization. Keep actual JIT artifacts local
    # through the runtime launcher, and never alter another user's /tmp files.
    spec = importlib.util.find_spec("aiter")
    if spec and spec.origin:
        configs = Path(spec.origin).parent / "configs"
        os.environ.setdefault("AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
                              str(configs / "a8w8_blockscale_tuned_gemm.csv"))
    from benchmarks._vllm import close_llm, configure_environment, write_json
    configure_environment(args.mode, args.batch_size)
    from vllm import LLM, SamplingParams
    backend = "ROCM_AITER_MLA_SPARSE"
    kwargs = dict(model=args.checkpoint, model_impl="vllm", dtype="bfloat16",
        kv_cache_dtype="bfloat16", load_format="dummy", skip_tokenizer_init=True,
        max_model_len=args.length + args.decode_tokens + 1,
        max_num_seqs=args.batch_size, max_num_batched_tokens=16384 + args.batch_size,
        long_prefill_token_threshold=min(16384, args.length + args.decode_tokens + 1), tensor_parallel_size=args.tp,
        kv_cache_memory_bytes=args.kv_cache_mib << 20, gpu_memory_utilization=0.15,
        enforce_eager=True, enable_prefix_caching=False, disable_log_stats=False,
        attention_config={"backend": backend}, seed=1234,
        compilation_config={"mode": 0})
    overrides = {}
    if args.ffn_width is not None:
        overrides["intermediate_size"] = args.ffn_width
    if args.exact_prefix or args.native_prefix:
        overrides["lod_native_prefix"] = True
    if overrides:
        kwargs["hf_overrides"] = overrides
    if args.mode != "full":
        kwargs["attention_config"] = {"backend": "CUSTOM", "backend_per_kind": {"mla_attention": "TRITON_MLA"}}
        kwargs["scheduler_cls"] = "vllm_lod_plugin.scheduler.LODChunkAlignedScheduler"
    if args.exact_all_history:
        kwargs.update(worker_cls="benchmarks._glm53_dense_attention.GLMDenseWorker",
                      attention_config={"backend": None,
                          "backend_per_kind": {"mla_attention": "TRITON_MLA"},
                          "mla_prefill_backend": "ROCM_AITER_FA"})
    elif args.exact_prefill or args.uncapped or args.exact_final_row:
        name = ("GLMExactFinalRowWorker" if args.exact_final_row else
                "GLMUncappedExactPrefillWorker" if args.exact_prefill and args.uncapped
                else "GLMExactPrefillWorker" if args.exact_prefill else "GLMUncappedWorker")
        kwargs["worker_cls"] = "benchmarks._glm53_lod_ablation." + name
    result = dict(mode=args.mode, length=args.length, batch=args.batch_size,
        tensor_parallel_size=args.tp,
        kv_cache_memory_bytes=kwargs["kv_cache_memory_bytes"],
        max_num_batched_tokens=kwargs["max_num_batched_tokens"],
        ffn_intermediate_size=ffn_width,
        random_weights=True, quality_evidence=False, fixture=args.checkpoint,
        timing=("diagnostic profiler; instrumentation overhead included"
                if args.profile_prefill else
                "one untimed exact-shape warmup followed by one synchronized pass"),
        exact_all_history=args.exact_all_history,
        exact_prefill=args.exact_prefill, exact_final_row=args.exact_final_row, uncapped=args.uncapped,
        projected_clustering=args.projected_clustering,
        reference=("exact all-history dense MLA" if args.exact_all_history else
                   "native learned sparse MLA (2048-token indexer budget), not dense attention"),
        initial_prefix=("native GLM learned sparse" if args.native_prefix else "exact")
            if args.mode != "full" else "native",
        status="initializing")
    if args.mode != "full":
        result["page_lookup"] = "vector control" if args.vector_page_lookup else "scalar per page"
        result["coarse_geometry"] = "absorbed512" if args.absorbed_coarse else "projected256-combined-kv"
        result["local_geometry"] = "absorbed512" if args.absorbed_coarse or args.latent_local else "projected256"
    if args.no_exact_prefix:
        result["scope"] = "diagnostic exact-16K-prefix bypass, not the release policy"
        result["initial_prefix"] = "ordinary 512-token exact front"
    write_json(args.output, result)
    llm = None
    try:
        llm = LLM(**kwargs)
        if args.projected_clustering:
            from benchmarks._glm53_projected_clustering import install_projected_key_clustering
            result["projected_clustering_weights"] = llm.collective_rpc(install_projected_key_clustering)
        if args.absorbed_coarse:
            llm.collective_rpc(use_absorbed_coarse)
        if args.latent_local:
            llm.collective_rpc(use_latent_local)
        if args.exact_prefix:
            llm.collective_rpc(disable_native_prefix)
        if args.no_exact_prefix:
            llm.collective_rpc(disable_exact_prefix)
        if args.vector_page_lookup:
            llm.collective_rpc(use_vector_page_lookup)
        result["worker_pids"] = llm.collective_rpc(enable_worker_trace)
        if args.mode == "full" and not args.exact_all_history:
            llm.collective_rpc(trace_native_indexer)
        result["workers"] = llm.collective_rpc(audit_worker)
        if args.exact_all_history:
            assert all(not m["sparse"] and not m["indexer"] and not m["lod_pool"]
                       for w in result["workers"] for m in w["mla"])
        assert all("torch.float8_e4m3fn" in x["parameter_dtypes"] or "torch.float8_e4m3fnuz" in x["parameter_dtypes"] for x in result["workers"]), "FP8 fixture silently became BF16"
        params = SamplingParams(temperature=0, max_tokens=args.decode_tokens, ignore_eos=True, logprobs=1)
        prompts = [dict(prompt_token_ids=[3 + ((i + row * 17) % 997) for i in range(args.length)]) for row in range(args.batch_size)]
        import torch
        for phase in ("warmup", "measured"):
            result["status"] = phase
            write_json(args.output, result)
            print("GLM53_PHASE " + phase, flush=True)
            if args.profile_prefill and phase == "measured":
                from benchmarks._prefill_profile import start_prefill_profile
                llm.collective_rpc(start_prefill_profile,
                                   kwargs={"collect_projection_usage": False})
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            assert len(outputs) == args.batch_size
            assert all(len(x.outputs[0].token_ids) == args.decode_tokens for x in outputs)
            assert all(math.isfinite(scores[token].logprob)
                       for x in outputs for token, scores in
                       zip(x.outputs[0].token_ids, x.outputs[0].logprobs, strict=True)), "nonfinite logits"
            assert all(x.metrics is not None and x.metrics.num_preemptions == 0
                       for x in outputs), "cache preemption invalidates the timing comparison"
            if phase == "measured":
                if args.profile_prefill:
                    from benchmarks._prefill_profile import stop_prefill_profile
                    result["diagnostic_profiles"] = llm.collective_rpc(stop_prefill_profile)
                result["seconds"] = seconds
                result["output_tokens"] = [list(x.outputs[0].token_ids) for x in outputs]
                result["first_token_logprobs"] = [x.outputs[0].logprobs[0][
                    x.outputs[0].token_ids[0]].logprob for x in outputs]
                metrics = [x.metrics for x in outputs]
                assert all(x is not None for x in metrics)
                scheduled = min(float(x.scheduled_ts) for x in metrics)
                first = max(float(x.first_token_ts) for x in metrics)
                last = max(float(x.last_token_ts) for x in metrics)
                result["prefill_seconds"] = first - scheduled
                result["prefill_tokens_per_second"] = args.batch_size * args.length / (first - scheduled)
                result["request_first_token_seconds"] = [float(x.first_token_ts) - scheduled for x in metrics]
                result["request_preemptions"] = [int(x.num_preemptions) for x in metrics]
                steps = args.decode_tokens - 1
                result["measured_decode_steps"] = steps
                result["decode_ms_per_step"] = (last - first) * 1000 / steps if steps else None
        result["post_generation_workers"] = llm.collective_rpc(audit_worker)
        if args.exact_prefill or args.exact_final_row:
            expected_tokens = 2 * args.batch_size * (1 if args.exact_final_row else args.length)
            assert all(layer["exact_prefill_tokens"] == expected_tokens
                       and layer["exact_prefill_histories"] == 0
                       for worker in result["post_generation_workers"] for layer in worker["mla"])
        if args.uncapped:
            assert all(pool["max_open_centroid_leaves"] is None
                       for worker in result["post_generation_workers"] for pool in worker["pools"])
        if args.projected_clustering:
            assert all(pool["projected_clustering_heads"] == 64 and pool["projected_clustering_calls"] > 0
                       for worker in result["post_generation_workers"] for pool in worker["pools"])
        if args.mode != "full" and args.batch_size == 1:
            expected_prefix_tokens = 2 * min(args.length, 16384) if args.native_prefix else 0
            assert all(layer["native_prefix_tokens"] == expected_prefix_tokens
                for worker in result["post_generation_workers"] for layer in worker["mla"]), (
                    "native attention did not stay confined to the two initial warmup/measured chunks")
        result["status"] = "complete"
    except Exception as exc:
        result.update(status="failed", error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        write_json(args.output, result)
        if llm is not None:
            close_llm(llm)
    print("GLM53_RESULT " + json.dumps({k: v for k, v in result.items()
                                      if k not in ("workers", "post_generation_workers", "output_tokens", "diagnostic_profiles")}), flush=True)


if __name__ == "__main__":
    main()
