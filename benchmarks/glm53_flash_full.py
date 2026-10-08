"""Matched trained GLM5.3-Flash TP4 speed, ProLong/LongBench or NIAH-S3 check.

Native attention is GLM's learned sparse attention, not all-history dense.
Weights remain native FP8; no fixture overrides or speculative model are used.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path


def audit_loaded_model(worker):
    from benchmarks.glm53_flash_fixture import audit_worker

    audit = audit_worker(worker)
    model = worker.model_runner.get_model()
    audit["daemon_resident_bytes"] = model._vllm_weight_cache_resident_bytes
    audit["daemon_endpoint"] = model._vllm_weight_cache_endpoint
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode", choices=("full", "two-tier", "three-tier-bf16", "three-tier-int4"), required=True)
    p.add_argument("--batch-size", type=int, choices=(1, 8), required=True)
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--length", type=int, default=65536)
    p.add_argument("--decode-tokens", type=int, default=1026)
    p.add_argument("--measure", choices=("speed", "niah-s3", "quality"), default="speed")
    p.add_argument("--quality-samples", type=int, default=8)
    p.add_argument("--sample-offset", type=int, default=8)
    p.add_argument("--longbench-samples", type=int, default=16)
    p.add_argument("--include-niah-s3", action="store_true",
                   help="Also run the matched NIAH smoke panel after quality, reusing the loaded model")
    p.add_argument("--preflight-only", action="store_true",
                   help="Prepare quality manifests without initializing the model")
    p.add_argument("--niah-samples", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--weight-cache-id", default="glm53-flash-tp4")
    p.add_argument("--kv-cache-gib", type=int, default=16)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--exact-all-history", action="store_true",
                   help="Benchmark-only dense MLA ablation; disable the learned indexer")
    prefill = p.add_mutually_exclusive_group()
    prefill.add_argument("--exact-prefill", action="store_true",
                   help="Quality-only isolation: exact all-history prefill, unchanged LoD decode")
    prefill.add_argument("--exact-final-row", action="store_true",
                   help="Quality-only isolation: exact attention on only the final prefill row")
    p.add_argument("--uncapped", action="store_true",
                   help="Quality-only isolation: remove leaf closure, keeping the same top-eight ranking")
    p.add_argument("--projected-clustering", action="store_true",
                   help="Quality-only diagnostic: cluster by mean cosine of all heads' projected keys")
    p.add_argument("--routing-diagnostic", action="store_true",
                   help="Capture real final-prefill queries; check dense equivalence and actual-leaf route oracles")
    p.add_argument("--diagnostic-save-dir", type=str,
                   help="Optional node-local location for captured tensors (large; never commit)")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.tp != 4 or min(args.length, args.kv_cache_gib, args.niah_samples,
                          args.max_new_tokens) < 1 or args.decode_tokens < 2:
        p.error("this matched panel requires TP4 and positive sizes / at least two outputs")
    if args.measure == "niah-s3" and args.niah_samples % args.batch_size:
        p.error("the matched NIAH cohort must contain complete execution batches")
    if args.measure == "quality" and (args.length <= 128 or args.quality_samples < 1 or
            args.longbench_samples < 3 or args.sample_offset < 0 or
            args.quality_samples % args.batch_size or args.longbench_samples % args.batch_size):
        p.error("quality length must exceed 128; panels must contain complete cohorts, and LB needs at least three samples")
    if args.preflight_only and args.measure != "quality":
        p.error("--preflight-only is for --measure quality")
    if args.include_niah_s3 and (args.measure != "quality" or args.niah_samples % args.batch_size):
        p.error("--include-niah-s3 requires quality and complete NIAH cohorts")
    if args.exact_all_history and args.mode != "full":
        p.error("--exact-all-history requires --mode full (no LoD)")
    if (args.exact_prefill or args.uncapped or args.exact_final_row or args.projected_clustering or args.routing_diagnostic) and (args.mode not in ("two-tier", "three-tier-bf16") or args.measure != "niah-s3"):
        p.error("LoD ablations require BF16 LoD --measure niah-s3 (not speed tests)")
    if (args.projected_clustering or args.routing_diagnostic) and args.mode != "two-tier":
        p.error("the projected-clustering/directory diagnostic currently requires two-tier")
    if args.exact_final_row and args.uncapped:
        p.error("--exact-final-row keeps the release cap unchanged; do not combine with --uncapped")
    if args.routing_diagnostic and (args.batch_size != 8 or args.niah_samples != 8 or
            args.exact_prefill or args.uncapped or args.exact_final_row or args.projected_clustering):
        p.error("routing diagnostic requires the ordinary matched eight-example LoD cohort")
    config = json.loads((Path(args.checkpoint) / "config.json").read_text())
    text = config.get("text_config", config)
    if (text.get("model_type"), text.get("num_hidden_layers"), text.get("n_routed_experts")) != (
            "glm5_next_text", 45, 288):
        p.error("expected the full 45-layer, 288-expert GLM5.3-Flash checkpoint")
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    os.environ["LOD_BENCHMARK_SYNC_PREFILL_CACHE"] = "1"
    spec = importlib.util.find_spec("aiter")
    if spec and spec.origin:
        configs = Path(spec.origin).parent / "configs"
        os.environ.setdefault("AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
                              str(configs / "a8w8_blockscale_tuned_gemm.csv"))
        # Avoid AITER's implicit merge into another user's root-owned /tmp
        # directory. Missing shapes still use the ordinary kernel defaults.
        os.environ.setdefault("AITER_CONFIG_FMOE", str(configs / "tuned_fmoe.csv"))
    from benchmarks._vllm import llm_kwargs, write_json, close_llm
    from benchmarks.prolong import (configure_synchronized_decode_environment,
        evaluate_speed, audit_worker_attention_mode, validate_worker_attention_mode)
    # Keep both quality arms on the same cohort schedule as the speed panel.
    # This also avoids this image's native GLM indexer mixed prefill/decode
    # offset bug. It does not pad prompts, force answers, or delay EOS after
    # prefill; shorter answers still finish naturally.
    configure_synchronized_decode_environment(enabled=True, batch_size=args.batch_size)
    max_outputs = args.decode_tokens if args.measure == "speed" else args.max_new_tokens
    kwargs = llm_kwargs(checkpoint=args.checkpoint, mode=args.mode,
        max_model_len=args.length + max_outputs + (16 if args.measure == "speed" else 128),
        batch_size=args.batch_size, tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.85, full_attention_backend="ROCM_AITER_MLA_SPARSE")
    kwargs.update(load_format="ipc_cache", enable_trace_replay=args.measure == "speed",
        enable_expert_parallel=True, disable_custom_all_reduce=False, seed=1234,
        kv_cache_memory_bytes=args.kv_cache_gib * 1024**3,
        model_loader_extra_config=dict(auto_start=False, cache_id=args.weight_cache_id,
            backing_load_format="auto", broker_timeout=1800.0),
        compilation_config=dict(cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=sorted(set((1, args.batch_size))),
            max_cudagraph_capture_size=args.batch_size))
    if args.enforce_eager:
        kwargs.update(enforce_eager=True, compilation_config={"mode": 0})
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
    result = dict(status="initializing",
                  kda_gluon_prefill=os.environ.get("LOD_GLM_KDA_PREFILL") == "1",
                  benchmark={"speed": "prolong-glm53-full-model",
                  "niah-s3": "niah-s3-glm53-full-model", "quality": "glm53-chat-and-prolong-quality"}[args.measure],
        checkpoint=args.checkpoint, mode=args.mode, batch_size=args.batch_size,
        tensor_parallel_size=args.tp, length=args.length, decode_tokens=args.decode_tokens,
        random_weights=False, weights="native FP8", weight_cache_id=args.weight_cache_id,
        attention_cache="INT4" if args.mode == "three-tier-int4" else "BF16",
        exact_all_history=args.exact_all_history,
        exact_prefill=args.exact_prefill, exact_final_row=args.exact_final_row, uncapped=args.uncapped,
        projected_clustering=args.projected_clustering,
        projected_leaves=os.environ.get("LOD_GLM_PROJECTED_LEAVES") == "1",
        routing_diagnostic=args.routing_diagnostic,
        reference=("exact all-history dense MLA; learned indexer disabled" if args.exact_all_history
                   else "native learned sparse attention (2048-token budget)"),
        scheduler_budget=kwargs["max_num_batched_tokens"],
        scheduler_chunk=kwargs["long_prefill_token_threshold"],
        synchronized_prefill_cohort=True,
        kv_cache_memory_bytes=kwargs["kv_cache_memory_bytes"],
        enforce_eager=args.enforce_eager, seed=1234, argv=sys.argv,
        aiter_fmoe_config=os.environ.get("AITER_CONFIG_FMOE"),
        aiter_gemm_config=os.environ.get("AITER_CONFIG_GEMM_A8W8_BLOCKSCALE"),
        hostname=platform.node(), measurements={})
    write_json(args.output, result)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    documents = None
    quality_panel = longbench_rows = None
    if args.measure == "quality":
        from benchmarks.prolong import select_quality_prompts, DATASET, DATASET_REVISION
        from benchmarks._glm53_quality import prepare_longbench, public_rows, common_quality_prefix
        from benchmarks import longbench_v2
        quality_panel, quality_length = common_quality_prefix(select_quality_prompts(
            tokenizer, length=args.length, samples=args.quality_samples,
            sample_offset=args.sample_offset, allow_short_documents=True))
        longbench_rows = prepare_longbench(tokenizer, max_input_tokens=args.length - 128,
            samples=args.longbench_samples)
        assert max(row["input_tokens"] for row in longbench_rows) + 32 < kwargs["max_model_len"]
        result.update(thinking=False, quality_samples=args.quality_samples,
            quality_effective_length=quality_length,
            sample_offset=args.sample_offset, longbench_samples=args.longbench_samples,
            prolong_dataset=DATASET, prolong_revision=DATASET_REVISION,
            prolong_chat_template=False, longbench_dataset=longbench_v2.DATASET,
            longbench_revision=longbench_v2.DATASET_REVISION,
            longbench_chat_template=True, longbench_max_content_tokens=args.length - 128,
            longbench_selection="metadata-stratified length bands/domain-spread smoke panel",
            longbench_max_new_tokens=32, longbench_guided_answers=True,
            prompt_manifest={"prolong": quality_panel[1], "longbench-v2": public_rows(longbench_rows)})
        result["status"] = "quality-preflight-complete"
        write_json(args.output, result)
        if args.preflight_only:
            return
    if args.measure == "niah-s3" or args.include_niah_s3:
        from benchmarks.niah_s3 import _load_ruler_generator, make_samples, evaluate_length
        from benchmarks.prolong import token_digest
        _, _, version = _load_ruler_generator()
        documents = make_samples(tokenizer, length=args.length,
                                 samples=args.niah_samples, sample_offset=0)
        tails = [tokenizer.decode(doc["prompt_token_ids"][-128:]) for doc in documents]
        assert all("<think></think>" in tail for tail in tails), "assistant thinking not closed"
        assert max(map(lambda doc: len(doc["prompt_token_ids"]), documents)) < kwargs["max_model_len"]
        niah_setup = dict(lm_eval_version=version, niah_samples=args.niah_samples,
            max_new_tokens=args.max_new_tokens, thinking=False,
            prompt_generation_seeds=dict(python=0, numpy=1234),
            prompt_manifest=[dict(index=doc["index"], target=doc["target"],
                input_tokens=len(doc["prompt_token_ids"]),
                token_sha256=token_digest(doc["prompt_token_ids"])) for doc in documents],
            prompt_suffix=tails[0])
        if args.include_niah_s3:
            result["niah_s3_setup"] = niah_setup
        else:
            result.update(niah_setup)
        write_json(args.output, result)
    llm = None
    try:
        from vllm import LLM
        llm = LLM(**kwargs)
        if args.routing_diagnostic:
            from functools import partial
            from benchmarks._glm53_routing_diagnostic import install_capture
            result["routing_diagnostic_capture"] = llm.collective_rpc(partial(install_capture, slot=1, queries=16))
        if args.projected_clustering:
            from benchmarks._glm53_projected_clustering import install_projected_key_clustering
            result["projected_clustering_weights"] = llm.collective_rpc(install_projected_key_clustering)
            assert all(len(w) == 11 and all(layer["heads"] == 64 for layer in w)
                       for w in result["projected_clustering_weights"])
        result["loaded_workers"] = llm.collective_rpc(audit_loaded_model)
        for worker in result["loaded_workers"]:
            assert len(worker["mla"]) == 11, "missing real-model MLA layers"
            assert any("float8" in name for name in worker["parameter_dtypes"]), "not native FP8"
            assert all(layer["heads"] == 16 and layer["direct_dim"] == 0
                       and layer["lod_pool"] == (args.mode != "full") for layer in worker["mla"])
            if args.exact_all_history:
                assert all(layer["backend"] == "TritonMLAImpl" and not layer["sparse"]
                           and not layer["indexer"]
                           and layer["prefill_backend"] == "AiterFlashAttnPrefillBackend"
                           for layer in worker["mla"])
        result["attention_audit"] = llm.collective_rpc(audit_worker_attention_mode)
        validate_worker_attention_mode(result["attention_audit"], mode=args.mode)
        from benchmarks.prolong import audit_worker_cohort_capacity, validate_cohort_capacity
        result["cohort_capacity"] = llm.collective_rpc(audit_worker_cohort_capacity)
        validate_cohort_capacity(result["cohort_capacity"], args.batch_size)
        if quality_panel is not None:
            from benchmarks.prolong import evaluate_quality
            from benchmarks._glm53_quality import evaluate_longbench
            for name in ("prolong", "longbench-v2"):
                result["status"] = "quality-evaluation:" + name
                def quality_progress(measured):
                    result["measurements"][name] = measured
                    write_json(args.output, result)
                write_json(args.output, result)
                if name == "prolong":
                    measured = evaluate_quality(llm, tokenizer, length=quality_length,
                        samples=args.quality_samples, sample_offset=args.sample_offset,
                        batch_size=args.batch_size, prompt_panel=quality_panel,
                        progress_callback=quality_progress)
                else:
                    measured = evaluate_longbench(llm, longbench_rows,
                        batch_size=args.batch_size, progress=quality_progress)
                quality_progress(measured)
            if args.include_niah_s3:
                result["status"] = "quality-evaluation:niah-s3"
                write_json(args.output, result)
                result["measurements"]["niah-s3"] = evaluate_length(llm, tokenizer,
                    length=args.length, samples=args.niah_samples, sample_offset=0,
                    batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                    documents=documents)
            result["post_quality_workers"] = llm.collective_rpc(audit_loaded_model)
            if result["kda_gluon_prefill"]:
                assert all(worker["kda_gluon_prefill"] and worker["kda_gluon_prefill_calls"] > 0
                    for worker in result["post_quality_workers"])
            if args.mode != "full":
                assert all(pool["prefill_topk"] == pool["decode_topk"] == 8 and
                           pool["max_open_centroid_leaves"] == 1024
                    for worker in result["post_quality_workers"] for pool in worker["pools"])
                if result["projected_leaves"]:
                    assert all(pool["projected_leaf_calls"] > 0
                        for worker in result["post_quality_workers"] for pool in worker["pools"])
            result["status"] = "complete"
            return
        if documents is not None:
            result["status"] = "quality-evaluation"
            write_json(args.output, result)
            result["measurements"][str(args.length)] = evaluate_length(llm, tokenizer,
                length=args.length, samples=args.niah_samples, sample_offset=0,
                batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                documents=documents)
            result["post_quality_workers"] = llm.collective_rpc(audit_loaded_model)
            if args.routing_diagnostic:
                from benchmarks._glm53_routing_diagnostic import analyze_worker
                result["status"] = "analyzing-trained-attention-tensors"
                write_json(args.output, result)
                result["routing_diagnostic_results"] = llm.collective_rpc(
                    partial(analyze_worker, save_dir=args.diagnostic_save_dir))
            for worker in result["post_quality_workers"]:
                assert all(pool["prefill_topk"] == pool["decode_topk"] == 8 for pool in worker["pools"])
                if args.projected_clustering:
                    assert all(pool["projected_clustering_heads"] == 64
                               and pool["projected_clustering_calls"] > 0 for pool in worker["pools"])
                if args.exact_prefill or args.exact_final_row:
                    expected_tokens = (len(documents) if args.exact_final_row else
                                       sum(len(doc["prompt_token_ids"]) for doc in documents))
                    assert all(layer["exact_prefill_tokens"] == expected_tokens
                               and layer["exact_prefill_histories"] == 0 for layer in worker["mla"])
                    if args.exact_final_row:
                        assert all(layer["exact_prefill_calls"] == len(documents) for layer in worker["mla"])
                if args.uncapped:
                    assert all(pool["max_open_centroid_leaves"] is None for pool in worker["pools"])
                elif args.mode != "full":
                    assert all(pool["max_open_centroid_leaves"] == 1024 for pool in worker["pools"])
            result["status"] = "complete"
            return
        result["status"] = "warmup-and-measurement"
        write_json(args.output, result)

        def progress(measurements):
            result["measurements"] = measurements
            write_json(args.output, result)

        result["measurements"] = evaluate_speed(llm, tokenizer, lengths=[args.length],
            batch_size=args.batch_size, samples=args.batch_size,
            decode_tokens=args.decode_tokens, repeats=1, seed=1234,
            fixed_decode_trace=True, retain_warmup_allocator=True,
            report_memory=True, progress_callback=progress)
        measured = result["measurements"][str(args.length)]
        result["post_speed_workers"] = llm.collective_rpc(audit_loaded_model)
        if result["kda_gluon_prefill"]:
            assert all(worker["kda_gluon_prefill"] and worker["kda_gluon_prefill_calls"] > 0
                for worker in result["post_speed_workers"])
        if result["projected_leaves"]:
            assert args.mode != "full"
            assert all(pool["projected_leaf_calls"] > 0
                for worker in result["post_speed_workers"] for pool in worker["pools"])
        if args.batch_size == 8:
            timing = measured["measured_batch_timings"][0][0]
            assert timing["last_token_spread_seconds"] == 0, "decode cohort finished in different steps"
            assert timing["all_requests_live_overlap_seconds"] == timing["decode_window_seconds"], (
                "not all eight requests remained live for the decode interval")
        if args.mode != "full" and args.decode_tokens == 1026 and args.length % 256 == 0:
            deltas = measured["measured_decode_update_counters"][0]
            assert len(deltas) == args.tp
            for worker in deltas:
                assert len(worker) == 11, "missing MLA update counters"
                assert all(value["catch_up_rows"] == 4 * args.batch_size
                           for value in worker.values()), "wrong global per-request update cadence"
        result["status"] = "complete"
    except Exception as exc:
        result.update(status="failed", error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        write_json(args.output, result)
        if llm is not None:
            close_llm(llm)
    print("GLM53_FULL_RESULT " + json.dumps({k: v for k, v in result.items()
        if k not in ("loaded_workers", "attention_audit")}), flush=True)


if __name__ == "__main__":
    main()
