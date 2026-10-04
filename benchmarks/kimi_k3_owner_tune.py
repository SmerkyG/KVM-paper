"""Tune prefill on one resident fixture, with exact-shape warmups."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[32768, 65536])
    parser.add_argument("--variants", nargs="+",
                        choices=("default", "tile64", "tile128", "routed", "fused", "sparse", "incremental", "tile_refine", "direct", "refine_direct", "final_reclaim", "final_fence", "final_only", "reuse_allocator", "reuse_group1", "reuse_group8", "reuse_group12", "reuse_native_local", "reuse_distributed8", "reuse_overlap_projection"),
                        default=["default", "tile64", "tile128", "routed"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--decode-context-parallel-size", type=int, default=1)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=3_221_225_472)
    parser.add_argument("--profile-length", type=int,
                        help="separate diagnostic pass after canonical timings")
    args = parser.parse_args()
    config = json.loads((Path(args.checkpoint) / "config.json").read_text())
    if not config.get("lod_attention_only_fixture"):
        parser.error("owner tuning requires the attention-stack fixture")
    os.environ["LOD_BENCHMARK_SYNC_PREFILL_CACHE"] = "1"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    from benchmarks._vllm import close_llm, llm_kwargs
    from vllm import LLM, SamplingParams

    kwargs = llm_kwargs(
        checkpoint=args.checkpoint, mode="two-tier", max_model_len=max(args.lengths) + 9,
        batch_size=args.batch_size, tensor_parallel_size=args.tensor_parallel_size,
        decode_context_parallel_size=args.decode_context_parallel_size,
        gpu_memory_utilization=0.1, full_attention_backend="ROCM_AITER_UNIFIED_ATTN",
    )
    kwargs.update(load_format="dummy", skip_tokenizer_init=True,
                  kv_cache_memory_bytes=args.kv_cache_memory_bytes)
    # Local-worker environment changes below deliberately reuse the model and
    # its weights. No vLLM engine reload is needed between leaf-kernel choices.
    llm = LLM(**kwargs)
    params = SamplingParams(temperature=0.0, max_tokens=1, seed=1234,
                            ignore_eos=True, detokenize=False)
    choices = {
        "default": (32, 1, False), "tile64": (64, 2, False),
        "tile128": (128, 4, False), "routed": (32, 2, True),
        "fused": (32, 1, False),
        "sparse": (32, 1, False),
        "incremental": (32, 1, False),
        "tile_refine": (32, 1, False),
        "direct": (32, 1, False),
        "refine_direct": (32, 1, False),
        "final_reclaim": (32, 1, False),
        "final_fence": (32, 1, False),
        "final_only": (32, 1, False),
        "reuse_allocator": (32, 1, False),
    }

    def change_variant(worker, variant):
        from vllm_lod_plugin import runtime

        overlap_projection = variant == "reuse_overlap_projection"
        os.environ["LOD_KIMI_OVERLAP_LEAF_PROJECTION"] = "1" if overlap_projection else "0"
        if overlap_projection:
            variant = "reuse_allocator"
        distributed = variant == "reuse_distributed8"
        runtime._DISTRIBUTED_PREFILL_BUILD = distributed
        if distributed:
            variant = "reuse_group8"
        native_local = variant == "reuse_native_local"
        if native_local:
            variant = "reuse_allocator"
        os.environ["LOD_KIMI_NATIVE_LOCAL_PREFILL"] = "1" if native_local else "0"
        group_size = 4
        if variant.startswith("reuse_group"):
            group_size = int(variant.removeprefix("reuse_group"))
            variant = "reuse_allocator"
        # Keep the construction grouping explicit in this fixture-only tuner;
        # the generic module constant is 12 but K3's effective default is 4.
        runtime._CROSS_LAYER_PREFILL_GROUP_OVERRIDDEN = True
        runtime._CROSS_LAYER_PREFILL_GROUP = group_size
        block_m, warps, routed = choices[variant]
        os.environ["LOD_KIMI_LEAF_BLOCK_M"] = str(block_m)
        os.environ["LOD_KIMI_LEAF_WARPS"] = str(warps)
        os.environ["LOD_KIMI_ROUTED_LEAF_PROJECTION"] = "1" if routed else "0"
        os.environ["LOD_KIMI_FUSED_LEAF_KV"] = "1" if variant == "fused" else "0"
        os.environ["LOD_KIMI_SPARSE_LEAF_PROJECTION"] = "1" if variant == "sparse" else "0"
        os.environ["LOD_KIMI_INCREMENTAL_LEAF_PROJECTION"] = "1" if variant == "incremental" else "0"
        refined = variant in ("tile_refine", "refine_direct", "final_reclaim", "final_fence", "final_only", "reuse_allocator")
        direct = variant in ("direct", "refine_direct", "final_reclaim", "final_fence", "final_only", "reuse_allocator")
        os.environ["LOD_KIMI_TILE_REFINE"] = "1" if refined else "0"
        os.environ["LOD_KIMI_DIRECT_LEAF_RESULT"] = "1" if direct else "0"
        runtime._PREFILL_RECLAIM_INTERVAL = 0 if variant in ("final_reclaim", "final_only", "reuse_allocator") else 32768
        os.environ["LOD_KIMI_REUSE_PREFILL_ALLOCATOR"] = "1" if variant == "reuse_allocator" else "0"
        # Final construction still synchronizes through mandatory final
        # reclaim/workspace release. Only redundant intermediate measurement
        # fences are disabled; consumers retain their per-request event wait.
        os.environ["LOD_BENCHMARK_SYNC_PREFILL_CACHE"] = "0" if variant in ("final_fence", "final_only") else "1"

    def memory_point(worker):
        import torch

        runner = worker.model_runner
        runtime = getattr(runner, "_vllm_lod_runtime", None)
        if runtime is None:
            runtime = getattr(getattr(runner, "model_state", None),
                              "_vllm_lod_runtime", None)
        return {"allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "construction_group_size": (
                    runtime.cross_layer_prefill_group_size if runtime else None),
                "device_free_bytes": torch.cuda.mem_get_info()[0]}

    def reset_memory_point(worker):
        import torch

        torch.cuda.reset_peak_memory_stats()

    result = {"scope": "24-MLA fixture, no MoE; fresh requests with final construction included",
              "batch_size": args.batch_size, "tensor_parallel_size": args.tensor_parallel_size,
              "decode_context_parallel_size": args.decode_context_parallel_size,
              "rotating_prefills": os.environ.get("LOD_BENCHMARK_ROTATE_PREFILLS") == "1",
              "rotating_cohort": int(os.environ.get("LOD_BENCHMARK_ROTATING_COHORT", args.batch_size)),
              "kv_cache_memory_bytes": args.kv_cache_memory_bytes, "measurements": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for variant in args.variants:
            llm.collective_rpc(change_variant, args=(variant,))
            run_name = variant
            occurrence = 1
            while run_name in result["measurements"]:
                occurrence += 1
                run_name = f"{variant}_{occurrence}"
            points = result["measurements"][run_name] = {}
            for length in args.lengths:
                prompt = [{"prompt_token_ids": [3 + (position + request) % 997
                                                for position in range(length)]}
                          for request in range(args.batch_size)]
                llm.collective_rpc(reset_memory_point)
                llm.generate(prompt, params, use_tqdm=False)
                start = time.perf_counter()
                outputs = llm.generate(prompt, params, use_tqdm=False)
                elapsed = time.perf_counter() - start
                if any(output.metrics is None for output in outputs):
                    raise RuntimeError("owner has no request metrics")
                points[str(length)] = {
                    "wall_seconds": elapsed,
                    "prefill_seconds": (max(output.metrics.first_token_ts for output in outputs)
                                        - min(output.metrics.scheduled_ts for output in outputs)),
                    "generated_token_ids": [output.outputs[0].token_ids for output in outputs],
                }
                points[str(length)]["worker_memory"] = llm.collective_rpc(memory_point)
                print("KIMI_TUNE_POINT " + json.dumps({
                    "variant": run_name, "length": length, **points[str(length)],
                }), flush=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
        if args.profile_length is not None:
            if args.profile_length not in args.lengths:
                raise ValueError("profile length must have received an exact-shape warmup")
            from benchmarks._prefill_profile import start_prefill_profile, stop_prefill_profile
            llm.collective_rpc(start_prefill_profile, args=(False,))
            prompt = [{"prompt_token_ids": [3 + (position + request) % 997
                                            for position in range(args.profile_length)]}
                      for request in range(args.batch_size)]
            llm.generate(prompt, params, use_tqdm=False)
            result["diagnostic_profiles"] = llm.collective_rpc(stop_prefill_profile)
        from benchmarks.prolong import audit_worker_attention_mode
        result["worker_attention_audit"] = llm.collective_rpc(audit_worker_attention_mode)
        for audit in result["worker_attention_audit"]:
            fused = [module for module in audit["loaded_kimi_lod_modules"]
                     if "_asyncbias_" in module["module"]]
            if not fused:
                raise RuntimeError("measured fixture did not load fused Kimi route/coarse")
            for module in fused:
                flags = module["route_build_flags"]
                expected = {"CK_TILE_FMHA_ROUTE_QUERY_NORMALIZE": "0",
                            "CK_TILE_FMHA_ROUTE_TOPK": "8",
                            "CK_TILE_FMHA_ROUTE_GLOBAL_TOPK": "0",
                            "CK_TILE_FMHA_ROUTE_TILE_MAX_ONLY": (
                                "1" if module["module"].endswith("_v13") else "0")}
                if flags != expected:
                    raise RuntimeError("measured fixture loaded a stale route/coarse binary")
        result["worker_attention_audit_status"] = "passed"
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        close_llm(llm)


if __name__ == "__main__":
    main()
