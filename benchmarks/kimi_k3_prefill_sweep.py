"""Measure matched K3 prefill and fixed-trace decode on the vLLM path."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from benchmarks._vllm import close_llm, llm_kwargs


# Spawned EngineCore workers import this module as well. A development-only
# watchdog locates a stuck startup without ptrace access to the child.
if os.environ.get("LOD_BENCHMARK_TRACE_TIMEOUT"):
    import faulthandler

    faulthandler.dump_traceback_later(
        float(os.environ["LOD_BENCHMARK_TRACE_TIMEOUT"]), repeat=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("full", "two-tier"), required=True)
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--decode-context-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--profile-length", type=int, default=None,
                        help="extra diagnostic pass after uninstrumented timing")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.lengths, args.batch_size, args.decode_tokens, args.repeats) < 1:
        raise ValueError("lengths, token counts, batch size, and repeats must be positive")
    if args.kv_cache_memory_bytes is not None and args.kv_cache_memory_bytes <= 0:
        raise ValueError("kv cache memory bytes must be positive")
    if args.profile_length is not None and args.profile_length not in args.lengths:
        raise ValueError("profile length must be one of the warmed measured lengths")
    os.environ["LOD_BENCHMARK_SYNC_PREFILL_CACHE"] = "1"
    # The local worker audit uses the same callable RPC as the ProLong runner.
    # Newer vLLM disables callable serialization unless explicitly enabled.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    from vllm import LLM, SamplingParams

    maximum = max(args.lengths)
    kwargs = llm_kwargs(
        checkpoint=args.checkpoint,
        mode=args.mode,
        max_model_len=maximum + args.decode_tokens + 8,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        decode_context_parallel_size=args.decode_context_parallel_size,
        dcp_comm_backend="ag_rs",
        gpu_memory_utilization=args.gpu_memory_utilization,
        full_attention_backend="ROCM_AITER_UNIFIED_ATTN",
    )
    kwargs.update(load_format="dummy", skip_tokenizer_init=True)
    if args.kv_cache_memory_bytes is not None:
        # Explicit native-cache sizing leaves room for several unfinished
        # request-owned LoD prefill shadows without changing the algorithm.
        kwargs["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    llm = LLM(**kwargs)
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.decode_tokens,
        seed=1234,
        ignore_eos=True,
        detokenize=False,
    )

    def prompts(length: int) -> list[dict[str, list[int]]]:
        return [
            {
                "prompt_token_ids": [
                    3 + ((position + request) % 997) for position in range(length)
                ]
            }
            for request in range(args.batch_size)
        ]

    try:
        measurements = {}
        result = {
            "checkpoint": args.checkpoint,
            "mode": args.mode,
            "batch_size": args.batch_size,
            "decode_tokens": args.decode_tokens,
            "tensor_parallel_size": args.tensor_parallel_size,
            "decode_context_parallel_size": args.decode_context_parallel_size,
            "repeats": args.repeats,
            "rotating_prefills": os.environ.get("LOD_BENCHMARK_ROTATE_PREFILLS") == "1",
            "rotating_cohort": int(os.environ.get("LOD_BENCHMARK_ROTATING_COHORT", args.batch_size)),
            "distributed_prefill_construction": os.environ.get("LOD_KIMI_DISTRIBUTED_PREFILL_BUILD") == "1",
            "cross_layer_prefill_group": os.environ.get("LOD_KIMI_CROSS_LAYER_PREFILL_GROUP"),
            "independent_local_dcp_prefill": (
                os.environ.get("LOD_KIMI_DCP_LOCAL_PREFILL") == "1"
            ),
            "replicated_summary_dcp_prefill": (
                os.environ.get("LOD_KIMI_DCP_SHARED_PREFILL") == "1"
            ),
            "measurements": measurements,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
            "worker_attention_audit_status": "pending",
            "measurement_status": "in_progress",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for length in args.lengths:
            # Different prefix lengths can select different DCP merge and
            # state-update specializations. Warm the exact measured length so
            # JIT compilation is never charged to serving latency.
            llm.generate(prompts(length), params, use_tqdm=False)
            elapsed_samples = []
            prefill_samples = []
            decode_samples = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                outputs = llm.generate(prompts(length), params, use_tqdm=False)
                elapsed_samples.append(time.perf_counter() - started)
                metrics = [output.metrics for output in outputs]
                if any(metric is None for metric in metrics):
                    raise RuntimeError("vLLM did not return request timing metrics")
                scheduled = min(float(metric.scheduled_ts) for metric in metrics)
                first_token = max(float(metric.first_token_ts) for metric in metrics)
                last_token = max(float(metric.last_token_ts) for metric in metrics)
                prefill_samples.append(first_token - scheduled)
                decode_steps = args.decode_tokens - 1
                if decode_steps:
                    decode_samples.append((last_token - first_token) / decode_steps)
            elapsed = statistics.median(elapsed_samples)
            prefill = statistics.median(prefill_samples)
            measurements[str(length)] = {
                "elapsed_seconds": elapsed,
                "prefill_seconds": prefill,
                "elapsed_samples_seconds": elapsed_samples,
                "prefill_samples_seconds": prefill_samples,
                "decode_ms_per_batch_step": (
                    1000.0 * statistics.median(decode_samples)
                    if decode_samples
                    else None
                ),
                "decode_samples_seconds_per_batch_step": decode_samples,
                "aggregate_prompt_tokens_per_second": (
                    args.batch_size * length / prefill
                ),
                "generated_token_ids": [output.outputs[0].token_ids for output in outputs],
            }
            print("KIMI_PREFILL_POINT " + json.dumps({
                "length": length, **measurements[str(length)],
            }), flush=True)
            # Keep completed points even if a later, larger cohort runs out
            # of memory. Pending audit status prevents treating a partial
            # artifact as a fully validated benchmark.
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        # Inspect the binary actually loaded by every worker, not just the
        # source's intended specialization name. A stale tile-max-only cache
        # once silently left seven candidate channels unwritten.
        from benchmarks.prolong import audit_worker_attention_mode

        result["measurement_status"] = "complete"
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        audits = llm.collective_rpc(audit_worker_attention_mode)
        result["worker_attention_audit"] = audits
        if args.mode == "two-tier" and maximum > 16_384:
            expected = {
                "CK_TILE_FMHA_ROUTE_QUERY_NORMALIZE": "0",
                "CK_TILE_FMHA_ROUTE_TOPK": "8",
                "CK_TILE_FMHA_ROUTE_GLOBAL_TOPK": "0",
                "CK_TILE_FMHA_ROUTE_TILE_MAX_ONLY": "0",
            }
            for audit in audits:
                fused = [
                    module for module in audit["loaded_kimi_lod_modules"]
                    if "_asyncbias_" in module["module"]
                ]
                if not fused or any(module["route_build_flags"] != expected for module in fused):
                    raise RuntimeError("measured Kimi worker did not load correct top-eight route/coarse")
        result["worker_attention_audit_status"] = "passed"
        if args.profile_length is not None:
            from benchmarks._prefill_profile import start_prefill_profile, stop_prefill_profile

            llm.collective_rpc(start_prefill_profile)
            llm.generate(prompts(args.profile_length), params, use_tqdm=False)
            result["diagnostic_gpu_profile"] = llm.collective_rpc(stop_prefill_profile)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        close_llm(llm)


if __name__ == "__main__":
    main()
