"""Measure eight independent attention owners, not a distributed-MoE model.

Each GPU runs one full-head attention-stack fixture. The cohort wall interval
includes its simultaneous release and final drain. This isolates the proposed
request-owner attention layout; it does not simulate the Q/output transfers
or claim full K3 model performance.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path


def configure_owner_scratch(worker, head_group_limit):
    """Opt-in scratch bound; preserve all heads and all eight selected regions."""
    state = getattr(worker.model_runner, "model_state", None)
    runtime = getattr(state, "_vllm_lod_runtime", None)
    pools = getattr(runtime, "pools", {})
    if not pools:
        raise RuntimeError("request-owner scratch probe has no LoD pools")
    for pool in pools.values():
        heads = int(pool.query_heads)
        if head_group_limit < 1 or heads % head_group_limit:
            raise ValueError("scratch head group must divide the query heads")
        pool.engine._lod_kimi_prefill_head_group_limit = head_group_limit
        pool.engine._lod_kimi_reduce_prefill_routes = True
    return {"layers": len(pools), "head_group_limit": head_group_limit, "reduced_fine_routes": True}


def owner(args, rank, gpu, barrier, results):
    # Resolve visibility before importing anything that can initialize HIP.
    os.environ["ROCR_VISIBLE_DEVICES"] = str(gpu)
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["LOD_BENCHMARK_SYNC_PREFILL_CACHE"] = "1"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    from benchmarks._vllm import close_llm, llm_kwargs
    from benchmarks.kimi_k3_prefill_sweep import peak_memory, reset_peak_memory
    from vllm import LLM, SamplingParams

    llm = None
    try:
        kwargs = llm_kwargs(
            checkpoint=args.checkpoint, mode=args.mode,
            max_model_len=max(args.lengths) + 9, batch_size=1,
            tensor_parallel_size=1, decode_context_parallel_size=1,
            gpu_memory_utilization=0.1,
            full_attention_backend="ROCM_AITER_UNIFIED_ATTN",
        )
        kwargs.update(load_format="dummy", skip_tokenizer_init=True,
                      kv_cache_memory_bytes=args.kv_cache_memory_bytes)
        llm = LLM(**kwargs)
        if args.fine_scratch_heads:
            llm.collective_rpc(configure_owner_scratch, args=(args.fine_scratch_heads,))
        params = SamplingParams(temperature=0.0, max_tokens=1, seed=1234,
                                ignore_eos=True, detokenize=False)
        for length in args.lengths:
            prompt = [{"prompt_token_ids": [
                3 + ((position + rank) % 997) for position in range(length)
            ]}]
            if args.report_memory:
                llm.collective_rpc(reset_peak_memory)
            llm.generate(prompt, params, use_tqdm=False)
            warmup_memory = llm.collective_rpc(peak_memory) if args.report_memory else None
            if args.report_memory:
                llm.collective_rpc(reset_peak_memory)
            barrier.wait(timeout=600)  # All exact-shape warmups completed.
            barrier.wait(timeout=600)  # Parent starts the cohort clock first.
            start = time.perf_counter()
            output = llm.generate(prompt, params, use_tqdm=False)[0]
            end = time.perf_counter()
            if output.metrics is None:
                raise RuntimeError("owner has no request timing metrics")
            point = {
                "kind": "point", "rank": rank, "gpu": gpu, "length": length,
                "started": start, "finished": end,
                "request_prefill_seconds": (
                    output.metrics.first_token_ts - output.metrics.scheduled_ts
                ),
                "generated_token_ids": output.outputs[0].token_ids,
            }
            if args.report_memory:
                point["warmup_worker_memory"] = warmup_memory
                point["worker_memory"] = llm.collective_rpc(peak_memory)
            results.put(point)
            barrier.wait(timeout=600)  # No next-shape warmup overlaps timing.
        from benchmarks.prolong import audit_worker_attention_mode
        results.put({"kind": "audit", "rank": rank,
                     "audits": llm.collective_rpc(audit_worker_attention_mode)})
    except BaseException:
        results.put({"kind": "error", "rank": rank,
                     "traceback": traceback.format_exc()})
        barrier.abort()
        raise
    finally:
        if llm is not None:
            close_llm(llm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("full", "two-tier"), required=True)
    parser.add_argument("--lengths", nargs="+", type=int, required=True)
    parser.add_argument("--owners", type=int, default=8)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=3_221_225_472)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-refine", action="store_true")
    parser.add_argument("--direct-leaf-result", action="store_true")
    parser.add_argument("--report-memory", action="store_true",
                        help="worker memory counters outside the cohort clock")
    parser.add_argument("--fine-scratch-heads", type=int, default=0,
                        help="experimental head-group scratch bound and early LSE route reduction")
    args = parser.parse_args()
    if args.mode == "full" and (args.tile_refine or args.direct_leaf_result or args.fine_scratch_heads):
        parser.error("LoD kernel variants cannot be used for a dense control")
    if args.fine_scratch_heads < 0:
        parser.error("fine scratch heads cannot be negative")
    os.environ["LOD_KIMI_TILE_REFINE"] = "1" if args.tile_refine else "0"
    os.environ["LOD_KIMI_DIRECT_LEAF_RESULT"] = "1" if args.direct_leaf_result else "0"
    if min(args.owners, args.kv_cache_memory_bytes, *args.lengths) < 1:
        parser.error("owners, cache bytes, and lengths must be positive")
    config = json.loads((Path(args.checkpoint) / "config.json").read_text())
    if not config.get("lod_attention_only_fixture"):
        parser.error("this layout experiment requires the attention-only fixture")
    visibility = next((os.environ[name] for name in (
        "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
    ) if os.environ.get(name)), None)
    gpus = visibility.split(",") if visibility else list(range(args.owners))
    if len(gpus) < args.owners:
        parser.error("fewer visible GPUs than requested owners")
    context = mp.get_context("spawn")
    barrier = context.Barrier(args.owners + 1)
    results = context.Queue()
    processes = [context.Process(target=owner, args=(
        args, rank, gpus[rank], barrier, results,
    )) for rank in range(args.owners)]
    result = {
        "scope": "attention-stack owners; no MoE or Q/output transfers",
        "checkpoint": args.checkpoint, "mode": args.mode,
        "owners": args.owners, "tensor_parallel_size_per_owner": 1,
        "decode_context_parallel_size_per_owner": 1,
        "scheduler_chunk_per_owner": 16_384,
        "kv_cache_memory_bytes_per_owner": args.kv_cache_memory_bytes,
        "exact_centroid_tile_refinement": args.tile_refine,
        "copy_free_leaf_result": args.direct_leaf_result,
        "fine_scratch_head_limit": args.fine_scratch_heads or None,
        "early_fine_route_lse_reduction": bool(args.fine_scratch_heads),
        "measurements": {}, "worker_attention_audit_status": "pending",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for process in processes:
        process.start()
    try:
        for length in args.lengths:
            barrier.wait(timeout=600)
            started = time.perf_counter()
            barrier.wait(timeout=600)
            points = [results.get(timeout=600) for _ in processes]
            if any(point["kind"] != "point" for point in points):
                raise RuntimeError(f"owner failed: {points}")
            if {point["rank"] for point in points} != set(range(args.owners)):
                raise RuntimeError("did not receive one result from every owner")
            if any(point["length"] != length for point in points):
                raise RuntimeError("owner length mismatch")
            seconds = max(point["finished"] for point in points) - started
            measurement = {
                "cohort_wall_seconds": seconds,
                "aggregate_prompt_tokens_per_second": args.owners * length / seconds,
                "owner_points": sorted(points, key=lambda point: point["rank"]),
            }
            if args.report_memory:
                memories = [memory for point in points for memory in
                            point["warmup_worker_memory"] + point["worker_memory"]]
                measurement.update(
                    peak_allocated_gib=max(memory["peak_torch_allocated_bytes"] for memory in memories) / 2**30,
                    peak_reserved_gib=max(memory["peak_torch_reserved_bytes"] for memory in memories) / 2**30,
                )
            result["measurements"][str(length)] = measurement
            print("KIMI_OWNER_POINT " + json.dumps({
                "length": length, **measurement,
            }), flush=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            barrier.wait(timeout=600)
        audits = [results.get(timeout=600) for _ in processes]
        if any(item["kind"] != "audit" for item in audits):
            raise RuntimeError(f"owner audit failed: {audits}")
        if args.mode == "two-tier" and max(args.lengths) > 16_384:
            expected = {
                "CK_TILE_FMHA_ROUTE_QUERY_NORMALIZE": "0",
                "CK_TILE_FMHA_ROUTE_TOPK": "8",
                "CK_TILE_FMHA_ROUTE_GLOBAL_TOPK": "0",
                "CK_TILE_FMHA_ROUTE_TILE_MAX_ONLY": "1" if args.tile_refine else "0",
            }
            for item in audits:
                for audit in item["audits"]:
                    fused = [module for module in audit["loaded_kimi_lod_modules"]
                             if "_asyncbias_" in module["module"]]
                    if not fused or any(module["route_build_flags"] != expected
                                        for module in fused):
                        raise RuntimeError("owner did not load the correct top-eight route")
        result["worker_attention_audit"] = audits
        result["worker_attention_audit_status"] = "passed"
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        for process in processes:
            process.join(timeout=10)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
    if any(process.exitcode != 0 for process in processes):
        raise RuntimeError("an attention owner exited unsuccessfully")


if __name__ == "__main__":
    main()
