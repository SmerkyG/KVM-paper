"""Same-engine trained K3 B8/2K prefill tile comparison.

Keep score-only 64-key routing, exact top-eight refinement, head grouping,
global 16K construction, native MoE and approved G8 direct-state KDA fixed.
Only CK's query-row tile changes. Warmup/dispatch instrumentation is untimed;
one ordinary serving measurement per arm has no profiler or internal timers.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def configure_tile_environment(head_group):
    from benchmarks.kimi_k3_decode_power2 import LOD_ENV

    os.environ.update(LOD_ENV, VLLM_ALLOW_INSECURE_SERIALIZATION="1",
        LOD_KIMI_OWNER_PREFILL_HEAD_GROUP=str(head_group),
        LOD_BENCHMARK_ADMISSION_COHORT="8",
        LOD_BENCHMARK_SYNCHRONIZED_DECODE="0",
        LOD_BENCHMARK_SYNC_PREFILL_CACHE="1")


def set_tile(worker, tile, *, audit=False):
    from benchmarks import kimi_k3_subtile_route as module

    if not hasattr(worker, "_lod_tile_factory_original"):
        worker._lod_tile_factory_original = module.serving_subtile_factory
    module.serving_subtile_factory = worker._lod_tile_factory_original
    os.environ["LOD_KIMI_COARSE_QUERY_TILE"] = str(tile)
    worker._lod_tile_calls = {}
    if audit:
        def factory(**kwargs):
            if (kwargs.get("query_tile"), kwargs.get("score_only"), kwargs.get("reuse_max")) != (
                    tile, True, False):
                raise AssertionError(f"wrong current-path prefill specialization: {kwargs}")
            original = worker._lod_tile_factory_original(**kwargs)
            def call(q, k, *args, **kw):
                geometry = f"queries={q.size(1)},heads={q.size(2)},states={k.size(1)}"
                worker._lod_tile_calls[geometry] = worker._lod_tile_calls.get(geometry, 0) + 1
                return original(q, k, *args, **kw)
            return call
        module.serving_subtile_factory = factory
    return dict(rank=worker.rank, query_tile=tile, audit=audit)


def finish_tile_audit(worker):
    from benchmarks import kimi_k3_subtile_route as module

    module.serving_subtile_factory = worker._lod_tile_factory_original
    runner = worker.model_runner
    runtime = getattr(runner, "_vllm_lod_runtime", None)
    if runtime is None:
        runtime = getattr(getattr(runner, "model_state", None), "_vllm_lod_runtime", None)
    return dict(rank=worker.rank, query_tile=int(os.environ["LOD_KIMI_COARSE_QUERY_TILE"]),
                coarse_calls=worker._lod_tile_calls,
                owner_query_sizes={name: getattr(pool, "_kimi_owner_query_sizes", {})
                                   for name, pool in runtime.pools.items()})


def frozen_prompts(documents, length, batch=8):
    if len(documents) < batch or not all(documents):
        raise ValueError("insufficient frozen ProLong documents")
    prompts = []
    for row in range(batch):
        tokens, cursor = [], row
        while len(tokens) < length:
            tokens.extend(documents[cursor % len(documents)])
            cursor += batch
        prompts.append(dict(prompt_token_ids=tokens[:length]))
    return prompts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--weight-cache-id", required=True)
    p.add_argument("--real-token-cache", type=Path, required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[32768, 65536])
    p.add_argument("--tiles", type=int, nargs="+", choices=(64, 128), default=[128, 64])
    p.add_argument("--head-group", type=int, choices=(6, 12), default=12)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if min(args.lengths) < 32768 or any(n % 16384 for n in args.lengths):
        p.error("use global 16K multiples >=32K so routing actually executes")
    configure_tile_environment(args.head_group)
    import torch
    from vllm import LLM, SamplingParams
    from benchmarks._vllm import close_llm, llm_kwargs, write_json
    from benchmarks._kimi_owner_local_mla import prepare_owner_tp_mla
    from benchmarks.kimi_k3_kda_dense_prefill import select_kda
    from benchmarks.prolong import DATASET, DATASET_REVISION, SPEED_SHUFFLE_SEED, timed_generate, token_digest

    cached = torch.load(args.real_token_cache, map_location="cpu", weights_only=False)
    if (cached["dataset"], cached["revision"], cached["seed"]) != (
            DATASET, DATASET_REVISION, SPEED_SHUFFLE_SEED):
        raise ValueError("not the frozen ProLong speed corpus")
    kwargs = llm_kwargs(checkpoint=args.checkpoint, mode="two-tier",
        max_model_len=max(args.lengths) + 1042, batch_size=8,
        tensor_parallel_size=8, decode_context_parallel_size=8, dcp_comm_backend="ag_rs",
        gpu_memory_utilization=0.8, full_attention_backend="ROCM_AITER_UNIFIED_ATTN")
    kwargs.update(load_format="ipc_cache", skip_tokenizer_init=True,
        disable_custom_all_reduce=False, enable_expert_parallel=True,
        kv_cache_memory_bytes=3 << 30,
        quantization_config={"moe": {"weight": "int4_per_group_32"}},
        model_loader_extra_config=dict(auto_start=True, cache_id=args.weight_cache_id,
            backing_load_format="auto", broker_timeout=1800.0),
        compilation_config=dict(cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[8], max_cudagraph_capture_size=8))
    result = dict(status="starting", scope=__doc__, engine_config=kwargs,
        head_group=args.head_group, kda_prefill_baseline="gluon_paged_G8",
        production_changed=False, measurements={})
    def save():
        write_json(args.output, result)
    save()
    llm = None
    try:
        llm = LLM(**kwargs)
        result["kda_hooks"] = llm.collective_rpc(select_kda, args=("gluon_paged", 8))
        result["owner_preparation"] = llm.collective_rpc(prepare_owner_tp_mla)
        params = SamplingParams(temperature=0, seed=0, max_tokens=1,
                                ignore_eos=True, detokenize=False)
        for length in args.lengths:
            prompts = frozen_prompts(cached["documents"], length)
            points = result["measurements"][str(length)] = {}
            result.setdefault("prompt_token_sha256", {})[str(length)] = [
                token_digest(p["prompt_token_ids"]) for p in prompts]
            for tile in args.tiles:
                label = f"q{tile}"
                if label in points:
                    label += "_repeat"
                result.update(status="warming", active_length=length, active_tile=tile)
                llm.collective_rpc(set_tile, args=(tile,), kwargs={"audit": True})
                save()
                llm.generate(prompts, params, use_tqdm=False)
                audit = llm.collective_rpc(finish_tile_audit)
                if {a["rank"] for a in audit} != set(range(8)) or any(
                        not a["coarse_calls"] or any("queries=2048,heads=96," not in shape
                        for shape in a["coarse_calls"]) for a in audit):
                    raise AssertionError("warmup did not use B8/2K owner current-path coarse attention")
                result["status"] = "measuring"
                save()
                elapsed, prefill, _, ids, _, timing = timed_generate(llm, prompts, params)
                points[label] = dict(prefill_seconds=prefill, elapsed_seconds=elapsed,
                    generated_token_ids=ids, request_timings=timing, untimed_dispatch_audit=audit)
                if "q128" in points:
                    points[label]["same_first_tokens_as_q128"] = ids == points["q128"]["generated_token_ids"]
                    points[label]["prefill_speedup"] = points["q128"]["prefill_seconds"] / prefill
                print(f"OWNER_TILE_PREFILL length={length} tile={tile} seconds={prefill:.6f}", flush=True)
                save()
        result["status"] = "complete"
        save()
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        save()
        raise
    finally:
        if llm is not None:
            try:
                llm.collective_rpc(set_tile, args=(128,))
                llm.collective_rpc(select_kda, args=("current", 8))
            finally:
                close_llm(llm)


if __name__ == "__main__":
    main()
