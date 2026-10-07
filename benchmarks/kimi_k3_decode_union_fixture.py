"""Same-engine owner decode comparison: separate versus fused top8/union.

Default: dummy attention-only weights, not model-quality evidence. Optional
checkpoint/cache arguments run the trained full model without weight reloads.
Each arm has its
own audited FULL decode graph, untimed warmup, then one 1,025-step serving
measurement including four global-256 updates in all layers on every rank.
The first fused warmup replay additionally checks real live-cache outputs
against separate routing; that check never runs in the measured pass.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path


def select_union(worker, fused):
    """Recapture instead of silently replaying the previous Python path."""
    import torch
    from unittest.mock import patch
    from lod_attention.kernels import paged_decode
    import vllm_lod_plugin.pool as pool_module

    torch.cuda.synchronize()
    runtime = worker.model_runner.model_state._vllm_lod_runtime
    pools = [pool.owner_decode_pool for pool in runtime.pools.values()]
    for pool in pools:
        if pool.dcp_world_size != 1 or pool.query_heads != 96:
            raise RuntimeError("this candidate requires local request-owner routing")
        pool.engine._kimi_fuse_compact_union = fused
    counts, calls = Counter(), []

    class AuditKernel:
        def __init__(self, kernel, name):
            self.kernel, self.name = kernel, name

        def __getitem__(self, grid):
            launch = self.kernel[grid]
            def checked(*args, **kwargs):
                if kwargs.get("QUERY_HEADS") == 96:
                    key = self.name
                    if self.name == "router":
                        key += "_fused" if kwargs.get("FUSE_UNION_INIT") else "_separate"
                    elif self.name == "reduce":
                        key += "_fused" if kwargs.get("FUSE_UNION_BUILD") else "_separate"
                    counts[key] += 1
                return launch(*args, **kwargs)
            return checked

    original = pool_module.fused_decode_paged_lod_attention
    def remember_call(*args, **kwargs):
        if args[0].shape[1] == 96:
            calls.append((args, dict(kwargs)))
        return original(*args, **kwargs)

    manager = worker.model_runner.cudagraph_manager
    manager.graphs = {}
    manager._graphs_captured = False
    with patch.object(paged_decode, "_decode_route_coarse_gqa_groups_kernel",
            AuditKernel(paged_decode._decode_route_coarse_gqa_groups_kernel, "router")), \
         patch.object(paged_decode, "_reduce_decode_route_topk_kernel",
            AuditKernel(paged_decode._reduce_decode_route_topk_kernel, "reduce")), \
         patch.object(paged_decode, "_decode_topk_gqa_union_kernel",
            AuditKernel(paged_decode._decode_topk_gqa_union_kernel, "union")), \
         patch.object(pool_module, "fused_decode_paged_lod_attention", remember_call):
        worker.model_runner.capture_model()
    if not counts or not calls:
        raise RuntimeError("candidate kernels were not observed during graph capture")
    suffix = "fused" if fused else "separate"
    if not counts[f"router_{suffix}"] or not counts[f"reduce_{suffix}"]:
        raise RuntimeError(f"wrong router/reducer captured: {dict(counts)}")
    if (fused and counts["union"]) or (not fused and not counts["union"]):
        raise RuntimeError(f"wrong union launch captured: {dict(counts)}")
    worker._kimi_union_fixture_calls = calls
    worker._kimi_union_fixture_oracle = None
    return dict(rank=worker.rank, fused=fused, capture_launch_counts=dict(counts),
                captured_tokens=manager.captured_token_counts(),
                layers=len(pools), graph_recaptured=True)


def arm_live_union_check(worker):
    """Run a one-shot numerical oracle after the first untimed graph replay."""
    import torch
    from lod_attention.kernels.paged_decode import fused_decode_paged_lod_attention

    manager = worker.model_runner.cudagraph_manager
    original = manager.run_fullgraph
    def replay(descriptor):
        manager.run_fullgraph = original
        result = original(descriptor)
        # Captured calls have fixed pointers. Deduplicate by the per-layer
        # query allocation: capture_model may have executed the body twice.
        seen, checks = set(), []
        for args, kwargs in worker._kimi_union_fixture_calls:
            if args[0].data_ptr() in seen:
                continue
            seen.add(args[0].data_ptr())
            kwargs = dict(kwargs)
            # The graph has already inserted this token and the pool advanced
            # recent length once. Reevaluate the now-complete cache without
            # inserting or advancing anything again.
            kwargs.update(new_k=None, new_v=None, store_new_kv=False,
                          advance_local_lens=False)
            kwargs["gqa_union_fuse_compact_route"] = False
            reference = fused_decode_paged_lod_attention(*args, **kwargs).clone()
            buffers = kwargs["buffers"]
            reference_lse = buffers["kimi_gluon_final_lse"].clone()
            counts = buffers["gqa_union_counts"].clone()
            union = buffers["gqa_union_slots"].clone()
            kwargs["gqa_union_fuse_compact_route"] = True
            actual = fused_decode_paged_lod_attention(*args, **kwargs).clone()
            torch.testing.assert_close(actual, reference, atol=0.003, rtol=0.015)
            torch.testing.assert_close(buffers["kimi_gluon_final_lse"], reference_lse,
                                       atol=3e-5, rtol=1e-6)
            if not torch.equal(counts, buffers["gqa_union_counts"]):
                raise RuntimeError("fused union changed live selected counts")
            for row in range(counts.numel()):
                count = int(counts[row])
                if not torch.equal(union[row, :count].sort().values,
                        buffers["gqa_union_slots"][row, :count].sort().values):
                    raise RuntimeError("fused union changed its live selected set")
            checks.append(dict(max_abs=float((actual.float()-reference.float()).abs().max()),
                               heads=int(actual.shape[1])))
        torch.cuda.synchronize()
        expected = len(worker.model_runner.model_state._vllm_lod_runtime.pools)
        if len(checks) != expected:
            raise RuntimeError(f"live oracle checked {len(checks)} layers, expected {expected}")
        worker._kimi_union_fixture_oracle = dict(rank=worker.rank, layers=checks,
                                                scope="untimed warmup only")
        return result
    manager.run_fullgraph = replay


def live_union_check(worker):
    if worker._kimi_union_fixture_oracle is None:
        raise RuntimeError("live union check did not observe a graph replay")
    return worker._kimi_union_fixture_oracle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--length", type=int, default=65536)
    p.add_argument("--checkpoint")
    p.add_argument("--weight-cache-id")
    p.add_argument("--real-token-cache", type=Path)
    p.add_argument("--reference-baseline", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    trained = args.checkpoint is not None
    trained_inputs = (args.checkpoint, args.weight_cache_id, args.real_token_cache, args.reference_baseline)
    if any(trained_inputs) and not all(trained_inputs):
        p.error("trained comparisons require checkpoint, weight cache, token cache and dense baseline")
    if trained:
        args.layers = 24
    if args.length < 16384 or args.length % 16384:
        p.error("use global 16K multiples")
    from vllm import LLM, SamplingParams
    from benchmarks._vllm import close_llm, llm_kwargs
    from benchmarks._decode_update_audit import read_decode_update_counters, decode_update_deltas
    from benchmarks._kimi_owner_local_mla import prepare_owner_tp_mla
    from benchmarks.kimi_k3_decode_fixture import fixture_overrides, audit_fixture
    from benchmarks.kimi_k3_decode_power2 import LOD_ENV
    from benchmarks.kimi_k3_decode_candidates import save_result
    from benchmarks.kimi_k3_prefill_sweep import (timed_sweep_generate,
        owner_decode_counters, owner_decode_graph_audit, owner_decode_graph_replays,
        validate_owner_decode_counts)
    from benchmarks.prolong import configure_synchronized_decode_environment, token_digest

    for name in tuple(os.environ):
        if name.startswith("LOD_KIMI_"):
            os.environ.pop(name)
    os.environ.update(LOD_ENV, VLLM_ALLOW_INSECURE_SERIALIZATION="1",
                      LOD_BENCHMARK_SYNC_PREFILL_CACHE="1")
    configure_synchronized_decode_environment(enabled=True, batch_size=8)
    kwargs = llm_kwargs(checkpoint=args.checkpoint or "tests/fixtures/kimi-k3-mla-stack", mode="two-tier",
        max_model_len=args.length + 1042, batch_size=8,
        tensor_parallel_size=8, decode_context_parallel_size=8, dcp_comm_backend="ag_rs",
        gpu_memory_utilization=0.8 if trained else 0.1, full_attention_backend="ROCM_AITER_UNIFIED_ATTN")
    kwargs.update(load_format="dummy", skip_tokenizer_init=True, enable_trace_replay=True,
        disable_custom_all_reduce=False, hf_overrides=fixture_overrides(args.layers),
        kv_cache_memory_bytes=3 << 30,
        compilation_config=dict(cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[8]))
    if trained:
        import torch
        from transformers import AutoTokenizer
        from benchmarks.prolong import DATASET, DATASET_REVISION, SPEED_SHUFFLE_SEED, SEPARATOR
        from benchmarks.kimi_k3_prefill_sweep import reference_trace_prefix
        kwargs.pop("hf_overrides")
        kwargs.update(load_format="ipc_cache", enable_expert_parallel=True,
            quantization_config={"moe": {"weight": "int4_per_group_32"}},
            model_loader_extra_config=dict(auto_start=True, cache_id=args.weight_cache_id,
                backing_load_format="auto", broker_timeout=1800.0))
        cached = torch.load(args.real_token_cache, map_location="cpu", weights_only=False)
        if (cached["dataset"], cached["revision"], cached["seed"]) != (
                DATASET, DATASET_REVISION, SPEED_SHUFFLE_SEED):
            raise ValueError("not the frozen ProLong speed corpus")
        documents = cached["documents"]
        reference = json.loads(args.reference_baseline.read_text())["measurements"][str(args.length)]["prompts"]
        if len(reference) != 8:
            raise ValueError("reference must be a fully live B8 cohort")
        separator = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)(
            SEPARATOR, add_special_tokens=False)["input_ids"]
    prompts = [dict(prompt_token_ids=[(i*17+row*31) % 2048 for i in range(args.length)]) for row in range(8)]
    traces = [[(i*19+row*23) % 2048 for i in range(1026)] for row in range(8)]
    if trained:
        prompts, traces = [], []
        for record in reference:
            stream = []
            for index in record["panel_source_stream_indices"]:
                if stream:
                    stream.extend(separator)
                stream.extend(documents[index])
            if token_digest(stream[:args.length]) != record["token_sha256"]:
                raise RuntimeError("reference prompt mismatch")
            prompts.append(dict(prompt_token_ids=stream[:args.length]))
            traces.append(reference_trace_prefix(stream, record, length=args.length, output_tokens=1026))
    parameters = [SamplingParams(temperature=0, max_tokens=1026, seed=0,
        ignore_eos=True, detokenize=False, trace_decode_token_ids=trace) for trace in traces]
    result = dict(status="starting", fixture_only=not trained, layers=args.layers,
        batch_size=8, physical_batch_per_gpu=1, length=args.length,
        timed_decode_steps=1025, updates_per_layer=4, production_changed=False,
        prompt_sha256=[token_digest(row["prompt_token_ids"]) for row in prompts],
        trace_sha256=[token_digest(row) for row in traces], engine_config=kwargs,
        reference_baseline=str(args.reference_baseline) if trained else None,
        kda_prefill_baseline="gluon_paged_G8" if trained else None, arms={})
    save_result(args.output, result)
    llm = None
    try:
        llm = LLM(**kwargs)
        if not trained:
            result["fixture_audit"] = llm.collective_rpc(audit_fixture, args=(args.layers,))
        else:
            from benchmarks.kimi_k3_kda_dense_prefill import select_kda
            result["kda_hooks"] = llm.collective_rpc(select_kda, args=("gluon_paged", 8))
        result["owner_preparation"] = llm.collective_rpc(prepare_owner_tp_mla)
        llm.collective_rpc(owner_decode_graph_replays)
        # Initialize all owner caches before explicit variant captures.
        timed_sweep_generate(llm, prompts, parameters, synchronized_decode=True)
        for label, fused in (("separate", False), ("fused", True)):
            print("KIMI_UNION_FIXTURE_PHASE " + label, flush=True)
            capture = llm.collective_rpc(select_union, args=(fused,))
            if fused:
                llm.collective_rpc(arm_live_union_check)
            timed_sweep_generate(llm, prompts, parameters, synchronized_decode=True)
            numerical = llm.collective_rpc(live_union_check) if fused else None
            before = llm.collective_rpc(read_decode_update_counters)
            owner_before = llm.collective_rpc(owner_decode_counters)
            graph_before = llm.collective_rpc(owner_decode_graph_replays)
            elapsed, prefill, decode, outputs, _, timing = timed_sweep_generate(
                llm, prompts, parameters, synchronized_decode=True)
            if [list(output) for output in outputs] != traces:
                raise RuntimeError("fixed continuation was not replayed")
            updates = validate_owner_decode_counts(owner_before,
                llm.collective_rpc(owner_decode_counters), steps=1025, world_size=8)
            replays = [end-begin for begin, end in zip(graph_before,
                llm.collective_rpc(owner_decode_graph_replays), strict=True)]
            if replays != [1025]*8:
                raise RuntimeError("each rank must replay the fully live B8 graph 1,025 times")
            arm = dict(decode_ms_per_batch_step=1000*decode/1025,
                prefill_seconds=prefill, elapsed_seconds=elapsed, measured_batch_timing=timing,
                capture_audit=capture, numerical_check=numerical,
                owner_update_counters=updates, graph_replays=replays,
                all_update_counters=decode_update_deltas(before,
                    llm.collective_rpc(read_decode_update_counters)),
                owner_graph_audit=llm.collective_rpc(owner_decode_graph_audit))
            result["arms"][label] = arm
            save_result(args.output, result)
            print("KIMI_UNION_FIXTURE_RESULT " + json.dumps(dict(variant=label,
                decode_ms_per_batch_step=arm["decode_ms_per_batch_step"])), flush=True)
        result["status"] = "complete"
        save_result(args.output, result)
    except BaseException as exc:
        result.update(status="failed", failure=repr(exc))
        save_result(args.output, result)
        raise
    finally:
        if llm is not None:
            close_llm(llm)


if __name__ == "__main__":
    main()
