"""Corrected TP8/DCP8 MLA-stack decode: serving timing, then a separate trace.

Dummy-weight fixture only. No full-model weights are loaded. The profiler
starts at the first real decode graph replay, never during prefill/capture;
its kernel durations explain work, but are not serving latency estimates.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path


def fixture_overrides(layers: int) -> dict:
    if layers < 1 or layers > 24:
        raise ValueError("fixture layers must be between one and 24")
    source = json.loads(Path("tests/fixtures/kimi-k3-mla-stack/config.json").read_text())
    return dict(num_hidden_layers=layers, first_k_dense_replace=layers,
                linear_attn_config=dict(source["linear_attn_config"],
                                        full_attn_layers=list(range(1, layers + 1))))


def audit_fixture(worker, layers: int):
    import torch

    decoders = [module for module in worker.model_runner.model.modules()
                if type(module).__name__ == "KimiDecoderLayer"]
    if len(decoders) != layers or any(
        not getattr(module, "_vllm_lod_attention_only_fixture", False)
        or not isinstance(module.mlp, torch.nn.Identity) for module in decoders
    ):
        raise RuntimeError("dense and LoD must both execute the attention-only fixture")
    runtime = getattr(worker.model_runner.model_state, "_vllm_lod_runtime", None) if (
        hasattr(worker.model_runner, "model_state")) else None
    geometry = {name: dict(head_dim=pool.head_dim,
                           centroid_tile=pool.engine.decode_route_group_size,
                           waves=pool.engine.decode_route_num_warps)
                for name, pool in runtime.pools.items()} if runtime is not None else {}
    return dict(rank=worker.rank, attention_only_layers=len(decoders), ffn_layers=0,
                router_geometry=geometry)


def arm_decode_trace(worker, batch: int, steps: int = 257):
    import torch

    manager = worker.model_runner.cudagraph_manager
    original = manager.run_fullgraph
    state = dict(original=original, manager=manager, profiler=None, replays=0,
                 profiler_stopped=False, population=None)
    worker._kimi_fixture_decode_trace = state

    def replay(descriptor):
        if descriptor.num_tokens != batch:
            raise RuntimeError("diagnostic decode graph does not match the live batch")
        if state["profiler"] is None and worker.rank == 0:
            torch.cuda.synchronize()
            state["profiler"] = torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            state["profiler"].start()
        state["replays"] += 1
        output = original(descriptor)
        if state["replays"] == steps:
            # Completed requests release their cache rows before collective_rpc.
            # Snapshot here, after the final diagnostic replay, never in the
            # uninstrumented serving measurement or inside the recorded trace.
            torch.cuda.synchronize()
            if state["profiler"] is not None:
                state["profiler"].stop()
                state["profiler_stopped"] = True
            state["population"] = route_population(worker)
        return output

    manager.run_fullgraph = replay


def summarize_trace(events, *, gpu_device):
    from benchmarks.kimi_k3_owner_diagnostic import interval_union_us

    kernels = defaultdict(lambda: dict(calls=0, summed_gpu_us=0.0))
    cpu = defaultdict(lambda: dict(calls=0, self_cpu_us=0.0))
    intervals = []
    for event in events:
        if event.device_type != gpu_device:
            if str(event.device_type).endswith("CPU"):
                cpu[event.name]["calls"] += 1
                cpu[event.name]["self_cpu_us"] += event.self_cpu_time_total
            continue
        if getattr(event, "is_user_annotation", False):
            continue
        entry = kernels[event.name]
        entry["calls"] += 1
        entry["summed_gpu_us"] += event.device_time_total
        intervals.append((event.time_range.start, event.time_range.end))
    span = (max(end for _, end in intervals) - min(start for start, _ in intervals)
            if intervals else 0.0)
    return dict(gpu_timeline_span_ms=span / 1000,
        gpu_activity_union_ms=interval_union_us(intervals) / 1000,
        kernels=sorted([dict(name=name, **entry) for name, entry in kernels.items()],
                       key=lambda entry: entry["summed_gpu_us"], reverse=True),
        cpu_events=sorted([dict(name=name, **entry) for name, entry in cpu.items()],
                          key=lambda entry: entry["self_cpu_us"], reverse=True)[:40])


def finish_decode_trace(worker, trace_path: str):
    import torch

    state = worker._kimi_fixture_decode_trace
    state["manager"].run_fullgraph = state["original"]
    result = dict(rank=worker.rank, actual_graph_replays=state["replays"],
                  live_route_population=state["population"])
    profiler = state["profiler"]
    if profiler is not None:
        torch.cuda.synchronize()
        if not state["profiler_stopped"]:
            profiler.stop()
        result.update(summarize_trace(profiler.events(), gpu_device=torch.autograd.DeviceType.CUDA))
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(path))
        result["trace_path"] = str(path)
    del worker._kimi_fixture_decode_trace
    return result


def route_population(worker):
    """Inspect completed caches outside timing; flag degenerate dummy routing."""
    import torch

    runtime = getattr(worker.model_runner.model_state, "_vllm_lod_runtime", None)
    populations = []
    if runtime:
        for name, pool in runtime.pools.items():
            pool = getattr(pool, "owner_decode_pool", pool)
            lengths = pool.state["page_cache"]["slot_lengths"].flatten()
            nonzero = lengths[lengths > 0].float()
            buffers = pool.dcp_decode_buffer_storage or {}
            populations.append(dict(layer=name, occupied_slots=int(nonzero.numel()),
                mean_leaves=float(nonzero.mean()) if nonzero.numel() else 0,
                max_leaves=int(nonzero.max()) if nonzero.numel() else 0,
                slots_above_cap=int((nonzero > 1024).sum()),
                state_lens=pool.state_lens.cpu().tolist(),
                buffers={name: dict(shape=list(value.shape), min=int(value.min()), max=int(value.max()))
                         for name, value in buffers.items()
                         if isinstance(value, torch.Tensor) and name in (
                             "gqa_union_counts", "gqa_union_token_counts")}))
    return dict(rank=worker.rank, layers=populations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "two-tier"), required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 8), required=True)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--length", type=int, default=65536)
    parser.add_argument("--profile-steps", type=int, default=257)
    parser.add_argument("--skip-profile", action="store_true",
                        help="Serving measurement only, without the separate diagnostic pass")
    layout = parser.add_mutually_exclusive_group()
    layout.add_argument("--owner-decode", action="store_true",
                        help="One request per GPU (already the B8 two-tier default)")
    layout.add_argument("--ordinary-dcp", action="store_true",
                        help="Explicit ordinary-DCP control instead of the B8 default")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "two-tier" and args.batch_size == 8 and not args.ordinary_dcp:
        args.owner_decode = True
    if args.length % 16384 or args.length < 16384 or not 257 <= args.profile_steps <= 1025:
        parser.error("use global 16K multiples and 257--1025 diagnostic steps")
    if args.owner_decode and (args.mode != "two-tier" or args.batch_size != 8):
        parser.error("owner decode requires B8 two-tier")
    from benchmarks._vllm import close_llm, llm_kwargs, write_json
    from benchmarks._decode_update_audit import read_decode_update_counters, decode_update_deltas
    from benchmarks.kimi_k3_decode_power2 import LOD_ENV
    from benchmarks.kimi_k3_prefill_sweep import audit_loaded_attention, timed_sweep_generate
    from benchmarks.prolong import configure_synchronized_decode_environment, token_digest
    from vllm import LLM, SamplingParams

    for name in tuple(os.environ):
        if name.startswith("LOD_KIMI_"):
            os.environ.pop(name)
    os.environ.update(LOD_ENV, VLLM_ALLOW_INSECURE_SERIALIZATION="1",
                      LOD_BENCHMARK_SYNC_PREFILL_CACHE="1")
    if not args.owner_decode:
        # Do not relabel the explicit ordinary-DCP control as request-owned B8.
        os.environ["LOD_KIMI_REQUEST_OWNER_PREFILL"] = "0"
    configure_synchronized_decode_environment(enabled=True, batch_size=args.batch_size)
    kwargs = llm_kwargs(checkpoint="tests/fixtures/kimi-k3-mla-stack", mode=args.mode,
        max_model_len=args.length + 1042, batch_size=args.batch_size,
        tensor_parallel_size=8, decode_context_parallel_size=8, dcp_comm_backend="ag_rs",
        gpu_memory_utilization=0.1, full_attention_backend="ROCM_AITER_UNIFIED_ATTN")
    kwargs.update(load_format="dummy", skip_tokenizer_init=True, enable_trace_replay=True,
        disable_custom_all_reduce=False, hf_overrides=fixture_overrides(args.layers),
        kv_cache_memory_bytes=(1 if args.batch_size == 1 else 3) << 30,
        compilation_config=dict(cudagraph_mode="FULL_DECODE_ONLY",
                                cudagraph_capture_sizes=[args.batch_size]))
    prompts = [dict(prompt_token_ids=[(i * 17 + row * 31) % 2048 for i in range(args.length)])
               for row in range(args.batch_size)]
    traces = [[(i * 19 + row * 23) % 2048 for i in range(1026)]
              for row in range(args.batch_size)]
    def parameters(outputs):
        return [SamplingParams(temperature=0, max_tokens=outputs, seed=0,
                ignore_eos=True, detokenize=False, trace_decode_token_ids=trace[:outputs])
                for trace in traces]

    result = dict(fixture_only=True, mode=args.mode, layers=args.layers, batch_size=args.batch_size,
        owner_decode=args.owner_decode,
        length=args.length, timed_decode_steps=1025, diagnostic_steps=args.profile_steps,
        input_policy="deterministic synthetic IDs; dummy weights, not quality evidence",
        prompt_sha256=[token_digest(row["prompt_token_ids"]) for row in prompts],
        environment={name: value for name, value in os.environ.items()
                     if name.startswith(("LOD_", "VLLM_", "TRITON_"))}, status="in_progress")
    def save():
        write_json(args.output, result)

    llm = None
    try:
        save()
        llm = LLM(**kwargs)
        result["fixture_audit"] = llm.collective_rpc(audit_fixture, args=(args.layers,))
        if args.owner_decode:
            from benchmarks._kimi_owner_local_mla import prepare_owner_tp_mla
            from benchmarks.kimi_k3_prefill_sweep import (owner_decode_counters,
                owner_decode_graph_audit, owner_decode_graph_replays, validate_owner_decode_counts)
            result["owner_preparation"] = llm.collective_rpc(prepare_owner_tp_mla)
            llm.collective_rpc(owner_decode_graph_replays)
        print("KIMI_FIXTURE_PHASE warmup", flush=True)
        timed_sweep_generate(llm, prompts, parameters(1026), synchronized_decode=True)
        print("KIMI_FIXTURE_PHASE measurement", flush=True)
        before = llm.collective_rpc(read_decode_update_counters)
        if args.owner_decode:
            owner_before = llm.collective_rpc(owner_decode_counters)
            graph_before = llm.collective_rpc(owner_decode_graph_replays)
        elapsed, prefill, decode, outputs, _, timing = timed_sweep_generate(
            llm, prompts, parameters(1026), synchronized_decode=True)
        if [list(output) for output in outputs] != traces:
            raise RuntimeError("timed fixture did not replay the fixed continuation")
        updates = decode_update_deltas(before, llm.collective_rpc(read_decode_update_counters))
        if args.mode == "two-tier" and not args.owner_decode and (len(updates) != 8 or any(
                len(worker) != args.layers or any(count != dict(catch_up_batches=4,
                    catch_up_rows=4 * args.batch_size) for count in worker.values()) for worker in updates)):
            raise RuntimeError("fixture did not measure four updates in every layer/rank")
        if args.owner_decode:
            result["owner_decode_update_counters"] = validate_owner_decode_counts(
                owner_before, llm.collective_rpc(owner_decode_counters), steps=1025, world_size=8)
            result["owner_graph_replays"] = [end - begin for begin, end in zip(
                graph_before, llm.collective_rpc(owner_decode_graph_replays), strict=True)]
            if result["owner_graph_replays"] != [1025] * 8:
                raise RuntimeError("owner fixture did not replay every live B8 decode graph")
            result["owner_graph_audit"] = llm.collective_rpc(owner_decode_graph_audit)
        result.update(prefill_seconds=prefill, elapsed_seconds=elapsed,
            decode_ms_per_batch_step=1000 * decode / 1025, measured_batch_timing=timing,
            measured_decode_update_counters=updates,
            worker_attention_audit=audit_loaded_attention(llm, mode=args.mode, length=args.length))
        save()
        if args.skip_profile:
            result["status"] = "complete"
            save()
            print("KIMI_FIXTURE_RESULT " + json.dumps(dict(mode=args.mode, batch=args.batch_size,
                owner=args.owner_decode, decode_ms_per_batch_step=result["decode_ms_per_batch_step"])), flush=True)
            return
        print("KIMI_FIXTURE_PHASE diagnostic", flush=True)
        llm.collective_rpc(arm_decode_trace, args=(args.batch_size, args.profile_steps))
        before = llm.collective_rpc(read_decode_update_counters)
        timed_sweep_generate(llm, prompts, parameters(args.profile_steps + 1), synchronized_decode=True)
        result["diagnostic_updates"] = decode_update_deltas(before, llm.collective_rpc(read_decode_update_counters))
        result["diagnostic_trace"] = llm.collective_rpc(finish_decode_trace,
            args=(str(args.output.with_suffix(".trace.json")),))
        if any(worker["actual_graph_replays"] != args.profile_steps for worker in result["diagnostic_trace"]):
            raise RuntimeError("trace did not capture the requested live decode graph count")
        result["diagnostic_scope"] = "separate instrumented pass; kernel sums are not serving latency"
        result["status"] = "complete"
        save()
        print("KIMI_FIXTURE_RESULT " + json.dumps(dict(mode=args.mode, batch=args.batch_size,
            layers=args.layers, decode_ms_per_batch_step=result["decode_ms_per_batch_step"])), flush=True)
    except Exception as error:
        result.update(status="failed", failure=str(error))
        save()
        raise
    finally:
        if llm is not None:
            close_llm(llm)


if __name__ == "__main__":
    main()
