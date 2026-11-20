"""Accumulate timed LoD prefill stages across one full ProLong request."""

import argparse
import json
import os
from pathlib import Path
from types import MethodType


def start_device_profile(model):
    import torch

    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    )
    profiler.start()
    model._lod_device_profiler = profiler


def finish_device_profile(model):
    import torch

    profiler = model._lod_device_profiler
    torch.cuda.synchronize()
    profiler.stop()
    del model._lod_device_profiler
    # Count GPU activities directly to avoid charging both operators and
    # their child kernels. Concurrent stream time can appear in both rows.
    kernels = {}
    for event in profiler.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            row = kernels.setdefault(event.name, dict(seconds=0., calls=0))
            row["seconds"] += event.device_time_total / 1.e6
            row["calls"] += 1
    return dict(sorted(kernels.items(), key=lambda item: -item[1]["seconds"]))


class RouteKernelTimer:
    """Time route dispatches in the active layer and context bucket."""

    def __init__(self):
        self.current = None

    def launch(self, name, function, *args, **kwargs):
        if self.current is None:
            return function(*args, **kwargs)
        import torch

        records, query_len, state_len = self.current
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function(*args, **kwargs)
        end.record()
        records.append((name, query_len, state_len, start, end))
        return result

    def instrument(self):
        import torch
        from lod_attention.kernels import threshold_prefill, threshold_prefill_streaming

        timer = self

        class TimedKernel:
            def __init__(self, kernel, name):
                self.kernel = kernel
                self.name = name

            def __getitem__(self, grid):
                def timed(*args, **kwargs):
                    return timer.launch(self.name, self.kernel[grid], *args, **kwargs)
                return timed

        for module, names in (
            (threshold_prefill, ("_means", "_select_grouped")),
            (threshold_prefill_streaming,
             ("_prepare_live_count", "_route_partials", "_route_cutoff",
              "_coarse_partials", "_compact_bits", "_merge_coarse")),
        ):
            for name in names:
                setattr(module, name, TimedKernel(getattr(module, name), "route" + name))

        if os.environ.get("LOD_AMD_PREFILL_SELECTOR") == "gluon":
            from lod_attention.kernels import threshold_prefill_select

            threshold_prefill_select.select_region_bits = TimedKernel(
                threshold_prefill_select.select_region_bits, "route_select_region_bits",
            )

        bmm = torch.bmm

        def timed_bmm(*args, **kwargs):
            name = "route_blas_qk" if kwargs.get("out_dtype") == torch.float32 else "route_blas_pv"
            return timer.launch(name, bmm, *args, **kwargs)

        torch.bmm = timed_bmm


def instrument(model, *, route_kernels=False):
    import torch

    engines = {}
    for module in model.modules():
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is not None:
            engines[id(pool.engine)] = pool.engine

    timer = RouteKernelTimer() if route_kernels else None
    if timer is not None:
        timer.instrument()

    def wrap(original, stage, records):
        def timed(engine, q, *args, **kwargs):
            if q.size(2) <= 1 or (stage == "remote_path" and engine._stage_inside_remote):
                return original(q, *args, **kwargs)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if stage == "remote_path":
                engine._stage_inside_remote = True
            time_route = stage == "route_and_coarse" and timer is not None
            if time_route:
                timer.current = records, q.size(2), kwargs.get("state_len")
            try:
                result = original(q, *args, **kwargs)
            finally:
                if time_route:
                    timer.current = None
                if stage == "remote_path":
                    engine._stage_inside_remote = False
            end.record()
            records.append((stage, q.size(2), kwargs.get("state_len"), start, end))
            return result
        return timed

    for engine in engines.values():
        engine._stage_totals = []
        engine._stage_inside_remote = False
        if os.environ.get("LOD_AMD_PREFILL_PROFILE_LEAVES") == "1":
            engine._lod_leaf_timing_events = {}
        for name, stage in (
            ("_two_level_attention", "remote_path"),
            ("_route_top_slots", "route_and_coarse"),
            ("_paged_leaf_attention", "exact_leaves_and_dispatch"),
        ):
            setattr(engine, name, MethodType(wrap(getattr(engine, name), stage, engine._stage_totals), engine))
    return len(engines)


def collect(model):
    import torch

    torch.cuda.synchronize()
    engines = {}
    for module in model.modules():
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is not None:
            engines[id(pool.engine)] = pool.engine

    records = {}
    for engine in engines.values():
        for stage, qlen, state_len, start, end in engine._stage_totals:
            stats = records.setdefault(stage, dict(seconds=0., calls=0, query_tokens=0))
            elapsed = start.elapsed_time(end) / 1000
            stats["seconds"] += elapsed
            stats["calls"] += 1
            stats["query_tokens"] += qlen
            if stage.startswith("route") and state_len is not None:
                buckets = stats.setdefault("by_state_len_seconds", {})
                buckets[state_len] = buckets.get(state_len, 0.) + elapsed
        engine._stage_totals.clear()
        events = getattr(engine, "_lod_leaf_timing_events", None)
        if events is not None:
            for name, intervals in events.items():
                stats = records.setdefault("leaf_" + name, dict(seconds=0., calls=0))
                stats["seconds"] += sum(begin.elapsed_time(end) for begin, end in intervals) / 1000
                stats["calls"] += len(intervals)
            events.clear()
    return records


def instrument_work(model):
    for module in model.modules():
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is not None:
            pool.engine._lod_leaf_work_stats = []


def collect_work(model):
    import torch

    engines = {}
    for module in model.modules():
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is not None:
            engines[id(pool.engine)] = pool.engine
    rows = []
    for engine in engines.values():
        rows.extend(engine._lod_leaf_work_stats)
        del engine._lod_leaf_work_stats
    names = (
        "query_region_pairs", "query_kv_pairs", "padded_query_kv_pairs",
        "tiled_kv_entries", "query_head_rows",
    )
    return dict(zip(names, torch.stack(rows).sum(0).tolist(), strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--length", type=int, default=65536)
    parser.add_argument("--audit-work", action="store_true")
    parser.add_argument("--route-kernels", action="store_true")
    parser.add_argument("--device-profile", action="store_true",
                        help="Record kernel activity on a separate warmed request")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    from functools import partial

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from benchmarks._identity import benchmark_identity
    from benchmarks._vllm import close_llm, default_gpu_memory_utilization, llm_kwargs
    from benchmarks.prolong import make_speed_prompts, timed_generate

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    prompts, metadata = make_speed_prompts(tokenizer, length=args.length, batch_size=1)
    kwargs = llm_kwargs(
        checkpoint=args.checkpoint, mode="two-tier", max_model_len=args.length + 18,
        batch_size=1, tensor_parallel_size=1,
        gpu_memory_utilization=default_gpu_memory_utilization(args.checkpoint, "two-tier"),
        full_attention_backend="ROCM_AITER_UNIFIED_ATTN",
    )
    params = SamplingParams(temperature=0, seed=0, max_tokens=2, ignore_eos=True)
    llm = LLM(**kwargs)
    try:
        _, warmup, _, _, _ = timed_generate(llm, prompts, params)
        (pools,) = llm.apply_model(partial(instrument, route_kernels=args.route_kernels))
        _, prefill, _, _, _ = timed_generate(llm, prompts, params)
        (stages,) = llm.apply_model(collect)
        if not {"remote_path", "route_and_coarse", "exact_leaves_and_dispatch"}.issubset(stages):
            raise RuntimeError(f"incomplete profile: {stages}")
        result = dict(
            checkpoint=args.checkpoint, length=args.length, lod_layers=pools, warmup_seconds=warmup,
            prefill_seconds=prefill, stages=stages, prompts=metadata,
            benchmark_identity=benchmark_identity(),
            options={k: v for k, v in os.environ.items() if k.startswith("LOD_AMD_")},
        )
        if args.audit_work:
            llm.apply_model(instrument_work)
            timed_generate(llm, prompts, params)
            (result["leaf_work"],) = llm.apply_model(collect_work)
        if args.device_profile:
            llm.apply_model(start_device_profile)
            _, profiled, _, _, _ = timed_generate(llm, prompts, params)
            (result["device_kernels"],) = llm.apply_model(finish_device_profile)
            result["device_profile_prefill_seconds"] = profiled
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        close_llm(llm)


if __name__ == "__main__":
    main()
